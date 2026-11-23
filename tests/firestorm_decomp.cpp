#include <windows.h>

#include "lodestar_firestorm_decomp.hpp"

#include <array>
#include <iostream>
#include <map>
#include <numeric>
#include <utility>

namespace {

using Cell = std::array<std::size_t, 3>;
using Planes = std::array<std::vector<float>, 3>;
using lodestar::firestorm::decompose;
using lodestar::firestorm::Hull;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void requireRejected(const auto& action) {
    try {
        action();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("The adapter accepted an unsupported input");
}

void forCells(const Planes& planes, const auto& visit) {
    for (std::size_t x = 0; x + 1 < planes[0].size(); ++x) {
        for (std::size_t y = 0; y + 1 < planes[1].size(); ++y) {
            for (std::size_t z = 0; z + 1 < planes[2].size(); ++z) visit(Cell{x, y, z});
        }
    }
}

// Emit just the outside faces of the occupied cells, sharing their indices.
Hull surface(const Planes& planes, const auto& occupied) {
    Hull result;
    std::map<Cell, uint32_t> corners;
    forCells(planes, [&](const Cell& cell) {
        if (!occupied(cell)) return;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto u = (axis + 1) % 3;
            const auto v = (axis + 2) % 3;
            for (const bool positive : {false, true}) {
                auto neighbor = cell;
                if (positive ? cell[axis] + 2 < planes[axis].size() : cell[axis] > 0) {
                    neighbor[axis] = positive ? cell[axis] + 1 : cell[axis] - 1;
                    if (occupied(neighbor)) continue;
                }
                const auto corner = [&](std::size_t i, std::size_t j) {
                    auto key = cell;
                    key[axis] += positive;
                    key[u] += i;
                    key[v] += j;
                    const auto [entry, added] = corners.try_emplace(key, static_cast<uint32_t>(corners.size()));
                    if (added) {
                        result.positions.push_back({{planes[0][key[0]], planes[1][key[1]], planes[2][key[2]]}});
                    }
                    return entry->second;
                };
                const auto a = corner(0, 0);
                auto b = corner(1, 0);
                const auto c = corner(1, 1);
                auto d = corner(0, 1);
                if (!positive) std::swap(b, d);
                result.indices.insert(result.indices.end(), {a, b, c, a, c, d});
            }
        }
    });
    return result;
}

using Point = std::array<float, 3>;

std::pair<Point, Point> verifiedBox(const Hull& hull, const Planes& planes) {
    require(hull.positions.size() == 8 && hull.indices.size() == 36, "Expected a closed eight-point box");
    Point low, high;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto [first, last] = std::ranges::minmax_element(hull.positions, {}, [axis](const auto& p) {
            return p.position[axis];
        });
        low[axis] = first->position[axis];
        high[axis] = last->position[axis];
        require(low[axis] < high[axis] && std::ranges::find(planes[axis], low[axis]) != planes[axis].end() &&
                    std::ranges::find(planes[axis], high[axis]) != planes[axis].end(),
                "Box bounds must match the source float coordinates");
    }
    for (std::size_t i = 0; i < hull.indices.size(); i += 3) {
        const auto& a = hull.positions[hull.indices[i]].position;
        const auto& b = hull.positions[hull.indices[i + 1]].position;
        const auto& c = hull.positions[hull.indices[i + 2]].position;
        const Point ab{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const Point ac{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const Point normal{ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                           ab[0] * ac[1] - ab[1] * ac[0]};
        float dot = 0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            dot += normal[axis] * (a[axis] - std::midpoint(low[axis], high[axis]));
        }
        require(dot > 0, "Hull face winding must stay outward across the module boundary");
    }
    return {low, high};
}

void verifyFixture(Planes planes, const auto& occupied, std::size_t expected,
                   const std::filesystem::path& modulePath) {
    // Match the viewer's per-axis normalization before crossing the DLL boundary.
    for (auto& axis : planes) {
        const float offset = (axis.front() + axis.back()) * -0.5f;
        const float scale = 1.0f / (axis.back() - axis.front());
        for (float& coordinate : axis) coordinate = (coordinate + offset) * scale;
    }
    auto source = surface(planes, occupied);
    const auto originalPositions = source.positions;
    const auto originalIndices = source.indices;
    const auto verifySource = [&] {
        require(source.indices == originalIndices, "Analyze changed source triangle indices");
        require(source.positions.size() == originalPositions.size(), "Analyze changed source vertex count");
        for (std::size_t i = 0; i < source.positions.size(); ++i) {
            require(std::ranges::equal(source.positions[i].position, originalPositions[i].position),
                    "Analyze changed source vertex positions");
        }
    };
    const auto hulls = decompose(source.positions, source.indices, expected, 256, modulePath);
    verifySource();
    const auto automatic = decompose(source.positions, source.indices, expected, modulePath);
    verifySource();
    require(automatic.size() == hulls.size(), "Automatic vertex selection changed the hull count");
    for (std::size_t i = 0; i < automatic.size(); ++i) {
        require(automatic[i].positions.size() == 8, "Automatic box conversion needs eight vertices");
        require(automatic[i].indices == hulls[i].indices, "Automatic vertex selection changed hull triangles");
        require(verifiedBox(automatic[i], planes) == verifiedBox(hulls[i], planes),
                "Automatic vertex selection changed the solid");
    }
    const auto moduleName = modulePath.empty() ? std::filesystem::path(L"lodestar_c.dll") : modulePath.filename();
    require(GetModuleHandleW(moduleName.c_str()) == nullptr, "Result must remain usable after Lodestar unloads");
    require(hulls.size() == expected, "Unexpected number of Lodestar physics hulls");
    std::vector<std::pair<Point, Point>> bounds;
    for (const auto& hull : hulls) bounds.push_back(verifiedBox(hull, planes));

    forCells(planes, [&](const Cell& cell) {
        const auto coverage = std::ranges::count_if(bounds, [&](const auto& box) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const auto center = std::midpoint(planes[axis][cell[axis]], planes[axis][cell[axis] + 1]);
                if (center <= box.first[axis] || center >= box.second[axis]) return false;
            }
            return true;
        });
        require(coverage == static_cast<int>(occupied(cell)), "Hull union changed occupied space");
    });

    // Feasible limits must preserve the same solid. Infeasible limits must fail
    // without distorting the source to force it into the requested budget.
    for (const std::size_t hullLimit : {expected, std::size_t{256}}) {
        for (const std::size_t vertexLimit : {8, 16, 256}) {
            const auto limited = decompose(source.positions, source.indices, hullLimit, vertexLimit, modulePath);
            verifySource();
            require(limited.size() == hulls.size(), "Changing feasible limits changed the hull count");
            for (std::size_t i = 0; i < limited.size(); ++i) {
                require(verifiedBox(limited[i], planes) == bounds[i], "Changing limits changed hull bounds");
                require(limited[i].indices == hulls[i].indices, "Changing limits changed hull triangles");
            }
        }
    }
    for (const std::size_t vertexLimit : {4, 7}) {
        requireRejected([&] { (void)decompose(source.positions, source.indices, expected, vertexLimit, modulePath); });
        verifySource();
    }
    requireRejected([&] { (void)decompose(source.positions, source.indices, expected - 1, 256, modulePath); });
    verifySource();
    const auto recovered = decompose(source.positions, source.indices, expected, 8, modulePath);
    require(recovered.size() == hulls.size(), "Analyze failed to recover after an infeasible budget");
    verifySource();
    for (std::size_t i = 0; i < source.indices.size(); i += 3) std::swap(source.indices[i], source.indices[i + 1]);
    requireRejected([&] { (void)decompose(source.positions, source.indices, expected, 256, modulePath); });
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path modulePath = argc > 1 ? argv[1] : "";
        const Planes doorway{{{-1.5f, -0.4f, 0.4f, 1.5f}, {0.0f, 2.0f, 2.25f}, {-0.125f, 0.125f}}};
        verifyFixture(doorway, [](const Cell& cell) { return cell[0] != 1 || cell[1] == 1; }, 3, modulePath);
        const Planes stairs{{{-0.5f, 0.5f}, {-0.25f, 0.0f, 0.17f, 0.34f, 0.51f}, {0.0f, 0.28f, 0.56f, 0.84f, 1.12f}}};
        verifyFixture(stairs, [](const Cell& cell) { return cell[1] <= cell[2]; }, 4, modulePath);
        const Hull octahedron{
            {{{1, 0, 0}}, {{-1, 0, 0}}, {{0, 1, 0}}, {{0, -1, 0}}, {{0, 0, 1}}, {{0, 0, -1}}},
            {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5}};
        const auto automatic = decompose(octahedron.positions, octahedron.indices, 1, modulePath);
        require(automatic.size() == 1 && automatic[0].positions.size() == 6 && automatic[0].indices.size() == 24,
                "Automatic vertex selection must follow the shape instead of a fixed vertex count");
        const Hull openFloor{{{{0, 0, 0}}, {{0, 0, 1}}, {{1, 0, 1}}, {{1, 0, 0}}}, {0, 1, 2, 0, 2, 3}};
        lodestar_coherent_collision_info info{};
        const auto cover = decompose(openFloor.positions, openFloor.indices, 256, 256, modulePath, .01, &info);
        require(!cover.empty() && info.exact_solid_conversion == 0 && info.surface_error_bound <= .01,
                "The bridge must accept coherent open elements with a bounded hull representation");
        for (const auto& hull : cover) {
            require(hull.positions.size() >= 4 && hull.positions.size() <= 256, "Surface cover violates hull format limits");
            require(std::ranges::all_of(hull.positions, [](const auto& p) { return p.position[1] <= 0; }),
                    "Bridge surface completion raised a floor contact");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
