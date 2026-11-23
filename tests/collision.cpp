#include <lodestar/lodestar.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace lodestar;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void appendQuad(Mesh& mesh, const std::array<glm::vec3, 4>& corners) {
    const auto first = static_cast<uint32_t>(mesh.vertices.size());
    for (const auto& corner : corners) {
        mesh.vertices.push_back({corner, glm::vec3(0.0f), glm::vec2(1.0f)});
    }
    mesh.indices.insert(mesh.indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
}

// Segment/triangle intersections for the axis-aligned test fixture.
bool blocksSegment(const CollisionMesh& mesh, const glm::dvec3& start, const glm::dvec3& end) {
    const auto direction = end - start;
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        std::array<glm::dvec3, 3> triangle;
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const auto& point = mesh.positions.at(mesh.indices[i + corner]);
            triangle[corner] = {point[0], point[1], point[2]};
        }
        const auto edgeA = triangle[1] - triangle[0];
        const auto edgeB = triangle[2] - triangle[0];
        const auto perpendicular = glm::cross(direction, edgeB);
        const auto denominator = glm::dot(edgeA, perpendicular);
        if (denominator == 0.0) continue;
        const auto offset = start - triangle[0];
        const auto across = glm::cross(offset, edgeA);
        const auto u = glm::dot(offset, perpendicular) / denominator;
        const auto v = glm::dot(direction, across) / denominator;
        const auto t = glm::dot(edgeB, across) / denominator;
        if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t >= 0.0 && t <= 1.0) return true;
    }
    return false;
}

void testDoorway() {
    Mesh source;
    source.hasUV = true;
    appendQuad(source, {{{-2, 0, 0}, {-1, 0, 0}, {-1, 0, 3}, {-2, 0, 3}}});
    appendQuad(source, {{{1, 0, 0}, {2, 0, 0}, {2, 0, 3}, {1, 0, 3}}});
    appendQuad(source, {{{-1, 0, 2}, {1, 0, 2}, {1, 0, 3}, {-1, 0, 3}}});
    appendQuad(source, {{{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}}});
    const auto collider = buildCollisionMesh(source);
    require(collider.indices == source.indices && collider.positions.size() == source.vertices.size(),
            "Collision preserves distinct source vertices, index winding and order");
    for (std::size_t i = 0; i < source.vertices.size(); ++i) {
        const auto& position = source.vertices[i].position;
        require(collider.positions[i] == std::array{position.x, position.y, position.z},
                "Collision follows source coordinates");
    }
    require(!blocksSegment(collider, {0, -1, 1}, {0, 1, 1}), "Doorway must remain open");
    require(blocksSegment(collider, {-1.5, -1, 1}, {-1.5, 1, 1}), "Wall must block passage");
    require(blocksSegment(collider, {0, -1, 2.5}, {0, 1, 2.5}), "Lintel must block passage");
    require(blocksSegment(collider, {0, 0.5, 1}, {0, 0.5, -1}), "Floor must support passage");
}

constexpr float stairRun = 0.32f;
constexpr float stairRise = 0.17f;

void requireStairContacts(const CollisionMesh& collider) {
    for (int step = 0; step < 4; ++step) {
        const double height = step * stairRise;
        const double center = (step + 0.5) * stairRun;
        require(blocksSegment(collider, {0.1, height + 0.01, center}, {0.1, height - 0.01, center}),
                "Each visible tread must support a contact at its authored height");
        if (step > 0) {
            const double previous = (step - 0.5) * stairRun;
            require(blocksSegment(collider, {0.1, height - 0.06, previous}, {0.1, height - 0.06, center}),
                    "Each visible riser must block movement below its next tread");
            require(!blocksSegment(collider, {0.1, height + 0.02, previous}, {0.1, height + 0.02, center}),
                    "The path just above each tread must stay clear");
        }
    }
}

void testStaircase() {
    Mesh source;
    for (int step = 0; step < 4; ++step) {
        const float start = step * stairRun;
        const float end = (step + 1) * stairRun;
        const float height = step * stairRise;
        appendQuad(source, {{{-0.5f, height, start}, {-0.5f, height, end},
                             {0.5f, height, end}, {0.5f, height, start}}});
        if (step > 0) {
            const float lower = (step - 1) * stairRise;
            appendQuad(source, {{{-0.5f, lower, start}, {-0.5f, height, start},
                                 {0.5f, height, start}, {0.5f, lower, start}}});
        }
    }
    requireStairContacts(buildCollisionMesh(source));
}

void requireInvalid(const Mesh& source) {
    try {
        (void)buildCollisionMesh(source);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Malformed collision input accepted");
}

void testDegeneratesAndValidation() {
    Mesh source;
    appendQuad(source, {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}});
    source.vertices.push_back({{2.0f, 0.0f, 0.0f}, glm::vec3(0.0f), glm::vec2(0.0f)});
    source.vertices.push_back({{std::numeric_limits<float>::infinity(), 0.0f, 0.0f},
                               glm::vec3(0.0f), glm::vec2(0.0f)});
    source.indices.insert(source.indices.end(), {0, 1, 4, 0, 0, 1});
    const auto collider = buildCollisionMesh(source);
    require(collider.positions.size() == 4 && collider.indices.size() == 6 && collider.skippedDegenerateTriangles == 2,
            "Zero-area triangles and unused positions stay out of the collider");

    source.indices = {0, 1, 4};
    const auto empty = buildCollisionMesh(source);
    require(empty.positions.empty() && empty.indices.empty() && empty.skippedDegenerateTriangles == 1,
            "A position-only line yields no collision triangles");
    require(buildCollisionMesh({}).indices.empty(), "Empty source yields an empty collider");

    source.indices = {0, 1};
    requireInvalid(source);
    source.indices = {0, 1, 6};
    requireInvalid(source);
    source.indices = {0, 1, 5};
    requireInvalid(source);
}

CollisionMesh denseBox(const glm::vec3& low, const glm::vec3& high) {
    Mesh source;
    for (int axis = 0; axis < 3; ++axis) {
        const int u = (axis + 1) % 3;
        const int v = (axis + 2) % 3;
        for (const bool upper : {false, true}) {
            const auto point = [&](int i, int j) {
                glm::vec3 p;
                p[axis] = upper ? high[axis] : low[axis];
                p[u] = std::lerp(low[u], high[u], i * 0.5f);
                p[v] = std::lerp(low[v], high[v], j * 0.5f);
                return p;
            };
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    std::array corners{point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)};
                    if (!upper) std::swap(corners[1], corners[3]);
                    appendQuad(source, corners);
                }
            }
        }
    }
    return buildCollisionMesh(source);
}

CollisionMesh clippedPrism() {
    // Clip a 1 cm bevel from the upright edge of a unit cube.
    constexpr std::array<std::array<float, 2>, 5> outline{{
        {0, 0}, {1, 0}, {1, 0.99f}, {0.99f, 1}, {0, 1},
    }};
    constexpr auto sides = static_cast<uint32_t>(outline.size());
    CollisionMesh hull;
    for (const float y : {0.0f, 1.0f}) {
        for (const auto& point : outline) hull.positions.push_back({point[0], y, point[1]});
    }
    for (uint32_t i = 0; i < sides; ++i) {
        const auto next = (i + 1) % sides;
        hull.indices.insert(hull.indices.end(), {i, i + sides, next + sides, i, next + sides, next});
    }
    for (uint32_t i = 1; i + 1 < sides; ++i) {
        hull.indices.insert(hull.indices.end(), {0, i, i + 1, sides, sides + i + 1, sides + i});
    }
    return hull;
}

void requireClosedHull(const CollisionMesh& mesh) {
    require(!mesh.positions.empty() && !mesh.indices.empty() && mesh.indices.size() % 3 == 0,
            "A simplified hull has complete triangles");
    glm::dvec3 center{0.0};
    const auto point = [&](uint32_t id) {
        const auto& p = mesh.positions.at(id);
        return glm::dvec3(p[0], p[1], p[2]);
    };
    for (const auto& p : mesh.positions) center += glm::dvec3(p[0], p[1], p[2]);
    center /= static_cast<double>(mesh.positions.size());
    std::map<std::pair<uint32_t, uint32_t>, std::pair<int, int>> edges;
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        const auto a = point(mesh.indices[i]);
        const auto b = point(mesh.indices[i + 1]);
        const auto c = point(mesh.indices[i + 2]);
        require(glm::dot(glm::cross(b - a, c - a), a - center) > 0,
                "A simplified hull has positive-area outward faces");
        for (std::size_t edge = 0; edge < 3; ++edge) {
            const auto from = mesh.indices[i + edge];
            const auto to = mesh.indices[i + (edge + 1) % 3];
            auto& [count, direction] = edges[std::minmax(from, to)];
            ++count;
            direction += from < to ? 1 : -1;
        }
    }
    for (const auto& [edge, uses] : edges) {
        require(uses == std::pair{2, 0}, "A simplified hull is watertight with consistent winding");
    }
}

void appendHull(CollisionMesh& destination, const CollisionMesh& mesh) {
    const auto offset = static_cast<uint32_t>(destination.positions.size());
    destination.positions.insert(destination.positions.end(), mesh.positions.begin(), mesh.positions.end());
    for (const auto index : mesh.indices) destination.indices.push_back(offset + index);
}

void testExactHullSimplification() {
    const auto dense = denseBox({-2, -1, 3}, {4, 5, 9});
    const auto simplified = simplifyConvexHull(dense, {.targetVertices = 8});
    require(simplified.metTarget && simplified.maxDeviation == 0 && simplified.mesh.positions.size() == 8
                && simplified.mesh.indices.size() == 36 && dense.indices.size() == 144,
            "Coplanar box tessellation reduces from 48 to 12 triangles with zero deviation");
    requireClosedHull(simplified.mesh);
    for (const float x : {-2.0f, 4.0f}) {
        for (const float y : {-1.0f, 5.0f}) {
            for (const float z : {3.0f, 9.0f}) {
                require(std::ranges::find(simplified.mesh.positions, std::array{x, y, z})
                            != simplified.mesh.positions.end(),
                        "Exact hull cleanup keeps each original box corner");
            }
        }
    }
}

void testHullErrorAndSupportLimits() {
    const auto bevel = clippedPrism();
    const auto tight = simplifyConvexHull(bevel, {.targetVertices = 9, .preserveVerticalSupport = false});
    require(!tight.metTarget && tight.mesh.positions.size() == 10 && tight.maxDeviation == 0,
            "The exact mode keeps a genuine bevel when the budget cannot be met without error");

    const auto loose = simplifyConvexHull(bevel, {.targetVertices = 9, .maxDeviation = 0.02,
                                                  .preserveVerticalSupport = false});
    require(loose.metTarget && loose.mesh.positions.size() <= 9 && loose.maxDeviation > 0
                && loose.maxDeviation <= 0.02,
            "An explicit tolerance permits bevel simplification within its deviation limit");
    requireClosedHull(loose.mesh);
    for (const auto& p : loose.mesh.positions) {
        require(std::ranges::find(bevel.positions, p) != bevel.positions.end(),
                "The approximation stays inside the original convex solid");
    }

    const auto protectedCap = simplifyConvexHull(bevel, {.targetVertices = 4, .maxDeviation = 2.0});
    require(!protectedCap.metTarget, "A protected five-corner tread cannot fit a four-vertex budget");
    for (const auto& p : bevel.positions) {
        if (p[1] == 1.0f) {
            require(std::ranges::find(protectedCap.mesh.positions, p) != protectedCap.mesh.positions.end(),
                    "The horizontal support face keeps its original footprint");
        }
    }
}

void testCompoundHullClearance() {
    CollisionMesh portal;
    for (const auto& piece : {
             denseBox({-1.5f, 0, -0.125f}, {-0.4f, 2.25f, 0.125f}),
             denseBox({0.4f, 0, -0.125f}, {1.5f, 2.25f, 0.125f}),
             denseBox({-0.4f, 2.0f, -0.125f}, {0.4f, 2.25f, 0.125f}),
         }) {
        const auto result = simplifyConvexHull(piece, {.targetVertices = 8});
        require(result.metTarget && result.mesh.positions.size() == 8, "Each doorway solid simplifies independently");
        appendHull(portal, result.mesh);
    }
    require(!blocksSegment(portal, {0, 1, -1}, {0, 1, 1}), "Simplified hulls preserve an open doorway");
    require(blocksSegment(portal, {0.75, 1, -1}, {0.75, 1, 1}), "Simplified hulls retain wall contact");
    require(blocksSegment(portal, {0, 2.125, -1}, {0, 2.125, 1}), "Simplified hulls retain lintel contact");
}

void testCompoundHullStaircase() {
    CollisionMesh stairs;
    for (int step = 0; step < 4; ++step) {
        const auto piece = denseBox({-0.5f, -0.25f, step * stairRun},
                                    {0.5f, step * stairRise, (step + 1) * stairRun});
        const auto result = simplifyConvexHull(piece, {.targetVertices = 8});
        require(result.metTarget && result.mesh.indices.size() == 36, "Each stair solid reduces to twelve triangles");
        appendHull(stairs, result.mesh);
    }
    requireStairContacts(stairs);
}

void requireInvalidHull(const CollisionMesh& mesh) {
    try {
        (void)simplifyConvexHull(mesh);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("A nonconvex or open mesh was accepted as one convex solid");
}

void testHullValidation() {
    auto open = denseBox({0, 0, 0}, {1, 1, 1});
    open.indices.resize(open.indices.size() - 3);
    requireInvalidHull(open);

    auto concave = denseBox({0, 0, 0}, {1, 1, 1});
    for (auto& p : concave.positions) {
        if (p == std::array{0.5f, 1.0f, 0.5f}) p[1] = 0.75f;
    }
    requireInvalidHull(concave);

    auto disconnected = denseBox({-2, 0, 0}, {-1, 1, 1});
    appendHull(disconnected, denseBox({1, 0, 0}, {2, 1, 1}));
    requireInvalidHull(disconnected);
}

using Cell = std::array<int, 3>;
using Grid = std::array<std::vector<float>, 3>;

// The fixture is one closed surface, with no faces inside the occupied cells.
CollisionMesh solidSurface(const Grid& planes, const auto& occupied) {
    Mesh source;
    const Cell size{static_cast<int>(planes[0].size()) - 1, static_cast<int>(planes[1].size()) - 1,
                    static_cast<int>(planes[2].size()) - 1};
    for (int x = 0; x < size[0]; ++x) {
        for (int y = 0; y < size[1]; ++y) {
            for (int z = 0; z < size[2]; ++z) {
                const Cell cell{x, y, z};
                if (!occupied(cell)) continue;
                for (int axis = 0; axis < 3; ++axis) {
                    const int u = (axis + 1) % 3;
                    const int v = (axis + 2) % 3;
                    for (const int step : {-1, 1}) {
                        auto neighbor = cell;
                        neighbor[axis] += step;
                        if (neighbor[axis] >= 0 && neighbor[axis] < size[axis] && occupied(neighbor)) continue;
                        const auto point = [&](int i, int j) {
                            glm::vec3 position;
                            position[axis] = planes[axis][cell[axis] + (step > 0 ? 1 : 0)];
                            position[u] = planes[u][cell[u] + i];
                            position[v] = planes[v][cell[v] + j];
                            return position;
                        };
                        std::array corners{point(0, 0), point(1, 0), point(1, 1), point(0, 1)};
                        if (step < 0) std::swap(corners[1], corners[3]);
                        appendQuad(source, corners);
                    }
                }
            }
        }
    }
    return buildCollisionMesh(source);
}

CollisionMesh requireExactBoxes(const std::vector<CollisionMesh>& hulls, const Grid& planes,
                                const auto& occupied) {
    CollisionMesh compound;
    std::vector<std::pair<glm::dvec3, glm::dvec3>> bounds;
    for (const auto& hull : hulls) {
        requireClosedHull(hull);
        require(hull.positions.size() == 8 && hull.indices.size() == 36, "Orthogonal pieces are closed boxes");
        const auto& first = hull.positions.front();
        glm::dvec3 low(first[0], first[1], first[2]);
        auto high = low;
        for (const auto& point : hull.positions) {
            const glm::dvec3 position(point[0], point[1], point[2]);
            low = glm::min(low, position);
            high = glm::max(high, position);
        }
        for (std::size_t axis = 0; axis < planes.size(); ++axis) {
            require(std::ranges::find(planes[axis], low[axis]) != planes[axis].end() &&
                        std::ranges::find(planes[axis], high[axis]) != planes[axis].end(),
                    "Box limits stay on original planes");
        }
        for (const auto& point : hull.positions) {
            for (std::size_t axis = 0; axis < planes.size(); ++axis) {
                require(point[axis] == low[axis] || point[axis] == high[axis], "Every box uses its bound corners");
            }
        }
        bounds.emplace_back(low, high);
        appendHull(compound, hull);
    }

    for (std::size_t x = 1; x < planes[0].size(); ++x) {
        for (std::size_t y = 1; y < planes[1].size(); ++y) {
            for (std::size_t z = 1; z < planes[2].size(); ++z) {
                const std::array cell{x, y, z};
                glm::dvec3 center;
                for (std::size_t axis = 0; axis < planes.size(); ++axis) {
                    center[axis] = 0.5 * (double(planes[axis][cell[axis] - 1]) + planes[axis][cell[axis]]);
                }
                const auto coverage = std::ranges::count_if(bounds, [&](const auto& box) {
                    return glm::all(glm::greaterThan(center, box.first)) && glm::all(glm::lessThan(center, box.second));
                });
                const int expected = occupied(Cell{static_cast<int>(x) - 1, static_cast<int>(y) - 1,
                                                   static_cast<int>(z) - 1}) ? 1 : 0;
                require(coverage == expected,
                        "Boxes cover each occupied cell once and leave empty cells open");
            }
        }
    }
    return compound;
}

void testIntegratedDoorwayConversion() {
    const Grid planes{{{-1.5f, -0.4f, 0.4f, 1.5f}, {0.0f, 2.0f, 2.25f}, {-0.125f, 0.125f}}};
    const auto occupied = [](const Cell& cell) { return cell[0] != 1 || cell[1] == 1; };
    const auto source = solidSurface(planes, occupied);
    requireInvalidHull(source);
    const auto compound = requireExactBoxes(decomposeCollisionHulls(source, 3), planes, occupied);
    require(!blocksSegment(compound, {0, 1, -1}, {0, 1, 1}), "Converted doorway remains open");
    require(blocksSegment(compound, {0.75, 1, -1}, {0.75, 1, 1}), "Converted doorway retains wall contact");
    require(blocksSegment(compound, {0, 2.125, -1}, {0, 2.125, 1}), "Converted doorway retains lintel contact");
    try {
        (void)decomposeCollisionHulls(source, 2);
    } catch (const std::length_error&) {
        return;
    }
    throw std::runtime_error("A hull budget below the exact doorway decomposition was accepted");
}

void testIntegratedStaircaseConversion() {
    Grid planes{{{-0.5f, 0.5f}, {-0.25f}, {0.0f}}};
    for (int step = 0; step < 4; ++step) {
        planes[1].push_back(step * stairRise);
        planes[2].push_back((step + 1) * stairRun);
    }
    const auto occupied = [](const Cell& cell) { return cell[1] <= cell[2]; };
    const auto source = solidSurface(planes, occupied);
    requireInvalidHull(source);
    const auto compound = requireExactBoxes(decomposeCollisionHulls(source), planes, occupied);
    requireStairContacts(compound);
}

void testCavityConversion() {
    const std::vector planesOnAxis{-1.0f, -0.25f, 0.25f, 1.0f};
    const Grid planes{planesOnAxis, planesOnAxis, planesOnAxis};
    const auto occupied = [](const Cell& cell) { return cell != Cell{1, 1, 1}; };
    const auto source = solidSurface(planes, occupied);
    const auto compound = requireExactBoxes(decomposeCollisionHulls(source), planes, occupied);
    require(!blocksSegment(compound, {-0.125, 0, 0}, {0.125, 0, 0}),
            "The inward-wound cavity shell preserves empty space");
    require(blocksSegment(compound, {0, 0, 0}, {1.25, 0, 0}), "The cavity wall remains solid");
}

void testDisconnectedConvexConversion() {
    auto source = denseBox({-4, 0, 0}, {-3, 1, 1});
    const CollisionMesh tetrahedron{{{3, 0, 0}, {4, 0.25f, 0}, {3, 1, 0.5f}, {3.25f, 0, 1}},
                                    {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3}};
    appendHull(source, tetrahedron);
    const auto hulls = decomposeCollisionHulls(source);
    require(hulls.size() == 2, "Separate solids on different axes stay in distinct convex pieces");
    require(std::ranges::any_of(hulls, [&](const auto& hull) {
                return hull.positions.size() == 4 && std::ranges::all_of(hull.positions, [&](const auto& point) {
                    return std::ranges::find(tetrahedron.positions, point) != tetrahedron.positions.end();
                });
            }),
            "A non-axis-aligned convex solid retains its corners");
    for (const auto& hull : hulls) requireClosedHull(hull);
}

void requireInvalidConversion(const CollisionMesh& source) {
    try {
        (void)decomposeCollisionHulls(source);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("An unsupported solid produced a compound collider");
}

void testCompoundConversionValidation() {
    auto open = denseBox({0, 0, 0}, {1, 1, 1});
    open.indices.resize(open.indices.size() - 3);
    requireInvalidConversion(open);

    auto slopedConcavity = denseBox({0, 0, 0}, {1, 1, 1});
    for (auto& point : slopedConcavity.positions) {
        if (point == std::array{0.5f, 1.0f, 0.5f}) point[1] = 0.75f;
    }
    requireInvalidConversion(slopedConcavity);
}

void testPointEnvelopesAndSmallSizePlans() {
    std::vector<std::array<float, 3>> points;
    for (float x : {0.f, .002f}) for (float y : {0.f, .004f}) for (float z : {0.f, .006f}) points.push_back({x, y, z});
    const auto original = points;
    const auto hull = buildConvexHull(points);
    requireClosedHull(hull);
    require(hull.positions.size() == 8 && hull.indices.size() == 36, "A millimetre-scale point cloud produces a valid box hull");
    const auto plan = planMeshSize(points);
    require(plan.requiresContainer && plan.requiresHullPhysics, "Small visible geometry needs a size container and hull physics");
    for (std::size_t axis = 0; axis < 3; ++axis) require(plan.containerSize[axis] == .01, "Every container axis reaches the SL minimum");
    for (const auto& point : points) {
        const glm::dvec3 p(point[0], point[1], point[2]);
        const auto normalized = (p - plan.center) / plan.containerSize;
        require(glm::length(plan.center + normalized * plan.containerSize - p) < 1e-15, "Container mapping preserves authored physical scale");
    }
    require(points == original, "Envelope and size planning do not mutate source vertices");
    const std::array atLimit{std::array{0.f, 0.f, 0.f}, std::array{.01f, .01f, .01f}};
    require(!planMeshSize(atLimit).requiresContainer, "Float representation of exactly 0.01 m must not trigger false padding");
    bool rejected = false;
    try { const std::array flat{std::array{0.f, 0.f, 0.f}, std::array{1.f, 0.f, 0.f}, std::array{0.f, 1.f, 0.f}, std::array{1.f, 1.f, 0.f}}; (void)buildConvexHull(flat); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "A flat point cloud is not presented as a valid volumetric hull");
}

void testCoherentOpenSurfaceShapes() {
    const CollisionMesh floor{{{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}}, {0, 1, 2, 0, 2, 3}};
    const auto original = floor.positions;
    const auto collider = buildCoherentCollisionShape(floor, {.01, 256});
    require(!collider.exactSolidConversion && collider.surfaceErrorBound <= .01, "Open surfaces have an explicit bounded cover report");
    require(!collider.hulls.empty(), "Open floor generates a coherent hull shape");
    for (const auto& hull : collider.hulls) {
        requireClosedHull(hull);
        for (const auto& point : hull.positions) require(point[1] <= 0, "Floor support height must not be raised");
    }
    require(floor.positions == original, "Source elements remain unchanged during shape generation");
    Mesh window;
    const std::array x{-1.f, -.5f, .5f, 1.f};
    const std::array y{0.f, .5f, 1.5f, 2.f};
    for (std::size_t row = 0; row < 3; ++row) for (std::size_t col = 0; col < 3; ++col) {
        if (row == 1 && col == 1) continue;
        appendQuad(window, {{{x[col], y[row], 0}, {x[col + 1], y[row], 0},
                             {x[col + 1], y[row + 1], 0}, {x[col], y[row + 1], 0}}});
    }
    const auto frame = buildCoherentCollisionShape(buildCollisionMesh(window), {.02, 256});
    bool openingBlocked = false, wallBlocked = false;
    for (const auto& hull : frame.hulls) {
        requireClosedHull(hull);
        openingBlocked |= blocksSegment(hull, {0, 1, -1}, {0, 1, 1});
        wallBlocked |= blocksSegment(hull, {-.75, 1, -1}, {-.75, 1, 1});
    }
    require(!openingBlocked && wallBlocked, "A coherent cover preserves a real opening and surrounding wall");
    auto rotated = floor;
    for (auto& point : rotated.positions) {
        const auto p = point;
        point = {.6f*p[0]-.8f*p[1], .48f*p[0]+.36f*p[1]-.8f*p[2], .64f*p[0]+.48f*p[1]+.6f*p[2]};
    }
    const auto sloped = buildCoherentCollisionShape(rotated, {.01, 256});
    require(sloped.surfaceErrorBound <= .01, "An arbitrarily oriented surface obeys the same metric error bound");
    for (const auto& hull : sloped.hulls) {
        requireClosedHull(hull);
        auto decoded = hull.positions;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            float low = std::numeric_limits<float>::infinity(), high = -low;
            for (const auto& p : decoded) { low = std::min(low, p[axis]); high = std::max(high, p[axis]); }
            for (auto& p : decoded) p[axis] = float(uint16_t((p[axis]-low)/(high-low)*65535.f))/65535.f*(high-low)+low;
        }
        requireClosedHull(buildConvexHull(decoded));
    }
    auto micro = floor;
    for (auto& point : micro.positions) for (auto& value : point) value *= 1e-4f;
    const auto tiny = buildCoherentCollisionShape(micro, {1e-6, 256});
    require(tiny.surfaceErrorBound <= 1e-6 && !tiny.hulls.empty(), "Surface completion is scale-aware below a millimetre");
    const std::vector planes{-1.f, -.25f, .25f, 1.f};
    auto hollow = solidSurface(Grid{planes,planes,planes}, [](const Cell& cell) { return cell != Cell{1,1,1}; });
    for (auto& p : hollow.positions) p[0] += .5f*p[1];
    const auto hollowShape = buildCoherentCollisionShape(hollow, {.02,256});
    for (const auto& hull : hollowShape.hulls) {
        bool containsCenter = true;
        for (std::size_t i=0;i<hull.indices.size();i+=3) {
            const auto p = [&](std::size_t j) { const auto& v=hull.positions[hull.indices[i+j]];return glm::dvec3(v[0],v[1],v[2]); };
            if (glm::dot(-p(0),glm::cross(p(1)-p(0),p(2)-p(0)))>1e-12) {containsCenter=false;break;}
        }
        require(!containsCenter,"An enclosing component must not fill a known inward-facing cavity");
    }
}

} // namespace

int main() {
    try {
        testDoorway();
        testStaircase();
        testDegeneratesAndValidation();
        testExactHullSimplification();
        testHullErrorAndSupportLimits();
        testCompoundHullClearance();
        testCompoundHullStaircase();
        testHullValidation();
        testIntegratedDoorwayConversion();
        testIntegratedStaircaseConversion();
        testCavityConversion();
        testDisconnectedConvexConversion();
        testCompoundConversionValidation();
        testPointEnvelopesAndSmallSizePlans();
        testCoherentOpenSurfaceShapes();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
