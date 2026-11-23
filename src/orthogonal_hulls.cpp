#include "orthogonal_hulls.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace lodestar::detail {
namespace {

using Point = std::array<float, 3>;
using Cell = std::array<std::size_t, 3>;
using Planes = std::array<std::vector<float>, 3>;
using Samples = std::array<std::vector<double>, 3>;

constexpr std::size_t maxCells = 1'000'000;
constexpr std::size_t maxTriangleSamples = 16'000'000;

struct ExactSum {
    std::array<double, 12> terms{};
    std::size_t count = 0;

    void add(double value) {
        std::size_t written = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const double sum = value + terms[i];
            const double rounded = sum - value;
            const double remainder = (value - (sum - rounded)) + (terms[i] - rounded);
            if (remainder != 0.0) terms[written++] = remainder;
            value = sum;
        }
        if (value != 0.0) terms[written++] = value;
        count = written;
    }

    void product(double a, double b) {
        const double value = a * b;
        add(std::fma(a, b, -value));
        add(value);
    }
};

int orientation(double ax, double ay, double bx, double by, double px, double py) {
    const double first = (bx - ax) * (py - ay);
    const double second = (by - ay) * (px - ax);
    const double determinant = first - second;
    const double roundoff = 8.0 * std::numeric_limits<double>::epsilon() * (std::abs(first) + std::abs(second));
    if (std::abs(determinant) > roundoff) return determinant > 0.0 ? 1 : -1;

    // Rays can hit a diagonal of the original tessellation. Resolve the sign
    // exactly for the stored floating-point coordinates, including ties.
    ExactSum exact;
    exact.product(ax, by);
    exact.product(bx, py);
    exact.product(px, ay);
    exact.product(-ay, bx);
    exact.product(-by, px);
    exact.product(-py, ax);
    return exact.count == 0 ? 0 : (exact.terms[exact.count - 1] > 0.0 ? 1 : -1);
}

struct BoundaryFace {
    std::array<uint32_t, 3> corners;
    std::size_t axis;
    int direction;
};

using Faces = std::array<std::vector<BoundaryFace>, 3>;

Faces collectFaces(const CollisionMesh& source, Planes& planes) {
    if (source.indices.empty() || source.indices.size() % 3 != 0) {
        throw std::invalid_argument("Expected orthogonal boundary triangles");
    }

    Faces faces;
    std::map<std::array<uint32_t, 2>, std::pair<std::size_t, int>> edgeOrientations;
    for (std::size_t i = 0; i < source.indices.size(); i += 3) {
        BoundaryFace face{{source.indices[i], source.indices[i + 1], source.indices[i + 2]}, 3, 0};
        for (const auto index : face.corners) {
            if (index >= source.positions.size() ||
                std::ranges::any_of(source.positions[index], [](float value) { return !std::isfinite(value); })) {
                throw std::invalid_argument("Invalid orthogonal boundary position");
            }
        }
        const auto& a = source.positions[face.corners[0]];
        const auto& b = source.positions[face.corners[1]];
        const auto& c = source.positions[face.corners[2]];
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (a[axis] == b[axis] && a[axis] == c[axis]) {
                face.axis = axis;
                break;
            }
        }
        if (face.axis == 3) {
            throw std::invalid_argument("Concave hull conversion requires axis-aligned boundary faces");
        }
        const auto u = (face.axis + 1) % 3;
        const auto v = (face.axis + 2) % 3;
        face.direction = orientation(a[u], a[v], b[u], b[v], c[u], c[v]);
        if (face.direction == 0) throw std::invalid_argument("Orthogonal boundary includes a degenerate triangle");

        // A coplanar patch may end only at a perpendicular grid plane.
        // Reject a folded patch whose adjacent coplanar faces point apart.
        for (std::size_t edge = 0; edge < face.corners.size(); ++edge) {
            const auto [low, high] = std::minmax(face.corners[edge], face.corners[(edge + 1) % 3]);
            const auto [entry, inserted] = edgeOrientations.try_emplace({low, high}, face.axis, face.direction);
            if (!inserted && entry->second.first == face.axis && entry->second.second != face.direction) {
                throw std::invalid_argument("Orthogonal boundary has an overlapping coplanar fold");
            }
        }

        if (face.direction < 0) std::swap(face.corners[1], face.corners[2]);
        faces[face.axis].push_back(face);
        planes[face.axis].push_back(a[face.axis]);
    }

    for (auto& axis : planes) {
        std::ranges::sort(axis);
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
        if (axis.size() < 2) throw std::invalid_argument("Orthogonal boundary encloses no volume");
    }
    return faces;
}

std::size_t cellIndex(const Cell& cell, const Cell& dimensions) {
    return (cell[0] * dimensions[1] + cell[1]) * dimensions[2] + cell[2];
}

bool covers(const std::array<std::array<double, 2>, 3>& triangle, double u, double v) {
    for (std::size_t edge = 0; edge < triangle.size(); ++edge) {
        const auto& a = triangle[edge];
        const auto& b = triangle[(edge + 1) % 3];
        const int side = orientation(a[0], a[1], b[0], b[1], u, v);
        // Shared edges belong to just one counterclockwise triangle.
        const bool onIncludedEdge = a[1] < b[1] || (a[1] == b[1] && a[0] > b[0]);
        if (side < 0 || (side == 0 && !onIncludedEdge)) return false;
    }
    return true;
}

void sweepAxis(const CollisionMesh& source, const Planes& planes, const Samples& samples,
               const std::vector<BoundaryFace>& faces, const Cell& dimensions, std::size_t axis,
               std::size_t& samplesRemaining, std::vector<uint8_t>& occupied) {
    const auto u = (axis + 1) % 3;
    const auto v = (axis + 2) % 3;
    const auto crossingIndex = [&](std::size_t plane, std::size_t row, std::size_t col) {
        return (row * dimensions[v] + col) * planes[axis].size() + plane;
    };
    std::vector<int8_t> crossings(planes[axis].size() * dimensions[u] * dimensions[v]);
    for (const auto& face : faces) {
        std::array<std::array<double, 2>, 3> triangle;
        for (std::size_t corner = 0; corner < triangle.size(); ++corner) {
            const auto& p = source.positions[face.corners[corner]];
            triangle[corner] = {p[u], p[v]};
            if (p[u] < planes[u].front() || p[u] > planes[u].back() ||
                p[v] < planes[v].front() || p[v] > planes[v].back()) {
                throw std::invalid_argument("Orthogonal boundary extends outside its grid planes");
            }
        }
        const auto offset = source.positions[face.corners[0]][axis];
        const auto plane = static_cast<std::size_t>(std::ranges::lower_bound(planes[axis], offset) - planes[axis].begin());
        const auto minU = std::min({triangle[0][0], triangle[1][0], triangle[2][0]});
        const auto maxU = std::max({triangle[0][0], triangle[1][0], triangle[2][0]});
        const auto minV = std::min({triangle[0][1], triangle[1][1], triangle[2][1]});
        const auto maxV = std::max({triangle[0][1], triangle[1][1], triangle[2][1]});
        const auto rowStart = static_cast<std::size_t>(std::ranges::lower_bound(samples[u], minU) - samples[u].begin());
        const auto rowEnd = static_cast<std::size_t>(std::ranges::upper_bound(samples[u], maxU) - samples[u].begin());
        const auto colStart = static_cast<std::size_t>(std::ranges::lower_bound(samples[v], minV) - samples[v].begin());
        const auto colEnd = static_cast<std::size_t>(std::ranges::upper_bound(samples[v], maxV) - samples[v].begin());
        const auto work = (rowEnd - rowStart) * (colEnd - colStart);
        if (work > samplesRemaining) throw std::length_error("Orthogonal hull conversion exceeds its triangle sampling limit");
        samplesRemaining -= work;
        for (auto row = rowStart; row < rowEnd; ++row) {
            for (auto col = colStart; col < colEnd; ++col) {
                if (!covers(triangle, samples[u][row], samples[v][col])) continue;
                auto& crossing = crossings[crossingIndex(plane, row, col)];
                if (crossing != 0) throw std::invalid_argument("Orthogonal boundary overlaps on a grid face");
                crossing = static_cast<int8_t>(face.direction);
            }
        }
    }

    Cell cell{};
    for (std::size_t row = 0; row < dimensions[u]; ++row) {
        cell[u] = row;
        for (std::size_t col = 0; col < dimensions[v]; ++col) {
            cell[v] = col;
            bool inside = false;
            for (std::size_t plane = 0; plane < planes[axis].size(); ++plane) {
                const auto crossing = crossings[crossingIndex(plane, row, col)];
                if (crossing != 0) {
                    if (inside != (crossing > 0)) {
                        throw std::invalid_argument("Orthogonal boundary lacks consistent solid winding");
                    }
                    inside = !inside;
                }
                if (plane == dimensions[axis]) break;
                cell[axis] = plane;
                auto& tile = occupied[cellIndex(cell, dimensions)];
                if (axis == 0) {
                    tile = inside;
                } else if ((tile != 0) != inside) {
                    throw std::invalid_argument("Orthogonal boundary has inconsistent interior across axes");
                }
            }
            if (inside) throw std::invalid_argument("Orthogonal boundary does not close");
        }
    }
}

CollisionMesh box(const Planes& planes, const Cell& start, const Cell& end) {
    const auto x0 = planes[0][start[0]];
    const auto y0 = planes[1][start[1]];
    const auto z0 = planes[2][start[2]];
    const auto x1 = planes[0][end[0]];
    const auto y1 = planes[1][end[1]];
    const auto z1 = planes[2][end[2]];
    return {{{x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0},
             {x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}},
            {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
             0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
             0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5}};
}

bool faceFilled(const Cell& start, const Cell& end, std::size_t axis,
                const Cell& dimensions, const std::vector<uint8_t>& occupied) {
    const auto u = (axis + 1) % 3;
    const auto v = (axis + 2) % 3;
    Cell cell{};
    cell[axis] = end[axis];
    for (cell[u] = start[u]; cell[u] < end[u]; ++cell[u]) {
        for (cell[v] = start[v]; cell[v] < end[v]; ++cell[v]) {
            if (!occupied[cellIndex(cell, dimensions)]) return false;
        }
    }
    return true;
}

std::vector<CollisionMesh> makeBoxes(const Planes& planes, const Cell& dimensions,
                                    std::vector<uint8_t> occupied, std::size_t maxHulls) {
    std::vector<CollisionMesh> hulls;
    Cell start{};
    for (start[0] = 0; start[0] < dimensions[0]; ++start[0]) {
        for (start[1] = 0; start[1] < dimensions[1]; ++start[1]) {
            for (start[2] = 0; start[2] < dimensions[2]; ++start[2]) {
                if (!occupied[cellIndex(start, dimensions)]) continue;
                if (hulls.size() == maxHulls) {
                    throw std::length_error("Exact orthogonal conversion exceeds the hull count limit");
                }
                Cell end{start[0] + 1, start[1] + 1, start[2] + 1};
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    while (end[axis] < dimensions[axis] && faceFilled(start, end, axis, dimensions, occupied)) {
                        ++end[axis];
                    }
                }
                Cell cell{};
                for (cell[0] = start[0]; cell[0] < end[0]; ++cell[0]) {
                    for (cell[1] = start[1]; cell[1] < end[1]; ++cell[1]) {
                        for (cell[2] = start[2]; cell[2] < end[2]; ++cell[2]) {
                            occupied[cellIndex(cell, dimensions)] = 0;
                        }
                    }
                }
                hulls.push_back(box(planes, start, end));
            }
        }
    }
    return hulls;
}

} // namespace

std::vector<CollisionMesh> decomposeOrthogonalSolid(const CollisionMesh& source, std::size_t maxHulls) {
    Planes planes;
    const auto faces = collectFaces(source, planes);
    Cell dimensions{};
    std::size_t cellCount = 1;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        dimensions[axis] = planes[axis].size() - 1;
        if (dimensions[axis] > maxCells / cellCount) {
            throw std::length_error("Orthogonal hull conversion exceeds its grid cell limit");
        }
        cellCount *= dimensions[axis];
    }

    Samples samples;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        samples[axis].reserve(dimensions[axis]);
        for (std::size_t i = 0; i < dimensions[axis]; ++i) {
            samples[axis].push_back(std::midpoint(double(planes[axis][i]), double(planes[axis][i + 1])));
        }
    }
    std::vector<uint8_t> occupied(cellCount);
    auto samplesRemaining = maxTriangleSamples;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        sweepAxis(source, planes, samples, faces[axis], dimensions, axis, samplesRemaining, occupied);
    }
    auto hulls = makeBoxes(planes, dimensions, std::move(occupied), maxHulls);
    if (hulls.empty()) throw std::invalid_argument("Orthogonal boundary encloses no volume");
    return hulls;
}

} // namespace lodestar::detail
