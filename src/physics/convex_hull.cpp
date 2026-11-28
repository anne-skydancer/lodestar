#include <lodestar/lodestar.hpp>

#include "orthogonal_hulls.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace lodestar {
namespace {

using Point = std::array<float, 3>;
using Face = std::array<uint32_t, 3>;
using Edge = std::array<uint32_t, 2>;
using Faces = std::vector<Face>;
constexpr auto missing = std::numeric_limits<std::size_t>::max();

glm::dvec3 position(const Point& point) {
    return {point[0], point[1], point[2]};
}

glm::dvec3 faceNormal(const std::vector<Point>& points, const Face& face) {
    const auto origin = position(points[face[0]]);
    return glm::cross(position(points[face[1]]) - origin, position(points[face[2]]) - origin);
}

int sign(double value) {
    return (value > 0.0) - (value < 0.0);
}

// The slow path sums exact terms when the ordinary determinant cannot resolve
// the sign. A product of two binary32 coordinates fits exactly in a double.
struct Expansion {
    std::array<double, 48> terms{};
    std::size_t count = 0;

    void add(double value) {
        std::size_t written = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const double sum = value + terms[i];
            const double rounded = sum - value;
            const double residual = (value - (sum - rounded)) + (terms[i] - rounded);
            if (residual != 0.0) terms[written++] = residual;
            value = sum;
        }
        if (value != 0.0) terms[written++] = value;
        count = written;
    }

    void addDeterminant(const Point& a, const Point& b, const Point& c, double direction) {
        for (glm::length_t axis = 0; axis < 3; ++axis) {
            const auto next = (axis + 1) % 3;
            const auto last = (axis + 2) % 3;
            addProduct(direction * a[axis], b[next], c[last]);
            addProduct(-direction * a[axis], b[last], c[next]);
        }
    }

    int result() const {
        return count ? sign(terms[count - 1]) : 0;
    }

private:
    void addProduct(double a, double b, double c) {
        const double ab = a * b;
        const double abc = ab * c;
        add(std::fma(ab, c, -abc));
        add(abc);
    }
};

int orientation(const Point& a, const Point& b, const Point& c, const Point& d) {
    const auto ab = position(b) - position(a);
    const auto ac = position(c) - position(a);
    const auto ad = position(d) - position(a);
    const auto first = glm::dvec3(ab.y * ac.z, ab.z * ac.x, ab.x * ac.y);
    const auto second = glm::dvec3(ab.z * ac.y, ab.x * ac.z, ab.y * ac.x);
    const double determinant = glm::dot(first - second, ad);
    const double scale = glm::dot(glm::abs(first) + glm::abs(second), glm::abs(ad));
    if (std::abs(determinant) > 16.0 * std::numeric_limits<double>::epsilon() * scale) {
        return sign(determinant);
    }
    if (scale == 0.0) return 0;
    Expansion exact;
    exact.addDeterminant(b, c, d, 1.0);
    exact.addDeterminant(a, c, d, -1.0);
    exact.addDeterminant(a, b, d, 1.0);
    exact.addDeterminant(a, b, c, -1.0);
    return exact.result();
}

int side(const std::vector<Point>& points, const Face& face, uint32_t vertex) {
    if (std::ranges::find(face, vertex) != face.end()) return 0;
    return orientation(points[face[0]], points[face[1]], points[face[2]], points[vertex]);
}

int turn(const Point& a, const Point& b, const Point& c, std::size_t x, std::size_t y) {
    Expansion exact;
    exact.add(double(a[x]) * b[y]);
    exact.add(double(b[x]) * c[y]);
    exact.add(double(c[x]) * a[y]);
    exact.add(-double(a[y]) * b[x]);
    exact.add(-double(b[y]) * c[x]);
    exact.add(-double(c[y]) * a[x]);
    return exact.result();
}

struct EdgeUse {
    std::size_t first;
    bool forward;
    std::size_t second = missing;
};

using EdgeMap = std::map<Edge, EdgeUse>;

Edge edgeKey(uint32_t first, uint32_t second) {
    return {std::min(first, second), std::max(first, second)};
}

std::vector<uint32_t> usedVertices(const Faces& faces, std::size_t vertexCount) {
    std::vector<bool> used(vertexCount);
    for (const auto& face : faces) {
        for (const auto vertex : face) used[vertex] = true;
    }
    std::vector<uint32_t> vertices;
    for (std::size_t i = 0; i < used.size(); ++i) {
        if (used[i]) vertices.push_back(static_cast<uint32_t>(i));
    }
    return vertices;
}

std::optional<EdgeMap> closedEdges(const Faces& faces) {
    if (faces.size() < 4) return std::nullopt;
    EdgeMap edges;
    for (std::size_t i = 0; i < faces.size(); ++i) {
        const auto& face = faces[i];
        if (face[0] == face[1] || face[0] == face[2] || face[1] == face[2]) return std::nullopt;
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const auto a = face[corner];
            const auto b = face[(corner + 1) % 3];
            const auto [entry, inserted] = edges.try_emplace(edgeKey(a, b), EdgeUse{i, a < b});
            if (!inserted) {
                if (entry->second.second != missing || entry->second.forward == (a < b)) return std::nullopt;
                entry->second.second = i;
            }
        }
    }
    if (std::ranges::any_of(edges, [](const auto& edge) { return edge.second.second == missing; })) {
        return std::nullopt;
    }
    return edges;
}

std::optional<EdgeMap> validateHull(const std::vector<Point>& points, const Faces& faces) {
    auto edges = closedEdges(faces);
    if (!edges) return std::nullopt;
    const auto vertices = usedVertices(faces, points.size());
    if (vertices.size() + faces.size() != edges->size() + 2) return std::nullopt;
    for (const auto& face : faces) {
        bool hasInterior = false;
        for (const auto vertex : vertices) {
            const int direction = side(points, face, vertex);
            if (direction > 0) return std::nullopt;
            hasInterior |= direction < 0;
        }
        if (!hasInterior) return std::nullopt;
    }
    return edges;
}

uint32_t oppositeCorner(const Face& face, const Edge& edge) {
    return *std::ranges::find_if(face, [&](uint32_t vertex) {
        return vertex != edge[0] && vertex != edge[1];
    });
}

std::vector<uint32_t> facetCorners(const std::vector<Point>& points, std::vector<uint32_t> vertices,
                                   const glm::dvec3& normal) {
    const auto axes = glm::abs(normal);
    auto dropped = axes.x > axes.y ? 0 : 1;
    if (axes.z > axes[dropped]) dropped = 2;
    std::size_t x = (dropped + 1) % 3;
    std::size_t y = (dropped + 2) % 3;
    if (normal[dropped] < 0.0) std::swap(x, y);
    std::ranges::sort(vertices, [&](uint32_t a, uint32_t b) {
        return std::pair{points[a][x], points[a][y]} < std::pair{points[b][x], points[b][y]};
    });
    vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());

    std::vector<uint32_t> corners;
    const auto append = [&](uint32_t next, std::size_t start) {
        while (corners.size() >= start + 2 &&
               turn(points[corners[corners.size() - 2]], points[corners.back()], points[next], x, y) <= 0) {
            corners.pop_back();
        }
        corners.push_back(next);
    };
    for (const auto vertex : vertices) append(vertex, 0);
    corners.pop_back();
    const auto start = corners.size();
    for (auto vertex = vertices.rbegin(); vertex != vertices.rend(); ++vertex) append(*vertex, start);
    corners.pop_back();
    return corners;
}

std::size_t groupRoot(std::vector<std::size_t>& groups, std::size_t index) {
    while (groups[index] != index) index = groups[index] = groups[groups[index]];
    return index;
}

Faces removeCoplanarSamples(const std::vector<Point>& points, const Faces& faces, const EdgeMap& edges) {
    std::vector<std::size_t> group(faces.size());
    std::iota(group.begin(), group.end(), 0);
    for (const auto& [edge, use] : edges) {
        if (side(points, faces[use.first], oppositeCorner(faces[use.second], edge)) == 0) {
            group[groupRoot(group, use.second)] = groupRoot(group, use.first);
        }
    }
    std::map<std::size_t, std::vector<uint32_t>> facets;
    for (std::size_t i = 0; i < faces.size(); ++i) {
        auto& vertices = facets[groupRoot(group, i)];
        vertices.insert(vertices.end(), faces[i].begin(), faces[i].end());
    }
    Faces result;
    for (auto& [face, vertices] : facets) {
        const auto corners = facetCorners(points, std::move(vertices), faceNormal(points, faces[face]));
        for (std::size_t i = 1; i + 1 < corners.size(); ++i) {
            result.push_back({corners[0], corners[i], corners[i + 1]});
        }
    }
    return result;
}

double squaredLength(const glm::dvec3& vector) {
    return glm::dot(vector, vector);
}

std::optional<std::array<uint32_t, 4>> tetrahedron(const std::vector<Point>& points,
                                                  const std::vector<uint32_t>& vertices) {
    if (vertices.size() < 4) return std::nullopt;
    const auto a = vertices.front();
    const auto origin = position(points[a]);
    const auto b = *std::ranges::max_element(vertices, {}, [&](uint32_t vertex) {
        return squaredLength(position(points[vertex]) - origin);
    });
    const auto edge = position(points[b]) - origin;
    const auto c = *std::ranges::max_element(vertices, {}, [&](uint32_t vertex) {
        return squaredLength(glm::cross(edge, position(points[vertex]) - origin));
    });
    const auto normal = glm::cross(edge, position(points[c]) - origin);
    const auto d = *std::ranges::max_element(vertices, {}, [&](uint32_t vertex) {
        return std::abs(glm::dot(normal, position(points[vertex]) - origin));
    });
    if (side(points, {a, b, c}, d) == 0) return std::nullopt;
    return std::array{a, b, c, d};
}

Faces convexSubset(const std::vector<Point>& points, const std::vector<uint32_t>& vertices) {
    const auto seed = tetrahedron(points, vertices);
    if (!seed) return {};
    const auto [a, b, c, d] = *seed;
    Faces faces{{a, b, c}, {a, c, d}, {a, d, b}, {b, d, c}};
    for (std::size_t i = 0; i < faces.size(); ++i) {
        const auto opposite = std::array{d, b, c, a}[i];
        if (side(points, faces[i], opposite) > 0) std::swap(faces[i][1], faces[i][2]);
    }
    for (const auto vertex : vertices) {
        if (std::ranges::find(*seed, vertex) != seed->end()) continue;
        std::map<Edge, Edge> horizon;
        const auto removed = std::remove_if(faces.begin(), faces.end(), [&](const Face& face) {
            if (side(points, face, vertex) <= 0) return false;
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const auto first = face[corner];
                const auto second = face[(corner + 1) % 3];
                const auto key = edgeKey(first, second);
                if (!horizon.erase(key)) horizon.emplace(key, Edge{first, second});
            }
            return true;
        });
        faces.erase(removed, faces.end());
        for (const auto& [key, directed] : horizon) {
            static_cast<void>(key);
            faces.push_back({directed[0], directed[1], vertex});
        }
    }
    return faces;
}

bool inside(const std::vector<Point>& points, const Faces& faces, uint32_t vertex) {
    return std::ranges::none_of(faces, [&](const Face& face) {
        return side(points, face, vertex) > 0;
    });
}

double segmentDistanceSquared(const glm::dvec3& point, const glm::dvec3& a, const glm::dvec3& b) {
    const auto edge = b - a;
    const auto offset = point - a;
    const double along = glm::dot(offset, edge) / squaredLength(edge);
    return squaredLength(offset - edge * std::clamp(along, 0.0, 1.0));
}

double triangleDistanceSquared(const glm::dvec3& point, const glm::dvec3& a,
                               const glm::dvec3& b, const glm::dvec3& c) {
    const auto ab = b - a;
    const auto ac = c - a;
    const auto offset = point - a;
    const auto normal = glm::cross(ab, ac);
    const double normalSquared = squaredLength(normal);
    const double u = glm::dot(glm::cross(offset, ac), normal) / normalSquared;
    const double v = glm::dot(glm::cross(ab, offset), normal) / normalSquared;
    if (u >= 0.0 && v >= 0.0 && u + v <= 1.0) {
        return squaredLength((offset - u * ab) - v * ac);
    }
    return std::min({segmentDistanceSquared(point, a, b), segmentDistanceSquared(point, b, c),
                     segmentDistanceSquared(point, c, a)});
}

std::optional<double> deviation(const std::vector<Point>& points, const Faces& faces,
                                const ConvexHullOptions& options, float minY, float maxY, double roundoff) {
    double largest = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (inside(points, faces, static_cast<uint32_t>(i))) continue;
        if (options.maxDeviation == 0.0 ||
            (options.preserveVerticalSupport && (points[i][1] == minY || points[i][1] == maxY))) {
            return std::nullopt;
        }
        double nearest = std::numeric_limits<double>::infinity();
        for (const auto& face : faces) {
            nearest = std::min(nearest, triangleDistanceSquared(position(points[i]), position(points[face[0]]),
                                                               position(points[face[1]]), position(points[face[2]])));
        }
        // Convexity of distance to the inner solid extends this vertex bound
        // to every point of the original solid.
        const double bound = std::sqrt(nearest) + roundoff;
        if (bound > options.maxDeviation) return std::nullopt;
        largest = std::max(largest, bound);
    }
    return largest;
}

CollisionMesh compactHull(const std::vector<Point>& points, const Faces& faces, std::size_t skipped) {
    CollisionMesh result;
    result.skippedDegenerateTriangles = skipped;
    std::vector<std::size_t> remap(points.size(), missing);
    result.indices.reserve(faces.size() * 3);
    for (const auto& face : faces) {
        for (const auto vertex : face) {
            if (remap[vertex] == missing) {
                remap[vertex] = result.positions.size();
                result.positions.push_back(points[vertex]);
            }
            result.indices.push_back(static_cast<uint32_t>(remap[vertex]));
        }
    }
    return result;
}

std::pair<std::vector<Point>, Faces> hullInput(const CollisionMesh& source, bool weld) {
    if (source.indices.size() % 3 != 0 || source.positions.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Invalid collision triangle list");
    }
    std::map<Point, uint32_t> unique;
    std::vector<Point> points;
    Faces faces;
    std::vector<std::size_t> remap(source.positions.size(), missing);
    faces.reserve(source.indices.size() / 3);
    for (std::size_t i = 0; i < source.indices.size(); i += 3) {
        Face face;
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const auto index = source.indices[i + corner];
            if (index >= source.positions.size()) throw std::invalid_argument("Convex hull index is out of range");
            if (remap[index] == missing) {
                const auto& point = source.positions[index];
                if (std::ranges::any_of(point, [](float value) { return !std::isfinite(value); })) {
                    throw std::invalid_argument("Convex hull position is not finite");
                }
                remap[index] = points.size();
                if (weld) {
                    const auto [entry, inserted] = unique.try_emplace(point, static_cast<uint32_t>(points.size()));
                    remap[index] = entry->second;
                    if (inserted) points.push_back(point);
                } else {
                    points.push_back(point);
                }
            }
            face[corner] = static_cast<uint32_t>(remap[index]);
        }
        faces.push_back(face);
    }
    return {std::move(points), std::move(faces)};
}

std::vector<Faces> connectedShells(const Faces& faces, const EdgeMap& edges) {
    std::vector<std::size_t> groups(faces.size());
    std::iota(groups.begin(), groups.end(), 0);
    for (const auto& [key, use] : edges) {
        groups[groupRoot(groups, use.second)] = groupRoot(groups, use.first);
    }
    std::vector<std::size_t> remap(faces.size(), missing);
    std::vector<Faces> shells;
    for (std::size_t i = 0; i < faces.size(); ++i) {
        auto& index = remap[groupRoot(groups, i)];
        if (index == missing) {
            index = shells.size();
            shells.emplace_back();
        }
        shells[index].push_back(faces[i]);
    }
    return shells;
}

} // namespace

ConvexHullResult simplifyConvexHull(const CollisionMesh& source, const ConvexHullOptions& options) {
    if (options.targetVertices < 4 || !std::isfinite(options.maxDeviation) || options.maxDeviation < 0.0) {
        throw std::invalid_argument("Invalid convex hull simplification input");
    }
    auto [points, sourceFaces] = hullInput(source, true);
    const auto sourceEdges = validateHull(points, sourceFaces);
    if (!sourceEdges) throw std::invalid_argument("Expected one closed, outward-wound convex collision solid");

    const auto [minY, maxY] = std::ranges::minmax(points | std::views::transform([](const Point& p) { return p[1]; }));
    auto best = removeCoplanarSamples(points, sourceFaces, *sourceEdges);
    ConvexHullOptions exact = options;
    exact.maxDeviation = 0.0;
    if (!validateHull(points, best) || !deviation(points, best, exact, minY, maxY, 0.0)) {
        best = std::move(sourceFaces);
    }
    auto result = compactHull(points, best, source.skippedDegenerateTriangles);
    double bestDeviation = 0.0;

    if (options.maxDeviation > 0.0 && result.positions.size() > options.targetVertices) {
        std::vector<float> coordinates;
        std::vector<uint8_t> locked;
        auto low = position(points.front());
        auto high = low;
        for (const auto& point : points) {
            low = glm::min(low, position(point));
            high = glm::max(high, position(point));
        }
        const auto span = high - low;
        const auto center = low + span * 0.5;
        const double scale = std::scalbn(1.0, std::ilogb(std::max({span.x, span.y, span.z})) + 1);
        // meshopt operates on float coordinates. Propose in a bounded frame
        // and certify the untouched source positions in their original units.
        for (const auto& point : points) {
            const auto coordinate = glm::vec3((position(point) - center) / scale);
            coordinates.insert(coordinates.end(), {coordinate.x, coordinate.y, coordinate.z});
            locked.push_back(options.preserveVerticalSupport && (point[1] == minY || point[1] == maxY));
        }
        const double roundoff = 64.0 * std::numeric_limits<double>::epsilon() * glm::length(span);
        std::vector<uint32_t> indices;
        indices.reserve(best.size() * 3);
        for (const auto& face : best) indices.insert(indices.end(), face.begin(), face.end());
        const float estimateLimit = static_cast<float>(std::min(options.maxDeviation / scale, 2.0));
        const auto originalVertexCount = result.positions.size();
        constexpr std::size_t attempts = 6;
        std::size_t lastAim = 0;
        for (std::size_t attempt = 0; attempt < attempts; ++attempt) {
            const auto aim = options.targetVertices + (originalVertexCount - options.targetVertices) * attempt / attempts;
            if (aim >= result.positions.size()) break;
            if (aim == lastAim) continue;
            lastAim = aim;
            const auto reduction = simplifyGeometry(coordinates, indices, locked, 2 * aim - 4, estimateLimit, false);
            Faces candidate;
            for (std::size_t i = 0; i < reduction.indices.size(); i += 3) {
                candidate.push_back({reduction.indices[i], reduction.indices[i + 1], reduction.indices[i + 2]});
            }
            const auto retained = usedVertices(candidate, points.size());
            if (retained.size() >= result.positions.size()) continue;
            auto edges = validateHull(points, candidate);
            if (!edges) {
                candidate = convexSubset(points, retained);
                edges = validateHull(points, candidate);
            }
            if (!edges) continue;
            auto facets = removeCoplanarSamples(points, candidate, *edges);
            if (facets.size() < candidate.size() && validateHull(points, facets)) candidate = std::move(facets);
            const auto error = deviation(points, candidate, options, minY, maxY, roundoff);
            if (!error) continue;
            result = compactHull(points, candidate, source.skippedDegenerateTriangles);
            bestDeviation = *error;
            if (result.positions.size() <= options.targetVertices) break;
        }
    }
    const bool metTarget = result.positions.size() <= options.targetVertices;
    return {std::move(result), source.positions.size(), bestDeviation, metTarget};
}

std::vector<CollisionMesh> decomposeCollisionHulls(const CollisionMesh& source, std::size_t maxHulls) {
    if (maxHulls == 0) throw std::invalid_argument("Convex conversion needs a positive hull budget");
    auto [points, faces] = hullInput(source, false);
    auto edges = closedEdges(faces);
    if (!edges) {
        std::tie(points, faces) = hullInput(source, true);
        edges = closedEdges(faces);
        if (!edges) throw std::invalid_argument("Convex conversion needs closed, consistently wound surfaces");
    }

    const auto shells = connectedShells(faces, *edges);
    std::vector<CollisionMesh> result;
    for (const auto& shell : shells) {
        if (result.size() == maxHulls) throw std::length_error("Convex conversion exceeds the hull budget");
        auto piece = compactHull(points, shell, 0);
        try {
            result.push_back(simplifyConvexHull(piece).mesh);
            continue;
        } catch (const std::invalid_argument&) {
            // A supported concave shell is split into a union of exact boxes.
        }
        try {
            auto boxes = detail::decomposeOrthogonalSolid(piece, maxHulls - result.size());
            std::ranges::move(boxes, std::back_inserter(result));
        } catch (const std::invalid_argument&) {
            // An orthogonal cavity can have a separate inward-wound shell.
            // Classify all its boundary surfaces together before making pieces.
            if (shells.size() == 1) throw;
            return detail::decomposeOrthogonalSolid(compactHull(points, faces, 0), maxHulls);
        }
    }
    return result;
}

CollisionMesh buildConvexHull(std::span<const std::array<float, 3>> input) {
    if (input.size() > 4096) throw std::length_error("Convex envelope exceeds its point work limit");
    std::vector<Point> points(input.begin(), input.end());
    for (const auto& point : points) {
        if (std::ranges::any_of(point, [](float value) { return !std::isfinite(value); })) {
            throw std::invalid_argument("Convex envelope requires finite points");
        }
    }
    std::ranges::sort(points);
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.size() < 4) throw std::invalid_argument("Convex envelope requires four noncoplanar points");
    const uint32_t a = 0;
    uint32_t b = 1, c = 0, d = 0;
    double best = -1;
    for (uint32_t i = 1; i < points.size(); ++i) {
        const auto delta = position(points[i]) - position(points[a]);
        const double length = glm::dot(delta, delta);
        if (length > best) { best = length; b = i; }
    }
    best = 0;
    for (uint32_t i = 0; i < points.size(); ++i) {
        const auto cross = glm::cross(position(points[b]) - position(points[a]), position(points[i]) - position(points[a]));
        const double area = glm::dot(cross, cross);
        if (area > best) { best = area; c = i; }
    }
    if (best == 0) throw std::invalid_argument("Convex envelope points are collinear");
    for (uint32_t i = 0; i < points.size(); ++i) {
        if (orientation(points[a], points[b], points[c], points[i]) != 0) { d = i; break; }
    }
    if (d == 0) throw std::invalid_argument("Convex envelope points are coplanar");
    Faces faces{{a, b, c}, {a, d, b}, {b, d, c}, {c, d, a}};
    const std::array initial{a, b, c, d};
    for (auto& face : faces) {
        const auto other = *std::ranges::find_if(initial, [&](uint32_t i) {
            return std::ranges::find(face, i) == face.end();
        });
        if (side(points, face, other) > 0) std::swap(face[1], face[2]);
    }
    std::size_t work = 0;
    for (uint32_t vertex = 0; vertex < points.size(); ++vertex) {
        std::vector<bool> visible(faces.size());
        bool outside = false;
        std::map<Edge, std::pair<Edge, unsigned>> horizon;
        for (std::size_t i = 0; i < faces.size(); ++i) {
            if (++work > 16'000'000) throw std::length_error("Convex envelope exceeds its orientation work limit");
            visible[i] = side(points, faces[i], vertex) > 0;
            outside |= visible[i];
            if (visible[i]) {
                for (std::size_t j = 0; j < 3; ++j) {
                    const auto first = faces[i][j], last = faces[i][(j + 1) % 3];
                    auto& edge = horizon[edgeKey(first, last)];
                    edge.first = {first, last};
                    ++edge.second;
                }
            }
        }
        if (!outside) continue;
        Faces next;
        for (std::size_t i = 0; i < faces.size(); ++i) if (!visible[i]) next.push_back(faces[i]);
        for (const auto& [key, edge] : horizon) {
            if (edge.second == 1) next.push_back({edge.first[0], edge.first[1], vertex});
        }
        faces = std::move(next);
    }
    auto edges = validateHull(points, faces);
    if (!edges) throw std::invalid_argument("Convex envelope failed closed-volume validation");
    for (const auto& face : faces) {
        for (uint32_t vertex = 0; vertex < points.size(); ++vertex) {
            if (++work > 16'000'000) throw std::length_error("Convex envelope exceeds its orientation work limit");
            if (side(points, face, vertex) > 0) throw std::invalid_argument("Convex envelope omitted a source point");
        }
    }
    auto reduced = removeCoplanarSamples(points, faces, *edges);
    if (validateHull(points, reduced)) faces = std::move(reduced);
    return compactHull(points, faces, 0);
}

namespace {

double pointTriangleDistance(const glm::dvec3& p, const std::array<glm::dvec3, 3>& triangle) {
    const auto& a = triangle[0]; const auto& b = triangle[1]; const auto& c = triangle[2];
    const auto ab = b - a, ac = c - a, ap = p - a;
    const auto normal = glm::cross(ab, ac);
    const double area2 = glm::dot(normal, normal);
    if (area2 > 0) {
        const auto projection = p - normal * (glm::dot(ap, normal) / area2);
        if (glm::dot(glm::cross(ab, projection - a), normal) >= 0 &&
            glm::dot(glm::cross(c - b, projection - b), normal) >= 0 &&
            glm::dot(glm::cross(a - c, projection - c), normal) >= 0) {
            return std::abs(glm::dot(ap, normal)) / std::sqrt(area2);
        }
    }
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < 3; ++i) {
        const auto start = triangle[i], delta = triangle[(i + 1) % 3] - start;
        const double length = glm::dot(delta, delta);
        const double t = length == 0 ? 0 : std::clamp(glm::dot(p - start, delta) / length, 0.0, 1.0);
        best = std::min(best, glm::length(p - start - t * delta));
    }
    return best;
}

using Triangle = std::array<glm::dvec3, 3>;

std::vector<Triangle> triangles(const CollisionMesh& mesh) {
    std::vector<Triangle> result;
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        result.push_back({position(mesh.positions[mesh.indices[i]]), position(mesh.positions[mesh.indices[i + 1]]),
                          position(mesh.positions[mesh.indices[i + 2]])});
    }
    return result;
}

std::optional<double> directedSurfaceBound(const std::vector<Triangle>& from, const std::vector<Triangle>& to,
                                           double limit, double allowance, std::size_t& work) {
    struct Cell { Triangle corners; unsigned depth; };
    std::vector<Cell> pending;
    for (const auto& triangle : from) pending.push_back({triangle, 0});
    double maximum = 0;
    while (!pending.empty()) {
        const auto cell = pending.back(); pending.pop_back();
        double upper = std::numeric_limits<double>::infinity();
        std::array<double, 3> lower; lower.fill(std::numeric_limits<double>::infinity());
        for (const auto& reference : to) {
            if (++work > 4'000'000) throw std::length_error("Surface cover verification exceeds its work limit");
            double distance = 0;
            for (std::size_t i = 0; i < 3; ++i) {
                const double value = pointTriangleDistance(cell.corners[i], reference);
                distance = std::max(distance, value);
                lower[i] = std::min(lower[i], value);
            }
            upper = std::min(upper, distance);
        }
        if (*std::max_element(lower.begin(), lower.end()) > limit + allowance) return std::nullopt;
        if (upper + allowance <= limit) { maximum = std::max(maximum, upper + allowance); continue; }
        if (cell.depth == 12) return std::nullopt;
        const auto& a = cell.corners[0]; const auto& b = cell.corners[1]; const auto& c = cell.corners[2];
        const auto ab = (a + b) * .5, bc = (b + c) * .5, ca = (c + a) * .5;
        const unsigned depth = cell.depth + 1;
        pending.push_back({{a, ab, ca}, depth}); pending.push_back({{ab, b, bc}, depth});
        pending.push_back({{ca, bc, c}, depth}); pending.push_back({{ab, bc, ca}, depth});
    }
    return maximum;
}

bool convexVolumesOverlap(const CollisionMesh& a, const CollisionMesh& b, double tolerance, std::size_t& work) {
    const auto first = triangles(a), second = triangles(b);
    const auto count = [&]() { if (++work > 4'000'000) throw std::length_error("Cavity verification exceeds its work limit"); };
    const auto separated = [&](const glm::dvec3& raw) {
        const double length = glm::length(raw); if (length == 0) return false;
        const auto axis = raw / length;
        double loA = std::numeric_limits<double>::infinity(), hiA = -loA, loB = loA, hiB = -loA;
        for (const auto& p : a.positions) { count(); const double t = glm::dot(position(p), axis); loA = std::min(loA,t); hiA = std::max(hiA,t); }
        for (const auto& p : b.positions) { count(); const double t = glm::dot(position(p), axis); loB = std::min(loB,t); hiB = std::max(hiB,t); }
        return std::min(hiA, hiB) - std::max(loA, loB) <= tolerance;
    };
    for (const auto axis : {glm::dvec3(1,0,0), glm::dvec3(0,1,0), glm::dvec3(0,0,1)}) if (separated(axis)) return false;
    for (const auto& face : first) if (separated(glm::cross(face[1]-face[0],face[2]-face[0]))) return false;
    for (const auto& face : second) if (separated(glm::cross(face[1]-face[0],face[2]-face[0]))) return false;
    for (const auto& faceA : first) for (std::size_t i=0;i<3;++i)
        for (const auto& faceB : second) for (std::size_t j=0;j<3;++j)
            if (separated(glm::cross(faceA[(i+1)%3]-faceA[i],faceB[(j+1)%3]-faceB[j]))) return false;
    return true;
}

} // namespace

CoherentCollisionResult buildCoherentCollisionShape(const CollisionMesh& source, const CoherentCollisionOptions& options) {
    if (!std::isfinite(options.maxSurfaceError) || options.maxSurfaceError < 0 || options.maxHulls == 0) {
        throw std::invalid_argument("A coherent collider requires a nonnegative surface bound and positive hull limit");
    }
    try {
        CoherentCollisionResult exact{decomposeCollisionHulls(source, options.maxHulls), 0.0, true};
        for (auto& hull : exact.hulls) {
            if (hull.positions.size() <= 256) continue;
            if (exact.hulls.size() != 1) throw std::invalid_argument("A compound contact must retain its source boundary vertices");
            auto reduced = simplifyConvexHull(hull, {256, options.maxSurfaceError, true});
            if (!reduced.metTarget) throw std::invalid_argument("Convex collider exceeds the vertex format within its fidelity bound");
            hull = std::move(reduced.mesh);
            exact.surfaceErrorBound = std::max(exact.surfaceErrorBound, reduced.maxDeviation);
            exact.exactSolidConversion &= reduced.maxDeviation == 0;
        }
        return exact;
    } catch (const std::invalid_argument&) {
        // Surface elements do not need to be closed; the generated pieces do.
    }
    if (options.maxSurfaceError == 0) throw std::invalid_argument("Surface completion requires a positive fidelity bound");
    auto [points, faces] = hullInput(source, true);
    if (faces.empty()) throw std::invalid_argument("A coherent collider requires source faces");
    glm::dvec3 low = position(points.front()), high = low;
    double magnitude = 0;
    for (const auto& point : points) { low = glm::min(low, position(point)); high = glm::max(high, position(point)); magnitude = std::max(magnitude, glm::length(position(point))); }
    const auto span = high - low;
    const double allowance = 128 * std::numeric_limits<double>::epsilon() * magnitude;
    const double inputRoundoff = 8 * std::numeric_limits<float>::epsilon() * magnitude;
    const double thickness = std::max(inputRoundoff, 4 * std::max({span.x, span.y, span.z}) / 65535.0);
    const double planeAllowance = std::min(options.maxSurfaceError, std::max(inputRoundoff, allowance));
    std::map<Edge, std::vector<std::size_t>> edges;
    std::vector<std::size_t> parent(faces.size()); std::iota(parent.begin(), parent.end(), 0);
    for (std::size_t i = 0; i < faces.size(); ++i) {
        if (glm::length(faceNormal(points, faces[i])) == 0) throw std::invalid_argument("Surface cover contains a zero-area source face");
        for (std::size_t j = 0; j < 3; ++j) edges[edgeKey(faces[i][j], faces[i][(j + 1) % 3])].push_back(i);
    }
    for (const auto& [edge, uses] : edges) for (auto index : uses) parent[groupRoot(parent, index)] = groupRoot(parent, uses.front());
    std::map<std::size_t, Faces> groups;
    for (std::size_t i = 0; i < faces.size(); ++i) groups[groupRoot(parent, i)].push_back(faces[i]);
    std::vector<Faces> pending;
    std::vector<CollisionMesh> cavities;
    // Inward-facing components describe empty space. An enclosing outer
    // component must not turn that cavity into an occupied hull. A convex
    // guard is conservative for unsupported concave cavities.
    for (const auto& [root, group] : groups) {
        const auto patch = compactHull(points, group, 0);
        const auto samples = triangles(patch);
        glm::dvec3 center(0);
        for (const auto& point : patch.positions) center += position(point);
        center /= double(patch.positions.size());
        double moment = 0, area = 0;
        for (const auto& face : samples) {
            const auto normal = glm::cross(face[1]-face[0],face[2]-face[0]);
            moment += glm::dot(face[0]-center, normal);
            area += glm::length(normal);
        }
        if (moment < -8 * inputRoundoff * area) {
            try { cavities.push_back(buildConvexHull(patch.positions)); }
            catch (const std::invalid_argument&) { }
        }
    }
    for (auto& [root, group] : groups) pending.push_back(std::move(group));
    CoherentCollisionResult result;
    std::size_t work = 0;
    while (!pending.empty()) {
        auto group = std::move(pending.back()); pending.pop_back();
        auto patch = compactHull(points, group, 0);
        std::optional<CollisionMesh> hull;
        const auto thicken = [&]() -> CollisionMesh {
            const auto normal = glm::normalize(faceNormal(points, group.front()));
            auto expanded = patch.positions;
            for (const auto& point : patch.positions) {
                const auto offset = position(point) - thickness * normal;
                expanded.push_back({static_cast<float>(offset.x), static_cast<float>(offset.y), static_cast<float>(offset.z)});
            }
            return buildConvexHull(expanded);
        };
        try { hull = buildConvexHull(patch.positions); }
        catch (const std::invalid_argument&) {
            if (thickness + allowance <= options.maxSurfaceError) {
                try { hull = thicken(); } catch (const std::invalid_argument&) { }
            }
        }
        catch (const std::length_error&) { }
        if (hull && thickness + allowance <= options.maxSurfaceError) {
            double minimumWidth = std::numeric_limits<double>::infinity();
            for (const auto& face : triangles(*hull)) {
                const auto normal = glm::normalize(glm::cross(face[1] - face[0], face[2] - face[0]));
                double first = std::numeric_limits<double>::infinity(), last = -first;
                for (const auto& p : hull->positions) { const double t = glm::dot(position(p), normal); first = std::min(first, t); last = std::max(last, t); }
                minimumWidth = std::min(minimumWidth, last - first);
            }
            // A rotated plane can gain a tiny artificial third dimension from
            // float import. Do not rely on that dimension surviving packing.
            if (minimumWidth < thickness) {
                try { hull = thicken(); } catch (const std::invalid_argument&) { hull.reset(); }
            }
        }
        std::optional<double> bound;
        if (hull && hull->positions.size() <= 256) {
            bool respectsFaces = true;
            for (const auto& cavity : cavities) if (convexVolumesOverlap(*hull, cavity, planeAllowance, work)) { respectsFaces = false; break; }
            for (const auto& face : group) {
                const auto n = faceNormal(points, face); const double length = glm::length(n);
                bool contactPlane = false;
                for (std::size_t axis = 0; axis < 3; ++axis) contactPlane |= points[face[0]][axis] == points[face[1]][axis] && points[face[0]][axis] == points[face[2]][axis];
                const double tolerance = contactPlane ? allowance : planeAllowance;
                for (const auto& vertex : hull->positions) {
                    if (glm::dot(position(vertex) - position(points[face[0]]), n) > tolerance * length) { respectsFaces = false; break; }
                }
                if (!respectsFaces) break;
            }
            if (respectsFaces) {
                const auto original = triangles(patch), candidate = triangles(*hull);
                const auto forward = directedSurfaceBound(original, candidate, options.maxSurfaceError, allowance, work);
                const auto backward = forward ? directedSurfaceBound(candidate, original, options.maxSurfaceError, allowance, work) : std::nullopt;
                if (forward && backward) bound = std::max(*forward, *backward);
            }
        }
        if (bound) {
            result.surfaceErrorBound = std::max(result.surfaceErrorBound, *bound);
            result.hulls.push_back(std::move(*hull));
        } else {
            if (group.size() == 1) throw std::invalid_argument("Surface cover cannot meet its fidelity or precision bound");
            glm::dvec3 centerLow(std::numeric_limits<double>::infinity()), centerHigh(-std::numeric_limits<double>::infinity());
            const auto centroid = [&](const Face& face) { return (position(points[face[0]]) + position(points[face[1]]) + position(points[face[2]])) / 3.0; };
            for (const auto& face : group) { centerLow = glm::min(centerLow, centroid(face)); centerHigh = glm::max(centerHigh, centroid(face)); }
            const auto extent = centerHigh - centerLow;
            const std::size_t axis = extent.x >= extent.y && extent.x >= extent.z ? 0 : (extent.y >= extent.z ? 1 : 2);
            std::stable_sort(group.begin(), group.end(), [&](const Face& a, const Face& b) { return centroid(a)[axis] < centroid(b)[axis]; });
            const auto middle = group.begin() + group.size() / 2;
            pending.emplace_back(group.begin(), middle); pending.emplace_back(middle, group.end());
        }
        if (result.hulls.size() + pending.size() > options.maxHulls) throw std::length_error("Coherent collision exceeds the hull limit within its fidelity bound");
    }
    return result;
}

} // namespace lodestar
