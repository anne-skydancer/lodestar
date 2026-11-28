// Internal triangle-surface helpers for the certified LOD metric: exact closest points,
// a bounding volume hierarchy and normal cones. Not installed.
// SPDX-License-Identifier: MIT
#pragma once

#include <glm/glm.hpp>
#include <lodestar/lodestar.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace lodestar::metric::detail {

using dvec3 = glm::dvec3;
inline constexpr double inf = std::numeric_limits<double>::infinity();

struct Tri {
    std::array<dvec3, 3> p;
    std::array<dvec3, 3> n{}; // unit vertex normals; zero when absent or degenerate
};

// The positive-area triangles of a mesh. `source` maps each kept triangle to its index in the mesh.
struct Surface {
    std::vector<Tri> tris;
    std::vector<uint32_t> source;
    dvec3 lo{inf}, hi{-inf};
    bool hasNormals = false;
};

Surface makeSurface(std::span<const glm::vec3> positions, std::span<const glm::vec3> normals,
                    std::span<const uint32_t> indices);
Surface makeSurface(const Mesh& mesh);

struct Closest {
    dvec3 point{};
    double distanceSquared = inf;
    uint32_t tri = 0;
};

// Closest point on triangle abc to p, with the barycentric weights of that point.
dvec3 closestOnTriangle(const dvec3& p, const dvec3& a, const dvec3& b, const dvec3& c,
                        dvec3* barycentric = nullptr);

enum class MatchMode {
    Inside,  // a subset of the matching set: matching vertices and fully matching triangles
    Outside, // a superset of the matching set: every triangle whose normal cone may match
};

class Bvh {
public:
    explicit Bvh(const Surface& surface);
    // Attach normal cones; required by nearestMatching.
    void buildCones();
    Closest nearest(const dvec3& p) const;
    // Nearest point whose normal lies within `angle` radians of `direction` (unit).
    Closest nearestMatching(const dvec3& p, const dvec3& direction, double angle, MatchMode mode) const;
    // Calls f(triangleIndex) for each triangle whose exact distance to p is at most radius.
    template <class F> void forEachWithin(const dvec3& p, double radius, F&& f) const {
        visitWithin(p, radius, [&](uint32_t t) { f(t); });
    }
    const Surface& surface() const { return *surface_; }

private:
    struct Node {
        dvec3 lo, hi;
        dvec3 axis{0.0};
        double halfAngle = 0.0; // pi when the cone is unusable
        uint32_t first = 0, count = 0; // leaf: range in order_ (count > 0)
        uint32_t left = 0, right = 0;  // inner node children
    };
    void visitWithin(const dvec3& p, double radius, const std::function<void(uint32_t)>& f) const;
    uint32_t build(uint32_t begin, uint32_t end);
    const Surface* surface_;
    std::vector<uint32_t> order_;
    std::vector<Node> nodes_;
    struct TriCone { dvec3 axis; double half; };
    std::vector<TriCone> cones_;
};

double angleBetween(const dvec3& a, const dvec3& b);

} // namespace lodestar::metric::detail
