// Certified two-sided shape bound by branch and bound.
//
// For a source piece (a sub-triangle) with centroid c and radius rho, every point of the piece is
// within rho of c, so its distance to the target surface is at most d(c) + rho. A second bound
// uses convexity: for any target triangle t, the distance to t is convex, so over a piece it is at
// most the largest distance of the piece's three corners to t. The smaller of the two is the
// piece's upper bound U. W is the largest distance actually evaluated at a point (a lower bound).
// The worst piece is split into four until U <= ratio * W, or below the 16-bit grid, or the work
// cap is reached; in every case U stays a valid bound.
// SPDX-License-Identifier: MIT
#include "metric_internal.hpp"

#include <algorithm>
#include <bit>
#include <unordered_map>
#include <cmath>

namespace lodestar::metric::detail {
namespace {

struct Piece {
    std::array<dvec3, 3> v;
    std::array<dvec3, 3> bary;
    uint32_t tri;
    double upper = 0.0;
    dvec3 offset{0.0}, radius{0.0};
    double rho = 0.0;
};

dvec3 absolute(const dvec3& v) { return glm::abs(v); }

struct PointKey {
    uint64_t x, y, z;
    bool operator==(const PointKey&) const = default;
};
struct PointHash {
    std::size_t operator()(const PointKey& k) const {
        uint64_t h = k.x * 0x9E3779B97F4A7C15ULL;
        h = (h ^ (h >> 32) ^ k.y) * 0xC2B2AE3D27D4EB4FULL;
        h = (h ^ (h >> 29) ^ k.z) * 0x165667B19E3779F9ULL;
        return static_cast<std::size_t>(h ^ (h >> 32));
    }
};

struct Searcher {
    const Bvh& to;
    double ratio;
    double gridSlack;
    double grid;
    double witness = 0.0;
    std::size_t queries = 0;
    // Piece corners are shared between neighbours and siblings (a midpoint is computed from the same
    // two endpoints), so their nearest-triangle queries are cached by exact position.
    std::unordered_map<PointKey, Closest, PointHash> corners;

    const Closest& cornerNearest(const dvec3& p) {
        const PointKey key{std::bit_cast<uint64_t>(p.x), std::bit_cast<uint64_t>(p.y), std::bit_cast<uint64_t>(p.z)};
        const auto [it, inserted] = corners.try_emplace(key);
        if (inserted) {
            it->second = to.nearest(p);
            ++queries;
        }
        return it->second;
    }

    // Fill a piece's bound from its corners. Returns after possibly refining with candidate triangles.
    void evaluate(Piece& p) {
        const dvec3 c = (p.v[0] + p.v[1] + p.v[2]) / 3.0;
        p.rho = std::max({glm::length(p.v[0] - c), glm::length(p.v[1] - c), glm::length(p.v[2] - c)});
        p.upper = inf;
        p.offset = p.radius = dvec3(inf);
        // The corners first: their distances are exact witnesses, and their nearest triangles are
        // the candidates for the convexity bound (no centroid query when that already suffices).
        std::array<uint32_t, 3> candidates{};
        int count = 0;
        for (int i = 0; i < 3; ++i) {
            const Closest& qi = cornerNearest(p.v[i]);
            if (!(qi.distanceSquared < inf)) return;
            witness = std::max(witness, std::sqrt(qi.distanceSquared));
            if (std::find(candidates.begin(), candidates.begin() + count, qi.tri) == candidates.begin() + count)
                candidates[count++] = qi.tri;
        }
        for (int i = 0; i < count; ++i) tryTriangle(p, candidates[i]);
        if (p.upper <= ratio * witness + gridSlack * grid) return;
        // The Lipschitz bound at the centroid.
        const Closest q = to.nearest(c);
        ++queries;
        if (!(q.distanceSquared < inf)) return;
        const double dc = std::sqrt(q.distanceSquared);
        witness = std::max(witness, dc);
        if (dc + p.rho < p.upper) {
            p.upper = dc + p.rho;
            p.offset = absolute(c - q.point);
            p.radius = glm::max(glm::max(absolute(p.v[0] - c), absolute(p.v[1] - c)), absolute(p.v[2] - c));
        }
        if (p.upper <= ratio * witness + gridSlack * grid) return;
        if (std::find(candidates.begin(), candidates.begin() + count, q.tri) == candidates.begin() + count)
            tryTriangle(p, q.tri);
    }

    // Bound by the convexity of the distance to one target triangle.
    void tryTriangle(Piece& p, uint32_t t) {
        const auto& tri = to.surface().tris[t];
        double worst = 0.0;
        dvec3 offset(0.0);
        for (int i = 0; i < 3; ++i) {
            const dvec3 y = closestOnTriangle(p.v[i], tri.p[0], tri.p[1], tri.p[2]);
            worst = std::max(worst, glm::length(p.v[i] - y));
            offset = glm::max(offset, absolute(p.v[i] - y));
        }
        if (worst < p.upper) {
            p.upper = worst;
            p.offset = offset;
            p.radius = dvec3(0.0);
        }
    }
};

bool lowerUpper(const Piece& a, const Piece& b) { return a.upper < b.upper; }

} // namespace

DirectedShape directedShape(const Prepared& from, const Prepared& to, double grid, const Options& options,
                            std::size_t queryCap, bool keepLeaves) {
    DirectedShape out;
    if (from.surface->tris.empty()) return out;
    if (to.surface->tris.empty()) {
        out.upper = out.witness = inf;
        return out;
    }
    Searcher search{*to.bvh, options.ratio, options.gridSlack, grid};
    std::vector<Piece> heap, finals;
    heap.reserve(from.surface->tris.size());
    for (std::size_t i = 0; i < from.surface->tris.size(); ++i) {
        Piece p;
        p.v = from.surface->tris[i].p;
        p.bary = {dvec3(1, 0, 0), dvec3(0, 1, 0), dvec3(0, 0, 1)};
        p.tri = from.surface->source[i];
        search.evaluate(p);
        (p.rho < grid ? finals : heap).push_back(p);
    }
    std::make_heap(heap.begin(), heap.end(), lowerUpper);
    while (!heap.empty()) {
        const double stop = options.ratio * search.witness + options.gridSlack * grid;
        if (heap.front().upper <= stop) break;
        if (search.queries >= queryCap) {
            out.capped = true;
            break;
        }
        std::pop_heap(heap.begin(), heap.end(), lowerUpper);
        const Piece parent = heap.back();
        heap.pop_back();
        const auto mid = [&](int a, int b) { return std::pair{(parent.v[a] + parent.v[b]) * 0.5, (parent.bary[a] + parent.bary[b]) * 0.5}; };
        const auto m01 = mid(0, 1), m12 = mid(1, 2), m20 = mid(2, 0);
        const std::array<std::array<std::pair<dvec3, dvec3>, 3>, 4> corners = {{
            {{{parent.v[0], parent.bary[0]}, m01, m20}},
            {{m01, {parent.v[1], parent.bary[1]}, m12}},
            {{m20, m12, {parent.v[2], parent.bary[2]}}},
            {{m01, m12, m20}},
        }};
        for (const auto& corner : corners) {
            Piece child;
            child.tri = parent.tri;
            for (int k = 0; k < 3; ++k) {
                child.v[k] = corner[k].first;
                child.bary[k] = corner[k].second;
            }
            search.evaluate(child);
            if (child.rho < grid) {
                finals.push_back(child);
            } else {
                heap.push_back(child);
                std::push_heap(heap.begin(), heap.end(), lowerUpper);
            }
        }
    }
    out.witness = search.witness;
    out.queries = search.queries;
    for (const auto* set : {&heap, &finals}) {
        for (const auto& p : *set) {
            out.upper = std::max(out.upper, p.upper);
            out.offset = glm::max(out.offset, p.offset);
            out.radius = glm::max(out.radius, p.radius);
            if (keepLeaves) out.leaves.push_back({p.v, p.bary, p.tri, p.upper});
        }
    }
    return out;
}

ShapeResult shapeBetween(const Prepared& a, const Prepared& b, double grid, const Options& options,
                         std::vector<Leaf>* leavesAB, std::vector<Leaf>* leavesBA) {
    const std::size_t triangles = a.surface->tris.size() + b.surface->tris.size();
    const std::size_t cap = options.baseQueries + options.queriesPerTriangle * triangles;
    auto ab = directedShape(a, b, grid, options, cap, leavesAB != nullptr);
    auto ba = directedShape(b, a, grid, options, cap, leavesBA != nullptr);
    if (leavesAB) *leavesAB = std::move(ab.leaves);
    if (leavesBA) *leavesBA = std::move(ba.leaves);
    const double allowance = allowanceFraction * (grid / options.gridFraction);
    ShapeResult r;
    r.upper = std::max(ab.upper, ba.upper) + allowance;
    r.witness = std::max(ab.witness, ba.witness);
    const dvec3 offset = glm::max(ab.offset, ba.offset), radius = glm::max(ab.radius, ba.radius);
    for (int i = 0; i < 3; ++i) {
        r.axisOffset[i] = offset[i];
        r.axisRadius[i] = radius[i];
    }
    // The per-axis sums include the allowance on the offsets.
    for (int i = 0; i < 3; ++i) r.axisOffset[i] += allowance;
    r.capped = ab.capped || ba.capped;
    if (a.surface->tris.empty() && b.surface->tris.empty()) r.upper = 0.0;
    return r;
}

double largestExtent(const Mesh& mesh) {
    const dvec3 e = boxExtent(mesh);
    return std::max({e.x, e.y, e.z});
}

dvec3 boxExtent(const Mesh& mesh) {
    dvec3 lo(inf), hi(-inf);
    for (const auto& v : mesh.vertices) {
        lo = glm::min(lo, dvec3(v.position));
        hi = glm::max(hi, dvec3(v.position));
    }
    return mesh.vertices.empty() ? dvec3(0.0) : hi - lo;
}

} // namespace lodestar::metric::detail

namespace lodestar::metric {

ShapeResult shapeBound(const Mesh& a, const Mesh& b, const Options& options) {
    using namespace detail;
    Prepared pa(a), pb(b);
    dvec3 lo = glm::min(pa.surface->lo, pb.surface->lo), hi = glm::max(pa.surface->hi, pb.surface->hi);
    double extent = 0.0;
    if (lo.x <= hi.x) extent = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
    if (!(extent > 0.0)) extent = 1.0;
    return shapeBetween(pa, pb, options.gridFraction * extent, options);
}

} // namespace lodestar::metric
