// Normal-pattern shift: the distance from a point to the nearest point of the other surface whose
// normal is within a small angle. W is always measured; U only where it can be certified.
//
// Two certificates, the smaller wins:
//  * UV overlay: where both surfaces cover each other's UV domain and the normal difference over
//    every overlay cell is within the angle, the same-UV partner of every point is a match, so the
//    shading shift is at most the texture deviation.
//  * Normal cones: a piece of one surface whose corner normals stay within `delta` of its reference
//    normal can use any target point whose normal is within (angle - delta) of that reference
//    (angles add on the sphere). The target candidates are its vertices and its triangles whose
//    three vertex normals all match (a convex cone contains every interpolated normal).
// SPDX-License-Identifier: MIT
#include "metric_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace lodestar::metric::detail {
namespace {

struct NPiece {
    std::array<dvec3, 3> v;
    std::array<dvec3, 3> n; // unnormalised interpolated normals at the corners
    double upper = inf;
    double rho = 0.0;
};

dvec3 unit(const dvec3& v) {
    const double l = glm::length(v);
    return l > 0.0 ? v / l : dvec3(0.0);
}

struct ConeSearch {
    const Bvh& to;
    double angle;
    std::size_t queries = 0;

    void evaluate(NPiece& p) {
        const dvec3 c = (p.v[0] + p.v[1] + p.v[2]) / 3.0;
        p.rho = std::max({glm::length(p.v[0] - c), glm::length(p.v[1] - c), glm::length(p.v[2] - c)});
        const dvec3 sum = p.n[0] + p.n[1] + p.n[2];
        const dvec3 ref = unit(sum);
        p.upper = inf;
        if (glm::dot(ref, ref) == 0.0) return;
        double delta = 0.0;
        for (const auto& n : p.n) {
            if (glm::dot(n, n) == 0.0) return;
            delta = std::max(delta, angleBetween(ref, unit(n)));
        }
        if (delta >= angle) return;
        const Closest q = to.nearestMatching(c, ref, angle - delta, MatchMode::Inside);
        ++queries;
        if (q.distanceSquared < inf) p.upper = std::sqrt(q.distanceSquared) + p.rho;
    }
};

bool lowerUpper(const NPiece& a, const NPiece& b) { return a.upper < b.upper; }

double directedCertified(const Prepared& from, const Prepared& to, double grid, double angle, double witness,
                         const Options& options, std::size_t cap) {
    if (from.surface->tris.empty()) return 0.0;
    if (to.surface->tris.empty()) return inf;
    ConeSearch search{*to.bvh, angle};
    std::vector<NPiece> heap, finals;
    for (const auto& t : from.surface->tris) {
        NPiece p;
        p.v = t.p;
        p.n = t.n;
        search.evaluate(p);
        (p.rho < grid ? finals : heap).push_back(p);
    }
    std::make_heap(heap.begin(), heap.end(), lowerUpper);
    while (!heap.empty() && heap.front().upper > std::max(options.ratio * witness, grid) && search.queries < cap) {
        std::pop_heap(heap.begin(), heap.end(), lowerUpper);
        const NPiece parent = heap.back();
        heap.pop_back();
        const dvec3 m01v = (parent.v[0] + parent.v[1]) * 0.5, m12v = (parent.v[1] + parent.v[2]) * 0.5,
                    m20v = (parent.v[2] + parent.v[0]) * 0.5;
        const dvec3 m01n = (parent.n[0] + parent.n[1]) * 0.5, m12n = (parent.n[1] + parent.n[2]) * 0.5,
                    m20n = (parent.n[2] + parent.n[0]) * 0.5;
        const std::array<NPiece, 4> children = {{
            {{parent.v[0], m01v, m20v}, {parent.n[0], m01n, m20n}},
            {{m01v, parent.v[1], m12v}, {m01n, parent.n[1], m12n}},
            {{m20v, m12v, parent.v[2]}, {m20n, m12n, parent.n[2]}},
            {{m01v, m12v, m20v}, {m01n, m12n, m20n}},
        }};
        for (NPiece child : children) {
            search.evaluate(child);
            if (child.rho < grid) {
                finals.push_back(child);
            } else {
                heap.push_back(child);
                std::push_heap(heap.begin(), heap.end(), lowerUpper);
            }
        }
    }
    double upper = 0.0;
    for (const auto* set : {&heap, &finals})
        for (const auto& p : *set) upper = std::max(upper, p.upper);
    return upper;
}

// Largest distance from a sample of `from` to the superset of matching points of `to`. A lower
// bound of the true shift, because the superset only brings points closer.
double directedWitness(const Prepared& from, const Prepared& to, double angle) {
    double worst = 0.0;
    const auto sample = [&](const dvec3& p, const dvec3& n) {
        if (glm::dot(n, n) == 0.0) return;
        const Closest q = to.bvh->nearestMatching(p, unit(n), angle, MatchMode::Outside);
        worst = std::max(worst, q.distanceSquared < inf ? std::sqrt(q.distanceSquared) : inf);
    };
    for (const auto& t : from.surface->tris) {
        for (int k = 0; k < 3; ++k) sample(t.p[k], t.n[k]);
        sample((t.p[0] + t.p[1] + t.p[2]) / 3.0, t.n[0] + t.n[1] + t.n[2]);
    }
    return worst;
}

} // namespace

ShadingResult shadingBetween(Prepared& high, Prepared& lod, const TextureResult& texture, double grid,
                             const Options& options) {
    const double angle = glm::radians(options.normalAngleDegrees);
    high.ensureCones();
    lod.ensureCones();
    ShadingResult r;
    r.witness = std::max(directedWitness(lod, high, angle), directedWitness(high, lod, angle));

    double overlay = inf;
    if (texture.hasUV && texture.overlay.covered && texture.overlay.targetCovered && !texture.overlay.limitExceeded &&
        texture.overlay.normalAngleDegrees <= options.normalAngleDegrees) {
        overlay = texture.upper;
    }
    double cones = inf;
    if (overlay > options.ratio * r.witness + grid) {
        // Try the cone certificate when the overlay one is absent or loose.
        const std::size_t triangles = lod.surface->tris.size() + high.surface->tris.size();
        const std::size_t cap = options.baseQueries + 16 * triangles;
        const double ab = directedCertified(lod, high, grid, angle, r.witness, options, cap);
        const double ba = std::isfinite(ab) ? directedCertified(high, lod, grid, angle, r.witness, options, cap) : inf;
        cones = std::max(ab, ba);
        if (std::isfinite(cones)) cones += allowanceFraction * (grid / options.gridFraction);
    }
    r.upper = std::min(overlay, cones);
    if (std::isfinite(r.upper)) r.upper = std::max(r.upper, r.witness);
    return r;
}

} // namespace lodestar::metric::detail

namespace lodestar::metric {

ShadingResult shadingBound(const Mesh& high, const Mesh& lod, const TextureResult& texture, const Options& options) {
    detail::Prepared h(high), l(lod);
    const double extent = std::max(detail::largestExtent(high), 1e-300);
    return detail::shadingBetween(h, l, texture, options.gridFraction * extent, options);
}

} // namespace lodestar::metric
