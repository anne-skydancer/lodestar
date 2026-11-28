// Rigged terms of the bound over every joint rotation, and a pose-sampling witness.
//
// Linear blend skinning moves a point p with weights w by sum_j w_j M_j p. For a High point and
// its LOD partner at bind-pose distance e_bind, with joint scale at most sigmaMax and weight
// vectors w_H and w_L, the posed distance is at most
//     sigmaMax * e_bind + 1/2 * |w_H - w_L|_1 * 2 * r_joint,
// where r_joint bounds the distance from the point to its joints' pivots. The weight term is
// bounded here by interval arithmetic over the pieces of the shape search and the target
// triangles within reach of each piece; r_joint is the largest corner-to-pivot distance.
// SPDX-License-Identifier: MIT
#include "metric_internal.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace lodestar::metric {
namespace {

using detail::dvec3;

struct Influence {
    std::array<uint16_t, 4> joint{};
    std::array<double, 4> weight{};
};

Influence influenceOf(const Skin& skin, uint32_t vertex) {
    Influence in;
    double sum = 0.0;
    for (int k = 0; k < 4; ++k) {
        in.joint[k] = skin.joints[vertex][k];
        in.weight[k] = std::max(0.0, double(skin.weights[vertex][k]));
        sum += in.weight[k];
    }
    if (sum > 0.0)
        for (auto& w : in.weight) w /= sum;
    return in;
}

bool validSkin(const Mesh& mesh, const Skin& skin) {
    if (skin.joints.size() != mesh.vertices.size() || skin.weights.size() != mesh.vertices.size()) return false;
    for (std::size_t v = 0; v < mesh.vertices.size(); ++v)
        for (int k = 0; k < 4; ++k)
            if (skin.weights[v][k] > 0.0f && skin.joints[v][k] >= skin.jointPivots.size()) return false;
    return true;
}

struct Interval {
    double lo = 0.0, hi = 0.0;
};

// Per-joint weight interval over a set of weight vectors given as (joint -> value) samples.
using Ranges = std::map<uint16_t, Interval>;

void include(Ranges& ranges, const std::vector<std::map<uint16_t, double>>& samples) {
    for (const auto& s : samples)
        for (const auto& [j, w] : s) ranges.try_emplace(j, Interval{});
    for (auto& [j, r] : ranges) {
        bool first = true;
        for (const auto& s : samples) {
            const auto it = s.find(j);
            const double w = it == s.end() ? 0.0 : it->second;
            r.lo = first ? w : std::min(r.lo, w);
            r.hi = first ? w : std::max(r.hi, w);
            first = false;
        }
    }
}

std::map<uint16_t, double> weightsOf(const Influence& in) {
    std::map<uint16_t, double> w;
    for (int k = 0; k < 4; ++k)
        if (in.weight[k] > 0.0) w[in.joint[k]] += in.weight[k];
    return w;
}

} // namespace

namespace detail {

RiggedTerms riggedBetween(const Prepared& high, const Skin& highSkin, const Prepared& lod, const Skin& lodSkin,
                          const std::vector<Leaf>& highLeaves, const std::vector<Leaf>& lodLeaves, double extent) {
    RiggedTerms terms;
    terms.weight = 0.0;
    if (!validSkin(*high.mesh, highSkin) || !validSkin(*lod.mesh, lodSkin)) return {1.0, inf};

    // r_joint over every triangle: a point inside is no farther from a pivot than the farthest corner.
    for (const auto* side : {&high, &lod}) {
        const Skin& skin = side == &high ? highSkin : lodSkin;
        const Mesh& mesh = *side->mesh;
        for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            std::array<uint32_t, 3> ids{mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]};
            std::map<uint16_t, double> joints;
            for (const auto id : ids) {
                if (id >= mesh.vertices.size()) continue;
                for (const auto& [j, w] : weightsOf(influenceOf(skin, id))) joints[j] += w;
            }
            for (const auto id : ids) {
                if (id >= mesh.vertices.size()) continue;
                for (const auto& [j, w] : joints)
                    terms.radius = std::max(terms.radius, double(glm::length(mesh.vertices[id].position - skin.jointPivots[j])));
            }
        }
    }

    const auto leafTerm = [&](const Leaf& leaf, const Prepared& from, const Skin& fromSkin, const Prepared& to,
                              const Skin& toSkin) {
        const dvec3 c = (leaf.v[0] + leaf.v[1] + leaf.v[2]) / 3.0;
        double rho = 0.0;
        for (const auto& v : leaf.v) rho = std::max(rho, glm::length(v - c));
        const double reach = (leaf.upper + rho) * (1.0 + 1e-9) + 1e-12 * extent;
        // Source weights at the piece corners.
        std::array<uint32_t, 3> ids{from.mesh->indices[3 * leaf.tri], from.mesh->indices[3 * leaf.tri + 1],
                                    from.mesh->indices[3 * leaf.tri + 2]};
        std::array<std::map<uint16_t, double>, 3> vertexWeights;
        for (int k = 0; k < 3; ++k) vertexWeights[k] = weightsOf(influenceOf(fromSkin, ids[k]));
        std::vector<std::map<uint16_t, double>> corners(3);
        for (int k = 0; k < 3; ++k)
            for (int m = 0; m < 3; ++m)
                for (const auto& [j, w] : vertexWeights[m]) corners[k][j] += leaf.bary[k][m] * w;
        Ranges source;
        include(source, corners);
        // Target weights over every triangle within reach.
        Ranges target;
        std::vector<std::map<uint16_t, double>> vertices;
        to.bvh->forEachWithin(c, reach, [&](uint32_t t) {
            const uint32_t mt = to.surface->source[t];
            for (int k = 0; k < 3; ++k) vertices.push_back(weightsOf(influenceOf(toSkin, to.mesh->indices[3 * mt + k])));
        });
        if (vertices.empty()) return 1.0;
        include(target, vertices);
        std::map<uint16_t, double> d;
        for (const auto& [j, r] : source) d[j] = 0.0;
        for (const auto& [j, r] : target) d[j] = 0.0;
        double sum = 0.0;
        for (auto& [j, value] : d) {
            const Interval s = source.contains(j) ? source[j] : Interval{};
            const Interval t = target.contains(j) ? target[j] : Interval{};
            value = std::max({s.hi - t.lo, t.hi - s.lo, 0.0});
            sum += value;
        }
        return std::min(1.0, 0.5 * sum);
    };

    for (const Leaf& leaf : highLeaves) terms.weight = std::max(terms.weight, leafTerm(leaf, high, highSkin, lod, lodSkin));
    for (const Leaf& leaf : lodLeaves) terms.weight = std::max(terms.weight, leafTerm(leaf, lod, lodSkin, high, highSkin));
    return terms;
}

} // namespace detail

RiggedTerms riggedTerms(const MeshWithSkin& high, const MeshWithSkin& lod, const Options& options) {
    if (!high.mesh || !lod.mesh || !high.skin || !lod.skin) return {1.0, detail::inf};
    detail::Prepared h(*high.mesh), l(*lod.mesh);
    const double extent = std::max(detail::largestExtent(*high.mesh), 1e-300);
    std::vector<detail::Leaf> highLeaves, lodLeaves;
    detail::shapeBetween(h, l, options.gridFraction * extent, options, &highLeaves, &lodLeaves);
    return detail::riggedBetween(h, *high.skin, l, *lod.skin, highLeaves, lodLeaves, extent);
}

namespace {

std::vector<glm::vec3> pose(const Mesh& mesh, const Skin& skin, std::span<const glm::mat4> matrices) {
    std::vector<glm::vec3> out(mesh.vertices.size());
    for (std::size_t v = 0; v < out.size(); ++v) {
        const Influence in = influenceOf(skin, static_cast<uint32_t>(v));
        glm::dvec3 p(0.0);
        double total = 0.0;
        for (int k = 0; k < 4; ++k) {
            if (in.weight[k] <= 0.0 || in.joint[k] >= matrices.size()) continue;
            p += in.weight[k] * glm::dvec3(matrices[in.joint[k]] * glm::vec4(mesh.vertices[v].position, 1.0f));
            total += in.weight[k];
        }
        out[v] = total > 0.0 ? glm::vec3(p / total) : mesh.vertices[v].position;
    }
    return out;
}

double sampleDistance(const detail::Surface& from, const detail::Bvh& to) {
    double worst = 0.0;
    for (const auto& t : from.tris) {
        for (const auto& p : t.p) worst = std::max(worst, std::sqrt(to.nearest(p).distanceSquared));
        worst = std::max(worst, std::sqrt(to.nearest((t.p[0] + t.p[1] + t.p[2]) / 3.0).distanceSquared));
    }
    return worst;
}

} // namespace

double poseWitness(const MeshWithSkin& high, const MeshWithSkin& lod, std::span<const glm::mat4> poses,
                   std::size_t jointCount, const Options&) {
    if (!high.mesh || !lod.mesh || !high.skin || !lod.skin || jointCount == 0) return 0.0;
    if (!validSkin(*high.mesh, *high.skin) || !validSkin(*lod.mesh, *lod.skin)) return 0.0;
    double worst = 0.0;
    for (std::size_t first = 0; first + jointCount <= poses.size(); first += jointCount) {
        const auto matrices = poses.subspan(first, jointCount);
        const auto hp = pose(*high.mesh, *high.skin, matrices);
        const auto lp = pose(*lod.mesh, *lod.skin, matrices);
        detail::Prepared a(detail::makeSurface(hp, {}, high.mesh->indices));
        detail::Prepared b(detail::makeSurface(lp, {}, lod.mesh->indices));
        if (a.surface->tris.empty() || b.surface->tris.empty()) continue;
        worst = std::max({worst, sampleDistance(*a.surface, *b.bvh), sampleDistance(*b.surface, *a.bvh)});
    }
    return worst;
}

} // namespace lodestar::metric
