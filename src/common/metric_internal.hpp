// Internal declarations shared by the metric translation units. Not installed.
// SPDX-License-Identifier: MIT
#pragma once

#include "metric_geometry.hpp"

#include <lodestar/metric.hpp>

#include <memory>
#include <utility>

namespace lodestar::metric::detail {

// A mesh prepared for queries: its positive-area triangles and a BVH over them.
struct Prepared {
    explicit Prepared(const Mesh& m) : mesh(&m), surface(std::make_unique<Surface>(makeSurface(m))),
                                      bvh(std::make_unique<Bvh>(*surface)) {}
    explicit Prepared(Surface s) : mesh(nullptr), surface(std::make_unique<Surface>(std::move(s))),
                                  bvh(std::make_unique<Bvh>(*surface)) {}
    void ensureCones() {
        if (!cones) {
            bvh->buildCones();
            cones = true;
        }
    }
    const Mesh* mesh;
    std::unique_ptr<Surface> surface;
    std::unique_ptr<Bvh> bvh;
    bool cones = false;
};

// A leaf piece of the shape search: a sub-triangle of source triangle `tri` with the barycentric
// coordinates of its corners in that triangle, and its certified bound.
struct Leaf {
    std::array<dvec3, 3> v;
    std::array<dvec3, 3> bary;
    uint32_t tri;
    double upper;
};

struct DirectedShape {
    double upper = 0.0;
    double witness = 0.0;
    dvec3 offset{0.0};
    dvec3 radius{0.0};
    bool capped = false;
    std::size_t queries = 0;
    std::vector<Leaf> leaves; // filled when requested
};

// One-sided bound: sup over the surface of `from` of the distance to the surface of `to`.
// Absolute units. `grid` is the 16-bit grid step; `queryCap` bounds the work.
DirectedShape directedShape(const Prepared& from, const Prepared& to, double grid, const Options& options,
                            std::size_t queryCap, bool keepLeaves);

ShapeResult shapeBetween(const Prepared& a, const Prepared& b, double grid, const Options& options,
                         std::vector<Leaf>* leavesAB = nullptr, std::vector<Leaf>* leavesBA = nullptr);

ShadingResult shadingBetween(Prepared& high, Prepared& lod, const TextureResult& texture, double grid,
                             const Options& options);

RiggedTerms riggedBetween(const Prepared& high, const Skin& highSkin, const Prepared& lod, const Skin& lodSkin,
                          const std::vector<Leaf>& highLeaves, const std::vector<Leaf>& lodLeaves, double extent);

// Floating-point allowance added to every certified bound, as a fraction of the High extent.
inline constexpr double allowanceFraction = 1e-10;

double largestExtent(const Mesh& mesh);
dvec3 boxExtent(const Mesh& mesh);

} // namespace lodestar::metric::detail
