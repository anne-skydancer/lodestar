// Certified LOD error metric. The viewer consumes the result as a Record; the uploader checks its
// LODs with the same code. Dependencies: glm and the standard library. Built with strict floating
// point so records agree across platforms. High stays authoritative: nothing here replaces it.
// SPDX-License-Identifier: MIT
#pragma once

#include <glm/glm.hpp>
#include <lodestar/lodestar.hpp>
#include <lodestar/record.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lodestar::metric {

struct Options {
    // Stop refining when U <= ratio * W.
    double ratio = 1.05;
    // The 16-bit position grid as a fraction of the largest High extent. Pieces and errors below
    // it are not refined further.
    double gridFraction = 1.0 / 65535.0;
    // The search also stops when U - ratio * W is below this many grid steps: two quantised meshes
    // cannot be told apart more finely than about a step.
    double gridSlack = 2.0;
    // Work cap: distance queries per source triangle, plus a fixed base.
    std::size_t queriesPerTriangle = 6;
    std::size_t baseQueries = 60000;
    // Work cap of the UV overlay (candidate triangle pairs).
    std::size_t overlayPairLimit = 300000;
    // A High normal matches a LOD normal within this angle.
    double normalAngleDegrees = 0.5;
};

// Skin data aligned with Mesh::vertices: up to four joint influences per vertex.
struct Skin {
    std::vector<std::array<uint16_t, 4>> joints;
    std::vector<std::array<float, 4>> weights;
    // Bind-pose pivot of each joint, in the mesh's units.
    std::vector<glm::vec3> jointPivots;
};

struct MeshWithSkin {
    const Mesh* mesh = nullptr;
    const Skin* skin = nullptr;
};

// ---- Shape -------------------------------------------------------------------------------------

struct ShapeResult {
    double upper = 0.0;   // certified two-sided Hausdorff bound, input units
    double witness = 0.0; // largest distance actually found, input units
    std::array<double, 3> axisOffset{}; // A, per axis, input units
    std::array<double, 3> axisRadius{}; // R, per axis, input units
    bool capped = false;
};

// Two-sided Hausdorff distance between the surfaces of a and b, by branch and bound.
ShapeResult shapeBound(const Mesh& a, const Mesh& b, const Options& options = {});

// ---- Texture -----------------------------------------------------------------------------------

// A UV chart in double precision. Vertex arrays are indexed by `triangles` (three per face).
struct Chart {
    std::span<const glm::dvec2> uv;
    std::span<const glm::dvec3> positions;
    std::span<const glm::dvec3> normals;
    std::span<const uint32_t> triangles;
    std::span<const uint32_t> materials; // optional, one per face: only equal materials overlay
};

struct Correspondence {
    // Largest position difference over the UV overlay, with the 1e-10 floating-point allowance.
    double surfaceBound = 0.0;
    // Lower bound of the same quantity (largest position difference at overlay-cell vertices, taking
    // the nearest of the target faces that contain the point), without the allowance.
    double witness = 0.0;
    double normalAngleDegrees = 0.0;
    std::size_t pairs = 0;
    // The target covers the whole UV domain of the source.
    bool covered = true;
    // The source covers the whole UV domain of the target (used to certify shading both ways).
    bool targetCovered = true;
    // The pair limit was exceeded; the other values are partial.
    bool limitExceeded = false;
    // A normal field was singular or crossed a hemisphere in an overlay cell; the angle is infinite.
    bool normalsDegenerate = false;
};

// Port of tools/lodestar_surface_fit.py correspondence_bound. Bounds the piecewise-linear position
// and normal differences between two surfaces on the complete UV overlay: each cell is convex and
// both interpolants are affine there, so the maximum is at a cell vertex. The Python raised on
// incomplete coverage and on the pair limit; this returns the flags instead.
Correspondence correspondenceBound(const Chart& source, const Chart& target, const Options& options = {});

struct TextureResult {
    double upper = 0.0; // infinity where not certifiable (High does not cover the UV domain)
    double witness = 0.0;
    Correspondence overlay;
    bool hasUV = false;
};

// Texture deviation of `lod` against `high`: the distance between points with the same texture
// coordinate. Exact on the overlay cells, wherever the LOD does or does not reuse High's UVs.
TextureResult textureBound(const Mesh& high, const Mesh& lod, const Options& options = {});

// ---- Shading -----------------------------------------------------------------------------------

struct ShadingResult {
    double upper = 0.0; // infinity where not certifiable
    double witness = 0.0;
};

// Normal-pattern shift: the distance from a point to the nearest point of the other surface whose
// normal is within options.normalAngleDegrees. W is always measured, U only where certifiable.
// `texture` is the overlay result for the same pair, used to certify through the UV correspondence.
ShadingResult shadingBound(const Mesh& high, const Mesh& lod, const TextureResult& texture,
                           const Options& options = {});

// ---- Rigged ------------------------------------------------------------------------------------

struct RiggedTerms {
    double weight = 1.0; // upper bound of half the L1 distance of the weights, in [0, 1]
    double radius = 0.0; // largest distance from a point to its joints' pivots, input units
};

// The weight and radius terms of the bound over every joint rotation. e_bind is the shape bound.
RiggedTerms riggedTerms(const MeshWithSkin& high, const MeshWithSkin& lod, const Options& options = {});

// A witnessed lower bound from sampled poses: skin both meshes with each pose and return the
// largest distance found between the posed surfaces, in input units. `poses` holds
// poses.size() / jointCount consecutive arrays of jointCount skinning matrices (joint transform
// times inverse bind). Any single pose that deforms is a valid witness, so the maximum over the
// poses given is a lower bound of the error over every pose. Poses must keep each joint's scale
// at most the sigmaMax used with the upper bound.
double poseWitness(const MeshWithSkin& high, const MeshWithSkin& lod, std::span<const glm::mat4> poses,
                   std::size_t jointCount, const Options& options = {});

// ---- Record ------------------------------------------------------------------------------------

// XXH64 of a canonical little-endian encoding of the mesh block (positions, normals, uvs, indices).
uint64_t hashMesh(const Mesh& mesh);
uint64_t xxh64(std::span<const uint8_t> bytes, uint64_t seed = 0);

// Measure every LOD (Medium first) against High and fill a Record. A LOD identical to High gets the
// copy flag and exact zeros without any search.
Record measureAsset(const MeshWithSkin& high, std::span<const MeshWithSkin> lods, const Options& options = {});
Record measureAsset(const Mesh& high, std::span<const Mesh* const> lods, const Options& options = {});

} // namespace lodestar::metric
