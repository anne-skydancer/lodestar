#pragma once

#include <lodestar/lodestar.hpp>

#include <cstddef>

namespace lodestar::metrics {

struct DirectionalDistance {
    double rms = 0.0;
    double maximum = 0.0;
};

struct SurfaceDistance {
    DirectionalDistance sourceToCandidate;
    DirectionalDistance candidateToSource;
    double sourceExtent = 0.0;
    std::size_t samplesPerDirection = 0;
};

// Sample each surface deterministically in proportion to triangle area and find
// the nearest point on the other surface for every sample. Report directional
// RMS and sampled maximum as fractions of the largest source AABB axis extent.
// This sampled maximum does not certify a Hausdorff bound.
// Both meshes must use the same coordinate system. Only positive-area triangles
// participate. Invalid meshes and zero sample counts throw std::invalid_argument.
SurfaceDistance measureSurfaceDistance(const Mesh& source, const Mesh& candidate,
                                       std::size_t samplesPerDirection);

} // namespace lodestar::metrics
