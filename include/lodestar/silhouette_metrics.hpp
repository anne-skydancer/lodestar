#pragma once

#include <lodestar/lodestar.hpp>

#include <cstddef>

namespace lodestar::metrics {

struct SilhouetteDistance {
    double meanIoUError;
    double maximumIoUError;
    std::size_t viewCount;
    unsigned resolution;
};

// Rasterize both meshes in the same square frame in each orthographic view.
// Errors are fractions in [0, 1]. A view with two empty masks has zero error.
SilhouetteDistance measureSilhouetteDistance(const Mesh& source, const Mesh& candidate,
                                              unsigned resolution = 192);

} // namespace lodestar::metrics
