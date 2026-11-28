#include <lodestar/lodestar.hpp>

#include <meshoptimizer.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lodestar {

GeometryReduction simplifyGeometry(std::span<const float> positions, std::span<const uint32_t> indices,
                                   std::span<const uint8_t> lockedVertices, std::size_t targetTriangles,
                                   float maxError, bool lockBoundary) {
    const auto vertexCount = positions.size() / 3;
    if (positions.empty() || positions.size() % 3 != 0 || vertexCount > std::numeric_limits<uint32_t>::max() ||
        indices.empty() || indices.size() % 3 != 0 ||
        (!lockedVertices.empty() && lockedVertices.size() != vertexCount) || targetTriangles == 0 ||
        !std::isfinite(maxError) || maxError < 0.0f ||
        std::ranges::any_of(positions, [](float value) { return !std::isfinite(value); }) ||
        std::ranges::any_of(indices, [=](uint32_t index) { return index >= vertexCount; }) ||
        std::ranges::any_of(lockedVertices, [](uint8_t value) { return value > 1; })) {
        throw std::invalid_argument("Invalid geometry simplification input");
    }
    if (targetTriangles >= indices.size() / 3) {
        return {{indices.begin(), indices.end()}, 0.0f};
    }
    GeometryReduction result;
    result.indices.resize(indices.size());
    const auto options = unsigned(meshopt_SimplifyErrorAbsolute) |
                         (lockBoundary ? unsigned(meshopt_SimplifyLockBorder) : 0u);
    const auto count = meshopt_simplifyWithAttributes(
        result.indices.data(), indices.data(), indices.size(), positions.data(), vertexCount, 3 * sizeof(float),
        nullptr, 0, nullptr, 0, lockedVertices.empty() ? nullptr : lockedVertices.data(), targetTriangles * 3,
        maxError, options, &result.estimatedError);
    result.indices.resize(count);
    return result;
}

} // namespace lodestar
