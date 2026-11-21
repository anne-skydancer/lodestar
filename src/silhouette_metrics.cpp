#include "silhouette_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lodestar::metrics {
namespace {

struct MeshProjection {
    const Mesh& mesh;
    const std::vector<uint32_t> referenced;
    std::vector<glm::dvec2> pixels;
    std::vector<uint8_t> mask;
};

std::vector<uint32_t> referencedVertices(const Mesh& mesh) {
    if (mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        throw std::invalid_argument("Silhouette measurement requires an indexed triangle mesh");
    }

    std::vector<bool> seen(mesh.vertices.size());
    std::vector<uint32_t> referenced;
    referenced.reserve(std::min(mesh.indices.size(), mesh.vertices.size()));
    for (const auto index : mesh.indices) {
        if (index >= mesh.vertices.size()) {
            throw std::invalid_argument("Silhouette mesh index is out of bounds");
        }
        if (seen[index]) continue;
        seen[index] = true;
        const auto& position = mesh.vertices[index].position;
        if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
            throw std::invalid_argument("Silhouette mesh has a non-finite position");
        }
        referenced.push_back(index);
    }
    return referenced;
}

struct Bounds {
    glm::dvec2 min{std::numeric_limits<double>::infinity()};
    glm::dvec2 max{-std::numeric_limits<double>::infinity()};
};

void project(MeshProjection& projection, const glm::dvec3& right, const glm::dvec3& up,
             const glm::dvec3& origin, Bounds& bounds) {
    for (const auto index : projection.referenced) {
        const auto position = glm::dvec3(projection.mesh.vertices[index].position) - origin;
        const glm::dvec2 point{glm::dot(position, right), glm::dot(position, up)};
        projection.pixels[index] = point;
        bounds.min = glm::min(bounds.min, point);
        bounds.max = glm::max(bounds.max, point);
    }
}

double cross(glm::dvec2 a, glm::dvec2 b) {
    return a.x * b.y - a.y * b.x;
}

void rasterize(MeshProjection& projection, const Bounds& bounds, unsigned resolution) {
    std::ranges::fill(projection.mask, uint8_t{0});
    const auto span = bounds.max - bounds.min;
    const double extent = std::max(span.x, span.y);
    if (extent == 0.0) return;

    const auto center = bounds.min + span * 0.5;
    const double scale = (resolution - 2.0) / extent;
    for (const auto index : projection.referenced) {
        auto& point = projection.pixels[index];
        point = (point - center) * scale + glm::dvec2(0.5 * resolution);
    }

    const auto& indices = projection.mesh.indices;
    for (std::size_t i = 0; i < indices.size(); i += 3) {
        const auto a = projection.pixels[indices[i]];
        auto b = projection.pixels[indices[i + 1]];
        auto c = projection.pixels[indices[i + 2]];
        const double area = cross(b - a, c - a);
        if (std::abs(area) <= 1e-12) continue;
        if (area < 0.0) std::swap(b, c);

        const auto min = glm::min(a, glm::min(b, c));
        const auto max = glm::max(a, glm::max(b, c));
        const auto x0 = static_cast<int>(std::ceil(min.x - 0.5));
        const auto y0 = static_cast<int>(std::ceil(min.y - 0.5));
        const auto x1 = static_cast<int>(std::floor(max.x - 0.5));
        const auto y1 = static_cast<int>(std::floor(max.y - 0.5));
        // The shared frame leaves a one-pixel border, so all of these indices are in bounds.
        constexpr double edgeTolerance = -1e-10;
        const auto ab = b - a;
        const auto bc = c - b;
        const auto ca = a - c;
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                auto& covered = projection.mask[static_cast<std::size_t>(y) * resolution + x];
                if (covered) continue;
                const glm::dvec2 pixel{x + 0.5, y + 0.5};
                covered = cross(ab, pixel - a) >= edgeTolerance &&
                          cross(bc, pixel - b) >= edgeTolerance &&
                          cross(ca, pixel - c) >= edgeTolerance;
            }
        }
    }
}

} // namespace

SilhouetteDistance measureSilhouetteDistance(const Mesh& source, const Mesh& candidate, unsigned resolution) {
    if (resolution < 16 || resolution > 1024) {
        throw std::invalid_argument("Silhouette resolution must be between 16 and 1024");
    }

    const auto maskSize = static_cast<std::size_t>(resolution) * resolution;
    MeshProjection reference{source, referencedVertices(source), std::vector<glm::dvec2>(source.vertices.size()),
                             std::vector<uint8_t>(maskSize)};
    MeshProjection comparison{candidate, referencedVertices(candidate),
                              std::vector<glm::dvec2>(candidate.vertices.size()), std::vector<uint8_t>(maskSize)};
    const glm::dvec3 origin = source.vertices[reference.referenced.front()].position;

    double totalError = 0.0;
    double maximumError = 0.0;
    std::size_t viewCount = 0;
    // Sample the axis, edge-diagonal, and corner-diagonal directions of a cube.
    // Opposite directions have the same orthographic coverage.
    for (int x = 0; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            for (int z = -1; z <= 1; ++z) {
                if (x == 0 && (y < 0 || (y == 0 && z <= 0))) continue;
                const auto direction = glm::normalize(glm::dvec3{x, y, z});
                const auto vertical = x == 0 && z == 0 ? glm::dvec3{0, 0, 1} : glm::dvec3{0, 1, 0};
                const auto right = glm::normalize(glm::cross(vertical, direction));
                const auto up = glm::cross(direction, right);
                Bounds bounds;
                project(reference, right, up, origin, bounds);
                project(comparison, right, up, origin, bounds);
                rasterize(reference, bounds, resolution);
                rasterize(comparison, bounds, resolution);

                std::size_t unionPixels = 0;
                std::size_t differingPixels = 0;
                for (std::size_t i = 0; i < maskSize; ++i) {
                    unionPixels += reference.mask[i] | comparison.mask[i];
                    differingPixels += reference.mask[i] ^ comparison.mask[i];
                }
                const double error = unionPixels ? static_cast<double>(differingPixels) / unionPixels : 0.0;
                totalError += error;
                maximumError = std::max(maximumError, error);
                ++viewCount;
            }
        }
    }
    return {totalError / viewCount, maximumError, viewCount, resolution};
}

} // namespace lodestar::metrics
