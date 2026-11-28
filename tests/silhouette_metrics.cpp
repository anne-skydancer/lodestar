#include <lodestar/silhouette_metrics.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

lodestar::Mesh square() {
    lodestar::Mesh mesh;
    for (auto position : {glm::vec3{-1, -1, 0}, glm::vec3{1, -1, 0},
                          glm::vec3{1, 1, 0}, glm::vec3{-1, 1, 0}}) {
        mesh.vertices.push_back({position, {0, 0, 1}, {0, 0}});
    }
    mesh.indices = {0, 1, 2, 0, 2, 3};
    return mesh;
}

} // namespace

int main() {
    using lodestar::metrics::measureSilhouetteDistance;
    try {
        const auto source = square();
        auto differentTriangulation = source;
        differentTriangulation.indices = {3, 2, 1, 3, 1, 0};
        const auto same = measureSilhouetteDistance(source, differentTriangulation, 128);
        require(same.meanIoUError == 0.0 && same.maximumIoUError == 0.0,
                "Polygon triangulation or winding affected its silhouette");
        require(same.viewCount == 13 && same.resolution == 128, "Unexpected view count or resolution");

        auto displaced = source;
        for (auto& vertex : displaced.vertices) vertex.position.x += 3.0f;
        const auto mismatch = measureSilhouetteDistance(source, displaced, 64);
        const auto reversed = measureSilhouetteDistance(displaced, source, 64);
        require(mismatch.maximumIoUError == 1.0 && mismatch.meanIoUError > 0.0,
                "Disjoint projected shapes must have a maximum error of one");
        require(mismatch.meanIoUError == reversed.meanIoUError, "Silhouette IoU must be symmetric");

        auto malformed = source;
        malformed.indices.front() = static_cast<uint32_t>(malformed.vertices.size());
        try {
            measureSilhouetteDistance(source, malformed);
            throw std::runtime_error("Invalid mesh indices were accepted");
        } catch (const std::invalid_argument&) {
        }
        try {
            measureSilhouetteDistance(source, source, 15);
            throw std::runtime_error("Invalid raster resolution was accepted");
        } catch (const std::invalid_argument&) {
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
