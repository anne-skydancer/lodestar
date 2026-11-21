#include <lodestar/lodestar.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace lodestar {

CollisionMesh buildCollisionMesh(const Mesh& source) {
    if (source.indices.size() % 3 != 0) {
        throw std::invalid_argument("Collision source indices must form a triangle list");
    }

    CollisionMesh result;
    if (source.indices.empty()) return result;

    constexpr auto unused = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> remap(source.vertices.size(), unused);
    result.indices.reserve(source.indices.size());

    for (std::size_t i = 0; i < source.indices.size(); i += 3) {
        const std::array corners{source.indices[i], source.indices[i + 1], source.indices[i + 2]};
        std::array<glm::dvec3, 3> positions;
        for (std::size_t corner = 0; corner < corners.size(); ++corner) {
            const auto index = corners[corner];
            if (index >= source.vertices.size()) {
                throw std::invalid_argument("Collision source index is out of range");
            }
            const auto& position = source.vertices[index].position;
            if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
                throw std::invalid_argument("Collision source position is not finite");
            }
            positions[corner] = glm::dvec3(position);
        }

        if (glm::cross(positions[1] - positions[0], positions[2] - positions[0]) == glm::dvec3(0.0)) {
            ++result.skippedDegenerateTriangles;
            continue;
        }

        for (const auto index : corners) {
            auto& mapped = remap[index];
            if (mapped == unused) {
                const auto& position = source.vertices[index].position;
                mapped = result.positions.size();
                result.positions.push_back({position.x, position.y, position.z});
            }
            result.indices.push_back(static_cast<uint32_t>(mapped));
        }
    }
    return result;
}

} // namespace lodestar
