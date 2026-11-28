#include <lodestar/lodestar.hpp>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lodestar {
namespace {

constexpr auto unmappedIndex = std::numeric_limits<uint32_t>::max();

bool finite(const glm::dvec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

struct Bounds {
    glm::dvec3 minimum{std::numeric_limits<double>::infinity()};
    glm::dvec3 maximum{-std::numeric_limits<double>::infinity()};

    void include(const glm::dvec3& position) {
        minimum = glm::min(minimum, position);
        maximum = glm::max(maximum, position);
    }

    glm::dvec3 center() const {
        return minimum + (maximum - minimum) * 0.5;
    }

    double halfExtent() const {
        const auto extent = maximum - minimum;
        return std::max({extent.x, extent.y, extent.z}) * 0.5;
    }
};

glm::dmat4 toGlm(const aiMatrix4x4& m) {
    return {{m.a1, m.b1, m.c1, m.d1}, {m.a2, m.b2, m.c2, m.d2},
            {m.a3, m.b3, m.c3, m.d3}, {m.a4, m.b4, m.c4, m.d4}};
}

glm::dvec3 toGlm(const aiVector3D& v) {
    return {v.x, v.y, v.z};
}

void appendMesh(const aiMesh& source, const glm::dmat4& transform,
                Mesh& mesh, std::vector<glm::dvec3>& worldPositions) {
    const glm::dmat3 linearTransform(transform);
    const double determinant = glm::determinant(linearTransform);
    const bool invertible = std::isfinite(determinant) && determinant != 0.0;
    const glm::dmat3 normalTransform =
        invertible ? glm::transpose(glm::inverse(linearTransform)) : glm::dmat3(0.0);

    std::vector<glm::dvec3> positions;
    positions.reserve(source.mNumVertices);
    for (unsigned i = 0; i < source.mNumVertices; ++i) {
        const glm::dvec3 position(transform * glm::dvec4(toGlm(source.mVertices[i]), 1.0));
        if (!finite(position)) {
            throw std::runtime_error("Asset contains non-finite coordinates");
        }
        positions.push_back(position);
    }

    const bool hasUV = std::ranges::any_of(source.mTextureCoords, [](const auto* channel) {
        return channel != nullptr;
    });
    const bool hasUV0 = source.HasTextureCoords(0);
    const bool hasAdditionalUV = (hasUV0 && source.mNumUVComponents[0] > 2) ||
        std::ranges::any_of(source.mTextureCoords + 1, source.mTextureCoords + AI_MAX_NUMBER_OF_TEXTURECOORDS,
                            [](const auto* channel) { return channel != nullptr; });
    const bool hasUnrepresentedAttributes =
        source.HasBones() || source.mNumAnimMeshes != 0 || source.mTangents || source.mBitangents ||
        std::ranges::any_of(source.mColors, [](const auto* channel) { return channel != nullptr; });
    const uint32_t surface = mesh.triangleSurfaceIds.empty() ? 0u : mesh.triangleSurfaceIds.back() + 1u;
    const size_t start = mesh.vertices.size();
    std::vector<uint32_t> remap(source.mNumVertices, unmappedIndex);
    for (unsigned i = 0; i < source.mNumFaces; ++i) {
        const aiFace& face = source.mFaces[i];
        if (face.mNumIndices != 3) {
            continue;
        }

        std::array<unsigned, 3> corners{face.mIndices[0], face.mIndices[1], face.mIndices[2]};
        for (const auto corner : corners) {
            if (!mesh.declaredSceneBounds) mesh.declaredSceneBounds = std::array{positions[corner], positions[corner]};
            (*mesh.declaredSceneBounds)[0] = glm::min((*mesh.declaredSceneBounds)[0], positions[corner]);
            (*mesh.declaredSceneBounds)[1] = glm::max((*mesh.declaredSceneBounds)[1], positions[corner]);
        }
        if (determinant < 0.0) {
            std::swap(corners[1], corners[2]);
        }
        const auto areaVector = glm::cross(positions[corners[1]] - positions[corners[0]],
                                           positions[corners[2]] - positions[corners[0]]);
        const double area = glm::length(areaVector);
        if (!std::isfinite(area)) {
            throw std::runtime_error("Asset coordinates exceed the supported range");
        }
        if (area == 0.0) {
            continue;
        }
        if (mesh.indices.size() > unmappedIndex - 3) {
            throw std::runtime_error("Asset exceeds the supported index count");
        }
        const glm::vec3 faceNormal(areaVector / area);
        for (const unsigned corner : corners) {
            auto& index = remap[corner];
            if (index == unmappedIndex) {
                if (mesh.vertices.size() >= unmappedIndex) {
                    throw std::runtime_error("Asset exceeds the supported vertex count");
                }
                index = static_cast<uint32_t>(mesh.vertices.size());
                const glm::vec2 uv = hasUV0 ? glm::vec2(toGlm(source.mTextureCoords[0][corner]))
                                           : glm::vec2(0.0f);
                mesh.vertices.push_back({glm::vec3(0.0f), glm::vec3(0.0f), uv});
                worldPositions.push_back(positions[corner]);
            }
            mesh.vertices[index].normal += faceNormal;
            mesh.indices.push_back(index);
        }
        mesh.triangleMaterials.push_back(source.mMaterialIndex);
        mesh.triangleSurfaceIds.push_back(surface);
    }

    for (unsigned i = 0; i < source.mNumVertices; ++i) {
        if (remap[i] == unmappedIndex) {
            continue;
        }
        auto& normal = mesh.vertices[remap[i]].normal;
        if (invertible && source.HasNormals()) {
            const glm::dvec3 transformed = normalTransform * toGlm(source.mNormals[i]);
            const double length = glm::length(transformed);
            if (std::isfinite(length) && length > 0.0) {
                normal = glm::vec3(transformed / length);
                continue;
            }
        }
        const float length = glm::length(normal);
        normal = length > 0.0f ? normal / length : glm::vec3(0.0f, 1.0f, 0.0f);
    }
    mesh.hasUV |= hasUV && mesh.vertices.size() != start;
    mesh.hasAdditionalUV |= hasAdditionalUV && mesh.vertices.size() != start;
    mesh.hasUnrepresentedVertexAttributes |= hasUnrepresentedAttributes && mesh.vertices.size() != start;
}

} // namespace

Mesh loadMesh(const std::string& path) {
    return loadMesh(path, ImportSpace::Normalized);
}

Mesh loadMesh(const std::string& path, ImportSpace space) {
    if (space != ImportSpace::Normalized && space != ImportSpace::Scene) {
        throw std::invalid_argument("Unknown mesh import coordinate space");
    }
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices |
                  aiProcess_ImproveCacheLocality | aiProcess_ValidateDataStructure);
    if (!scene || !scene->mRootNode) {
        throw std::runtime_error(importer.GetErrorString());
    }

    struct Instance {
        const aiNode* node;
        glm::dmat4 parent;
    };

    Mesh mesh;
    std::vector<glm::dvec3> worldPositions;
    std::vector<Instance> pending{{scene->mRootNode, glm::dmat4(1.0)}};
    while (!pending.empty()) {
        const auto [node, parent] = pending.back();
        pending.pop_back();
        const auto transform = parent * toGlm(node->mTransformation);
        for (unsigned i = 0; i < node->mNumMeshes; ++i) {
            appendMesh(*scene->mMeshes[node->mMeshes[i]], transform, mesh, worldPositions);
        }
        for (unsigned i = node->mNumChildren; i > 0; --i) {
            pending.push_back({node->mChildren[i - 1], transform});
        }
    }
    if (mesh.indices.empty()) {
        throw std::runtime_error("Asset has no non-degenerate triangles");
    }

    glm::dvec3 center(0.0);
    double scale = 1.0;
    if (space == ImportSpace::Normalized) {
        Bounds bounds;
        for (const auto& position : worldPositions) {
            bounds.include(position);
        }
        center = bounds.center();
        scale = bounds.halfExtent();
        if (!finite(center) || !std::isfinite(scale) || scale <= 0.0) {
            throw std::runtime_error("Degenerate geometry");
        }
    }
    const glm::dvec3 maximumCoordinate(std::numeric_limits<float>::max());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        const auto position = (worldPositions[i] - center) / scale;
        if (glm::any(glm::greaterThan(glm::abs(position), maximumCoordinate))) {
            throw std::runtime_error("Scene positions exceed the supported float32 range");
        }
        mesh.vertices[i].position = glm::vec3(position);
    }
    return mesh;
}

void normalizeMesh(Mesh& mesh) {
    Bounds bounds;
    for (const auto& vertex : mesh.vertices) {
        if (!finite(vertex.position)) {
            throw std::runtime_error("Mesh contains non-finite coordinates");
        }
        bounds.include(vertex.position);
    }
    const auto center = bounds.center();
    const double scale = bounds.halfExtent();
    if (!finite(center) || !std::isfinite(scale) || scale <= 0.0) {
        throw std::runtime_error("Degenerate geometry");
    }
    for (auto& vertex : mesh.vertices) {
        vertex.position = glm::vec3((glm::dvec3(vertex.position) - center) / scale);
    }
}

} // namespace lodestar
