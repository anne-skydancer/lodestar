#pragma once

#include <lodestar/lodestar.hpp>

#include <limits>

namespace lodestar::detail {

// Compact a validated mesh after changing its triangle indices and marking discarded triangle slots.
inline void compactEditedMesh(Mesh& mesh, const std::vector<bool>& removed) {
    std::size_t faceDestination = 0;
    for (std::size_t face = 0; face < removed.size(); ++face) {
        if (removed[face]) continue;
        for (std::size_t corner = 0; corner < 3; ++corner) {
            mesh.indices[faceDestination * 3 + corner] = mesh.indices[face * 3 + corner];
        }
        if (!mesh.triangleMaterials.empty()) {
            mesh.triangleMaterials[faceDestination] = mesh.triangleMaterials[face];
        }
        if (!mesh.triangleSurfaceIds.empty()) {
            mesh.triangleSurfaceIds[faceDestination] = mesh.triangleSurfaceIds[face];
        }
        ++faceDestination;
    }
    mesh.indices.resize(faceDestination * 3);
    if (!mesh.triangleMaterials.empty()) mesh.triangleMaterials.resize(faceDestination);
    if (!mesh.triangleSurfaceIds.empty()) mesh.triangleSurfaceIds.resize(faceDestination);

    constexpr auto unused = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> remap(mesh.vertices.size(), unused);
    for (const auto index : mesh.indices) remap[index] = 0;
    uint32_t vertexDestination = 0;
    for (std::size_t vertex = 0; vertex < mesh.vertices.size(); ++vertex) {
        if (remap[vertex] == unused) continue;
        mesh.vertices[vertexDestination] = mesh.vertices[vertex];
        if (!mesh.vertexPositionIds.empty()) {
            mesh.vertexPositionIds[vertexDestination] = mesh.vertexPositionIds[vertex];
        }
        remap[vertex] = vertexDestination++;
    }
    mesh.vertices.resize(vertexDestination);
    if (!mesh.vertexPositionIds.empty()) mesh.vertexPositionIds.resize(vertexDestination);
    for (auto& index : mesh.indices) index = remap[index];
}

} // namespace lodestar::detail
