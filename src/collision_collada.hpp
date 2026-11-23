#pragma once

#include <filesystem>
#include <span>

namespace lodestar {

struct CollisionMesh;

// Export a scene-space collider as one COLLADA geometry at meter scale.
void writeCollisionCollada(const CollisionMesh& mesh, const std::filesystem::path& path);

// Export each piece as its own scene node in the same metre scene space.
void writeHullCollada(std::span<const CollisionMesh> hulls, const std::filesystem::path& path);

} // namespace lodestar
