#pragma once

#include <filesystem>

namespace lodestar {

struct CollisionMesh;

// Export a scene-space collider as one COLLADA geometry at meter scale.
void writeCollisionCollada(const CollisionMesh& mesh, const std::filesystem::path& path);

} // namespace lodestar
