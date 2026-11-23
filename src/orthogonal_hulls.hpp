#pragma once

#include <lodestar/lodestar.hpp>

namespace lodestar::detail {

// The input's geometric edges must already be welded and paired with
// consistent orientations. Coordinate compression preserves the mesh's
// occupied volume exactly for closed, axis-aligned polygonal boundaries.
std::vector<CollisionMesh> decomposeOrthogonalSolid(const CollisionMesh& source, std::size_t maxHulls);

} // namespace lodestar::detail
