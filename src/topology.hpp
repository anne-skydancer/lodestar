#pragma once

#include <lodestar/lodestar.hpp>

namespace lodestar::detail {

struct TopologyAnalysis {
    TopologyReport report;
    // Empty when no edge can be recovered. Otherwise parallel to Mesh::vertices,
    // mapping an index to the least original index of its proven equivalent corners.
    std::vector<uint32_t> canonicalCorners;
};

TopologyAnalysis analyzeTopology(const Mesh& mesh);

} // namespace lodestar::detail
