#include <lodestar/topology.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace lodestar {
namespace {

constexpr auto unknownVertex = std::numeric_limits<uint32_t>::max();
using Corners = std::array<uint32_t, 3>;
using EdgeCorners = std::array<uint32_t, 2>;

template <glm::length_t Size>
bool finite(const glm::vec<Size, float>& value) {
    for (glm::length_t i = 0; i < Size; ++i) {
        if (!std::isfinite(value[i])) {
            return false;
        }
    }
    return true;
}

uint32_t positionBits(float coordinate) {
    return coordinate == 0.0f ? 0u : std::bit_cast<uint32_t>(coordinate);
}

Corners positionKey(const glm::vec3& p) {
    return {positionBits(p.x), positionBits(p.y), positionBits(p.z)};
}

template <glm::length_t Size>
bool sameAttribute(const glm::vec<Size, float>& a, const glm::vec<Size, float>& b) {
    for (glm::length_t i = 0; i < Size; ++i) {
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) {
            return false;
        }
    }
    return true;
}

struct CornerHash {
    std::size_t operator()(const Corners& key) const {
        std::size_t hash = 0;
        for (const auto part : key) {
            hash ^= std::size_t(part) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct Face {
    Corners corners;
    bool duplicate = false;
    bool degenerate = false;
};

struct EdgeOccurrence {
    uint64_t geometryEdge;
    std::size_t face;
    // Ordered by geometric endpoint ID, consistent across this geometric edge.
    EdgeCorners corners;
    bool forward;
};

bool validProvenance(const Mesh& mesh, std::size_t faceCount) {
    if ((!mesh.triangleSurfaceIds.empty() && mesh.triangleSurfaceIds.size() != faceCount) ||
        (!mesh.vertexPositionIds.empty() && mesh.vertexPositionIds.size() != mesh.vertices.size())) {
        return false;
    }
    std::unordered_map<uint64_t, Corners> origins;
    for (std::size_t i = 0; i < mesh.vertexPositionIds.size(); ++i) {
        const auto id = mesh.vertexPositionIds[i];
        if (id == 0) {
            continue;
        }
        const auto& position = mesh.vertices[i].position;
        if (!finite(position)) {
            return false;
        }
        const auto key = positionKey(position);
        const auto [entry, inserted] = origins.try_emplace(id, key);
        if (!inserted && entry->second != key) {
            return false;
        }
    }
    return true;
}

uint32_t geometricCorner(const Mesh& mesh, uint32_t corner, std::vector<uint32_t>& positions,
                         std::unordered_map<Corners, uint32_t, CornerHash>& geometricPositions) {
    auto& group = positions[corner];
    if (group == unknownVertex) {
        const auto key = positionKey(mesh.vertices[corner].position);
        group = geometricPositions.try_emplace(key, corner).first->second;
    }
    return group;
}

bool collectFaces(const Mesh& mesh, std::vector<Face>& faces, std::vector<EdgeOccurrence>& edges,
                  std::vector<bool>& degenerateCorners, TopologyReport& report) {
    std::vector<uint32_t> positions(mesh.vertices.size(), unknownVertex);
    std::unordered_map<Corners, uint32_t, CornerHash> geometricPositions;
    std::unordered_map<Corners, std::size_t, CornerHash> geometricFaces;
    geometricFaces.reserve(faces.size());
    geometricPositions.reserve(mesh.vertices.size());
    edges.reserve(mesh.indices.size());

    for (std::size_t i = 0; i < faces.size(); ++i) {
        auto& face = faces[i];
        Corners groups{};
        for (std::size_t j = 0; j < groups.size(); ++j) {
            const auto corner = mesh.indices[i * 3 + j];
            if (corner >= mesh.vertices.size()) {
                return false;
            }
            const auto& vertex = mesh.vertices[corner];
            if (!finite(vertex.position) || !finite(vertex.normal) || (mesh.hasUV && !finite(vertex.uv))) {
                return false;
            }
            face.corners[j] = corner;
            groups[j] = geometricCorner(mesh, corner, positions, geometricPositions);
        }
        const glm::dvec3 a(mesh.vertices[face.corners[0]].position);
        const glm::dvec3 b(mesh.vertices[face.corners[1]].position);
        const glm::dvec3 c(mesh.vertices[face.corners[2]].position);
        if (glm::cross(b - a, c - a) == glm::dvec3(0.0)) {
            face.degenerate = true;
            ++report.degenerateTriangles;
            for (const auto corner : face.corners) {
                degenerateCorners[corner] = true;
            }
            continue;
        }

        auto faceKey = groups;
        std::ranges::sort(faceKey);
        const auto [entry, inserted] = geometricFaces.try_emplace(faceKey, i);
        if (!inserted) {
            face.duplicate = true;
            faces[entry->second].duplicate = true;
        }
        for (std::size_t j = 0; j < groups.size(); ++j) {
            const auto k = (j + 1) % groups.size();
            const bool forward = groups[j] < groups[k];
            const auto low = forward ? j : k;
            const auto high = forward ? k : j;
            edges.push_back({(uint64_t(groups[low]) << 32) | groups[high], i,
                             {face.corners[low], face.corners[high]}, forward});
        }
    }
    std::ranges::sort(edges, {}, &EdgeOccurrence::geometryEdge);
    return true;
}

std::size_t edgeEnd(const std::vector<EdgeOccurrence>& edges, std::size_t first) {
    auto end = first + 1;
    while (end < edges.size() && edges[end].geometryEdge == edges[first].geometryEdge) {
        ++end;
    }
    return end;
}

std::size_t countIndexedBoundaries(const std::vector<EdgeOccurrence>& edges, std::size_t first,
                                   std::size_t end, std::vector<EdgeCorners>& indexed) {
    const auto count = end - first;
    if (count == 1) {
        return 1;
    }
    if (count == 2) {
        return edges[first].corners == edges[first + 1].corners ? 0 : 2;
    }
    indexed.clear();
    for (auto i = first; i < end; ++i) {
        indexed.push_back(edges[i].corners);
    }
    std::ranges::sort(indexed);
    std::size_t boundaries = 0;
    for (std::size_t i = 0; i < indexed.size(); ++i) {
        if ((i == 0 || indexed[i - 1] != indexed[i]) && (i + 1 == indexed.size() || indexed[i + 1] != indexed[i])) {
            ++boundaries;
        }
    }
    return boundaries;
}

bool matchingOrigin(const Mesh& mesh, uint32_t a, uint32_t b) {
    return a == b || (!mesh.vertexPositionIds.empty() && mesh.vertexPositionIds[a] != 0 &&
                      mesh.vertexPositionIds[a] == mesh.vertexPositionIds[b]);
}

bool classifyPair(const Mesh& mesh, const EdgeOccurrence& a, const EdgeOccurrence& b,
                  const std::vector<bool>& degenerateCorners, TopologyReport& report) {
    if (a.forward == b.forward) {
        ++report.windingConflictEdges;
        return false;
    }
    if (a.corners == b.corners) {
        ++report.sharedEdges;
    }
    bool normalSeam = false;
    bool uvSeam = false;
    for (std::size_t i = 0; i < a.corners.size(); ++i) {
        const auto& va = mesh.vertices[a.corners[i]];
        const auto& vb = mesh.vertices[b.corners[i]];
        normalSeam |= !sameAttribute(va.normal, vb.normal);
        uvSeam |= mesh.hasUV && !sameAttribute(va.uv, vb.uv);
    }
    const bool materialSeam = !mesh.triangleMaterials.empty() &&
        mesh.triangleMaterials[a.face] != mesh.triangleMaterials[b.face];
    report.normalSeamEdges += normalSeam;
    report.uvSeamEdges += uvSeam;
    report.materialSeamEdges += materialSeam;
    if (a.corners == b.corners || normalSeam || uvSeam || materialSeam) {
        return false;
    }

    ++report.compatibleSplitEdges;
    if (!mesh.triangleSurfaceIds.empty() && mesh.triangleSurfaceIds[a.face] != mesh.triangleSurfaceIds[b.face]) {
        ++report.crossSurfaceSplitEdges;
        return false;
    }
    if (mesh.hasAdditionalUV || mesh.hasUnrepresentedVertexAttributes) {
        return false;
    }
    for (std::size_t i = 0; i < a.corners.size(); ++i) {
        if (!matchingOrigin(mesh, a.corners[i], b.corners[i]) ||
            degenerateCorners[a.corners[i]] || degenerateCorners[b.corners[i]]) {
            return false;
        }
    }
    return true;
}

std::vector<std::size_t> classifyEdges(const Mesh& mesh, const std::vector<Face>& faces,
                                       const std::vector<EdgeOccurrence>& edges,
                                       const std::vector<bool>& degenerateCorners, TopologyReport& report) {
    std::vector<std::size_t> recoveryEdges;
    std::vector<EdgeCorners> indexed;
    for (std::size_t first = 0; first < edges.size();) {
        const auto end = edgeEnd(edges, first);
        const auto count = end - first;
        ++report.geometricEdges;
        report.indexedBoundaryEdges += countIndexedBoundaries(edges, first, end, indexed);
        if (count == 1) {
            ++report.boundaryEdges;
        } else if (count > 2) {
            ++report.nonManifoldEdges;
            ++report.ambiguousEdges;
        } else if (faces[edges[first].face].duplicate || faces[edges[first + 1].face].duplicate) {
            ++report.ambiguousEdges;
        } else if (classifyPair(mesh, edges[first], edges[first + 1], degenerateCorners, report)) {
            recoveryEdges.push_back(first);
        }
        first = end;
    }
    return recoveryEdges;
}

uint32_t representative(std::vector<uint32_t>& canonical, uint32_t corner) {
    while (canonical[corner] != corner) {
        canonical[corner] = canonical[canonical[corner]];
        corner = canonical[corner];
    }
    return corner;
}

std::vector<uint32_t> remapCorners(std::size_t vertexCount, const std::vector<EdgeOccurrence>& edges,
                                  const std::vector<std::size_t>& recoveryEdges) {
    std::vector<uint32_t> canonical(vertexCount);
    std::iota(canonical.begin(), canonical.end(), 0u);
    for (const auto edge : recoveryEdges) {
        const auto& a = edges[edge].corners;
        const auto& b = edges[edge + 1].corners;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const auto ra = representative(canonical, a[i]);
            const auto rb = representative(canonical, b[i]);
            canonical[std::max(ra, rb)] = std::min(ra, rb);
        }
    }
    for (std::size_t i = 0; i < vertexCount; ++i) {
        canonical[i] = representative(canonical, static_cast<uint32_t>(i));
    }
    return canonical;
}

bool preservesOtherEdges(const std::vector<EdgeOccurrence>& edges, const std::vector<uint32_t>& canonical,
                         const std::vector<std::size_t>& recoveryEdges) {
    using EdgeChange = std::pair<EdgeCorners, EdgeCorners>;
    std::vector<EdgeChange> changes;
    std::size_t nextRecovery = 0;
    for (std::size_t first = 0; first < edges.size();) {
        const auto end = edgeEnd(edges, first);
        const bool accepted = nextRecovery < recoveryEdges.size() && recoveryEdges[nextRecovery] == first;
        if (accepted) {
            ++nextRecovery;
        } else if (end - first > 1) {
            changes.clear();
            for (auto i = first; i < end; ++i) {
                const auto& raw = edges[i].corners;
                changes.push_back({{canonical[raw[0]], canonical[raw[1]]}, raw});
            }
            std::ranges::sort(changes);
            for (std::size_t i = 1; i < changes.size(); ++i) {
                if (changes[i - 1].first == changes[i].first && changes[i - 1].second != changes[i].second) {
                    return false;
                }
            }
        }
        first = end;
    }
    return true;
}

bool isSingleFan(std::vector<EdgeCorners>& links, std::vector<uint32_t>& incoming) {
    std::ranges::sort(links);
    if (std::ranges::adjacent_find(links, {}, [](const auto& link) { return link[0]; }) != links.end()) {
        return false;
    }
    incoming.clear();
    for (const auto& [from, to] : links) {
        incoming.push_back(to);
    }
    std::ranges::sort(incoming);
    if (std::ranges::adjacent_find(incoming) != incoming.end()) {
        return false;
    }
    uint32_t openStart = unknownVertex;
    for (const auto& link : links) {
        if (!std::ranges::binary_search(incoming, link[0])) {
            if (openStart != unknownVertex) {
                return false;
            }
            openStart = link[0];
        }
    }
    const uint32_t start = openStart == unknownVertex ? links.front()[0] : openStart;
    uint32_t current = start;
    const auto from = [](const auto& link) { return link[0]; };
    for (std::size_t i = 0; i < links.size(); ++i) {
        const auto link = std::ranges::lower_bound(links, current, {}, from);
        if (link == links.end() || (*link)[0] != current) {
            return false;
        }
        current = (*link)[1];
        if (current == start && i + 1 != links.size()) {
            return false;
        }
    }
    if (openStart == unknownVertex) {
        return current == start;
    }
    const auto end = std::ranges::lower_bound(links, current, {}, from);
    return end == links.end() || (*end)[0] != current;
}

bool preservesVertexFans(const std::vector<Face>& faces, const std::vector<uint32_t>& canonical) {
    std::vector<bool> merged(canonical.size());
    for (std::size_t i = 0; i < canonical.size(); ++i) {
        if (canonical[i] != i) {
            merged[canonical[i]] = true;
        }
    }
    std::vector<std::vector<EdgeCorners>> links(canonical.size());
    for (const auto& face : faces) {
        if (face.degenerate) {
            continue;
        }
        for (std::size_t i = 0; i < face.corners.size(); ++i) {
            const auto corner = canonical[face.corners[i]];
            if (merged[corner]) {
                links[corner].push_back({canonical[face.corners[(i + 1) % 3]],
                                         canonical[face.corners[(i + 2) % 3]]});
            }
        }
    }
    std::vector<uint32_t> incoming;
    return std::ranges::all_of(links, [&](auto& fan) {
        return fan.empty() || isSingleFan(fan, incoming);
    });
}

} // namespace

detail::TopologyAnalysis detail::analyzeTopology(const Mesh& mesh) {
    const auto faceCount = mesh.indices.size() / 3;
    if (mesh.vertices.size() > unknownVertex || mesh.indices.size() % 3 != 0 ||
        (!mesh.triangleMaterials.empty() && mesh.triangleMaterials.size() != faceCount) ||
        !validProvenance(mesh, faceCount)) {
        return {};
    }
    TopologyAnalysis result;
    std::vector<Face> faces(faceCount);
    std::vector<EdgeOccurrence> edges;
    std::vector<bool> degenerateCorners(mesh.vertices.size());
    if (!collectFaces(mesh, faces, edges, degenerateCorners, result.report)) {
        return {};
    }
    result.report.valid = true;
    const auto recoveryEdges = classifyEdges(mesh, faces, edges, degenerateCorners, result.report);
    if (!recoveryEdges.empty()) {
        auto canonical = remapCorners(mesh.vertices.size(), edges, recoveryEdges);
        if (preservesOtherEdges(edges, canonical, recoveryEdges) && preservesVertexFans(faces, canonical)) {
            result.report.recoverableSplitEdges = recoveryEdges.size();
            result.canonicalCorners = std::move(canonical);
        }
    }
    return result;
}

TopologyReport inspectTopology(const Mesh& mesh) {
    return detail::analyzeTopology(mesh).report;
}

} // namespace lodestar
