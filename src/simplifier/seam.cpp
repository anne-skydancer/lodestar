#include <lodestar/lodestar.hpp>

#include <lodestar/mesh_compaction.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

namespace lodestar {
namespace {

constexpr auto absent = std::numeric_limits<std::size_t>::max();
constexpr std::size_t maximumFanTriangles = 64;
using Corners = std::array<uint32_t, 3>;

std::size_t nextCorner(std::size_t corner) {
    return corner % 3 == 2 ? corner - 2 : corner + 1;
}

std::size_t previousCorner(std::size_t corner) {
    return corner % 3 == 0 ? corner + 2 : corner - 1;
}

template <glm::length_t N>
bool finite(const glm::vec<N, float>& value) {
    for (glm::length_t i = 0; i < N; ++i) {
        if (!std::isfinite(value[i])) {
            return false;
        }
    }
    return true;
}

uint32_t positionBits(float value) {
    return value == 0.0f ? 0u : std::bit_cast<uint32_t>(value);
}

struct KeyHash {
    std::size_t operator()(const Corners& key) const {
        std::size_t hash = 0;
        for (const auto part : key) {
            hash ^= std::size_t(part) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

uint64_t edgeKey(uint32_t a, uint32_t b) {
    return (uint64_t(std::min(a, b)) << 32) | std::max(a, b);
}

struct Edge {
    std::size_t first;
    std::size_t second = absent;
    std::size_t count = 1;
};

using Edges = std::unordered_map<uint64_t, Edge>;

void addEdge(Edges& edges, uint64_t key, std::size_t corner) {
    const auto [entry, inserted] = edges.try_emplace(key, Edge{corner});
    if (!inserted) {
        auto& edge = entry->second;
        ++edge.count;
        edge.second = corner;
    }
}

struct Position {
    glm::dvec3 value;
    std::vector<std::size_t> incident;
    bool unsafe = false;
};

struct Incidence {
    std::vector<Position> positions;
    std::vector<uint32_t> geometricCorner;
    std::vector<std::size_t> chartPartner;
    Edges geometricEdges;
};

bool sameSide(const Mesh& mesh, std::size_t a, std::size_t b) {
    return (mesh.triangleMaterials.empty() || mesh.triangleMaterials[a] == mesh.triangleMaterials[b]) &&
           (mesh.triangleSurfaceIds.empty() || mesh.triangleSurfaceIds[a] == mesh.triangleSurfaceIds[b]);
}

bool collectIncidence(const Mesh& mesh, Incidence& incidence) {
    const auto faceCount = mesh.indices.size() / 3;
    if (mesh.hasAdditionalUV || mesh.hasUnrepresentedVertexAttributes ||
        mesh.vertices.size() > std::numeric_limits<uint32_t>::max() || mesh.indices.size() % 3 != 0 ||
        (!mesh.triangleMaterials.empty() && mesh.triangleMaterials.size() != faceCount) ||
        (!mesh.triangleSurfaceIds.empty() && mesh.triangleSurfaceIds.size() != faceCount) ||
        (!mesh.vertexPositionIds.empty() && mesh.vertexPositionIds.size() != mesh.vertices.size())) {
        return false;
    }
    Edges indexedEdges;
    std::unordered_map<Corners, uint32_t, KeyHash> groupedPositions;
    std::unordered_map<Corners, std::size_t, KeyHash> geometricFaces;
    incidence.geometricCorner.resize(mesh.indices.size());
    incidence.chartPartner.resize(mesh.indices.size(), absent);
    for (std::size_t i = 0; i < mesh.indices.size(); ++i) {
        const auto index = mesh.indices[i];
        if (index >= mesh.vertices.size()) {
            return false;
        }
        const auto& vertex = mesh.vertices[index];
        if (!finite(vertex.position) || !finite(vertex.normal) || (mesh.hasUV && !finite(vertex.uv))) {
            return false;
        }
        const Corners key{positionBits(vertex.position.x), positionBits(vertex.position.y),
                          positionBits(vertex.position.z)};
        const auto [entry, inserted] = groupedPositions.try_emplace(
            key, static_cast<uint32_t>(incidence.positions.size()));
        if (inserted) {
            incidence.positions.push_back({glm::dvec3(vertex.position), {}});
        }
        incidence.geometricCorner[i] = entry->second;
        incidence.positions[entry->second].incident.push_back(i);
    }

    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        Corners corners{incidence.geometricCorner[i], incidence.geometricCorner[i + 1],
                        incidence.geometricCorner[i + 2]};
        const auto& a = incidence.positions[corners[0]].value;
        const auto& b = incidence.positions[corners[1]].value;
        const auto& c = incidence.positions[corners[2]].value;
        const bool degenerate = glm::cross(b - a, c - a) == glm::dvec3(0.0);
        for (std::size_t j = 0; j < corners.size(); ++j) {
            const auto h = i + j;
            incidence.positions[corners[j]].unsafe |= degenerate;
            if (corners[j] != corners[(j + 1) % 3]) {
                addEdge(indexedEdges, edgeKey(mesh.indices[h], mesh.indices[nextCorner(h)]), h);
                addEdge(incidence.geometricEdges, edgeKey(corners[j], corners[(j + 1) % 3]), h);
            }
        }
        std::ranges::sort(corners);
        if (!geometricFaces.try_emplace(corners, i).second) {
            for (const auto corner : corners) {
                incidence.positions[corner].unsafe = true;
            }
        }
    }
    for (const auto& [key, edge] : incidence.geometricEdges) {
        if (edge.count > 2 || (edge.count == 2 &&
                incidence.geometricCorner[edge.first] == incidence.geometricCorner[edge.second])) {
            incidence.positions[key >> 32].unsafe = true;
            incidence.positions[static_cast<uint32_t>(key)].unsafe = true;
        }
    }
    for (const auto& [key, edge] : indexedEdges) {
        if (edge.count == 2 && mesh.indices[edge.first] != mesh.indices[edge.second] &&
            sameSide(mesh, edge.first / 3, edge.second / 3)) {
            incidence.chartPartner[edge.first] = edge.second;
            incidence.chartPartner[edge.second] = edge.first;
        }
    }
    return true;
}

// Error-free floating-point transformations on float-derived dyadics. If an
// intermediate needed as a single double rounds, we decline the edit.
std::optional<double> difference(double a, double b) {
    const double high = a - b;
    const double restoreB = a - high;
    const double low = (a - (high + restoreB)) + (restoreB - b);
    return low == 0.0 ? std::optional(high) : std::nullopt;
}

template <glm::length_t N>
std::optional<glm::vec<N, double>> subtract(const glm::vec<N, double>& a, const glm::vec<N, double>& b) {
    glm::vec<N, double> delta;
    for (glm::length_t i = 0; i < N; ++i) {
        const auto d = difference(a[i], b[i]);
        if (!d) {
            return std::nullopt;
        }
        delta[i] = *d;
    }
    return delta;
}

struct Product {
    double high;
    double low;

    Product(double a, double b) : high(a * b), low(std::fma(a, b, -high)) {}
    bool operator==(const Product&) const = default;
};

std::optional<double> cross2(const glm::dvec2& a, const glm::dvec2& b) {
    const Product left(a.x, b.y), right(a.y, b.x);
    if (left.low != 0.0 || right.low != 0.0) {
        return std::nullopt;
    }
    return difference(left.high, right.high);
}

std::optional<double> orientedArea(const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c) {
    const auto ab = subtract(b, a);
    const auto ac = subtract(c, a);
    return ab && ac ? cross2(*ab, *ac) : std::nullopt;
}

std::optional<bool> strictlyBetween(const glm::dvec3& point, const glm::dvec3& a, const glm::dvec3& b) {
    if (point == a || point == b || glm::any(glm::lessThan(point, glm::min(a, b))) ||
        glm::any(glm::greaterThan(point, glm::max(a, b)))) {
        return false;
    }
    const auto fromA = subtract(point, a);
    const auto segment = subtract(b, a);
    if (!fromA || !segment) {
        return std::nullopt;
    }
    for (glm::length_t i = 0; i < 3; ++i) {
        const auto j = (i + 1) % 3;
        if (Product((*fromA)[i], (*segment)[j]) != Product((*fromA)[j], (*segment)[i])) {
            return false;
        }
    }
    return true;
}

bool sameAffineValue(double area, double value, double a, double deltaA, double b, double deltaB) {
    const Product target(area, value);
    const Product termA(a, deltaA), termB(b, deltaB);
    const auto high = difference(termA.high, -termB.high);
    const auto low = difference(termA.low, -termB.low);
    return high && low && *high == target.high && *low == target.low;
}

template <glm::length_t N>
bool sameAffineValues(double area, const glm::vec<N, float>& origin, const glm::vec<N, float>& vertex,
                      double a, const glm::vec<N, float>& first, double b, const glm::vec<N, float>& second) {
    const auto delta = subtract(glm::vec<N, double>(vertex), glm::vec<N, double>(origin));
    const auto deltaA = subtract(glm::vec<N, double>(first), glm::vec<N, double>(origin));
    const auto deltaB = subtract(glm::vec<N, double>(second), glm::vec<N, double>(origin));
    if (!delta || !deltaA || !deltaB) {
        return false;
    }
    for (glm::length_t i = 0; i < N; ++i) {
        if (!sameAffineValue(area, (*delta)[i], a, (*deltaA)[i], b, (*deltaB)[i])) {
            return false;
        }
    }
    return true;
}

struct RingVertex {
    uint32_t index;
    uint32_t geometry;
    glm::dvec2 projection{0.0};
};

struct Side {
    uint32_t center;
    std::vector<std::size_t> faces;
    // Oriented outer path, from the first boundary spoke to the second.
    std::vector<RingVertex> ring;
};

std::vector<Side> traceSides(const Mesh& mesh, const Incidence& incidence, uint32_t middle) {
    const auto& position = incidence.positions[middle];
    if (position.unsafe || position.incident.size() > maximumFanTriangles * 2) {
        return {};
    }
    std::vector<Side> sides;
    std::size_t represented = 0;
    for (const auto start : position.incident) {
        if (incidence.chartPartner[start] != absent) {
            continue;
        }
        if (sides.size() == 2) {
            return {};
        }
        Side side{mesh.indices[start], {}, {}};
        auto current = start;
        auto next = nextCorner(start);
        side.ring.push_back({mesh.indices[next], incidence.geometricCorner[next]});
        while (current != absent && side.faces.size() < maximumFanTriangles) {
            next = previousCorner(current);
            side.faces.push_back(current / 3);
            side.ring.push_back({mesh.indices[next], incidence.geometricCorner[next]});
            current = incidence.chartPartner[next];
        }
        if (current != absent || side.faces.size() < 2) {
            return {};
        }
        represented += side.faces.size();
        sides.push_back(std::move(side));
    }
    if (represented != position.incident.size()) {
        return {};
    }
    return sides;
}

bool matchedSides(const std::vector<Side>& sides) {
    if (sides.size() == 1) {
        return true;
    }
    const auto& a = sides[0];
    const auto& b = sides[1];
    if (a.ring.front().geometry != b.ring.back().geometry || a.ring.back().geometry != b.ring.front().geometry) {
        return false;
    }
    // The new segment must carry exactly the same indexed connection as
    // both old segments. In particular, no vertex-only contact can be lost.
    const bool sharedCenter = a.center == b.center;
    const bool sharedStart = a.ring.front().index == b.ring.back().index;
    const bool sharedEnd = a.ring.back().index == b.ring.front().index;
    return sharedCenter ? sharedStart && sharedEnd : !(sharedStart && sharedEnd);
}

bool validateSide(const Mesh& mesh, Side& side) {
    const auto& origin = mesh.vertices[side.center];
    const auto& first = mesh.vertices[side.ring[0].index];
    const auto& second = mesh.vertices[side.ring[1].index];
    const auto tA = subtract(glm::dvec3(first.position), glm::dvec3(origin.position));
    const auto tB = subtract(glm::dvec3(second.position), glm::dvec3(origin.position));
    if (!tA || !tB) {
        return false;
    }
    const auto normal = glm::cross(*tA, *tB);
    const auto magnitude = glm::abs(normal);
    glm::length_t dropped = magnitude.x >= magnitude.y ? 0 : 1;
    if (magnitude.z > magnitude[dropped]) {
        dropped = 2;
    }
    auto x = (dropped + 1) % 3;
    auto y = (dropped + 2) % 3;
    if (normal[dropped] < 0.0) {
        std::swap(x, y);
    }
    const glm::dvec2 a2((*tA)[x], (*tA)[y]);
    const glm::dvec2 b2((*tB)[x], (*tB)[y]);
    const auto area = cross2(a2, b2);
    if (!area || *area <= 0.0) {
        return false;
    }

    for (std::size_t i = 0; i < side.ring.size(); ++i) {
        auto& point = side.ring[i];
        const auto& vertex = mesh.vertices[point.index];
        const auto delta = subtract(glm::dvec3(vertex.position), glm::dvec3(origin.position));
        if (!delta) {
            return false;
        }
        point.projection = {(*delta)[x], (*delta)[y]};
        const auto a = cross2(point.projection, b2);
        const auto b = cross2(a2, point.projection);
        if (!a || !b || (i > 0 && i + 1 < side.ring.size() && *b <= 0.0) ||
            !sameAffineValue(*area, (*delta)[dropped], *a, (*tA)[dropped], *b, (*tB)[dropped]) ||
            !sameAffineValues(*area, origin.normal, vertex.normal, *a, first.normal, *b, second.normal) ||
            (mesh.hasUV && !sameAffineValues(*area, origin.uv, vertex.uv, *a, first.uv, *b, second.uv))) {
            return false;
        }
        if (i > 0) {
            const auto wedge = cross2(side.ring[i - 1].projection, point.projection);
            if (!wedge || *wedge <= 0.0) {
                return false;
            }
        }
    }
    // Each ray lies in one half-plane and consecutive wedges have positive
    // area. Together they cover this simple polygon up to its straight seam.
    return true;
}

bool triangulate(const Side& side, const Edges& existing, std::vector<uint64_t>& newDiagonals,
                 std::vector<Corners>& triangles) {
    auto ring = side.ring;
    while (ring.size() > 3) {
        bool clipped = false;
        for (std::size_t i = 0; i < ring.size(); ++i) {
            const auto previous = (i + ring.size() - 1) % ring.size();
            const auto next = (i + 1) % ring.size();
            const auto& a = ring[previous];
            const auto& b = ring[i];
            const auto& c = ring[next];
            const auto area = orientedArea(a.projection, b.projection, c.projection);
            const auto diagonal = edgeKey(a.geometry, c.geometry);
            if (!area || *area <= 0.0 || existing.contains(diagonal) ||
                std::ranges::find(newDiagonals, diagonal) != newDiagonals.end()) {
                continue;
            }
            bool obstructed = false;
            for (std::size_t j = 0; j < ring.size(); ++j) {
                if (j == previous || j == i || j == next) {
                    continue;
                }
                const auto& point = ring[j].projection;
                const auto ab = orientedArea(a.projection, b.projection, point);
                const auto bc = orientedArea(b.projection, c.projection, point);
                const auto ca = orientedArea(c.projection, a.projection, point);
                if (!ab || !bc || !ca || (*ab >= 0.0 && *bc >= 0.0 && *ca >= 0.0)) {
                    obstructed = true;
                    break;
                }
            }
            if (!obstructed) {
                newDiagonals.push_back(diagonal);
                triangles.push_back({a.index, b.index, c.index});
                ring.erase(ring.begin() + i);
                clipped = true;
                break;
            }
        }
        if (!clipped) {
            return false;
        }
    }
    const auto area = orientedArea(ring[0].projection, ring[1].projection, ring[2].projection);
    if (!area || *area <= 0.0) {
        return false;
    }
    triangles.push_back({ring[0].index, ring[1].index, ring[2].index});
    return true;
}

struct Patch {
    std::vector<std::size_t> slots;
    std::vector<Corners> replacements;
};

using Edit = std::vector<Patch>;

class SegmentIndex {
public:
    explicit SegmentIndex(const Incidence& incidence) : positions_(incidence.positions) {
        segments_.reserve(incidence.geometricEdges.size());
        for (const auto& [key, edge] : incidence.geometricEdges) {
            const auto a = static_cast<uint32_t>(key >> 32);
            const auto b = static_cast<uint32_t>(key);
            segments_.push_back({a, b});
        }
        build(0, segments_.size());
    }

    bool touchesUnsplitEdge(uint32_t middle) const {
        return touches(0, middle, positions_[middle].value);
    }

private:
    static constexpr std::size_t leafSize = 8;
    using Segment = std::array<uint32_t, 2>;

    struct Node {
        glm::dvec3 minimum{std::numeric_limits<double>::infinity()};
        glm::dvec3 maximum{-std::numeric_limits<double>::infinity()};
        std::size_t first = 0;
        std::size_t end = 0;
        std::size_t left = absent;
        std::size_t right = absent;
    };

    std::size_t build(std::size_t first, std::size_t end) {
        Node node;
        node.first = first;
        node.end = end;
        for (auto i = first; i < end; ++i) {
            const auto& [a, b] = segments_[i];
            const auto& p = positions_[a].value;
            const auto& q = positions_[b].value;
            node.minimum = glm::min(node.minimum, glm::min(p, q));
            node.maximum = glm::max(node.maximum, glm::max(p, q));
        }
        const auto id = nodes_.size();
        nodes_.push_back(node);
        if (end - first <= leafSize) {
            return id;
        }
        const auto size = node.maximum - node.minimum;
        glm::length_t axis = size.x >= size.y ? 0 : 1;
        if (size.z > size[axis]) {
            axis = 2;
        }
        const auto middle = first + (end - first) / 2;
        const auto center = [&](const Segment& segment) {
            return positions_[segment[0]].value[axis] + positions_[segment[1]].value[axis];
        };
        const auto begin = segments_.begin();
        std::nth_element(begin + first, begin + middle, begin + end, [&](const Segment& a, const Segment& b) {
            const double centerA = center(a), centerB = center(b);
            return centerA != centerB ? centerA < centerB : a < b;
        });
        const auto left = build(first, middle);
        const auto right = build(middle, end);
        nodes_[id].left = left;
        nodes_[id].right = right;
        return id;
    }

    bool touches(std::size_t id, uint32_t middle, const glm::dvec3& point) const {
        const auto& node = nodes_[id];
        if (glm::any(glm::lessThan(point, node.minimum)) ||
            glm::any(glm::greaterThan(point, node.maximum))) {
            return false;
        }
        if (node.left != absent) {
            return touches(node.left, middle, point) || touches(node.right, middle, point);
        }
        for (auto i = node.first; i < node.end; ++i) {
            const auto& [a, b] = segments_[i];
            if (a != middle && b != middle &&
                strictlyBetween(point, positions_[a].value, positions_[b].value).value_or(true)) {
                return true;
            }
        }
        return false;
    }

    const std::vector<Position>& positions_;
    std::vector<Segment> segments_;
    std::vector<Node> nodes_;
};

Edit planAt(const Mesh& mesh, const Incidence& incidence, uint32_t middle, std::vector<bool>& reserved,
            std::optional<SegmentIndex>& segments) {
    if (reserved[middle]) {
        return {};
    }
    auto sides = traceSides(mesh, incidence, middle);
    if (sides.empty() || !matchedSides(sides)) {
        return {};
    }
    const auto a = sides.front().ring.front().geometry;
    const auto b = sides.front().ring.back().geometry;
    if (incidence.geometricEdges.contains(edgeKey(a, b)) ||
        !strictlyBetween(incidence.positions[middle].value, incidence.positions[a].value,
                         incidence.positions[b].value).value_or(false)) {
        return {};
    }
    Edit edit;
    std::vector<uint64_t> newDiagonals;
    for (auto& side : sides) {
        if (std::ranges::any_of(side.ring, [&](const RingVertex& point) {
                return reserved[point.geometry] || incidence.positions[point.geometry].unsafe;
            }) || !validateSide(mesh, side)) {
            return {};
        }
        Patch patch;
        if (!triangulate(side, incidence.geometricEdges, newDiagonals, patch.replacements)) {
            return {};
        }
        patch.slots = std::move(side.faces);
        edit.push_back(std::move(patch));
    }
    if (!segments) {
        segments.emplace(incidence);
    }
    if (segments->touchesUnsplitEdge(middle)) {
        return {};
    }
    // Disjoint geometric closed stars make every accepted edit independent.
    reserved[middle] = true;
    for (const auto& side : sides) {
        for (const auto& point : side.ring) {
            reserved[point.geometry] = true;
        }
    }
    return edit;
}

std::vector<Edit> planEdits(const Mesh& mesh) {
    Incidence incidence;
    if (std::fegetround() != FE_TONEAREST || !collectIncidence(mesh, incidence)) {
        return {};
    }
    std::vector<bool> reserved(incidence.positions.size());
    std::optional<SegmentIndex> segments;
    std::vector<Edit> edits;
    for (std::size_t i = 0; i < incidence.positions.size(); ++i) {
        auto edit = planAt(mesh, incidence, static_cast<uint32_t>(i), reserved, segments);
        if (!edit.empty()) {
            edits.push_back(std::move(edit));
        }
    }
    return edits;
}

} // namespace

struct SeamAnalysis::Data {
    Mesh original;
    std::vector<Edit> edits;
    std::size_t minimum = 0;
};

SeamAnalysis::SeamAnalysis(std::shared_ptr<const Data> data) : data_(std::move(data)) {}

const Mesh& SeamAnalysis::original() const { return data_->original; }
std::size_t SeamAnalysis::minimumTriangles() const { return data_->minimum; }
std::size_t SeamAnalysis::removablePositions() const { return data_->edits.size(); }

SeamAnalysis analyzeSeamMesh(Mesh source) {
    auto data = std::make_shared<SeamAnalysis::Data>();
    data->edits = planEdits(source);
    data->minimum = source.indices.size() / 3;
    for (const auto& edit : data->edits) {
        data->minimum -= edit.size();
    }
    data->original = std::move(source);
    return SeamAnalysis(std::move(data));
}

Mesh generateSeamMesh(const SeamAnalysis& analysis, std::size_t targetTriangles) {
    auto mesh = analysis.original();
    const auto sourceTriangles = mesh.indices.size() / 3;
    auto currentTriangles = sourceTriangles;
    if (analysis.data_->edits.empty() || currentTriangles <= targetTriangles) {
        return mesh;
    }
    std::vector<bool> removed(sourceTriangles);
    for (const auto& edit : analysis.data_->edits) {
        if (currentTriangles <= targetTriangles) {
            break;
        }
        for (const auto& patch : edit) {
            for (std::size_t i = 0; i < patch.replacements.size(); ++i) {
                std::ranges::copy(patch.replacements[i], mesh.indices.begin() + patch.slots[i] * 3);
            }
            removed[patch.slots.back()] = true;
        }
        currentTriangles -= edit.size();
    }
    detail::compactEditedMesh(mesh, removed);
    return mesh;
}

Mesh simplifySeamMesh(const Mesh& mesh, std::size_t targetTriangles) {
    if (targetTriangles >= mesh.indices.size() / 3) {
        return mesh;
    }
    return generateSeamMesh(analyzeSeamMesh(mesh), targetTriangles);
}

} // namespace lodestar
