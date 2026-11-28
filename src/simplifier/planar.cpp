#include <lodestar/lodestar.hpp>

#include <lodestar/mesh_compaction.hpp>
#include <lodestar/topology.hpp>

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lodestar {
namespace {

constexpr auto noFace = std::numeric_limits<std::size_t>::max();
constexpr auto noVertex = std::numeric_limits<uint32_t>::max();
constexpr std::size_t maxFanSize = 64;
constexpr double positionTolerance = 1e-7;
constexpr double attributeTolerance = 1e-6;

using Corners = std::array<uint32_t, 3>;

struct Face {
    Corners corners;
    uint32_t material;
    uint32_t surface;
    std::size_t region = noFace;
};

struct Edit {
    std::vector<std::size_t> slots;
    std::vector<Corners> replacements;
};

template <glm::length_t Size>
bool finite(const glm::vec<Size, float>& value) {
    for (glm::length_t i = 0; i < Size; ++i) {
        if (!std::isfinite(value[i])) {
            return false;
        }
    }
    return true;
}

template <glm::length_t Size>
bool attributeMatches(const glm::vec<Size, double>& actual, const glm::vec<Size, double>& expected) {
    return glm::all(glm::lessThanEqual(glm::abs(actual - expected),
                                      glm::vec<Size, double>(attributeTolerance)));
}

double cross2(const glm::dvec2& a, const glm::dvec2& b) {
    return a.x * b.y - a.y * b.x;
}

double orient(const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c) {
    return cross2(b - a, c - a);
}

struct Projection {
    glm::length_t x = 0;
    glm::length_t y = 1;

    glm::dvec2 operator()(const glm::dvec3& p) const { return {p[x], p[y]}; }
};

struct PlaneMapping {
    Projection project;
    glm::dvec3 normal;
    glm::dvec3 origin;
    glm::dvec2 edgeA;
    glm::dvec2 edgeB;
    glm::dvec3 normalOrigin;
    glm::dvec3 normalA;
    glm::dvec3 normalB;
    glm::dvec2 uvOrigin;
    glm::dvec2 uvA;
    glm::dvec2 uvB;
    double area = 0.0;
    double edgeScale = 0.0;
    uint32_t material = 0;
    uint32_t surface = 0;

    PlaneMapping(const Mesh& mesh, const Face& face) : material(face.material), surface(face.surface) {
        const auto& a = mesh.vertices[face.corners[0]];
        const auto& b = mesh.vertices[face.corners[1]];
        const auto& c = mesh.vertices[face.corners[2]];
        origin = glm::dvec3(a.position);
        const auto ab = glm::dvec3(b.position) - origin;
        const auto ac = glm::dvec3(c.position) - origin;
        const auto unnormalized = glm::cross(ab, ac);
        const auto magnitudes = glm::abs(unnormalized);
        glm::length_t axis = magnitudes.x >= magnitudes.y ? 0 : 1;
        if (magnitudes.z > magnitudes[axis]) {
            axis = 2;
        }
        if (magnitudes[axis] == 0.0) {
            return;
        }
        project = {(axis + 1) % 3, (axis + 2) % 3};
        if (unnormalized[axis] < 0.0) {
            std::swap(project.x, project.y);
        }
        normal = glm::normalize(unnormalized);
        edgeA = project(ab);
        edgeB = project(ac);
        area = cross2(edgeA, edgeB);
        edgeScale = std::max({glm::length(ab), glm::length(ac), glm::length(ac - ab)});
        normalOrigin = glm::dvec3(a.normal);
        normalA = glm::dvec3(b.normal) - normalOrigin;
        normalB = glm::dvec3(c.normal) - normalOrigin;
        // Untextured meshes need no UV compatibility test.
        if (mesh.hasUV) {
            uvOrigin = glm::dvec2(a.uv);
            uvA = glm::dvec2(b.uv) - uvOrigin;
            uvB = glm::dvec2(c.uv) - uvOrigin;
        }
    }

    bool contains(const Mesh& mesh, const Face& face, double planeTolerance) const {
        if (face.material != material || face.surface != surface) {
            return false;
        }
        std::array<glm::dvec2, 3> points;
        for (std::size_t i = 0; i < points.size(); ++i) {
            const auto& vertex = mesh.vertices[face.corners[i]];
            const auto offset = glm::dvec3(vertex.position) - origin;
            if (std::abs(glm::dot(offset, normal)) > planeTolerance) {
                return false;
            }
            points[i] = project(offset);
            const double a = cross2(points[i], edgeB) / area;
            const double b = cross2(edgeA, points[i]) / area;
            if (!attributeMatches(glm::dvec3(vertex.normal), normalOrigin + a * normalA + b * normalB) ||
                (mesh.hasUV && !attributeMatches(glm::dvec2(vertex.uv), uvOrigin + a * uvA + b * uvB))) {
                return false;
            }
        }
        return orient(points[0], points[1], points[2]) > 0.0;
    }
};

struct Edge {
    std::size_t first;
    uint32_t direction;
    std::size_t second = noFace;
    bool nonmanifold = false;
};

uint64_t edgeKey(uint32_t from, uint32_t to) {
    return (uint64_t(std::min(from, to)) << 32) | std::max(from, to);
}

using EdgeMap = std::unordered_map<uint64_t, Edge>;
using VertexFaces = std::vector<std::vector<std::size_t>>;

bool prepare(const Mesh& mesh, std::vector<Face>& faces, VertexFaces& incident, EdgeMap& edges,
             double& extent) {
    const std::size_t triangles = mesh.indices.size() / 3;
    if (mesh.hasAdditionalUV || mesh.hasUnrepresentedVertexAttributes || mesh.indices.size() % 3 != 0 ||
        mesh.vertices.size() > noVertex ||
        (!mesh.triangleMaterials.empty() && mesh.triangleMaterials.size() != triangles) ||
        (!mesh.triangleSurfaceIds.empty() && mesh.triangleSurfaceIds.size() != triangles)) {
        return false;
    }
    if (triangles < 3) {
        return false;
    }

    faces.reserve(triangles);
    incident.resize(mesh.vertices.size());
    edges.reserve(mesh.indices.size());
    const double infinity = std::numeric_limits<double>::infinity();
    glm::dvec3 minimum(infinity), maximum(-infinity);

    for (std::size_t index = 0; index < triangles; ++index) {
        const Corners corners{mesh.indices[index * 3], mesh.indices[index * 3 + 1], mesh.indices[index * 3 + 2]};
        const auto material = mesh.triangleMaterials.empty() ? 0u : mesh.triangleMaterials[index];
        for (const auto corner : corners) {
            if (corner >= mesh.vertices.size()) {
                return false;
            }
            const auto& vertex = mesh.vertices[corner];
            if (!finite(vertex.position) || !finite(vertex.normal) || (mesh.hasUV && !finite(vertex.uv))) {
                return false;
            }
            const glm::dvec3 point(vertex.position);
            minimum = glm::min(minimum, point);
            maximum = glm::max(maximum, point);
        }
        if (corners[0] == corners[1] || corners[1] == corners[2] || corners[2] == corners[0]) {
            return false;
        }
        for (std::size_t i = 0; i < corners.size(); ++i) {
            const auto from = corners[i];
            const auto to = corners[(i + 1) % corners.size()];
            incident[from].push_back(index);
            const auto [entry, inserted] = edges.try_emplace(edgeKey(from, to), Edge{index, from});
            if (!inserted) {
                auto& edge = entry->second;
                if (edge.second != noFace || edge.direction == from) {
                    edge.nonmanifold = true;
                } else {
                    edge.second = index;
                }
            }
        }
        const auto surface = mesh.triangleSurfaceIds.empty() ? 0u : mesh.triangleSurfaceIds[index];
        faces.push_back({corners, material, surface});
    }
    const auto size = maximum - minimum;
    extent = std::max({size.x, size.y, size.z});
    return extent > 0.0;
}

struct RegionMap {
    std::vector<Projection> projections;
    std::vector<bool> blockedVertices;
};

RegionMap groupFaces(const Mesh& mesh, std::vector<Face>& faces, const EdgeMap& edges,
                     double extent) {
    RegionMap regions;
    regions.blockedVertices.resize(mesh.vertices.size());
    for (const auto& face : faces) {
        for (std::size_t i = 0; i < face.corners.size(); ++i) {
            const auto key = edgeKey(face.corners[i], face.corners[(i + 1) % face.corners.size()]);
            if (edges.at(key).nonmanifold) {
                for (const auto corner : face.corners) {
                    regions.blockedVertices[corner] = true;
                }
                break;
            }
        }
    }
    std::vector<std::size_t> queue;
    for (std::size_t i = 0; i < faces.size(); ++i) {
        auto& seed = faces[i];
        if (seed.region != noFace) {
            continue;
        }
        const PlaneMapping mapping(mesh, seed);
        const auto region = regions.projections.size();
        regions.projections.push_back(mapping.project);
        seed.region = region;
        if (mapping.area <= 0.0) {
            for (const auto corner : seed.corners) {
                regions.blockedVertices[corner] = true;
            }
            continue;
        }

        // The region's plane and attribute map stay fixed across every later edit.
        const double planeTolerance = positionTolerance * std::min(extent, mapping.edgeScale);
        queue.clear();
        queue.push_back(i);
        for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
            const auto current = queue[cursor];
            const auto& face = faces[current];
            for (std::size_t j = 0; j < face.corners.size(); ++j) {
                const auto key = edgeKey(face.corners[j], face.corners[(j + 1) % face.corners.size()]);
                const auto& edge = edges.at(key);
                if (edge.nonmanifold || edge.second == noFace) {
                    continue;
                }
                const auto neighbor = edge.first == current ? edge.second : edge.first;
                auto& adjacent = faces[neighbor];
                if (adjacent.region == noFace && mapping.contains(mesh, adjacent, planeTolerance)) {
                    adjacent.region = region;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    return regions;
}

bool hasCorner(const Face& face, uint32_t vertex) {
    return std::ranges::find(face.corners, vertex) != face.corners.end();
}

bool edgeExists(const std::vector<Face>& faces, const VertexFaces& incident, uint32_t a, uint32_t b) {
    return std::ranges::any_of(incident[a], [&](std::size_t index) {
        return hasCorner(faces[index], b);
    });
}

bool triangleExists(const std::vector<Face>& faces, const VertexFaces& incident, const Corners& corners) {
    return std::ranges::any_of(incident[corners[0]], [&](std::size_t index) {
        return hasCorner(faces[index], corners[1]) && hasCorner(faces[index], corners[2]);
    });
}

struct RingVertex {
    uint32_t index;
    glm::dvec2 position;
};

bool triangulate(std::vector<RingVertex> ring, const std::vector<Face>& faces,
                 const VertexFaces& incident, std::vector<Corners>& replacements) {
    while (ring.size() > 3) {
        bool clipped = false;
        for (std::size_t i = 0; i < ring.size(); ++i) {
            const auto previous = (i + ring.size() - 1) % ring.size();
            const auto next = (i + 1) % ring.size();
            const auto& a = ring[previous];
            const auto& b = ring[i];
            const auto& c = ring[next];
            if (orient(a.position, b.position, c.position) <= 0.0 ||
                edgeExists(faces, incident, a.index, c.index)) {
                continue;
            }
            const bool obstructed = std::ranges::any_of(ring, [&](const RingVertex& point) {
                return point.index != a.index && point.index != b.index && point.index != c.index &&
                    orient(a.position, b.position, point.position) >= 0.0 &&
                    orient(b.position, c.position, point.position) >= 0.0 &&
                    orient(c.position, a.position, point.position) >= 0.0;
            });
            if (obstructed) {
                continue;
            }
            replacements.push_back({a.index, b.index, c.index});
            ring.erase(ring.begin() + i);
            clipped = true;
            break;
        }
        if (!clipped) {
            return false;
        }
    }
    const Corners last{ring[0].index, ring[1].index, ring[2].index};
    if (orient(ring[0].position, ring[1].position, ring[2].position) <= 0.0 ||
        triangleExists(faces, incident, last)) {
        return false;
    }
    replacements.push_back(last);
    return true;
}

Edit planRemoval(uint32_t center, const Mesh& mesh, const std::vector<Face>& faces,
                 const VertexFaces& incident, const RegionMap& regions) {
    const auto& adjacent = incident[center];
    if (adjacent.size() < 3 || adjacent.size() > maxFanSize || regions.blockedVertices[center]) {
        return {};
    }
    const auto region = faces[adjacent.front()].region;
    const auto project = regions.projections[region];
    const auto origin = project(glm::dvec3(mesh.vertices[center].position));

    // The directed edges opposite the center must describe one closed vertex fan.
    std::vector<std::pair<uint32_t, uint32_t>> links;
    links.reserve(adjacent.size());
    for (const auto index : adjacent) {
        const auto& face = faces[index];
        if (face.region != region) {
            return {};
        }
        const auto corner = std::ranges::find(face.corners, center) - face.corners.begin();
        links.emplace_back(face.corners[(corner + 1) % 3], face.corners[(corner + 2) % 3]);
    }
    std::ranges::sort(links);
    for (std::size_t i = 1; i < links.size(); ++i) {
        if (links[i - 1].first == links[i].first) {
            return {};
        }
    }
    const uint32_t first = links.front().first;
    uint32_t current = first;
    std::vector<RingVertex> ring;
    ring.reserve(links.size());
    double sweep = 0.0;
    for (std::size_t i = 0; i < links.size(); ++i) {
        const auto link = std::ranges::lower_bound(links, current, {}, &std::pair<uint32_t, uint32_t>::first);
        if (link == links.end() || link->first != current) {
            return {};
        }
        const auto a = project(glm::dvec3(mesh.vertices[current].position)) - origin;
        const auto b = project(glm::dvec3(mesh.vertices[link->second].position)) - origin;
        const double cross = cross2(a, b);
        if (cross <= 0.0) {
            return {};
        }
        sweep += std::atan2(cross, glm::dot(a, b));
        ring.push_back({current, a});
        current = link->second;
        if (current == first && i + 1 != links.size()) {
            return {};
        }
    }
    // Positive wedges covering one revolution give a simple polygon around the center.
    if (current != first || std::abs(sweep - glm::two_pi<double>()) > 1e-9) {
        return {};
    }

    Edit edit;
    edit.replacements.reserve(ring.size() - 2);
    if (!triangulate(std::move(ring), faces, incident, edit.replacements)) {
        return {};
    }
    edit.slots = adjacent;
    return edit;
}

void applyToTopology(const Edit& edit, std::vector<Face>& faces, VertexFaces& incident) {
    for (const auto slot : edit.slots) {
        for (const auto vertex : faces[slot].corners) {
            auto& references = incident[vertex];
            references.erase(std::ranges::find(references, slot));
        }
    }
    for (std::size_t i = 0; i < edit.replacements.size(); ++i) {
        const auto slot = edit.slots[i];
        auto& face = faces[slot];
        face.corners = edit.replacements[i];
        for (const auto vertex : face.corners) {
            incident[vertex].push_back(slot);
        }
    }
}

std::vector<Edit> planEdits(const Mesh& mesh) {
    std::vector<Face> faces;
    VertexFaces incident;
    RegionMap regions;
    {
        EdgeMap edges;
        double extent = 0.0;
        if (!prepare(mesh, faces, incident, edges, extent)) {
            return {};
        }
        regions = groupFaces(mesh, faces, edges, extent);
    }

    std::deque<uint32_t> queue;
    std::vector<bool> queued(mesh.vertices.size());
    const auto enqueue = [&](uint32_t vertex) {
        const auto degree = incident[vertex].size();
        if (!queued[vertex] && !regions.blockedVertices[vertex] && degree >= 3 && degree <= maxFanSize) {
            queued[vertex] = true;
            queue.push_back(vertex);
        }
    };
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        enqueue(static_cast<uint32_t>(i));
    }
    std::vector<Edit> edits;
    while (!queue.empty()) {
        const auto center = queue.front();
        queue.pop_front();
        queued[center] = false;
        auto edit = planRemoval(center, mesh, faces, incident, regions);
        if (edit.slots.empty()) {
            continue;
        }
        applyToTopology(edit, faces, incident);
        for (const auto& replacement : edit.replacements) {
            for (const auto vertex : replacement) {
                enqueue(vertex);
            }
        }
        edits.push_back(std::move(edit));
    }
    return edits;
}

void remapCorners(Mesh& mesh, const std::vector<uint32_t>& canonical) {
    if (canonical.empty()) return;
    for (auto& index : mesh.indices) index = canonical[index];
}

} // namespace

struct PlanarAnalysis::Data {
    Mesh original;
    std::vector<Edit> edits;
    std::vector<uint32_t> canonicalCorners;
};

PlanarAnalysis::PlanarAnalysis(std::shared_ptr<const Data> data) : data_(std::move(data)) {}

const Mesh& PlanarAnalysis::original() const { return data_->original; }

std::size_t PlanarAnalysis::minimumTriangles() const {
    return data_->original.indices.size() / 3 - data_->edits.size() * 2;
}

PlanarAnalysis analyzePlanarMesh(Mesh source) {
    auto data = std::make_shared<PlanarAnalysis::Data>();
    data->original = std::move(source);
    const auto& original = data->original;
    if (original.vertexPositionIds.empty()) {
        data->edits = planEdits(original);
    } else {
        auto topology = detail::analyzeTopology(original);
        if (topology.report.valid) {
            if (topology.canonicalCorners.empty()) {
                data->edits = planEdits(original);
            } else {
                auto connected = original;
                remapCorners(connected, topology.canonicalCorners);
                data->edits = planEdits(connected);
                if (!data->edits.empty()) data->canonicalCorners = std::move(topology.canonicalCorners);
            }
        }
    }
    return PlanarAnalysis(std::move(data));
}

Mesh generatePlanarMesh(const PlanarAnalysis& analysis, std::size_t targetTriangles) {
    Mesh mesh = analysis.original();
    const auto sourceTriangles = mesh.indices.size() / 3;
    const auto reduction = sourceTriangles - std::min(targetTriangles, sourceTriangles);
    // Each interior vertex removal saves two triangles. Round up to meet the budget.
    const auto edits = std::min(analysis.data_->edits.size(), reduction / 2 + reduction % 2);
    if (edits == 0) {
        return mesh;
    }

    remapCorners(mesh, analysis.data_->canonicalCorners);
    std::vector<bool> removed(sourceTriangles);
    for (std::size_t i = 0; i < edits; ++i) {
        const auto& edit = analysis.data_->edits[i];
        for (std::size_t j = 0; j < edit.replacements.size(); ++j) {
            std::ranges::copy(edit.replacements[j], mesh.indices.begin() + edit.slots[j] * 3);
        }
        removed[edit.slots[edit.slots.size() - 2]] = true;
        removed[edit.slots.back()] = true;
    }
    detail::compactEditedMesh(mesh, removed);
    return mesh;
}

Mesh simplifyPlanarMesh(const Mesh& mesh, std::size_t targetTriangles) {
    if (targetTriangles >= mesh.indices.size() / 3) {
        return mesh;
    }
    return generatePlanarMesh(analyzePlanarMesh(mesh), targetTriangles);
}

} // namespace lodestar
