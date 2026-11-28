#include <lodestar/lodestar.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <utility>

namespace lodestar {
namespace {

constexpr auto absent = std::numeric_limits<std::size_t>::max();
using Corners = std::array<uint32_t, 3>;

std::size_t nextHalfEdge(std::size_t edge) {
    return edge % 3 == 2 ? edge - 2 : edge + 1;
}

std::size_t previousHalfEdge(std::size_t edge) {
    return edge % 3 == 0 ? edge + 2 : edge - 1;
}

bool finite(const glm::vec3& position) {
    return std::isfinite(position.x) && std::isfinite(position.y) && std::isfinite(position.z);
}

std::size_t representative(std::vector<std::size_t>& components, std::size_t face) {
    while (components[face] != face) {
        components[face] = components[components[face]];
        face = components[face];
    }
    return face;
}

void join(std::vector<std::size_t>& components, std::size_t a, std::size_t b) {
    const auto first = representative(components, a);
    const auto second = representative(components, b);
    components[std::max(first, second)] = std::min(first, second);
}

struct IndexedEdge {
    uint64_t key;
    std::size_t halfEdge;
};

struct Incidence {
    std::vector<std::size_t> indexedPartner;
    std::vector<std::size_t> chartPartner;
    std::vector<std::size_t> faceComponents;
    std::vector<bool> invalidVertices;
};

uint64_t edgeKey(uint32_t from, uint32_t to) {
    return (uint64_t(std::min(from, to)) << 32) | std::max(from, to);
}

bool sameChartLabels(const Mesh& mesh, std::size_t a, std::size_t b) {
    return (mesh.triangleMaterials.empty() || mesh.triangleMaterials[a] == mesh.triangleMaterials[b]) &&
           (mesh.triangleSurfaceIds.empty() || mesh.triangleSurfaceIds[a] == mesh.triangleSurfaceIds[b]);
}

bool collectIncidence(const Mesh& mesh, Incidence& incidence) {
    incidence.indexedPartner.resize(mesh.indices.size(), absent);
    incidence.chartPartner.resize(mesh.indices.size(), absent);
    incidence.invalidVertices.resize(mesh.vertices.size());
    incidence.faceComponents.resize(mesh.indices.size() / 3);
    std::iota(incidence.faceComponents.begin(), incidence.faceComponents.end(), std::size_t{0});
    std::vector<IndexedEdge> edges;
    std::vector<Corners> faces;
    edges.reserve(mesh.indices.size());
    faces.reserve(incidence.faceComponents.size());

    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        Corners corners{mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]};
        const bool invalid = std::ranges::any_of(corners, [&](uint32_t index) {
            return index >= mesh.vertices.size() || !finite(mesh.vertices[index].position);
        });
        if (invalid) {
            return false;
        }
        const glm::dvec3 a(mesh.vertices[corners[0]].position);
        const glm::dvec3 b(mesh.vertices[corners[1]].position);
        const glm::dvec3 c(mesh.vertices[corners[2]].position);
        if (glm::cross(b - a, c - a) == glm::dvec3(0.0)) {
            for (const auto corner : corners) {
                incidence.invalidVertices[corner] = true;
            }
            continue;
        }
        for (std::size_t j = 0; j < corners.size(); ++j) {
            edges.push_back({edgeKey(corners[j], corners[(j + 1) % 3]), i + j});
        }
        std::ranges::sort(corners);
        faces.push_back(corners);
    }

    std::ranges::sort(faces);
    for (std::size_t i = 1; i < faces.size(); ++i) {
        if (faces[i - 1] == faces[i]) {
            for (const auto vertex : faces[i]) {
                incidence.invalidVertices[vertex] = true;
            }
        }
    }

    std::ranges::sort(edges, {}, &IndexedEdge::key);
    for (std::size_t first = 0; first < edges.size();) {
        auto end = first + 1;
        while (end < edges.size() && edges[end].key == edges[first].key) {
            ++end;
        }
        if (end - first == 2) {
            const auto a = edges[first].halfEdge;
            const auto b = edges[first + 1].halfEdge;
            if (mesh.indices[a] != mesh.indices[b]) {
                incidence.indexedPartner[a] = b;
                incidence.indexedPartner[b] = a;
                if (sameChartLabels(mesh, a / 3, b / 3)) {
                    incidence.chartPartner[a] = b;
                    incidence.chartPartner[b] = a;
                    join(incidence.faceComponents, a / 3, b / 3);
                }
            } else {
                incidence.invalidVertices[mesh.indices[a]] = true;
                incidence.invalidVertices[mesh.indices[nextHalfEdge(a)]] = true;
            }
        } else if (end - first > 2) {
            incidence.invalidVertices[edges[first].key >> 32] = true;
            incidence.invalidVertices[static_cast<uint32_t>(edges[first].key)] = true;
        }
        first = end;
    }
    return true;
}

struct Fan {
    bool valid = false;
    std::size_t entrance = absent;
    std::size_t exit = absent;
};

Fan traceFan(std::span<const std::size_t> corners, const std::vector<std::size_t>& partners) {
    auto start = corners.front();
    auto entrance = absent;
    for (const auto corner : corners) {
        const auto incoming = previousHalfEdge(corner);
        if (partners[incoming] == absent) {
            if (entrance != absent) {
                return {};
            }
            entrance = incoming;
            start = corner;
        }
    }

    auto current = start;
    for (std::size_t i = 0; i < corners.size(); ++i) {
        if (partners[current] == absent) {
            return entrance != absent && i + 1 == corners.size() ? Fan{true, entrance, current} : Fan{};
        }
        current = nextHalfEdge(partners[current]);
        if (current == start) {
            return entrance == absent && i + 1 == corners.size() ? Fan{true} : Fan{};
        }
    }
    return {};
}

template <typename Key, typename Function>
void forCornerGroups(const std::vector<std::size_t>& corners, Key key, Function function) {
    for (std::size_t first = 0; first < corners.size();) {
        auto end = first + 1;
        const auto group = key(corners[first]);
        while (end < corners.size() && key(corners[end]) == group) {
            ++end;
        }
        function(group, std::span(corners).subspan(first, end - first));
        first = end;
    }
}

void validateIndexedFans(const Mesh& mesh, Incidence& incidence, std::vector<std::size_t>& corners) {
    const auto vertex = [&](std::size_t corner) { return mesh.indices[corner]; };
    std::ranges::sort(corners, {}, vertex);
    forCornerGroups(corners, vertex, [&](auto index, auto fan) {
        if (!incidence.invalidVertices[index] && !traceFan(fan, incidence.indexedPartner).valid) {
            incidence.invalidVertices[index] = true;
        }
    });
}

struct Chart {
    std::size_t faces = 0;
    std::size_t vertices = 0;
    std::size_t boundaryVertices = 0;
    std::size_t boundaryEdges = 0;
    std::size_t boundaryLoops = 0;
    bool valid = true;
};

std::vector<Chart> collectCharts(const Mesh& mesh, Incidence& incidence, std::vector<std::size_t>& chartsByFace) {
    std::vector<std::size_t> chartForComponent(chartsByFace.size(), absent);
    std::vector<Chart> charts;
    for (std::size_t face = 0; face < chartsByFace.size(); ++face) {
        const auto component = representative(incidence.faceComponents, face);
        auto& id = chartForComponent[component];
        if (id == absent) {
            id = charts.size();
            charts.emplace_back();
        }
        chartsByFace[face] = id;
        auto& chart = charts[id];
        ++chart.faces;
        for (std::size_t corner = face * 3; corner < face * 3 + 3; ++corner) {
            chart.boundaryEdges += incidence.chartPartner[corner] == absent;
            chart.valid &= !incidence.invalidVertices[mesh.indices[corner]];
        }
    }
    return charts;
}

void validateChartFans(const Mesh& mesh, const Incidence& incidence, const std::vector<std::size_t>& chartsByFace,
                       std::vector<std::size_t>& corners, std::vector<Chart>& charts,
                       std::vector<std::size_t>& boundarySuccessors) {
    const auto key = [&](std::size_t corner) {
        return std::pair{chartsByFace[corner / 3], mesh.indices[corner]};
    };
    std::ranges::sort(corners, {}, key);
    forCornerGroups(corners, key, [&](const auto& group, auto cornersAtVertex) {
        auto& chart = charts[group.first];
        if (!chart.valid) {
            return;
        }
        const auto fan = traceFan(cornersAtVertex, incidence.chartPartner);
        if (!fan.valid) {
            chart.valid = false;
            return;
        }
        ++chart.vertices;
        if (fan.entrance != absent) {
            ++chart.boundaryVertices;
            boundarySuccessors[fan.entrance] = fan.exit;
        }
    });
}

void traceBoundaryLoops(const std::vector<std::size_t>& boundarySuccessors,
                        const std::vector<std::size_t>& chartsByFace, std::vector<Chart>& charts) {
    std::vector<bool> visited(boundarySuccessors.size());
    for (std::size_t edge = 0; edge < boundarySuccessors.size(); ++edge) {
        if (boundarySuccessors[edge] == absent || visited[edge]) {
            continue;
        }
        const auto chartId = chartsByFace[edge / 3];
        auto& chart = charts[chartId];
        if (!chart.valid) {
            continue;
        }
        std::size_t length = 0;
        auto current = edge;
        while (current != absent && chartsByFace[current / 3] == chartId && !visited[current]) {
            visited[current] = true;
            ++length;
            current = boundarySuccessors[current];
        }
        if (current != edge || length < 3) {
            chart.valid = false;
        } else {
            ++chart.boundaryLoops;
        }
    }
}

std::optional<std::size_t> chartLowerBound(const Chart& chart) {
    if (!chart.valid || chart.boundaryVertices != chart.boundaryEdges ||
        (chart.boundaryEdges == 0) != (chart.boundaryLoops == 0)) {
        return std::nullopt;
    }
    const auto faces = static_cast<int64_t>(chart.faces);
    const auto boundary = static_cast<int64_t>(chart.boundaryEdges);
    const auto edgeIncidences = 3 * faces + boundary;
    if (edgeIncidences % 2 != 0) {
        return std::nullopt;
    }
    // 3F = 2E - B. Keeping the chart type and boundary fixes chi and B, so
    // F = 2V - B - 2chi and each distinct retained boundary vertex contributes to V.
    const auto chi = static_cast<int64_t>(chart.vertices) - edgeIncidences / 2 + faces;
    const auto twiceGenus = 2 - static_cast<int64_t>(chart.boundaryLoops) - chi;
    if (twiceGenus < 0 || twiceGenus % 2 != 0) {
        return std::nullopt;
    }
    const auto minimum = std::max<int64_t>(boundary == 0 ? 4 : 1, boundary - 2 * chi);
    if (minimum > faces) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(minimum);
}

} // namespace

BoundaryBudgetReport inspectBoundaryBudget(const Mesh& mesh) {
    const auto faceCount = mesh.indices.size() / 3;
    constexpr auto incidenceLimit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 4;
    if (mesh.indices.size() % 3 != 0 || mesh.indices.size() > incidenceLimit ||
        (!mesh.triangleMaterials.empty() && mesh.triangleMaterials.size() != faceCount) ||
        (!mesh.triangleSurfaceIds.empty() && mesh.triangleSurfaceIds.size() != faceCount)) {
        return {};
    }
    Incidence incidence;
    if (!collectIncidence(mesh, incidence)) {
        return {};
    }
    std::vector<std::size_t> corners(mesh.indices.size());
    std::iota(corners.begin(), corners.end(), std::size_t{0});
    validateIndexedFans(mesh, incidence, corners);
    std::vector<std::size_t> chartsByFace(faceCount);
    auto charts = collectCharts(mesh, incidence, chartsByFace);
    std::vector<std::size_t> boundarySuccessors(corners.size(), absent);
    validateChartFans(mesh, incidence, chartsByFace, corners, charts, boundarySuccessors);
    traceBoundaryLoops(boundarySuccessors, chartsByFace, charts);

    BoundaryBudgetReport report;
    report.valid = true;
    report.sourceTriangles = faceCount;
    report.upperBoundTriangles = faceCount;
    report.chartCount = charts.size();
    for (const auto& chart : charts) {
        if (const auto minimum = chartLowerBound(chart)) {
            ++report.certifiedCharts;
            report.closedCharts += chart.boundaryEdges == 0;
            report.protectedBoundaryEdges += chart.boundaryEdges;
            report.boundaryLoops += chart.boundaryLoops;
            report.interiorVertices += chart.vertices - chart.boundaryVertices;
            report.lowerBoundTriangles += *minimum;
        } else {
            ++report.unsupportedCharts;
            report.unsupportedTriangles += chart.faces;
        }
    }
    report.unchangedFallbackLowerBoundTriangles = report.lowerBoundTriangles + report.unsupportedTriangles;
    return report;
}

} // namespace lodestar
