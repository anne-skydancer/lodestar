#include <lodestar/lodestar.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace lodestar;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

bool sameMesh(const Mesh& a, const Mesh& b) {
    return a.hasUV == b.hasUV && a.hasAdditionalUV == b.hasAdditionalUV &&
        a.indices == b.indices && a.triangleMaterials == b.triangleMaterials &&
        a.vertexPositionIds == b.vertexPositionIds && a.triangleSurfaceIds == b.triangleSurfaceIds &&
        a.hasUnrepresentedVertexAttributes == b.hasUnrepresentedVertexAttributes &&
        std::ranges::equal(a.vertices, b.vertices, [](const Vertex& left, const Vertex& right) {
            return left.position == right.position && left.normal == right.normal && left.uv == right.uv;
        });
}

Vertex vertexAt(float x, float y) {
    return {{x, y, 0.0f}, {x / 128.0f, y / 128.0f, 1.0f}, {x / 8.0f, y / 8.0f}};
}

Mesh grid(uint32_t cells, bool split = false, bool hole = false) {
    Mesh mesh;
    mesh.hasUV = true;
    const auto side = cells + 1;
    for (uint32_t y = 0; y < side; ++y) {
        for (uint32_t x = 0; x < side; ++x) {
            mesh.vertices.push_back(vertexAt(float(x), float(y)));
        }
    }
    for (uint32_t y = 0; y < cells; ++y) {
        for (uint32_t x = 0; x < cells; ++x) {
            if (hole && x == cells / 2 && y == cells / 2) {
                continue;
            }
            const auto first = y * side + x;
            mesh.indices.insert(mesh.indices.end(), {first, first + 1, first + side,
                                                     first + 1, first + side + 1, first + side});
            const auto material = split && x >= cells / 2 ? 23u : 17u;
            mesh.triangleMaterials.insert(mesh.triangleMaterials.end(), {material, material});
        }
    }
    return mesh;
}

using Point = std::array<float, 3>;
using Edge = std::pair<Point, Point>;

Point positionOf(const Mesh& mesh, uint32_t index) {
    const auto& position = mesh.vertices.at(index).position;
    return {position.x, position.y, position.z};
}

std::map<Edge, std::vector<uint32_t>> boundaries(const Mesh& mesh) {
    std::map<Edge, std::vector<uint32_t>> edges;
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        for (std::size_t j = 0; j < 3; ++j) {
            auto a = positionOf(mesh, mesh.indices[i + j]);
            auto b = positionOf(mesh, mesh.indices[i + (j + 1) % 3]);
            if (b < a) {
                std::swap(a, b);
            }
            const auto material = mesh.triangleMaterials.empty() ? 0u : mesh.triangleMaterials.at(i / 3);
            edges[{a, b}].push_back(material);
        }
    }
    std::erase_if(edges, [](const auto& entry) {
        const auto& materials = entry.second;
        return materials.size() == 2 && materials.front() == materials.back();
    });
    for (auto& [edge, materials] : edges) {
        std::ranges::sort(materials);
    }
    return edges;
}

std::map<uint32_t, double> materialAreas(const Mesh& mesh) {
    std::map<uint32_t, double> areas;
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        const auto a = positionOf(mesh, mesh.indices[i]);
        const auto b = positionOf(mesh, mesh.indices[i + 1]);
        const auto c = positionOf(mesh, mesh.indices[i + 2]);
        const double area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
        require(area > 0.0, "Retriangulated mesh must keep the original winding and positive area");
        const auto material = mesh.triangleMaterials.empty() ? 0u : mesh.triangleMaterials.at(i / 3);
        areas[material] += area;
    }
    return areas;
}

void requirePreservation(const Mesh& source, const Mesh& result) {
    require(boundaries(source) == boundaries(result), "Boundary and material-seam segments must be preserved");
    require(materialAreas(source) == materialAreas(result), "Each material's area must be preserved");
    for (const auto& vertex : result.vertices) {
        const auto expected = vertexAt(vertex.position.x, vertex.position.y);
        require(vertex.position == expected.position && vertex.normal == expected.normal && vertex.uv == expected.uv,
                "Positions, normals, and UV coordinates must match existing vertices");
    }
}

void testCachedTargets() {
    const auto source = grid(4);
    const auto plan = analyzePlanarMesh(source);
    require(sameMesh(plan.original(), source), "Plan owns an unchanged copy of the source");
    require(plan.minimumTriangles() == 14, "4-by-4 grid reduces to a boundary-only triangulation");
    const auto coarsest = generatePlanarMesh(plan, 0);
    require(coarsest.indices.size() == plan.minimumTriangles() * 3 && coarsest.vertices.size() == 16,
            "Only original boundary vertices remain at the minimum");
    requirePreservation(source, coarsest);
    const auto partial = generatePlanarMesh(plan, 19);
    require(partial.indices.size() == 18 * 3 && partial.vertices.size() == 18,
            "A cached edit prefix meets the triangle budget with two-triangle edits");
    requirePreservation(source, partial);
    require(sameMesh(partial, generatePlanarMesh(plan, 19)), "Plan replay is deterministic");
    require(sameMesh(source, generatePlanarMesh(plan, 40)), "Target above input returns the unchanged source");
    require(sameMesh(coarsest, simplifyPlanarMesh(source, 0)), "One-call simplification matches cached plan");
}

void testProvenSplitCorners() {
    Mesh source;
    source.hasUV = true;
    source.vertices = {vertexAt(0, 0), vertexAt(2, 0), vertexAt(2, 2), vertexAt(0, 2),
                       vertexAt(1, 1), vertexAt(1, 1)};
    source.indices = {0, 1, 4, 1, 2, 4, 2, 3, 5, 3, 0, 5};
    source.triangleMaterials = {17, 17, 17, 17};
    source.triangleSurfaceIds = {7, 7, 7, 7};

    const auto unproven = inspectTopology(source);
    require(unproven.valid && unproven.compatibleSplitEdges == 2 && unproven.recoverableSplitEdges == 0,
            "Exact coincidence does not prove shared source identity");
    require(sameMesh(source, simplifyPlanarMesh(source, 2)), "Unproven split is retained exactly");

    source.vertexPositionIds = {1, 2, 3, 4, 5, 5};
    const auto proven = inspectTopology(source);
    require(proven.valid && proven.recoverableSplitEdges == 2, "Two split edges have proven endpoints");
    const auto plan = analyzePlanarMesh(source);
    require(sameMesh(plan.original(), source) && sameMesh(generatePlanarMesh(plan, 4), source),
            "Preparation and full-budget generation retain the immutable source");
    const auto reduced = generatePlanarMesh(plan, 2);
    require(plan.minimumTriangles() == 2 && reduced.indices.size() == 6 &&
                reduced.vertexPositionIds == std::vector<uint64_t>{1, 2, 3, 4} &&
                reduced.triangleSurfaceIds == std::vector<uint32_t>{7, 7},
            "Proven split removal preserves position identities and surface labels");
    requirePreservation(source, reduced);

    source.triangleSurfaceIds[2] = source.triangleSurfaceIds[3] = 8;
    require(sameMesh(source, simplifyPlanarMesh(source, 2)), "Split crossing scopes is retained exactly");
    source.triangleSurfaceIds[2] = source.triangleSurfaceIds[3] = 7;
    source.hasUnrepresentedVertexAttributes = true;
    require(sameMesh(source, simplifyPlanarMesh(source, 2)), "Unsupported vertex attributes prevent editing");
}

void testSeamsAndHoles() {
    const auto split = grid(4, true);
    const auto reduced = simplifyPlanarMesh(split, 0);
    require(reduced.indices.size() < split.indices.size(), "Interiors simplify on either side of material boundaries");
    requirePreservation(split, reduced);

    const auto holed = grid(6, false, true);
    const auto ring = simplifyPlanarMesh(holed, 0);
    require(ring.indices.size() < holed.indices.size(), "A region with a hole simplifies outside its boundary");
    requirePreservation(holed, ring);

    Mesh notch;
    notch.hasUV = true;
    notch.vertices.push_back(vertexAt(0, 0));
    for (const auto& p : std::array<glm::vec2, 6>{{{2, 0}, {2, 2}, {0, 1}, {-2, 2}, {-2, -2}, {2, -2}}}) {
        notch.vertices.push_back(vertexAt(p.x, p.y));
    }
    for (uint32_t i = 1; i <= 6; ++i) {
        notch.indices.insert(notch.indices.end(), {0, i, i % 6 + 1});
    }
    const auto clipped = simplifyPlanarMesh(notch, 0);
    require(clipped.indices.size() == 4 * 3, "Concave closed fan can be retriangulated");
    requirePreservation(notch, clipped);

    auto uvSeam = grid(4);
    std::array<uint32_t, 5> seam;
    for (uint32_t y = 0; y < seam.size(); ++y) {
        auto original = uvSeam.vertices[y * 5 + 2];
        original.uv.x += 1.0f;
        seam[y] = static_cast<uint32_t>(uvSeam.vertices.size());
        uvSeam.vertices.push_back(original);
        uvSeam.vertices[y * 5 + 3].uv.x += 1.0f;
        uvSeam.vertices[y * 5 + 4].uv.x += 1.0f;
    }
    for (std::size_t i = 0; i < uvSeam.indices.size(); ++i) {
        const auto cell = i / 6;
        auto& corner = uvSeam.indices[i];
        if (cell % 4 >= 2 && corner % 5 == 2) {
            corner = seam[corner / 5];
        }
    }
    const auto seamResult = simplifyPlanarMesh(uvSeam, 0);
    require(seamResult.indices.size() < uvSeam.indices.size(), "Planar fans simplify beside UV seams");
    for (const auto position : seam) {
        const auto& seamVertex = uvSeam.vertices[position];
        const bool originalSide = std::ranges::any_of(seamResult.vertices, [&](const Vertex& vertex) {
            return vertex.position == seamVertex.position && vertex.uv == seamVertex.uv - glm::vec2(1, 0);
        });
        const bool offsetSide = std::ranges::any_of(seamResult.vertices, [&](const Vertex& vertex) {
            return vertex.position == seamVertex.position && vertex.uv == seamVertex.uv;
        });
        require(originalSide && offsetSide, "Duplicated UV seam vertices remain on both sides");
    }
}

void testUnsupportedRegions() {
    const auto source = grid(2);
    auto mesh = source;
    mesh.hasAdditionalUV = true;
    require(sameMesh(mesh, simplifyPlanarMesh(mesh, 0)), "Extra UV components keep the original mesh");

    mesh = source;
    mesh.vertices[4].uv.x += 0.125f;
    require(sameMesh(mesh, simplifyPlanarMesh(mesh, 0)), "Non-affine UV interpolation prevents edits");
    mesh = source;
    mesh.vertices[4].normal.x += 0.125f;
    require(sameMesh(mesh, simplifyPlanarMesh(mesh, 0)), "Non-affine normal interpolation prevents edits");
    mesh = source;
    mesh.vertices[4].position.z += 0.125f;
    require(sameMesh(mesh, simplifyPlanarMesh(mesh, 0)), "A crease prevents edits");

    mesh = source;
    mesh.indices.insert(mesh.indices.end(), {1, 4, 3});
    mesh.triangleMaterials.push_back(17);
    require(sameMesh(mesh, simplifyPlanarMesh(mesh, 0)), "Nonmanifold fan keeps its original faces");
    mesh = source;
    mesh.vertices.push_back(vertexAt(1, -1));
    mesh.vertices.push_back(vertexAt(2, -1));
    mesh.indices.insert(mesh.indices.end(), {2, 1, 9, 1, 2, 10});
    mesh.triangleMaterials.insert(mesh.triangleMaterials.end(), {17, 17});
    require(sameMesh(mesh, simplifyPlanarMesh(mesh, 0)), "Fans touching nonmanifold boundary edges keep their faces");

    mesh = source;
    mesh.vertices[4].position.z = 1e-5f;
    mesh.vertices.push_back(vertexAt(100000, 0));
    mesh.vertices.push_back(vertexAt(100001, 0));
    mesh.vertices.push_back(vertexAt(100000, 1));
    mesh.indices.insert(mesh.indices.end(), {9, 10, 11});
    mesh.triangleMaterials.push_back(17);
    require(sameMesh(mesh, simplifyPlanarMesh(mesh, 0)), "Distant geometry does not obscure small features");
    mesh = source;
    std::swap(mesh.indices[3], mesh.indices[4]);
    require(sameMesh(mesh, simplifyPlanarMesh(mesh, 0)), "Inconsistent fan winding keeps its original faces");
    mesh = source;
    mesh.triangleMaterials.pop_back();
    require(sameMesh(mesh, simplifyPlanarMesh(mesh, 0)), "Malformed material metadata keeps source exactly");
}

} // namespace

int main() {
    try {
        testCachedTargets();
        testProvenSplitCorners();
        testSeamsAndHoles();
        testUnsupportedRegions();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
