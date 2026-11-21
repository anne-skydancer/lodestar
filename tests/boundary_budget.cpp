#include <lodestar/lodestar.hpp>

#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace lodestar;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void addVertex(Mesh& mesh, float x, float y, float z = 0.0f) {
    mesh.vertices.push_back({{x, y, z}, {0, 0, 1}, {x, y}});
}

Mesh grid(uint32_t cells, bool hole = false) {
    Mesh mesh;
    const auto side = cells + 1;
    for (uint32_t y = 0; y < side; ++y) {
        for (uint32_t x = 0; x < side; ++x) {
            addVertex(mesh, float(x), float(y));
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
        }
    }
    return mesh;
}

void testDiskAndHole() {
    const auto disk = inspectBoundaryBudget(grid(4));
    require(disk.valid && disk.chartCount == 1 && disk.certifiedCharts == 1 &&
                disk.protectedBoundaryEdges == 16 && disk.boundaryLoops == 1 && disk.interiorVertices == 9,
            "Disk certifies one closed boundary and its removable interior vertex count");
    require(disk.lowerBoundTriangles == 14 && disk.unchangedFallbackLowerBoundTriangles == 14 &&
                disk.upperBoundTriangles == 32 && disk.sourceTriangles == 32,
            "Disk floor is B - 2; source is an achievable upper bound");

    const auto ring = inspectBoundaryBudget(grid(5, true));
    require(ring.valid && ring.certifiedCharts == 1 && ring.boundaryLoops == 2 &&
                ring.protectedBoundaryEdges == 24 && ring.interiorVertices == 12,
            "Annulus keeps both outer and hole boundary loops");
    require(ring.lowerBoundTriangles == 24 && ring.upperBoundTriangles == 48,
            "For one hole the Euler lower bound equals the boundary edge count");

    const auto tinyRing = inspectBoundaryBudget(grid(3, true));
    require(tinyRing.valid && tinyRing.certifiedCharts == 1 && tinyRing.interiorVertices == 0 &&
                tinyRing.lowerBoundTriangles == tinyRing.upperBoundTriangles &&
                tinyRing.upperBoundTriangles == 16,
            "A chart with no interior vertices cannot reduce while keeping its boundaries");
}

void testProtectedCuts() {
    auto mesh = grid(4);
    for (std::size_t cell = 0; cell < 16; ++cell) {
        const auto material = cell % 4 < 2 ? 17u : 23u;
        mesh.triangleMaterials.insert(mesh.triangleMaterials.end(), {material, material});
    }
    const auto material = inspectBoundaryBudget(mesh);
    require(material.valid && material.certifiedCharts == 2 && material.boundaryLoops == 2 &&
                material.protectedBoundaryEdges == 24 && material.lowerBoundTriangles == 20,
            "Material seam counts as a fixed boundary on both sides");

    mesh.triangleSurfaceIds = std::move(mesh.triangleMaterials);
    mesh.triangleMaterials.clear();
    const auto surface = inspectBoundaryBudget(mesh);
    require(surface.valid && surface.certifiedCharts == 2 && surface.protectedBoundaryEdges == 24 &&
                surface.lowerBoundTriangles == 20,
            "Source surface scopes cannot be merged across an indexed seam");

    mesh = grid(1);
    mesh.hasUV = true;
    mesh.vertices.push_back(mesh.vertices[1]);
    mesh.vertices.push_back(mesh.vertices[2]);
    mesh.vertices[4].uv.x += 1.0f;
    mesh.indices[3] = 4;
    mesh.indices[5] = 5;
    mesh.vertexPositionIds = {1, 2, 3, 4, 2, 3};
    const auto split = inspectBoundaryBudget(mesh);
    require(split.valid && split.certifiedCharts == 2 && split.protectedBoundaryEdges == 6 &&
                split.boundaryLoops == 2 && split.lowerBoundTriangles == 2,
            "Each split corner side is a separate protected indexed boundary");
}

void testClosedChart() {
    Mesh mesh;
    addVertex(mesh, 0, 0, 0);
    addVertex(mesh, 1, 0, 0);
    addVertex(mesh, 0, 1, 0);
    addVertex(mesh, 0, 0, 1);
    mesh.indices = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
    const auto tetrahedron = inspectBoundaryBudget(mesh);
    require(tetrahedron.valid && tetrahedron.certifiedCharts == 1 && tetrahedron.closedCharts == 1 &&
                tetrahedron.boundaryLoops == 0 && tetrahedron.interiorVertices == 4 &&
                tetrahedron.lowerBoundTriangles == 4 && tetrahedron.upperBoundTriangles == 4,
            "A closed simplicial chart has a nontrivial floor of four triangles");
}

void testUnsupportedCharts() {
    auto mesh = grid(1);
    std::swap(mesh.indices[3], mesh.indices[4]);
    const auto winding = inspectBoundaryBudget(mesh);
    require(winding.valid && winding.certifiedCharts == 0 && winding.unsupportedTriangles == 2 &&
                winding.lowerBoundTriangles == 0 && winding.unchangedFallbackLowerBoundTriangles == 2,
            "Winding conflict is unknown unless the unsafe chart must be retained");

    mesh = grid(1);
    addVertex(mesh, 0, 0, 1);
    mesh.indices.insert(mesh.indices.end(), {1, 2, 4});
    const auto nonManifold = inspectBoundaryBudget(mesh);
    require(nonManifold.valid && nonManifold.certifiedCharts == 0 && nonManifold.unsupportedTriangles == 3,
            "Three indexed faces on one edge cannot certify a manifold");

    mesh = grid(1);
    mesh.indices = {0, 1, 2, 0, 2, 1};
    const auto doubles = inspectBoundaryBudget(mesh);
    require(doubles.valid && doubles.certifiedCharts == 0 && doubles.unsupportedTriangles == 2,
            "Back-to-back duplicate face pair is not a closed simplicial chart");

    mesh = grid(2);
    mesh.indices = {0, 1, 4, 0, 7, 6};
    const auto pointContact = inspectBoundaryBudget(mesh);
    require(pointContact.valid && pointContact.chartCount == 2 && pointContact.unsupportedCharts == 2,
            "Two charts sharing one disconnected indexed fan both stay unsupported");

    mesh = {};
    addVertex(mesh, 0, 0);
    addVertex(mesh, 1, 0);
    addVertex(mesh, 1, 1);
    addVertex(mesh, -1, 1);
    addVertex(mesh, -1, 0);
    mesh.indices = {0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 0};
    const auto pinch = inspectBoundaryBudget(mesh);
    require(pinch.valid && pinch.chartCount == 1 && pinch.unsupportedCharts == 1 &&
                pinch.unsupportedTriangles == 4,
            "Disconnected vertex link in one edge-connected chart remains unsupported");

    mesh = grid(4);
    for (std::size_t cell = 0; cell < 16; ++cell) {
        const auto material = cell == 5 || cell == 10 ? 23u : 17u;
        mesh.triangleMaterials.insert(mesh.triangleMaterials.end(), {material, material});
    }
    const auto cutPinch = inspectBoundaryBudget(mesh);
    require(cutPinch.valid && cutPinch.chartCount == 3 && cutPinch.certifiedCharts == 2 &&
                cutPinch.unsupportedTriangles == 28 && cutPinch.lowerBoundTriangles == 4,
            "Cuts can split a chart's vertex link even if the full indexed mesh is manifold");

    mesh = grid(1);
    mesh.indices.insert(mesh.indices.end(), {0, 0, 0});
    addVertex(mesh, 3, 0);
    addVertex(mesh, 4, 0);
    addVertex(mesh, 3, 1);
    mesh.indices.insert(mesh.indices.end(), {4, 5, 6});
    const auto mixed = inspectBoundaryBudget(mesh);
    require(mixed.valid && mixed.certifiedCharts == 1 && mixed.unsupportedCharts == 2 &&
                mixed.unsupportedTriangles == 3 && mixed.lowerBoundTriangles == 1 &&
                mixed.unchangedFallbackLowerBoundTriangles == 4,
            "Degenerate triangles and affected charts do not contaminate unrelated certified bounds");
}

void testInvalidInput() {
    const auto empty = inspectBoundaryBudget({});
    require(empty.valid && empty.sourceTriangles == 0 && empty.lowerBoundTriangles == 0,
            "Empty indexed surface has a zero budget");
    auto mesh = grid(1);
    mesh.indices.push_back(0);
    require(!inspectBoundaryBudget(mesh).valid, "Incomplete indexed triangles are invalid");
    mesh.indices.pop_back();
    mesh.triangleSurfaceIds = {1};
    require(!inspectBoundaryBudget(mesh).valid, "Bad surface metadata is structurally invalid");
    mesh.triangleSurfaceIds.clear();
    mesh.indices.back() = 33;
    require(!inspectBoundaryBudget(mesh).valid, "Out-of-range indices are invalid");
    mesh = grid(1);
    mesh.vertices.front().position.x = std::numeric_limits<float>::infinity();
    require(!inspectBoundaryBudget(mesh).valid, "Non-finite referenced positions are invalid");
}

} // namespace

int main() {
    try {
        testDiskAndHole();
        testProtectedCuts();
        testClosedChart();
        testUnsupportedCharts();
        testInvalidInput();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
