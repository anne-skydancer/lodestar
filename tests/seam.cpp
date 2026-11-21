#include <lodestar/lodestar.hpp>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
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

bool sameMesh(const Mesh& a, const Mesh& b) {
    return a.indices == b.indices && a.triangleMaterials == b.triangleMaterials &&
        a.triangleSurfaceIds == b.triangleSurfaceIds && a.vertexPositionIds == b.vertexPositionIds &&
        a.hasUV == b.hasUV && a.hasAdditionalUV == b.hasAdditionalUV &&
        a.hasUnrepresentedVertexAttributes == b.hasUnrepresentedVertexAttributes &&
        std::ranges::equal(a.vertices, b.vertices, [](const Vertex& left, const Vertex& right) {
            return left.position == right.position && left.normal == right.normal && left.uv == right.uv;
        });
}

void append(Mesh& mesh, float x, float y, bool right = false, float z = 0.0f) {
    const auto normal = right ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1);
    const auto uv = right ? glm::vec2(2.0f + x * 0.5f, 3.0f + y * 0.25f)
                          : glm::vec2(x * 0.25f, y * 0.5f);
    mesh.vertices.push_back({{x, y, z}, normal, uv});
    mesh.vertexPositionIds.push_back(mesh.vertices.size());
}

Mesh splitSides() {
    Mesh mesh;
    mesh.hasUV = true;
    append(mesh, 0, 0);
    append(mesh, 0, 1);
    append(mesh, -1, 0);
    append(mesh, 0, -1);
    append(mesh, 0, -0.0f, true);
    append(mesh, 0, -1, true);
    append(mesh, 1, 0, true);
    append(mesh, 0, 1, true);
    mesh.indices = {0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7};
    mesh.triangleMaterials = {3, 3, 7, 7};
    mesh.triangleSurfaceIds = {11, 11, 13, 13};
    return mesh;
}

std::map<uint32_t, double> materialAreas(const Mesh& mesh) {
    std::map<uint32_t, double> areas;
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
        const glm::dvec3 a(mesh.vertices.at(mesh.indices[i]).position);
        const glm::dvec3 b(mesh.vertices.at(mesh.indices[i + 1]).position);
        const glm::dvec3 c(mesh.vertices.at(mesh.indices[i + 2]).position);
        const auto area = glm::cross(b - a, c - a).z * 0.5;
        require(area > 0.0, "Positive output winding");
        areas[mesh.triangleMaterials.empty() ? 0 : mesh.triangleMaterials.at(i / 3)] += area;
    }
    return areas;
}

void requireFallback(const Mesh& mesh, std::string_view message) {
    const auto analysis = analyzeSeamMesh(mesh);
    require(analysis.removablePositions() == 0 && sameMesh(mesh, generateSeamMesh(analysis, 0)), message);
}

void testCoupledSeam() {
    const auto mesh = splitSides();
    const auto analysis = analyzeSeamMesh(mesh);
    require(analysis.removablePositions() == 1 && analysis.minimumTriangles() == 2,
            "One coupled position saves one face on each seam side");
    require(sameMesh(analysis.original(), mesh) && sameMesh(mesh, generateSeamMesh(analysis, 4)),
            "Source and no-change ceiling remain intact");
    const auto reduced = generateSeamMesh(analysis, 2);
    require(sameMesh(reduced, generateSeamMesh(analysis, 3)), "A coupled edit can cross an odd ceiling by one");
    require(reduced.vertices.size() == 6 && reduced.indices.size() == 6 &&
                reduced.triangleMaterials == std::vector<uint32_t>{3, 7} &&
                reduced.triangleSurfaceIds == std::vector<uint32_t>{11, 13} &&
                reduced.vertexPositionIds == std::vector<uint64_t>{2, 3, 4, 6, 7, 8},
            "Both seam sides retain original corner attributes, labels, and provenance");
    require(materialAreas(mesh) == materialAreas(reduced), "Each material's triangle union keeps its area");
    require(sameMesh(reduced, simplifySeamMesh(mesh, 2)), "Cached and one-shot outputs agree");
    for (std::size_t i = 0; i < reduced.vertices.size(); ++i) {
        const auto& vertex = reduced.vertices[i];
        const auto& original = mesh.vertices[reduced.vertexPositionIds[i] - 1];
        require(vertex.position == original.position && vertex.normal == original.normal && vertex.uv == original.uv,
                "No retained corner attribute is rebuilt");
    }
}

void testExteriorAndSharedIndex() {
    auto mesh = splitSides();
    mesh.indices.resize(6);
    mesh.triangleMaterials.resize(2);
    mesh.triangleSurfaceIds.resize(2);
    const auto exterior = simplifySeamMesh(mesh, 0);
    require(exterior.indices.size() == 3 && materialAreas(mesh) == materialAreas(exterior),
            "One-sided collinear exterior fan may remove its midpoint");

    mesh = splitSides();
    mesh.indices[6] = 0;
    mesh.indices[7] = 3;
    mesh.indices[9] = 0;
    mesh.indices[11] = 1;
    mesh.vertices[6].normal = {0, 0, 1};
    mesh.vertices[6].uv = {0.25f, 0.0f};
    const auto reduced = simplifySeamMesh(mesh, 0);
    require(reduced.vertices.size() == 4 && reduced.indices.size() == 6 && materialAreas(mesh) == materialAreas(reduced),
            "Both originally indexed material seam segments become one indexed segment");
}

void testConcaveFan() {
    Mesh mesh;
    mesh.hasUV = true;
    append(mesh, 0, 0);
    append(mesh, 0, 1);
    append(mesh, -2, 1);
    append(mesh, -1, 0);
    append(mesh, -2, -1);
    append(mesh, 0, -1);
    mesh.indices = {0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5};
    const auto result = simplifySeamMesh(mesh, 0);
    require(result.vertices.size() == 5 && result.indices.size() == 9 &&
                materialAreas(mesh) == materialAreas(result),
            "A nonconvex simple fan can be retriangulated exactly");
    mesh.vertices[3].position.z = 0.125f;
    requireFallback(mesh, "A non-planar fan keeps its original triangles");
}

void testIndependentSeams() {
    const auto tile = splitSides();
    Mesh mesh;
    mesh.hasUV = true;
    constexpr int copies = 128;
    for (int i = 0; i < copies; ++i) {
        const auto offset = static_cast<uint32_t>(mesh.vertices.size());
        for (auto vertex : tile.vertices) {
            vertex.position.x += static_cast<float>(i * 4);
            mesh.vertices.push_back(vertex);
            mesh.vertexPositionIds.push_back(mesh.vertices.size());
        }
        for (const auto index : tile.indices) {
            mesh.indices.push_back(offset + index);
        }
        mesh.triangleMaterials.insert(mesh.triangleMaterials.end(),
                                      tile.triangleMaterials.begin(), tile.triangleMaterials.end());
        mesh.triangleSurfaceIds.insert(mesh.triangleSurfaceIds.end(),
                                       tile.triangleSurfaceIds.begin(), tile.triangleSurfaceIds.end());
    }
    const auto analysis = analyzeSeamMesh(mesh);
    require(analysis.removablePositions() == copies && analysis.minimumTriangles() == copies * 2 &&
                generateSeamMesh(analysis, 0).indices.size() == copies * 6,
            "Disjoint seam edits remain independent across AABB partitions");
}

void testConservativeFallbacks() {
    const auto source = splitSides();
    auto mesh = source;
    mesh.vertices[4].uv.x = std::nextafter(mesh.vertices[4].uv.x, 3.0f);
    requireFallback(mesh, "One ULP of non-affine UV blocks both seam sides");
    mesh = source;
    mesh.vertices[4].normal.x = 0.125f;
    requireFallback(mesh, "Non-affine authored normal blocks both seam sides");
    mesh = source;
    mesh.vertices[0].position.x = 1.0f / 8192.0f;
    mesh.vertices[4].position.x = mesh.vertices[0].position.x;
    requireFallback(mesh, "A near-straight seam may not lose an off-line point");
    mesh = source;
    mesh.indices[6] = 0;
    mesh.indices[9] = 0;
    requireFallback(mesh, "Removing a shared midpoint must not break an indexed vertex-only contact");
    mesh = source;
    mesh.indices[7] = 3;
    mesh.indices[11] = 1;
    requireFallback(mesh, "Replacing two split segments must not weld their indexed sides");
    mesh = source;
    std::swap(mesh.indices[6], mesh.indices[7]);
    requireFallback(mesh, "Winding ambiguity blocks seam elimination");
    mesh = source;
    mesh.hasAdditionalUV = true;
    requireFallback(mesh, "Unknown UV components block edits");
    mesh = source;
    mesh.hasUnrepresentedVertexAttributes = true;
    requireFallback(mesh, "Unknown vertex attributes block edits");
    mesh = source;
    mesh.triangleMaterials.pop_back();
    requireFallback(mesh, "Malformed material labels preserve the original mesh");
    mesh = source;
    append(mesh, 0, -2, true);
    append(mesh, 0, 2, true);
    append(mesh, 2, 1, true);
    mesh.indices.insert(mesh.indices.end(), {8, 9, 10});
    mesh.triangleMaterials.push_back(9);
    mesh.triangleSurfaceIds.push_back(17);
    requireFallback(mesh, "A coincident longer edge makes coupling ambiguous at the candidate point");

    const int originalRound = std::fegetround();
    if (std::fesetround(FE_UPWARD) == 0) {
        const bool declined = analyzeSeamMesh(source).removablePositions() == 0;
        std::fesetround(originalRound);
        require(declined, "Unknown rounding assumptions cause exact fallback");
    }
}

} // namespace

int main() {
    try {
        testCoupledSeam();
        testExteriorAndSharedIndex();
        testConcaveFan();
        testIndependentSeams();
        testConservativeFallbacks();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
