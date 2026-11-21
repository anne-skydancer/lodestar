#include <lodestar/lodestar.hpp>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace lodestar;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void append(Mesh& mesh, const glm::vec3& position, uint64_t id) {
    mesh.vertices.push_back({position, {0.0f, 0.0f, 1.0f}, {position.x, position.y}});
    mesh.vertexPositionIds.push_back(id);
}

Mesh splitFan() {
    Mesh mesh;
    mesh.hasUV = true;
    const std::array<glm::vec3, 4> ring{{{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}}};
    for (uint32_t i = 0; i < ring.size(); ++i) {
        const auto next = (i + 1) % ring.size();
        append(mesh, glm::vec3(0.0f), 1);
        append(mesh, ring[i], i + 2);
        append(mesh, ring[next], next + 2);
        mesh.indices.insert(mesh.indices.end(), {i * 3, i * 3 + 1, i * 3 + 2});
        mesh.triangleMaterials.push_back(17);
        mesh.triangleSurfaceIds.push_back(33);
    }
    mesh.vertices[4].position.z = -0.0f;
    return mesh;
}

Mesh splitPair() {
    Mesh mesh;
    mesh.hasUV = true;
    append(mesh, {0, 0, 0}, 1);
    append(mesh, {1, 0, 0}, 2);
    append(mesh, {0, 1, 0}, 3);
    append(mesh, {1, 0, 0}, 2);
    append(mesh, {0, 0, 0}, 1);
    append(mesh, {0, -1, 0}, 4);
    mesh.indices = {0, 1, 2, 3, 4, 5};
    mesh.triangleMaterials = {17, 17};
    mesh.triangleSurfaceIds = {33, 33};
    return mesh;
}

void testProvenRecovery() {
    const auto mesh = splitFan();
    const auto report = inspectTopology(mesh);
    require(report.valid && report.geometricEdges == 8 && report.boundaryEdges == 4 &&
                report.indexedBoundaryEdges == 12 && report.sharedEdges == 0,
            "Exact topology groups signed-zero position duplicates without a tolerance");
    require(report.compatibleSplitEdges == 4 && report.recoverableSplitEdges == 4,
            "Corroborated split fan can recover all four inner edges");
    const auto plan = analyzePlanarMesh(mesh);
    require(plan.minimumTriangles() == 2 && plan.original().indices == mesh.indices,
            "Planar mode uses proven connectivity and retains the original surface");
    const auto generated = generatePlanarMesh(plan, 2);
    require(generated.indices.size() == 6 && generated.vertices.size() == 4 &&
                generated.triangleMaterials == std::vector<uint32_t>{17, 17} &&
                generated.triangleSurfaceIds == std::vector<uint32_t>{33, 33} &&
                generated.vertexPositionIds.size() == generated.vertices.size(),
            "Recovered fan replay preserves material/surface provenance and remaps vertex origins");

    auto unknown = mesh;
    unknown.vertexPositionIds.clear();
    const auto unproven = inspectTopology(unknown);
    require(unproven.valid && unproven.compatibleSplitEdges == 4 && unproven.recoverableSplitEdges == 0,
            "Coincident corners with matching attributes are insufficient evidence");
    require(simplifyPlanarMesh(unknown, 2).indices == unknown.indices,
            "Unproven split fan stays unchanged even inside one imported scope");

    auto separate = mesh;
    separate.triangleSurfaceIds = {1, 2, 3, 4};
    const auto crossSurface = inspectTopology(separate);
    require(crossSurface.valid && crossSurface.compatibleSplitEdges == 4 &&
                crossSurface.crossSurfaceSplitEdges == 4 && crossSurface.recoverableSplitEdges == 0,
            "Position identities do not join distinct source scopes");

    auto hidden = mesh;
    hidden.hasUnrepresentedVertexAttributes = true;
    const auto hiddenReport = inspectTopology(hidden);
    require(hiddenReport.compatibleSplitEdges == 4 && hiddenReport.recoverableSplitEdges == 0 &&
                simplifyPlanarMesh(hidden, 2).indices == hidden.indices,
            "Unrepresented vertex attributes block edge recovery and planar edits");
    hidden.hasUnrepresentedVertexAttributes = false;
    hidden.hasAdditionalUV = true;
    require(inspectTopology(hidden).recoverableSplitEdges == 0, "Extra UV components block edge recovery");
}

void testSeamsAndWinding() {
    auto pair = splitPair();
    pair.vertices[3].normal = {0, 1, 0};
    pair.vertices[3].uv.x += 0.25f;
    pair.triangleMaterials[1] = 19;
    const auto seams = inspectTopology(pair);
    require(seams.valid && seams.uvSeamEdges == 1 && seams.normalSeamEdges == 1 &&
                seams.materialSeamEdges == 1 && seams.compatibleSplitEdges == 0 &&
                seams.recoverableSplitEdges == 0,
            "The same geometric edge can carry more than one kind of seam");

    pair = splitPair();
    pair.indices[3] = 1;
    pair.indices[4] = 0;
    pair.triangleMaterials[1] = 19;
    const auto indexed = inspectTopology(pair);
    require(indexed.sharedEdges == 1 && indexed.indexedBoundaryEdges == 4 &&
                indexed.materialSeamEdges == 1 && indexed.compatibleSplitEdges == 0,
            "A material seam can already be shared in indexed topology");

    pair = splitPair();
    std::swap(pair.indices[3], pair.indices[4]);
    const auto reversed = inspectTopology(pair);
    require(reversed.windingConflictEdges == 1 && reversed.compatibleSplitEdges == 0 &&
                reversed.recoverableSplitEdges == 0,
            "Same-direction incidence cannot be recovered");

    pair = splitPair();
    pair.vertexPositionIds.clear();
    pair.vertices[3].position.x = std::nextafter(1.0f, 2.0f);
    const auto near = inspectTopology(pair);
    require(near.valid && near.geometricEdges == 6 && near.boundaryEdges == 6 &&
                near.compatibleSplitEdges == 0,
            "Near-coincident positions are never geometrically matched");
}

void testAmbiguities() {
    auto pair = splitPair();
    pair.vertices[5].position = pair.vertices[2].position;
    const auto doubled = inspectTopology(pair);
    require(doubled.valid && doubled.ambiguousEdges == 3 && doubled.nonManifoldEdges == 0 &&
                doubled.recoverableSplitEdges == 0,
            "Back-to-back duplicate faces are ambiguous even with two opposed occurrences");

    pair = splitPair();
    append(pair, {1, 0, 1}, 5);
    pair.indices.insert(pair.indices.end(), {0, 1, 6});
    pair.triangleMaterials.push_back(17);
    pair.triangleSurfaceIds.push_back(33);
    const auto nonManifold = inspectTopology(pair);
    require(nonManifold.valid && nonManifold.ambiguousEdges == 1 &&
                nonManifold.nonManifoldEdges == 1 && nonManifold.recoverableSplitEdges == 0,
            "An edge with three geometric incidences is ambiguous and nonmanifold");

    pair = splitPair();
    append(pair, {-1, 0, 0}, 5);
    append(pair, {-1, -1, 0}, 6);
    pair.indices.insert(pair.indices.end(), {4, 6, 7});
    pair.triangleMaterials.push_back(17);
    pair.triangleSurfaceIds.push_back(33);
    const auto disconnected = inspectTopology(pair);
    require(disconnected.valid && disconnected.compatibleSplitEdges == 1 &&
                disconnected.recoverableSplitEdges == 0,
            "Recovery cannot make a merged vertex contain disconnected fans");

    auto fan = splitFan();
    fan.indices[11] = 1;
    append(fan, {0, 0, 0}, 1);
    append(fan, {-1, -1, 0}, 2);
    append(fan, {0, 0, 1}, 6);
    fan.indices.insert(fan.indices.end(), {12, 13, 14});
    fan.triangleMaterials.push_back(17);
    fan.triangleSurfaceIds.push_back(33);
    const auto indirect = inspectTopology(fan);
    require(indirect.valid && indirect.compatibleSplitEdges == 3 && indirect.nonManifoldEdges == 1 &&
                indirect.recoverableSplitEdges == 0,
            "Transitive corner recovery cannot implicitly merge a rejected geometric edge");
}

void testMalformedInputs() {
    auto pair = splitPair();
    pair.vertexPositionIds.pop_back();
    require(!inspectTopology(pair).valid, "Mismatched vertex origin metadata is invalid");
    pair = splitPair();
    pair.vertexPositionIds[1] = 1;
    require(!inspectTopology(pair).valid, "One origin cannot denote different exact positions");
    pair = splitPair();
    pair.triangleSurfaceIds.pop_back();
    require(!inspectTopology(pair).valid, "Mismatched triangle source metadata is invalid");
    pair = splitPair();
    pair.indices[0] = 100;
    require(!inspectTopology(pair).valid, "Out-of-range indices are invalid");
    pair = splitPair();
    pair.vertices[2].position.y = std::numeric_limits<float>::infinity();
    require(!inspectTopology(pair).valid, "Non-finite source positions are invalid");

    pair = splitPair();
    pair.indices.insert(pair.indices.end(), {0, 1, 0});
    pair.triangleMaterials.push_back(17);
    pair.triangleSurfaceIds.push_back(33);
    const auto degenerate = inspectTopology(pair);
    require(degenerate.valid && degenerate.degenerateTriangles == 1 && degenerate.geometricEdges == 5 &&
                degenerate.compatibleSplitEdges == 1 && degenerate.recoverableSplitEdges == 0,
            "Zero-area faces are counted separately and prevent recovery at touched corners");
}

} // namespace

int main() {
    try {
        testProvenRecovery();
        testSeamsAndWinding();
        testAmbiguities();
        testMalformedInputs();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
