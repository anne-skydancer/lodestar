#include <lodestar/lodestar.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
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
    return a.hasUV == b.hasUV && a.hasAdditionalUV == b.hasAdditionalUV &&
        a.indices == b.indices && a.triangleMaterials == b.triangleMaterials &&
        a.vertexPositionIds == b.vertexPositionIds && a.triangleSurfaceIds == b.triangleSurfaceIds &&
        a.hasUnrepresentedVertexAttributes == b.hasUnrepresentedVertexAttributes &&
        std::ranges::equal(a.vertices, b.vertices, [](const Vertex& left, const Vertex& right) {
            return left.position == right.position && left.normal == right.normal && left.uv == right.uv;
        });
}

void requireFallback(const Asset& asset, SphereEligibility expected) {
    require(asset.eligibility == expected, "Unexpected sphere eligibility");
    require(sameMesh(asset.original, generateMesh(asset, Mode::Sphere, 4)),
            "Ineligible sphere mode must retain the original mesh");
}

void testSample(const Asset& sample) {
    const auto& original = sample.original;
    require(sample.eligibility == SphereEligibility::Eligible, "Sample is eligible");
    require(sample.fit.valid && sample.fit.maxRelativeError < 1e-6f, "Sample fits the unit sphere");
    require(sample.maxFaceInwardRelativeError && *sample.maxFaceInwardRelativeError > 0.0f &&
                *sample.maxFaceInwardRelativeError < 0.01f,
            "Sample face deviation is small and positive");
    require(original.indices.size() == 3 * 3968, "Sample triangle count");
    require(sameMesh(original, generateMesh(sample, Mode::Original, minimumQuality)),
            "Original mode preserves all mesh data");

    require(baselineFraction(minimumQuality - 1) == 1.0f / 32.0f, "Minimum baseline quality clamps");
    require(baselineFraction(maximumQuality + 1) == 1.0f, "Maximum baseline quality clamps");
    const auto baseline = generateMesh(sample, Mode::Baseline, 4);
    require(!baseline.indices.empty() && baseline.indices.size() < original.indices.size(),
            "Baseline reduces triangle count");
    require(baseline.vertices.size() < original.vertices.size() &&
                *std::ranges::max_element(baseline.indices) < baseline.vertices.size(),
            "Baseline compacts vertex storage and updates references");
    require(sameMesh(baseline, generateMesh(sample, Mode::Baseline, 4)), "Baseline is repeatable");
    require(sameMesh(original, generateMesh(sample, Mode::Baseline, maximumQuality)),
            "Highest baseline quality returns the original");

    const auto analytic = generateMesh(sample, Mode::Sphere, 4);
    require(analytic.indices.size() == 3 * 960 && analytic.vertices.size() == 482,
            "Analytic sphere triangle and vertex counts at quality 4");
    require(generateMesh(sample, Mode::Sphere, minimumQuality - 1).indices.size() == 3 * 48,
            "Analytic sphere minimum quality clamps");
    require(generateMesh(sample, Mode::Sphere, maximumQuality + 1).indices.size() == 3 * 9024,
            "Analytic sphere maximum quality clamps");
    for (size_t i = 0; i < analytic.indices.size(); i += 3) {
        const auto a = analytic.vertices.at(analytic.indices[i]).position - sample.fit.center;
        const auto b = analytic.vertices.at(analytic.indices[i + 1]).position - sample.fit.center;
        const auto c = analytic.vertices.at(analytic.indices[i + 2]).position - sample.fit.center;
        require(glm::dot(a, glm::cross(b, c)) > 0.0f, "Analytic triangle winding faces outward");
    }
}

Mesh cubeMesh() {
    constexpr std::array<std::array<unsigned, 6>, 6> faces{
        {{0, 2, 1, 1, 2, 3}, {4, 5, 6, 5, 7, 6}, {0, 1, 4, 1, 5, 4},
         {2, 6, 3, 3, 6, 7}, {0, 4, 2, 2, 4, 6}, {1, 3, 5, 3, 7, 5}}};
    Mesh cube;
    for (const auto& face : faces) {
        for (const unsigned corner : face) {
            const glm::vec3 position{corner & 1 ? 1.0f : -1.0f, corner & 2 ? 1.0f : -1.0f,
                                     corner & 4 ? 1.0f : -1.0f};
            cube.indices.push_back(static_cast<uint32_t>(cube.vertices.size()));
            cube.vertices.push_back({position, glm::vec3(0.0f), glm::vec2(0.0f)});
        }
    }
    return cube;
}

void testFallbacks(const Asset& sample) {
    auto mesh = sample.original;
    mesh.hasUV = true;
    mesh.vertices.front().uv = glm::vec2(0.5f, 0.25f);
    const auto textured = analyze(std::move(mesh));
    requireFallback(textured, SphereEligibility::HasTextureCoordinates);
    require(textured.maxFaceInwardRelativeError == sample.maxFaceInwardRelativeError,
            "Texture coordinates do not affect geometric error");

    mesh = sample.original;
    mesh.indices.clear();
    for (size_t i = 0; i < sample.original.indices.size(); i += 3) {
        const std::array triangle{sample.original.indices[i], sample.original.indices[i + 1],
                                  sample.original.indices[i + 2]};
        if (std::ranges::find(triangle, 0u) == triangle.end()) {
            mesh.indices.insert(mesh.indices.end(), triangle.begin(), triangle.end());
        }
    }
    const auto open = analyze(std::move(mesh));
    requireFallback(open, SphereEligibility::IncompleteSurface);
    require(open.maxFaceInwardRelativeError.has_value(), "Open sphere still has a face error");

    mesh = sample.original;
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        std::swap(mesh.indices[i], mesh.indices[i + 1]);
    }
    const auto reversed = analyze(std::move(mesh));
    requireFallback(reversed, SphereEligibility::IncompleteSurface);
    require(reversed.maxFaceInwardRelativeError == sample.maxFaceInwardRelativeError,
            "Winding does not affect geometric error");

    const auto cube = analyze(cubeMesh());
    requireFallback(cube, SphereEligibility::IncompleteSurface);
    require(cube.fit.valid && cube.fit.maxRelativeError < 1e-6f, "Cube vertices alone fit a sphere");
    const float cubeFaceError = 1.0f - 1.0f / std::sqrt(3.0f);
    require(cube.maxFaceInwardRelativeError &&
                std::abs(*cube.maxFaceInwardRelativeError - cubeFaceError) < 1e-6f,
            "Cube face error matches the geometric distance to a face center");

    mesh = sample.original;
    for (auto& vertex : mesh.vertices) {
        vertex.position.z = 0.0f;
    }
    const auto planar = analyze(std::move(mesh));
    requireFallback(planar, SphereEligibility::NoFit);
    require(!planar.fit.valid && !planar.maxFaceInwardRelativeError, "Planar fit is rank-deficient");

    mesh = sample.original;
    mesh.vertices.front().position.y *= 1.5f;
    const auto radial = analyze(std::move(mesh));
    requireFallback(radial, SphereEligibility::RadialError);
    require(!radial.maxFaceInwardRelativeError, "Radial outliers do not trigger surface assessment");
}

void testSceneImport(const char* path) {
    const auto world = loadMesh(path, ImportSpace::Scene);
    const auto normalized = loadMesh(path);
    const auto x = [](const Vertex& vertex) { return vertex.position.x; };
    const auto [worldMin, worldMax] = std::ranges::minmax_element(world.vertices, {}, x);
    const auto [normalizedMin, normalizedMax] = std::ranges::minmax_element(normalized.vertices, {}, x);
    require(x(*worldMin) == 5.0f && x(*worldMax) == 11.0f, "Scene import retains node translations");
    require(x(*normalizedMin) == -1.0f && x(*normalizedMax) == 1.0f, "Default import still normalizes the mesh");
    require(world.hasUV && !world.hasAdditionalUV && world.indices.size() == 8 * 3 &&
                world.triangleMaterials.size() == 8 && world.triangleSurfaceIds.size() == 8 &&
                world.triangleSurfaceIds.front() != world.triangleSurfaceIds.back() && world.vertexPositionIds.empty(),
            "Imported mesh has UV0 and eight material-tagged triangles");
    const auto red = world.triangleMaterials[0];
    const auto blue = world.triangleMaterials[4];
    require(red != blue && std::ranges::count(world.triangleMaterials, red) == 4 &&
                std::ranges::count(world.triangleMaterials, blue) == 4,
            "Material labels survive scene instancing");

    const auto plan = analyzePlanarMesh(world);
    const auto result = generatePlanarMesh(plan, 4);
    require(plan.minimumTriangles() == 4 && result.vertices.size() == 8 && result.indices.size() == 4 * 3,
            "Planar reduction removes each panel's interior vertex");
    require(result.hasUV && std::ranges::count(result.triangleMaterials, red) == 2 &&
                std::ranges::count(result.triangleMaterials, blue) == 2,
            "Planar reduction keeps the two materials and UV mapping");
    for (size_t triangle = 0; triangle < result.indices.size() / 3; ++triangle) {
        const float centerX = result.triangleMaterials[triangle] == red ? 6.0f : 10.0f;
        for (size_t corner = 0; corner < 3; ++corner) {
            const auto& vertex = result.vertices.at(result.indices[triangle * 3 + corner]);
            require(vertex.position.z == 3.0f && vertex.normal == glm::vec3(0, 0, 1) &&
                        vertex.uv == glm::vec2((vertex.position.x - centerX + 1) / 2, (vertex.position.y - 1) / 2),
                    "Planar output retains source position, winding orientation, and UV0");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto sample = lodestar::sampleAsset();
        testSample(sample);
        testFallbacks(sample);
        if (argc > 1) testSceneImport(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
