#include <lodestar/lodestar.hpp>
#include <lodestar/lodestar_c.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

struct lodestar_asset {
    explicit lodestar_asset(lodestar::Asset asset) : value(std::move(asset)) {}

    lodestar::Asset value;
    mutable std::once_flag preparePlanar;
    mutable std::optional<lodestar::PlanarAnalysis> planar;
    mutable std::once_flag prepareSeam;
    mutable std::optional<lodestar::SeamAnalysis> seam;
};

struct lodestar_mesh {
    std::vector<lodestar_vertex> vertices;
    std::vector<uint32_t> indices;
    bool hasUV = false;
    bool hasAdditionalUV = false;
    std::vector<uint32_t> triangleMaterials;
    std::vector<uint64_t> vertexPositionIds;
    std::vector<uint32_t> triangleSurfaceIds;
    bool hasUnrepresentedVertexAttributes = false;
};

struct lodestar_collision_mesh {
    std::vector<lodestar_collision_vertex> positions;
    std::vector<uint32_t> indices;
    size_t skippedDegenerateTriangles = 0;
};

struct lodestar_convex_compound {
    std::vector<lodestar_collision_mesh> hulls;
};

static_assert(sizeof(lodestar_collision_vertex) == sizeof(float[3]));
static_assert(alignof(lodestar_collision_vertex) == alignof(float));

namespace {

thread_local char lastError[1024]{};

lodestar_status fail(lodestar_status status, const char* message) noexcept {
    std::snprintf(lastError, sizeof(lastError), "%s", message);
    return status;
}

template <typename Operation>
lodestar_status checked(Operation operation) noexcept {
    lastError[0] = '\0';
    try {
        operation();
        return LODESTAR_OK;
    } catch (const std::invalid_argument& error) {
        return fail(LODESTAR_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(LODESTAR_ERROR, error.what());
    } catch (...) {
        return fail(LODESTAR_ERROR, "Unknown native error");
    }
}

lodestar::Mode modeFromAbi(lodestar_mode mode) {
    switch (mode) {
    case LODESTAR_ORIGINAL: return lodestar::Mode::Original;
    case LODESTAR_BASELINE: return lodestar::Mode::Baseline;
    case LODESTAR_SPHERE: return lodestar::Mode::Sphere;
    case LODESTAR_PLANAR: return lodestar::Mode::Planar;
    case LODESTAR_SEAM: return lodestar::Mode::Seam;
    }
    throw std::invalid_argument("Invalid mesh representation");
}

lodestar_sphere_eligibility eligibility(lodestar::SphereEligibility value) {
    switch (value) {
    case lodestar::SphereEligibility::Eligible: return LODESTAR_SPHERE_ELIGIBLE;
    case lodestar::SphereEligibility::HasTextureCoordinates: return LODESTAR_SPHERE_HAS_UV;
    case lodestar::SphereEligibility::NoFit: return LODESTAR_SPHERE_NO_FIT;
    case lodestar::SphereEligibility::RadialError: return LODESTAR_SPHERE_RADIAL_ERROR;
    case lodestar::SphereEligibility::IncompleteSurface: return LODESTAR_SPHERE_INCOMPLETE;
    }
    throw std::logic_error("Unknown sphere eligibility");
}

lodestar::ImportSpace spaceFromAbi(lodestar_import_space space) {
    switch (space) {
    case LODESTAR_IMPORT_NORMALIZED: return lodestar::ImportSpace::Normalized;
    case LODESTAR_IMPORT_SCENE: return lodestar::ImportSpace::Scene;
    }
    throw std::invalid_argument("Invalid mesh import coordinate space");
}

lodestar::Mesh planarMesh(const lodestar_asset& asset, size_t targetTriangles) {
    if (targetTriangles >= asset.value.original.indices.size() / 3) return asset.value.original;
    std::call_once(asset.preparePlanar, [&] {
        asset.planar.emplace(lodestar::analyzePlanarMesh(asset.value.original));
    });
    return lodestar::generatePlanarMesh(*asset.planar, targetTriangles);
}

lodestar::Mesh seamMesh(const lodestar_asset& asset, size_t targetTriangles) {
    if (targetTriangles >= asset.value.original.indices.size() / 3) return asset.value.original;
    std::call_once(asset.prepareSeam, [&] {
        asset.seam.emplace(lodestar::analyzeSeamMesh(asset.value.original));
    });
    return lodestar::generateSeamMesh(*asset.seam, targetTriangles);
}

template <typename Value>
std::vector<Value> copyValues(const Value* values, size_t count) {
    if (count == 0) return {};
    if (!values) throw std::invalid_argument("Nonempty mesh buffers require a pointer");
    return {values, values + count};
}

lodestar::CollisionMesh copyCollisionInput(std::span<const lodestar_collision_vertex> positions,
                                           const uint32_t* indices, size_t indexCount) {
    lodestar::CollisionMesh mesh;
    mesh.positions.reserve(positions.size());
    for (const auto& vertex : positions) {
        const auto& point = vertex.position;
        mesh.positions.push_back({point[0], point[1], point[2]});
    }
    mesh.indices = copyValues(indices, indexCount);
    return mesh;
}

lodestar_collision_mesh collisionOutput(lodestar::CollisionMesh mesh) {
    lodestar_collision_mesh output;
    output.positions.reserve(mesh.positions.size());
    for (const auto& position : mesh.positions) {
        output.positions.push_back({{position[0], position[1], position[2]}});
    }
    output.indices = std::move(mesh.indices);
    output.skippedDegenerateTriangles = mesh.skippedDegenerateTriangles;
    return output;
}

lodestar_collision_mesh_view collisionView(const lodestar_collision_mesh& mesh) {
    return {mesh.positions.data(), mesh.positions.size(), mesh.indices.data(), mesh.indices.size(),
            mesh.skippedDegenerateTriangles};
}

lodestar::Mesh copyInput(const lodestar_mesh_input& input) {
    const auto& view = input.mesh;
    const auto& metadata = input.metadata;
    const auto& provenance = input.provenance;
    if (!view.vertices || !view.vertex_count || view.vertex_count > std::numeric_limits<uint32_t>::max() ||
        !view.index_count || view.index_count % 3 != 0 ||
        (metadata.triangle_material_count != 0 && metadata.triangle_material_count != view.index_count / 3) ||
        (provenance.vertex_position_id_count != 0 && provenance.vertex_position_id_count != view.vertex_count) ||
        (provenance.triangle_surface_id_count != 0 && provenance.triangle_surface_id_count != view.index_count / 3)) {
        throw std::invalid_argument("Mesh input has invalid buffer counts");
    }
    lodestar::Mesh mesh;
    mesh.vertices.reserve(view.vertex_count);
    for (size_t i = 0; i < view.vertex_count; ++i) {
        const auto& vertex = view.vertices[i];
        mesh.vertices.push_back({{vertex.position[0], vertex.position[1], vertex.position[2]},
                                 {vertex.normal[0], vertex.normal[1], vertex.normal[2]},
                                 {vertex.uv[0], vertex.uv[1]}});
    }
    mesh.indices = copyValues(view.indices, view.index_count);
    mesh.hasUV = view.has_uv != 0 || metadata.has_additional_uv != 0;
    mesh.hasAdditionalUV = metadata.has_additional_uv != 0;
    mesh.triangleMaterials = copyValues(metadata.triangle_materials, metadata.triangle_material_count);
    mesh.vertexPositionIds = copyValues(provenance.vertex_position_ids, provenance.vertex_position_id_count);
    mesh.triangleSurfaceIds = copyValues(provenance.triangle_surface_ids, provenance.triangle_surface_id_count);
    mesh.hasUnrepresentedVertexAttributes = provenance.has_unrepresented_vertex_attributes != 0;
    const auto topology = lodestar::inspectTopology(mesh);
    if (!topology.valid || topology.degenerateTriangles == view.index_count / 3) {
        throw std::invalid_argument("Mesh input has invalid topology or provenance");
    }
    return mesh;
}

std::unique_ptr<lodestar_mesh> meshOutput(lodestar::Mesh mesh) {
    auto output = std::make_unique<lodestar_mesh>();
    output->vertices.reserve(mesh.vertices.size());
    for (const auto& vertex : mesh.vertices) {
        output->vertices.push_back({{vertex.position.x, vertex.position.y, vertex.position.z},
                                    {vertex.normal.x, vertex.normal.y, vertex.normal.z},
                                    {vertex.uv.x, vertex.uv.y}});
    }
    output->indices = std::move(mesh.indices);
    output->hasUV = mesh.hasUV;
    output->hasAdditionalUV = mesh.hasAdditionalUV;
    output->triangleMaterials = std::move(mesh.triangleMaterials);
    output->vertexPositionIds = std::move(mesh.vertexPositionIds);
    output->triangleSurfaceIds = std::move(mesh.triangleSurfaceIds);
    output->hasUnrepresentedVertexAttributes = mesh.hasUnrepresentedVertexAttributes;
    return output;
}

template <typename Generate>
lodestar_status generateOwnedMesh(const lodestar_asset* asset, lodestar_mesh** result, Generate generate) {
    if (!result) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Mesh output is required");
    }
    *result = nullptr;
    if (!asset) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Asset is required");
    }
    return checked([&] {
        *result = meshOutput(generate(*asset)).release();
    });
}

lodestar::BezierTriangle copyPatch(const lodestar_bezier_triangle& input) {
    lodestar::BezierTriangle patch;
    for (std::size_t i = 0; i < patch.controls.size(); ++i) {
        patch.controls[i] = {input.controls[i][0], input.controls[i][1], input.controls[i][2]};
    }
    return patch;
}

} // namespace

lodestar_status lodestar_differentiate_surface(const lodestar_bezier_triangle* patch, double u, double v,
                                               lodestar_surface_differential* result) {
    if (!result) return fail(LODESTAR_INVALID_ARGUMENT, "Surface derivative output is required");
    *result = {};
    if (!patch) return fail(LODESTAR_INVALID_ARGUMENT, "A surface patch is required");
    return checked([&] {
        const auto d = lodestar::differentiateSurface(copyPatch(*patch), u, v);
        const std::array values{d.position, d.du, d.dv, d.duu, d.duv, d.dvv};
        const std::array<double*, 6> outputs{result->position, result->du, result->dv,
                                           result->duu, result->duv, result->dvv};
        for (std::size_t i = 0; i < values.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) outputs[i][axis] = values[i][axis];
        }
    });
}

lodestar_status lodestar_calculate_surface_lod(const lodestar_bezier_triangle* patches, size_t patchCount,
                                               double worldUnitsPerPixel, double maximumPixelError,
                                               double relativeTolerance, uint32_t maximumDepth,
                                               lodestar_surface_lod_budget* result) {
    if (!result) return fail(LODESTAR_INVALID_ARGUMENT, "Surface budget output is required");
    *result = {};
    if (!patches || patchCount == 0) return fail(LODESTAR_INVALID_ARGUMENT, "Surface patches are required");
    return checked([&] {
        std::vector<lodestar::BezierTriangle> inputs;
        inputs.reserve(patchCount);
        for (std::size_t i = 0; i < patchCount; ++i) inputs.push_back(copyPatch(patches[i]));
        const auto budget = lodestar::calculateSurfaceLod(inputs, worldUnitsPerPixel, maximumPixelError,
                                                         relativeTolerance, maximumDepth);
        const auto& integral = budget.integral;
        *result = {integral.area, integral.absoluteCurvature, integral.estimatedAreaError,
                   integral.estimatedCurvatureError, budget.worldError, budget.triangleDensityIntegral,
                   budget.interiorVertexDensityIntegral, budget.suggestedTriangles, integral.evaluations,
                   integral.converged};
    });
}

lodestar_status lodestar_tessellate_surface(const lodestar_bezier_triangle* patch, double worldUnitsPerPixel,
                                           double maximumPixelError, uint32_t maximumDepth,
                                           lodestar_mesh** result, double* maximumDeviationBound) {
    if (result) *result = nullptr;
    if (maximumDeviationBound) *maximumDeviationBound = 0.0;
    if (!patch || !result || !maximumDeviationBound) {
        return fail(LODESTAR_INVALID_ARGUMENT, "A patch, mesh output and deviation output are required");
    }
    return checked([&] {
        auto tessellation = lodestar::tessellateSurface(copyPatch(*patch), worldUnitsPerPixel,
                                                       maximumPixelError, maximumDepth);
        auto output = meshOutput(std::move(tessellation.mesh));
        *maximumDeviationBound = tessellation.maximumDeviationBound;
        *result = output.release();
    });
}

const char* lodestar_last_error(void) {
    return lastError;
}

lodestar_status lodestar_fit_surface(const double* parameters, const double* positions, size_t count,
                                    lodestar_polynomial_surface* result) {
    if (!result) return fail(LODESTAR_INVALID_ARGUMENT, "Surface fit output is required");
    *result = {};
    if (!parameters || !positions || count < 3 || count > 1000000) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Expected 3 to 1000000 parameter/position pairs");
    }
    return checked([&] {
        std::vector<glm::dvec2> uv;
        std::vector<glm::dvec3> points;
        uv.reserve(count);
        points.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            uv.push_back({parameters[2*i], parameters[2*i + 1]});
            points.push_back({positions[3*i], positions[3*i + 1], positions[3*i + 2]});
        }
        const auto fitted = lodestar::fitSurface(uv, points);
        for (size_t i = 0; i < fitted.coefficients.size(); ++i) {
            for (size_t axis = 0; axis < 3; ++axis) result->coefficients[i][axis] = fitted.coefficients[i][axis];
        }
        result->maximum_vertex_residual = fitted.maximumVertexResidual;
        result->degree = fitted.degree;
    });
}

lodestar_status lodestar_polynomial_surface_patch(const lodestar_polynomial_surface* surface,
                                                 const double* domain, lodestar_bezier_triangle* result) {
    if (!result) return fail(LODESTAR_INVALID_ARGUMENT, "Surface patch output is required");
    *result = {};
    if (!surface || !domain) return fail(LODESTAR_INVALID_ARGUMENT, "A fitted surface and triangle domain are required");
    return checked([&] {
        lodestar::PolynomialSurface fitted;
        for (size_t i = 0; i < fitted.coefficients.size(); ++i) {
            fitted.coefficients[i] = {surface->coefficients[i][0], surface->coefficients[i][1], surface->coefficients[i][2]};
        }
        const std::array<glm::dvec2, 3> triangle{{{domain[0], domain[1]}, {domain[2], domain[3]}, {domain[4], domain[5]}}};
        const auto patch = lodestar::surfacePatch(fitted, triangle);
        for (size_t i = 0; i < patch.controls.size(); ++i) {
            for (size_t axis = 0; axis < 3; ++axis) result->controls[i][axis] = patch.controls[i][axis];
        }
    });
}

lodestar_status lodestar_asset_load(const char* path, lodestar_asset** result) {
    return lodestar_asset_load_in_space(path, LODESTAR_IMPORT_NORMALIZED, result);
}

lodestar_status lodestar_asset_load_in_space(const char* path, lodestar_import_space space,
                                           lodestar_asset** result) {
    if (!result) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Asset output is required");
    }
    *result = nullptr;
    if (!path || !*path) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Mesh path is required");
    }
    return checked([&] { *result = new lodestar_asset{lodestar::loadAsset(path, spaceFromAbi(space))}; });
}

lodestar_status lodestar_asset_analyze_mesh(const lodestar_mesh_input* input, lodestar_asset** result) {
    if (!result) return fail(LODESTAR_INVALID_ARGUMENT, "Asset output is required");
    *result = nullptr;
    if (!input) return fail(LODESTAR_INVALID_ARGUMENT, "Mesh input is required");
    return checked([&] { *result = new lodestar_asset{lodestar::analyze(copyInput(*input))}; });
}

lodestar_status lodestar_asset_sample(lodestar_asset** result) {
    if (!result) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Asset output is required");
    }
    *result = nullptr;
    return checked([&] { *result = new lodestar_asset{lodestar::sampleAsset()}; });
}

void lodestar_asset_destroy(lodestar_asset* asset) {
    delete asset;
}

lodestar_status lodestar_asset_get_info(const lodestar_asset* asset, lodestar_asset_info* result) {
    if (!asset || !result) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Asset and analysis output are required");
    }
    return checked([&] {
        const auto& value = asset->value;
        const auto& fit = value.fit;
        const auto faceError = value.maxFaceInwardRelativeError;
        *result = {value.original.indices.size() / 3, value.original.hasUV, fit.valid,
                   eligibility(value.eligibility), faceError.has_value(),
                   {fit.center.x, fit.center.y, fit.center.z}, fit.radius,
                   fit.maxRelativeError, fit.rmsRelativeError, faceError.value_or(0.0f)};
    });
}

lodestar_status lodestar_asset_inspect_topology(const lodestar_asset* asset, lodestar_topology_report* result) {
    if (!asset || !result) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Asset and topology output are required");
    }
    return checked([&] {
        const auto report = lodestar::inspectTopology(asset->value.original);
        *result = {
            report.geometricEdges, report.boundaryEdges, report.indexedBoundaryEdges, report.sharedEdges,
            report.compatibleSplitEdges, report.recoverableSplitEdges, report.crossSurfaceSplitEdges,
            report.uvSeamEdges, report.normalSeamEdges, report.materialSeamEdges, report.windingConflictEdges,
            report.ambiguousEdges, report.nonManifoldEdges, report.degenerateTriangles, report.valid
        };
    });
}

lodestar_status lodestar_asset_inspect_boundary_budget(const lodestar_asset* asset,
                                                       lodestar_boundary_budget_report* result) {
    if (!asset || !result) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Asset and boundary budget output are required");
    }
    return checked([&] {
        const auto report = lodestar::inspectBoundaryBudget(asset->value.original);
        *result = {
            report.sourceTriangles, report.chartCount, report.certifiedCharts, report.unsupportedCharts,
            report.unsupportedTriangles, report.protectedBoundaryEdges, report.boundaryLoops, report.interiorVertices,
            report.closedCharts, report.lowerBoundTriangles, report.unchangedFallbackLowerBoundTriangles,
            report.upperBoundTriangles, report.valid
        };
    });
}

lodestar_status lodestar_asset_generate(const lodestar_asset* asset, lodestar_mode mode,
                                        int32_t quality, lodestar_mesh** result) {
    return generateOwnedMesh(asset, result, [&](const lodestar_asset& input) {
        if (mode == LODESTAR_PLANAR) {
            return planarMesh(input, lodestar::planarTriangleTarget(input.value.original.indices.size() / 3, quality));
        }
        if (mode == LODESTAR_SEAM) {
            return seamMesh(input, lodestar::planarTriangleTarget(input.value.original.indices.size() / 3, quality));
        }
        return lodestar::generateMesh(input.value, modeFromAbi(mode), quality);
    });
}

lodestar_status lodestar_asset_generate_planar(const lodestar_asset* asset, size_t targetTriangles,
                                              lodestar_mesh** result) {
    return generateOwnedMesh(asset, result, [&](const lodestar_asset& input) {
        return planarMesh(input, targetTriangles);
    });
}

lodestar_status lodestar_asset_generate_seam(const lodestar_asset* asset, size_t targetTriangles,
                                             lodestar_mesh** result) {
    return generateOwnedMesh(asset, result, [&](const lodestar_asset& input) {
        return seamMesh(input, targetTriangles);
    });
}

lodestar_status lodestar_simplify_geometry(const float* positions, size_t vertexCount, const uint32_t* indices,
                                         size_t indexCount, const uint8_t* vertexLocks, size_t targetTriangles,
                                         float maxAbsoluteError, uint32_t lockBoundary, uint32_t* outputIndices,
                                         size_t* outputCount, float* estimatedError) {
    if (!positions || !indices || !outputIndices || !outputCount || !estimatedError || lockBoundary > 1 ||
        vertexCount > std::numeric_limits<size_t>::max() / 3) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Geometry simplification requires valid buffers and outputs");
    }
    return checked([&] {
        const auto locks = vertexLocks ? std::span(vertexLocks, vertexCount) : std::span<const uint8_t>{};
        const auto result = lodestar::simplifyGeometry(std::span(positions, vertexCount * 3),
            std::span(indices, indexCount), locks, targetTriangles, maxAbsoluteError, lockBoundary != 0);
        std::copy(result.indices.begin(), result.indices.end(), outputIndices);
        *outputCount = result.indices.size();
        *estimatedError = result.estimatedError;
    });
}

lodestar_status lodestar_asset_generate_collision(const lodestar_asset* asset, lodestar_collision_mesh** result) {
    if (!result) return fail(LODESTAR_INVALID_ARGUMENT, "Collision mesh output is required");
    *result = nullptr;
    if (!asset) return fail(LODESTAR_INVALID_ARGUMENT, "Asset is required");
    return checked([&] {
        *result = new lodestar_collision_mesh(collisionOutput(lodestar::buildCollisionMesh(asset->value.original)));
    });
}

lodestar_status lodestar_simplify_convex_hull(
    const lodestar_collision_vertex* positions, size_t positionCount, const uint32_t* indices, size_t indexCount,
    size_t targetVertices, double maxDeviation, uint32_t preserveVerticalSupport,
    lodestar_collision_mesh** result, lodestar_convex_hull_info* info) {
    if (!result) return fail(LODESTAR_INVALID_ARGUMENT, "Collision mesh output is required");
    *result = nullptr;
    if (!positions || !indices || !info || preserveVerticalSupport > 1 ||
        positionCount > std::numeric_limits<uint32_t>::max()) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Convex hull simplification requires valid buffers and outputs");
    }
    return checked([&] {
        const auto source = copyCollisionInput(std::span(positions, positionCount), indices, indexCount);
        auto reduced = lodestar::simplifyConvexHull(source, {targetVertices, maxDeviation, preserveVerticalSupport != 0});
        auto output = std::make_unique<lodestar_collision_mesh>(collisionOutput(std::move(reduced.mesh)));
        *info = {reduced.inputVertices, reduced.maxDeviation, reduced.metTarget};
        *result = output.release();
    });
}

lodestar_status lodestar_build_convex_hull(
    const lodestar_collision_vertex* positions, size_t positionCount, lodestar_collision_mesh** result) {
    if (!result) return fail(LODESTAR_INVALID_ARGUMENT, "Convex envelope output is required");
    *result = nullptr;
    if (!positions || !positionCount) return fail(LODESTAR_INVALID_ARGUMENT, "Convex envelope points are required");
    return checked([&] {
        std::vector<std::array<float, 3>> points;
        points.reserve(positionCount);
        for (const auto& point : std::span(positions, positionCount)) {
            points.push_back({point.position[0], point.position[1], point.position[2]});
        }
        *result = new lodestar_collision_mesh(collisionOutput(lodestar::buildConvexHull(points)));
    });
}

lodestar_status lodestar_decompose_collision_hulls(
    const lodestar_collision_vertex* positions, size_t positionCount, const uint32_t* indices, size_t indexCount,
    size_t maxHulls, lodestar_convex_compound** result) {
    if (!result) return fail(LODESTAR_INVALID_ARGUMENT, "Convex compound output is required");
    *result = nullptr;
    if (!positions || !positionCount || !indices || !indexCount || indexCount % 3 != 0 || !maxHulls ||
        positionCount > std::numeric_limits<uint32_t>::max()) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Hull conversion requires triangle buffers and a positive hull limit");
    }
    return checked([&] {
        const auto source = copyCollisionInput(std::span(positions, positionCount), indices, indexCount);
        auto hulls = lodestar::decomposeCollisionHulls(source, maxHulls);
        auto output = std::make_unique<lodestar_convex_compound>();
        output->hulls.reserve(hulls.size());
        for (auto& hull : hulls) output->hulls.push_back(collisionOutput(std::move(hull)));
        *result = output.release();
    });
}

lodestar_status lodestar_plan_mesh_size(
    const lodestar_collision_vertex* positions, size_t positionCount, double minimumDimension,
    double trianglePhysicsMinimum, lodestar_mesh_size_plan* result) {
    if (!positions || !positionCount || !result) return fail(LODESTAR_INVALID_ARGUMENT, "Mesh size planning needs points and an output");
    return checked([&] {
        std::vector<std::array<float, 3>> points;
        points.reserve(positionCount);
        for (const auto& point : std::span(positions, positionCount)) points.push_back({point.position[0], point.position[1], point.position[2]});
        const auto plan = lodestar::planMeshSize(points, minimumDimension, trianglePhysicsMinimum);
        lodestar_mesh_size_plan output{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            output.visible_minimum[axis] = plan.visibleMinimum[axis]; output.visible_maximum[axis] = plan.visibleMaximum[axis];
            output.container_minimum[axis] = plan.containerMinimum[axis]; output.container_maximum[axis] = plan.containerMaximum[axis];
            output.center[axis] = plan.center[axis]; output.visible_size[axis] = plan.visibleSize[axis];
            output.container_size[axis] = plan.containerSize[axis]; output.visible_fraction[axis] = plan.visibleFraction[axis];
        }
        output.requires_container = plan.requiresContainer; output.requires_hull_physics = plan.requiresHullPhysics;
        *result = output;
    });
}

lodestar_status lodestar_build_coherent_collision(
    const lodestar_collision_vertex* positions, size_t positionCount, const uint32_t* indices, size_t indexCount,
    size_t maxHulls, double maxSurfaceError, lodestar_convex_compound** result, lodestar_coherent_collision_info* info) {
    if (!result) return fail(LODESTAR_INVALID_ARGUMENT, "Coherent collision output is required");
    *result = nullptr;
    if (!positions || !positionCount || !indices || !indexCount || indexCount % 3 || !info ||
        positionCount > std::numeric_limits<uint32_t>::max()) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Coherent collision needs source triangles and report outputs");
    }
    return checked([&] {
        const auto source = copyCollisionInput(std::span(positions, positionCount), indices, indexCount);
        auto shape = lodestar::buildCoherentCollisionShape(source, {maxSurfaceError, maxHulls});
        auto output = std::make_unique<lodestar_convex_compound>();
        for (auto& hull : shape.hulls) output->hulls.push_back(collisionOutput(std::move(hull)));
        *info = {shape.surfaceErrorBound, shape.exactSolidConversion};
        *result = output.release();
    });
}

lodestar_status lodestar_convex_compound_get_hull_count(const lodestar_convex_compound* compound, size_t* result) {
    if (!compound || !result) return fail(LODESTAR_INVALID_ARGUMENT, "Convex compound and hull count output are required");
    lastError[0] = '\0';
    *result = compound->hulls.size();
    return LODESTAR_OK;
}

lodestar_status lodestar_convex_compound_get_hull_view(const lodestar_convex_compound* compound, size_t hullIndex,
                                                      lodestar_collision_mesh_view* result) {
    if (!compound || !result) return fail(LODESTAR_INVALID_ARGUMENT, "Convex compound and hull view output are required");
    if (hullIndex >= compound->hulls.size()) return fail(LODESTAR_INVALID_ARGUMENT, "Convex hull index is out of range");
    lastError[0] = '\0';
    *result = collisionView(compound->hulls[hullIndex]);
    return LODESTAR_OK;
}

void lodestar_convex_compound_destroy(lodestar_convex_compound* compound) {
    delete compound;
}

lodestar_status lodestar_collision_mesh_get_view(const lodestar_collision_mesh* mesh,
                                                 lodestar_collision_mesh_view* result) {
    if (!mesh || !result) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Collision mesh and view output are required");
    }
    lastError[0] = '\0';
    *result = collisionView(*mesh);
    return LODESTAR_OK;
}

void lodestar_collision_mesh_destroy(lodestar_collision_mesh* mesh) {
    delete mesh;
}

lodestar_status lodestar_mesh_get_view(const lodestar_mesh* mesh, lodestar_mesh_view* result) {
    if (!mesh || !result) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Mesh and view output are required");
    }
    lastError[0] = '\0';
    *result = {mesh->vertices.data(), mesh->vertices.size(), mesh->indices.data(), mesh->indices.size(), mesh->hasUV};
    return LODESTAR_OK;
}

lodestar_status lodestar_mesh_get_metadata(const lodestar_mesh* mesh, lodestar_mesh_metadata* result) {
    if (!mesh || !result) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Mesh and metadata output are required");
    }
    lastError[0] = '\0';
    *result = {mesh->triangleMaterials.data(), mesh->triangleMaterials.size(), mesh->hasAdditionalUV};
    return LODESTAR_OK;
}

lodestar_status lodestar_mesh_get_provenance(const lodestar_mesh* mesh, lodestar_mesh_provenance* result) {
    if (!mesh || !result) {
        return fail(LODESTAR_INVALID_ARGUMENT, "Mesh and provenance output are required");
    }
    lastError[0] = '\0';
    *result = {mesh->vertexPositionIds.data(), mesh->vertexPositionIds.size(),
               mesh->triangleSurfaceIds.data(), mesh->triangleSurfaceIds.size(), mesh->hasUnrepresentedVertexAttributes};
    return LODESTAR_OK;
}

void lodestar_mesh_destroy(lodestar_mesh* mesh) {
    delete mesh;
}
