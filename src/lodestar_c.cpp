#include <lodestar/lodestar.hpp>
#include <lodestar/lodestar_c.h>

#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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
        auto mesh = generate(*asset);
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
        *result = output.release();
    });
}

} // namespace

const char* lodestar_last_error(void) {
    return lastError;
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
