#ifndef LODESTAR_C_H
#define LODESTAR_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(LODESTAR_C_EXPORTS)
#define LODESTAR_C_API __declspec(dllexport)
#else
#define LODESTAR_C_API __declspec(dllimport)
#endif
#else
#define LODESTAR_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lodestar_asset lodestar_asset;
typedef struct lodestar_mesh lodestar_mesh;

typedef uint32_t lodestar_status;
enum {
    LODESTAR_OK = 0,
    LODESTAR_INVALID_ARGUMENT = 1,
    LODESTAR_ERROR = 2
};

typedef uint32_t lodestar_mode;
enum {
    LODESTAR_ORIGINAL = 0,
    LODESTAR_BASELINE = 1,
    LODESTAR_SPHERE = 2,
    LODESTAR_PLANAR = 3,
    LODESTAR_SEAM = 4
};

typedef uint32_t lodestar_import_space;
enum {
    LODESTAR_IMPORT_NORMALIZED = 0,
    LODESTAR_IMPORT_SCENE = 1
};

typedef uint32_t lodestar_sphere_eligibility;
enum {
    LODESTAR_SPHERE_ELIGIBLE = 0,
    LODESTAR_SPHERE_HAS_UV = 1,
    LODESTAR_SPHERE_NO_FIT = 2,
    LODESTAR_SPHERE_RADIAL_ERROR = 3,
    LODESTAR_SPHERE_INCOMPLETE = 4
};

typedef struct lodestar_vertex {
    float position[3];
    float normal[3];
    float uv[2];
} lodestar_vertex;

typedef struct lodestar_asset_info {
    size_t original_triangles;
    uint32_t has_uv;
    uint32_t fit_valid;
    lodestar_sphere_eligibility sphere_eligibility;
    uint32_t max_face_inward_error_valid;
    float sphere_center[3];
    float sphere_radius;
    float max_relative_error;
    float rms_relative_error;
    float max_face_inward_relative_error;
} lodestar_asset_info;

typedef struct lodestar_mesh_view {
    const lodestar_vertex* vertices;
    size_t vertex_count;
    const uint32_t* indices;
    size_t index_count;
    uint32_t has_uv;
} lodestar_mesh_view;

typedef struct lodestar_mesh_metadata {
    const uint32_t* triangle_materials;
    size_t triangle_material_count;
    uint32_t has_additional_uv;
} lodestar_mesh_metadata;

typedef struct lodestar_mesh_provenance {
    const uint64_t* vertex_position_ids;
    size_t vertex_position_id_count;
    const uint32_t* triangle_surface_ids;
    size_t triangle_surface_id_count;
    uint32_t has_unrepresented_vertex_attributes;
} lodestar_mesh_provenance;

typedef struct lodestar_mesh_input {
    lodestar_mesh_view mesh;
    lodestar_mesh_metadata metadata;
    lodestar_mesh_provenance provenance;
} lodestar_mesh_input;

typedef struct lodestar_topology_report {
    size_t geometric_edges;
    size_t boundary_edges;
    size_t indexed_boundary_edges;
    size_t shared_edges;
    size_t compatible_split_edges;
    size_t recoverable_split_edges;
    size_t cross_surface_split_edges;
    size_t uv_seam_edges;
    size_t normal_seam_edges;
    size_t material_seam_edges;
    size_t winding_conflict_edges;
    size_t ambiguous_edges;
    size_t non_manifold_edges;
    size_t degenerate_triangles;
    uint32_t valid;
} lodestar_topology_report;

typedef struct lodestar_boundary_budget_report {
    size_t source_triangles;
    size_t chart_count;
    size_t certified_charts;
    size_t unsupported_charts;
    size_t unsupported_triangles;
    size_t protected_boundary_edges;
    size_t boundary_loops;
    size_t interior_vertices;
    size_t closed_charts;
    size_t lower_bound_triangles;
    size_t unchanged_fallback_lower_bound_triangles;
    size_t upper_bound_triangles;
    uint32_t valid;
} lodestar_boundary_budget_report;

/* Errors are thread-local. The returned message is valid until the next operation on that thread. */
LODESTAR_C_API const char* lodestar_last_error(void);

/* Path is a null-terminated UTF-8 string. Successful calls return owned handles. */
LODESTAR_C_API lodestar_status lodestar_asset_load(const char* path, lodestar_asset** result);
LODESTAR_C_API lodestar_status lodestar_asset_load_in_space(const char* path, lodestar_import_space space,
                                                         lodestar_asset** result);
/* Copies an indexed mesh in the supplied coordinate system. Optional arrays may be empty.
 * Equal nonzero position IDs assert a shared authored vertex. Zero means unknown. */
LODESTAR_C_API lodestar_status lodestar_asset_analyze_mesh(const lodestar_mesh_input* input, lodestar_asset** result);
LODESTAR_C_API lodestar_status lodestar_asset_sample(lodestar_asset** result);
LODESTAR_C_API void lodestar_asset_destroy(lodestar_asset* asset);
LODESTAR_C_API lodestar_status lodestar_asset_get_info(const lodestar_asset* asset, lodestar_asset_info* result);
/* Classifies the source mesh. A successful call sets valid=0 if its topology is malformed. */
LODESTAR_C_API lodestar_status lodestar_asset_inspect_topology(const lodestar_asset* asset,
                                                            lodestar_topology_report* result);
/* Inspect the triangle budget while keeping every indexed chart boundary edge fixed. */
LODESTAR_C_API lodestar_status lodestar_asset_inspect_boundary_budget(const lodestar_asset* asset,
                                                                    lodestar_boundary_budget_report* result);

/* Quality is clamped to 1–12. An ineligible sphere request returns the original mesh. */
LODESTAR_C_API lodestar_status lodestar_asset_generate(const lodestar_asset* asset, lodestar_mode mode,
                                                    int32_t quality, lodestar_mesh** result);
/* Reuses the asset's planar analysis. The returned count can exceed the requested target. */
LODESTAR_C_API lodestar_status lodestar_asset_generate_planar(const lodestar_asset* asset, size_t target_triangles,
                                                           lodestar_mesh** result);
/* Reuses the asset's seam analysis. The returned count can exceed the requested target. */
LODESTAR_C_API lodestar_status lodestar_asset_generate_seam(const lodestar_asset* asset, size_t target_triangles,
                                                         lodestar_mesh** result);
/* A view borrows the mesh buffers until lodestar_mesh_destroy is called. */
LODESTAR_C_API lodestar_status lodestar_mesh_get_view(const lodestar_mesh* mesh, lodestar_mesh_view* result);
/* Material IDs are provided in triangle order when available. The mesh owns the IDs. */
LODESTAR_C_API lodestar_status lodestar_mesh_get_metadata(const lodestar_mesh* mesh, lodestar_mesh_metadata* result);
/* The mesh owns the optional position and surface IDs. */
LODESTAR_C_API lodestar_status lodestar_mesh_get_provenance(const lodestar_mesh* mesh, lodestar_mesh_provenance* result);
LODESTAR_C_API void lodestar_mesh_destroy(lodestar_mesh* mesh);

#ifdef __cplusplus
}
#endif

#endif
