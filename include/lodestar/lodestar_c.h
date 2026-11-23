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
typedef struct lodestar_collision_mesh lodestar_collision_mesh;
typedef struct lodestar_convex_compound lodestar_convex_compound;

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

typedef struct lodestar_collision_vertex {
    float position[3];
} lodestar_collision_vertex;

typedef struct lodestar_collision_mesh_view {
    const lodestar_collision_vertex* positions;
    size_t position_count;
    const uint32_t* indices;
    size_t index_count;
    size_t skipped_degenerate_triangles;
} lodestar_collision_mesh_view;

typedef struct lodestar_convex_hull_info {
    size_t input_vertices;
    double max_deviation;
    uint32_t met_target;
} lodestar_convex_hull_info;

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

typedef struct lodestar_bezier_triangle {
    double controls[10][3];
} lodestar_bezier_triangle;

typedef struct lodestar_polynomial_surface {
    double coefficients[10][3];
    double maximum_vertex_residual;
    uint32_t degree;
} lodestar_polynomial_surface;

/* Parameters contain two doubles and positions contain three doubles per sample.
 * Vertex residuals alone do not bound the complete fitted surface. */
LODESTAR_C_API lodestar_status lodestar_fit_surface(
    const double* parameters, const double* positions, size_t count, lodestar_polynomial_surface* result);
LODESTAR_C_API lodestar_status lodestar_polynomial_surface_patch(
    const lodestar_polynomial_surface* surface, const double* domain, lodestar_bezier_triangle* result);

typedef struct lodestar_surface_differential {
    double position[3];
    double du[3];
    double dv[3];
    double duu[3];
    double duv[3];
    double dvv[3];
} lodestar_surface_differential;

typedef struct lodestar_surface_lod_budget {
    double area;
    double absolute_curvature;
    double estimated_area_error;
    double estimated_curvature_error;
    double world_error;
    double triangle_density_integral;
    double interior_vertex_density_integral;
    size_t suggested_triangles;
    size_t evaluations;
    uint32_t converged;
} lodestar_surface_lod_budget;

/* Controls follow the cubic triangular Bezier order documented in surface_calculus.hpp.
 * Outputs initialize to zero before validation. Integration estimates guide planning;
 * callers certify the patch fit and the resulting mesh independently. */
LODESTAR_C_API lodestar_status lodestar_differentiate_surface(
    const lodestar_bezier_triangle* patch, double u, double v, lodestar_surface_differential* result);
LODESTAR_C_API lodestar_status lodestar_calculate_surface_lod(
    const lodestar_bezier_triangle* patches, size_t patch_count, double world_units_per_pixel,
    double maximum_pixel_error, double relative_tolerance, uint32_t maximum_depth,
    lodestar_surface_lod_budget* result);
/* Generate an owned, untextured mesh from one regular authoritative patch.
 * The vertex count follows the calculated density and checked interpolation bound.
 * Failure leaves *result null and *maximum_deviation_bound zero. */
LODESTAR_C_API lodestar_status lodestar_tessellate_surface(
    const lodestar_bezier_triangle* patch, double world_units_per_pixel, double maximum_pixel_error,
    uint32_t maximum_depth, lodestar_mesh** result, double* maximum_deviation_bound);

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
/* Experimental geometry-only simplification. Positions contain 3 floats per vertex.
 * Optional vertex_locks contains one 0 or 1 byte per vertex. Output has capacity for
 * index_count entries. The final indices refer to the original position buffer. */
LODESTAR_C_API lodestar_status lodestar_simplify_geometry(
    const float* positions, size_t vertex_count, const uint32_t* indices, size_t index_count,
    const uint8_t* vertex_locks, size_t target_triangles, float max_absolute_error, uint32_t lock_boundary,
    uint32_t* output_indices, size_t* output_count, float* estimated_error);
/* Build an owned static triangle collider in the imported source coordinate space. */
LODESTAR_C_API lodestar_status lodestar_asset_generate_collision(const lodestar_asset* asset,
                                                               lodestar_collision_mesh** result);
/* Reduce one closed, outward-wound convex piece. Deviation uses the input position units.
 * A successful call returns an owned mesh and its report. met_target can be zero.
 * Set preserve_vertical_support to 1 to retain the footprint at each extreme Y plane. */
LODESTAR_C_API lodestar_status lodestar_simplify_convex_hull(
    const lodestar_collision_vertex* positions, size_t position_count, const uint32_t* indices, size_t index_count,
    size_t target_vertices, double max_deviation, uint32_t preserve_vertical_support,
    lodestar_collision_mesh** result, lodestar_convex_hull_info* info);
/* Construct a convex point envelope. No mesh conformity or opening guarantee
 * is implied. Coplanar inputs fail; success returns an owned collision mesh. */
LODESTAR_C_API lodestar_status lodestar_build_convex_hull(
    const lodestar_collision_vertex* positions, size_t position_count, lodestar_collision_mesh** result);
typedef struct lodestar_coherent_collision_info {
    double surface_error_bound;
    uint32_t exact_solid_conversion;
} lodestar_coherent_collision_info;
typedef struct lodestar_mesh_size_plan {
    double visible_minimum[3];
    double visible_maximum[3];
    double container_minimum[3];
    double container_maximum[3];
    double center[3];
    double visible_size[3];
    double container_size[3];
    double visible_fraction[3];
    uint32_t requires_container;
    uint32_t requires_hull_physics;
} lodestar_mesh_size_plan;
/* A proposal only: a physics bbox alone cannot set visual normalization. */
LODESTAR_C_API lodestar_status lodestar_plan_mesh_size(
    const lodestar_collision_vertex* positions, size_t position_count,
    double minimum_dimension, double triangle_physics_minimum, lodestar_mesh_size_plan* result);
/* Generate closed convex pieces from coherent surface elements, including open
 * inputs. max_surface_error is in input units. A non-exact surface cover does
 * not claim preservation of occupied interior volume. Failure leaves result null. */
LODESTAR_C_API lodestar_status lodestar_build_coherent_collision(
    const lodestar_collision_vertex* positions, size_t position_count,
    const uint32_t* indices, size_t index_count, size_t max_hulls, double max_surface_error,
    lodestar_convex_compound** result, lodestar_coherent_collision_info* info);
/* Convert a supported closed collision mesh to an owned compound of convex pieces.
 * max_hulls is a positive limit. Failure leaves *result null. */
LODESTAR_C_API lodestar_status lodestar_decompose_collision_hulls(
    const lodestar_collision_vertex* positions, size_t position_count, const uint32_t* indices, size_t index_count,
    size_t max_hulls, lodestar_convex_compound** result);
LODESTAR_C_API lodestar_status lodestar_convex_compound_get_hull_count(
    const lodestar_convex_compound* compound, size_t* result);
/* A hull view borrows the compound buffers until lodestar_convex_compound_destroy is called. */
LODESTAR_C_API lodestar_status lodestar_convex_compound_get_hull_view(
    const lodestar_convex_compound* compound, size_t hull_index, lodestar_collision_mesh_view* result);
LODESTAR_C_API void lodestar_convex_compound_destroy(lodestar_convex_compound* compound);
/* A view borrows the collider buffers until lodestar_collision_mesh_destroy is called. */
LODESTAR_C_API lodestar_status lodestar_collision_mesh_get_view(const lodestar_collision_mesh* mesh,
                                                               lodestar_collision_mesh_view* result);
LODESTAR_C_API void lodestar_collision_mesh_destroy(lodestar_collision_mesh* mesh);
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
