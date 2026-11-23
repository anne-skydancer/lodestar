//! Rust bindings for Lodestar's native mesh conversion library.
//!
//! Build the native `lodestar_c` target with CMake before compiling this crate.

use std::ffi::{c_char, CStr, CString};
use std::fmt;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::slice;

/// Cubic triangular Bezier control points in the order
/// 300, 030, 003, 210, 120, 201, 102, 021, 012, 111.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BezierTriangle {
    pub controls: [[f64; 3]; 10],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PolynomialSurface {
    /// Basis 1, u, v, u^2, uv, v^2, u^3, u^2v, uv^2, v^3.
    pub coefficients: [[f64; 3]; 10],
    /// Residual at the input vertices. Complete surface bounds require separate checks.
    pub maximum_vertex_residual: f64,
    pub degree: u32,
}

impl PolynomialSurface {
    /// Restrict this polynomial to a triangular parameter domain.
    pub fn patch(&self, domain: [[f64; 2]; 3]) -> Result<BezierTriangle, Error> {
        let mut output = MaybeUninit::uninit();
        check(unsafe {
            lodestar_polynomial_surface_patch(self, domain.as_ptr(), output.as_mut_ptr())
        })?;
        // Success initializes all ten control points.
        Ok(unsafe { output.assume_init() })
    }
}

/// Fit a polynomial chart with native column-pivoted QR and retain fit diagnostics.
pub fn fit_surface(
    parameters: &[[f64; 2]],
    positions: &[[f64; 3]],
) -> Result<PolynomialSurface, Error> {
    if parameters.len() != positions.len() {
        return Err(Error(
            "Surface fitting needs paired parameters and positions".to_owned(),
        ));
    }
    let mut output = MaybeUninit::uninit();
    check(unsafe {
        lodestar_fit_surface(
            parameters.as_ptr(),
            positions.as_ptr(),
            positions.len(),
            output.as_mut_ptr(),
        )
    })?;
    // Success initializes the coefficients, residual and degree.
    Ok(unsafe { output.assume_init() })
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SurfaceDifferential {
    pub position: [f64; 3],
    pub du: [f64; 3],
    pub dv: [f64; 3],
    pub duu: [f64; 3],
    pub duv: [f64; 3],
    pub dvv: [f64; 3],
}

/// Surface integrals and continuous LOD density estimates for supplied patches.
/// The caller verifies the fitted surface and generated geometry independently.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SurfaceLodBudget {
    pub area: f64,
    pub absolute_curvature: f64,
    pub estimated_area_error: f64,
    pub estimated_curvature_error: f64,
    pub world_error: f64,
    pub triangle_density_integral: f64,
    pub interior_vertex_density_integral: f64,
    pub suggested_triangles: usize,
    pub evaluations: usize,
    converged: u32,
}

impl SurfaceLodBudget {
    pub fn converged(&self) -> bool {
        self.converged != 0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceLodOptions {
    pub world_units_per_pixel: f64,
    pub maximum_pixel_error: f64,
    pub relative_tolerance: f64,
    pub maximum_depth: u32,
}

/// Evaluate the analytic first and second derivatives at (u,v), u+v <= 1.
pub fn differentiate_surface(
    patch: &BezierTriangle,
    u: f64,
    v: f64,
) -> Result<SurfaceDifferential, Error> {
    let mut output = MaybeUninit::uninit();
    check(unsafe { lodestar_differentiate_surface(patch, u, v, output.as_mut_ptr()) })?;
    // Success initializes every output field.
    Ok(unsafe { output.assume_init() })
}

/// Integrate surface area and curvature to estimate a screen-space LOD budget.
pub fn calculate_surface_lod(
    patches: &[BezierTriangle],
    options: SurfaceLodOptions,
) -> Result<SurfaceLodBudget, Error> {
    let mut output = MaybeUninit::uninit();
    check(unsafe {
        lodestar_calculate_surface_lod(
            patches.as_ptr(),
            patches.len(),
            options.world_units_per_pixel,
            options.maximum_pixel_error,
            options.relative_tolerance,
            options.maximum_depth,
            output.as_mut_ptr(),
        )
    })?;
    // Success initializes every output field.
    Ok(unsafe { output.assume_init() })
}

#[derive(Debug)]
pub struct SurfaceTessellation {
    pub mesh: Mesh,
    pub maximum_deviation_bound: f64,
}

/// Generate a view-dependent mesh from one authoritative regular patch.
/// The result owns its geometry and the supplied patch remains reusable.
pub fn tessellate_surface(
    patch: &BezierTriangle,
    world_units_per_pixel: f64,
    maximum_pixel_error: f64,
    maximum_depth: u32,
) -> Result<SurfaceTessellation, Error> {
    let mut handle = ptr::null_mut();
    let mut bound = 0.0;
    check(unsafe {
        lodestar_tessellate_surface(
            patch,
            world_units_per_pixel,
            maximum_pixel_error,
            maximum_depth,
            &mut handle,
            &mut bound,
        )
    })?;
    Ok(SurfaceTessellation {
        mesh: Mesh::from_native(handle)?,
        maximum_deviation_bound: bound,
    })
}

/// The mesh representation to generate. An ineligible sphere uses the original mesh.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Original = 0,
    Baseline = 1,
    Sphere = 2,
    Planar = 3,
    Seam = 4,
}

/// Controls whether imports are centered and scaled or retain scene coordinates.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImportSpace {
    Normalized = 0,
    Scene = 1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SphereEligibility {
    Eligible,
    HasTextureCoordinates,
    NoFit,
    RadialError,
    IncompleteSurface,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereFit {
    pub center: [f32; 3],
    pub radius: f32,
    pub max_relative_error: f32,
    pub rms_relative_error: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Analysis {
    pub original_triangles: usize,
    pub has_uv: bool,
    pub fit: Option<SphereFit>,
    pub sphere_eligibility: SphereEligibility,
    pub max_face_inward_relative_error: Option<f32>,
}

/// Edge counts in the imported source mesh. An edge may have multiple seam types.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopologyReport {
    pub geometric_edges: usize,
    pub boundary_edges: usize,
    pub indexed_boundary_edges: usize,
    pub shared_edges: usize,
    pub compatible_split_edges: usize,
    pub recoverable_split_edges: usize,
    pub cross_surface_split_edges: usize,
    pub uv_seam_edges: usize,
    pub normal_seam_edges: usize,
    pub material_seam_edges: usize,
    pub winding_conflict_edges: usize,
    pub ambiguous_edges: usize,
    pub non_manifold_edges: usize,
    pub degenerate_triangles: usize,
    valid: u32,
}

impl TopologyReport {
    pub fn is_valid(&self) -> bool {
        self.valid != 0
    }
}

/// A necessary triangle-count bound for preserving each current indexed chart.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundaryBudgetReport {
    pub source_triangles: usize,
    pub chart_count: usize,
    pub certified_charts: usize,
    pub unsupported_charts: usize,
    pub unsupported_triangles: usize,
    pub protected_boundary_edges: usize,
    pub boundary_loops: usize,
    pub interior_vertices: usize,
    pub closed_charts: usize,
    pub lower_bound_triangles: usize,
    pub unchanged_fallback_lower_bound_triangles: usize,
    pub upper_bound_triangles: usize,
    valid: u32,
}

impl BoundaryBudgetReport {
    pub fn is_valid(&self) -> bool {
        self.valid != 0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// Borrowed source geometry copied by `Asset::from_mesh`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshInput<'a> {
    pub vertices: &'a [Vertex],
    pub indices: &'a [u32],
    pub has_uv: bool,
    pub has_additional_uv: bool,
    pub triangle_materials: &'a [u32],
    /// Equal nonzero IDs assert the same authored geometric vertex. Zero is unknown.
    pub vertex_position_ids: &'a [u64],
    /// Optional partition IDs in triangle order.
    pub triangle_surface_ids: &'a [u32],
    pub has_unrepresented_vertex_attributes: bool,
}

#[derive(Debug)]
pub struct Mesh {
    _handle: MeshHandle,
    view: NativeMeshView,
    metadata: NativeMeshMetadata,
    provenance: NativeMeshProvenance,
}

impl Mesh {
    fn from_native(handle: *mut NativeMesh) -> Result<Self, Error> {
        let handle = MeshHandle(
            NonNull::new(handle)
                .ok_or_else(|| Error("Native mesh handle is missing".to_owned()))?,
        );
        let mut view = MaybeUninit::uninit();
        check(unsafe { lodestar_mesh_get_view(handle.0.as_ptr(), view.as_mut_ptr()) })?;
        let mut metadata = MaybeUninit::uninit();
        check(unsafe { lodestar_mesh_get_metadata(handle.0.as_ptr(), metadata.as_mut_ptr()) })?;
        let mut provenance = MaybeUninit::uninit();
        check(unsafe { lodestar_mesh_get_provenance(handle.0.as_ptr(), provenance.as_mut_ptr()) })?;
        // Success initializes the views. The owned handle keeps their buffers alive.
        Ok(Self {
            _handle: handle,
            view: unsafe { view.assume_init() },
            metadata: unsafe { metadata.assume_init() },
            provenance: unsafe { provenance.assume_init() },
        })
    }

    pub fn vertices(&self) -> &[Vertex] {
        if self.view.vertex_count == 0 {
            return &[];
        }
        // The native buffer stays allocated and immutable for the lifetime of this mesh.
        unsafe { slice::from_raw_parts(self.view.vertices, self.view.vertex_count) }
    }

    pub fn indices(&self) -> &[u32] {
        if self.view.index_count == 0 {
            return &[];
        }
        // The native buffer stays allocated and immutable for the lifetime of this mesh.
        unsafe { slice::from_raw_parts(self.view.indices, self.view.index_count) }
    }

    pub fn has_uv(&self) -> bool {
        self.view.has_uv != 0
    }

    /// Whether the source has UV components outside the exposed UV0.xy.
    pub fn has_additional_uv(&self) -> bool {
        self.metadata.has_additional_uv != 0
    }

    /// Material IDs in triangle order when available. IDs are local to the source asset.
    pub fn triangle_materials(&self) -> &[u32] {
        if self.metadata.triangle_material_count == 0 {
            return &[];
        }
        // The native buffer stays allocated and immutable for the lifetime of this mesh.
        unsafe {
            slice::from_raw_parts(
                self.metadata.triangle_materials,
                self.metadata.triangle_material_count,
            )
        }
    }

    /// Authored position IDs in vertex order, when supplied by the input.
    pub fn vertex_position_ids(&self) -> &[u64] {
        if self.provenance.vertex_position_id_count == 0 {
            return &[];
        }
        // The native mesh owns this immutable array.
        unsafe {
            slice::from_raw_parts(
                self.provenance.vertex_position_ids,
                self.provenance.vertex_position_id_count,
            )
        }
    }

    /// Surface IDs in triangle order, when available.
    pub fn triangle_surface_ids(&self) -> &[u32] {
        if self.provenance.triangle_surface_id_count == 0 {
            return &[];
        }
        // The native mesh owns this immutable array.
        unsafe {
            slice::from_raw_parts(
                self.provenance.triangle_surface_ids,
                self.provenance.triangle_surface_id_count,
            )
        }
    }

    pub fn has_unrepresented_vertex_attributes(&self) -> bool {
        self.provenance.has_unrepresented_vertex_attributes != 0
    }

    pub fn triangle_count(&self) -> usize {
        self.view.index_count / 3
    }
}

/// Owned positions and triangle indices for static collision queries.
#[derive(Debug)]
pub struct CollisionMesh {
    _handle: CollisionMeshHandle,
    view: NativeCollisionMeshView,
}

impl CollisionMesh {
    fn from_native(handle: *mut NativeCollisionMesh) -> Result<Self, Error> {
        let handle = CollisionMeshHandle(
            NonNull::new(handle)
                .ok_or_else(|| Error("Native collision mesh handle is missing".to_owned()))?,
        );
        let mut view = MaybeUninit::uninit();
        check(unsafe { lodestar_collision_mesh_get_view(handle.0.as_ptr(), view.as_mut_ptr()) })?;
        // Success initializes a view borrowing the owned collider buffers.
        Ok(Self {
            _handle: handle,
            view: unsafe { view.assume_init() },
        })
    }

    pub fn positions(&self) -> &[[f32; 3]] {
        self.as_view().positions()
    }

    pub fn indices(&self) -> &[u32] {
        self.as_view().indices()
    }

    pub fn triangle_count(&self) -> usize {
        self.view.index_count / 3
    }

    pub fn skipped_degenerate_triangles(&self) -> usize {
        self.view.skipped_degenerate_triangles
    }

    /// Borrow the geometry for the lifetime of the collider.
    pub fn as_view(&self) -> CollisionMeshView<'_> {
        // The mesh owns the immutable buffers described by its native view.
        unsafe { self.view.borrow() }
    }
}

/// Borrowed positions and triangle indices for a collision piece.
#[derive(Clone, Copy, Debug)]
pub struct CollisionMeshView<'a> {
    positions: &'a [[f32; 3]],
    indices: &'a [u32],
    skipped_degenerate_triangles: usize,
}

impl<'a> CollisionMeshView<'a> {
    pub fn positions(&self) -> &'a [[f32; 3]] {
        self.positions
    }

    pub fn indices(&self) -> &'a [u32] {
        self.indices
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn skipped_degenerate_triangles(&self) -> usize {
        self.skipped_degenerate_triangles
    }
}

/// Options for simplifying one convex collision piece in its source coordinate space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvexHullOptions {
    /// Desired vertex count, with a minimum of four.
    pub target_vertices: usize,
    /// Maximum inward deviation in the same units as the vertex positions.
    pub max_deviation: f64,
    /// Preserve the contact footprints on the minimum and maximum Y planes.
    pub preserve_vertical_support: bool,
}

impl Default for ConvexHullOptions {
    fn default() -> Self {
        Self {
            target_vertices: 256,
            max_deviation: 0.0,
            preserve_vertical_support: true,
        }
    }
}

/// Owned collision geometry and the outcome of the convex hull simplification.
#[derive(Debug)]
pub struct ConvexHullResult {
    pub mesh: CollisionMesh,
    pub input_vertices: usize,
    pub max_deviation: f64,
    pub met_target: bool,
}

/// Simplify one closed, outward-wound convex collision piece.
///
/// The result can exceed the requested vertex count when further reduction would
/// exceed the geometric tolerance or alter the protected contact footprints.
pub fn simplify_convex_hull(
    positions: &[[f32; 3]],
    indices: &[u32],
    options: ConvexHullOptions,
) -> Result<ConvexHullResult, Error> {
    let mut handle = ptr::null_mut();
    let mut info = MaybeUninit::uninit();
    // Native code copies the supplied slices and initializes both outputs on success.
    check(unsafe {
        lodestar_simplify_convex_hull(
            positions.as_ptr(),
            positions.len(),
            indices.as_ptr(),
            indices.len(),
            options.target_vertices,
            options.max_deviation,
            u32::from(options.preserve_vertical_support),
            &mut handle,
            info.as_mut_ptr(),
        )
    })?;
    let mesh = CollisionMesh::from_native(handle)?;
    let info = unsafe { info.assume_init() };
    Ok(ConvexHullResult {
        mesh,
        input_vertices: info.input_vertices,
        max_deviation: info.max_deviation,
        met_target: info.met_target != 0,
    })
}

/// Owned convex pieces that together represent one closed collision mesh.
#[derive(Debug)]
pub struct ConvexCompound {
    _handle: ConvexCompoundHandle,
    views: Vec<NativeCollisionMeshView>,
}

impl ConvexCompound {
    fn from_native(raw: *mut NativeConvexCompound) -> Result<Self, Error> {
        let handle = ConvexCompoundHandle(
            NonNull::new(raw).ok_or_else(|| Error("Native convex compound handle is missing".to_owned()))?,
        );
        let mut count = 0;
        check(unsafe { lodestar_convex_compound_get_hull_count(handle.0.as_ptr(), &mut count) })?;
        let mut views = Vec::with_capacity(count);
        for index in 0..count {
            let mut view = MaybeUninit::uninit();
            check(unsafe { lodestar_convex_compound_get_hull_view(handle.0.as_ptr(), index, view.as_mut_ptr()) })?;
            views.push(unsafe { view.assume_init() });
        }
        Ok(Self { _handle: handle, views })
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }

    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// Borrow the hull at this zero-based index.
    pub fn hull(&self, index: usize) -> Option<CollisionMeshView<'_>> {
        // Every native view borrows the buffers owned by this compound.
        self.views.get(index).map(|view| unsafe { view.borrow() })
    }

    pub fn hulls(&self) -> impl ExactSizeIterator<Item = CollisionMeshView<'_>> + '_ {
        self.views.iter().map(|view| unsafe { view.borrow() })
    }
}

/// Construct a convex point envelope. This alone does not certify mesh conformity.
pub fn build_convex_hull(positions: &[[f32; 3]]) -> Result<CollisionMesh, Error> {
    let mut handle = ptr::null_mut();
    check(unsafe { lodestar_build_convex_hull(positions.as_ptr(), positions.len(), &mut handle) })?;
    CollisionMesh::from_native(handle)
}

/// Convert a supported closed solid into convex pieces with the same occupied volume.
pub fn decompose_collision_hulls(
    positions: &[[f32; 3]],
    indices: &[u32],
    max_hulls: usize,
) -> Result<ConvexCompound, Error> {
    let mut handle = ptr::null_mut();
    // Native code copies the positions and indices before returning.
    check(unsafe {
        lodestar_decompose_collision_hulls(
            positions.as_ptr(),
            positions.len(),
            indices.as_ptr(),
            indices.len(),
            max_hulls,
            &mut handle,
        )
    })?;
    ConvexCompound::from_native(handle)
}

#[derive(Debug)]
pub struct CoherentCollisionShape {
    pub compound: ConvexCompound,
    pub surface_error_bound: f64,
    pub exact_solid_conversion: bool,
}

/// Build coherent closed convex pieces from surface elements, including open inputs.
/// A non-exact surface cover does not certify preservation of occupied interior volume.
pub fn build_coherent_collision_shape(positions: &[[f32; 3]], indices: &[u32], max_hulls: usize,
                                      max_surface_error: f64) -> Result<CoherentCollisionShape, Error> {
    let mut handle = ptr::null_mut();
    let mut info = MaybeUninit::<NativeCoherentCollisionInfo>::uninit();
    check(unsafe { lodestar_build_coherent_collision(positions.as_ptr(), positions.len(), indices.as_ptr(),
        indices.len(), max_hulls, max_surface_error, &mut handle, info.as_mut_ptr()) })?;
    let compound = ConvexCompound::from_native(handle)?;
    let info = unsafe { info.assume_init() };
    Ok(CoherentCollisionShape { compound, surface_error_bound: info.surface_error_bound,
        exact_solid_conversion: info.exact_solid_conversion != 0 })
}

#[derive(Debug)]
pub struct MeshSizePlan {
    pub visible_minimum: [f64; 3],
    pub visible_maximum: [f64; 3],
    pub container_minimum: [f64; 3],
    pub container_maximum: [f64; 3],
    pub center: [f64; 3],
    pub visible_size: [f64; 3],
    pub container_size: [f64; 3],
    pub visible_fraction: [f64; 3],
    pub requires_container: bool,
    pub requires_hull_physics: bool,
}

/// Propose a shared visual/size container while retaining authored physical scale.
/// A physics bounding box alone cannot establish the rendered mesh's normalization.
pub fn plan_mesh_size(positions: &[[f32; 3]], minimum_dimension: f64,
                      triangle_physics_minimum: f64) -> Result<MeshSizePlan, Error> {
    let mut plan = MaybeUninit::<NativeMeshSizePlan>::uninit();
    check(unsafe { lodestar_plan_mesh_size(positions.as_ptr(), positions.len(), minimum_dimension,
        triangle_physics_minimum, plan.as_mut_ptr()) })?;
    let plan = unsafe { plan.assume_init() };
    Ok(MeshSizePlan { visible_minimum: plan.visible_minimum, visible_maximum: plan.visible_maximum,
        container_minimum: plan.container_minimum, container_maximum: plan.container_maximum, center: plan.center,
        visible_size: plan.visible_size, container_size: plan.container_size, visible_fraction: plan.visible_fraction,
        requires_container: plan.requires_container != 0, requires_hull_physics: plan.requires_hull_physics != 0 })
}

#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// A source mesh and its cached analysis. Generated meshes own their buffers.
#[derive(Debug)]
pub struct Asset {
    handle: AssetHandle,
    analysis: Analysis,
}

impl Asset {
    /// Import a COLLADA, glTF, or GLB asset from a UTF-8 path.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::load_in_space(path, ImportSpace::Normalized)
    }

    /// Import with either normalized or scene coordinates.
    pub fn load_in_space(path: impl AsRef<Path>, space: ImportSpace) -> Result<Self, Error> {
        let path = path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error("Mesh path must be UTF-8".to_owned()))?;
        let path =
            CString::new(path).map_err(|_| Error("Mesh path contains a null byte".to_owned()))?;
        Self::from_native(|result| unsafe {
            lodestar_asset_load_in_space(path.as_ptr(), space as u32, result)
        })
    }

    /// Copy a mesh in its current coordinates, with optional authored topology IDs.
    pub fn from_mesh(mesh: MeshInput<'_>) -> Result<Self, Error> {
        let input = NativeMeshInput {
            mesh: NativeMeshView {
                vertices: mesh.vertices.as_ptr(),
                vertex_count: mesh.vertices.len(),
                indices: mesh.indices.as_ptr(),
                index_count: mesh.indices.len(),
                has_uv: u32::from(mesh.has_uv),
            },
            metadata: NativeMeshMetadata {
                triangle_materials: mesh.triangle_materials.as_ptr(),
                triangle_material_count: mesh.triangle_materials.len(),
                has_additional_uv: u32::from(mesh.has_additional_uv),
            },
            provenance: NativeMeshProvenance {
                vertex_position_ids: mesh.vertex_position_ids.as_ptr(),
                vertex_position_id_count: mesh.vertex_position_ids.len(),
                triangle_surface_ids: mesh.triangle_surface_ids.as_ptr(),
                triangle_surface_id_count: mesh.triangle_surface_ids.len(),
                has_unrepresented_vertex_attributes: u32::from(
                    mesh.has_unrepresented_vertex_attributes,
                ),
            },
        };
        // The mesh slices stay borrowed until the native call has copied their contents.
        Self::from_native(|result| unsafe { lodestar_asset_analyze_mesh(&input, result) })
    }

    /// Load the same built-in UV-free sphere as the Lodestar viewer.
    pub fn sample() -> Result<Self, Error> {
        Self::from_native(|result| unsafe { lodestar_asset_sample(result) })
    }

    pub fn analysis(&self) -> Analysis {
        self.analysis
    }

    /// Inspect source boundaries, seams, and splits with exact position matches.
    pub fn inspect_topology(&self) -> Result<TopologyReport, Error> {
        let mut report = MaybeUninit::uninit();
        check(unsafe {
            lodestar_asset_inspect_topology(self.handle.0.as_ptr(), report.as_mut_ptr())
        })?;
        // Success initializes every report field.
        Ok(unsafe { report.assume_init() })
    }

    /// Bound triangle counts with all current indexed chart boundary segments fixed.
    pub fn inspect_boundary_budget(&self) -> Result<BoundaryBudgetReport, Error> {
        let mut report = MaybeUninit::uninit();
        check(unsafe {
            lodestar_asset_inspect_boundary_budget(self.handle.0.as_ptr(), report.as_mut_ptr())
        })?;
        // Success initializes every report field.
        Ok(unsafe { report.assume_init() })
    }

    /// Generate an owned mesh. Quality is clamped to the range 1–12.
    pub fn generate(&self, mode: Mode, quality: i32) -> Result<Mesh, Error> {
        self.generate_using(|asset, result| unsafe {
            lodestar_asset_generate(asset, mode as u32, quality, result)
        })
    }

    /// Generate a planar mesh with a triangle target, reusing analysis on later calls.
    /// The result may stay above the target when further edits would change the source surface.
    pub fn generate_planar(&self, target_triangles: usize) -> Result<Mesh, Error> {
        self.generate_using(|asset, result| unsafe {
            lodestar_asset_generate_planar(asset, target_triangles, result)
        })
    }

    /// Shorten eligible straight seam chains and reuse analysis on later calls.
    /// The result may stay above the target when the remaining seams need all their vertices.
    pub fn generate_seam(&self, target_triangles: usize) -> Result<Mesh, Error> {
        self.generate_using(|asset, result| unsafe {
            lodestar_asset_generate_seam(asset, target_triangles, result)
        })
    }

    /// Build a static collider in the source coordinate space, independent of visual LOD.
    pub fn generate_collision(&self) -> Result<CollisionMesh, Error> {
        let mut handle = ptr::null_mut();
        check(unsafe { lodestar_asset_generate_collision(self.handle.0.as_ptr(), &mut handle) })?;
        CollisionMesh::from_native(handle)
    }

    fn generate_using(
        &self,
        generate: impl FnOnce(*const NativeAsset, *mut *mut NativeMesh) -> u32,
    ) -> Result<Mesh, Error> {
        let mut handle = ptr::null_mut();
        check(generate(self.handle.0.as_ptr(), &mut handle))?;
        Mesh::from_native(handle)
    }

    fn from_native(create: impl FnOnce(*mut *mut NativeAsset) -> u32) -> Result<Self, Error> {
        use SphereEligibility::*;

        let mut handle = ptr::null_mut();
        check(create(&mut handle))?;
        let handle = AssetHandle(
            NonNull::new(handle)
                .ok_or_else(|| Error("Native asset handle is missing".to_owned()))?,
        );
        let mut info = MaybeUninit::uninit();
        check(unsafe { lodestar_asset_get_info(handle.0.as_ptr(), info.as_mut_ptr()) })?;
        // Success initializes every analysis field.
        let info = unsafe { info.assume_init() };
        let sphere_eligibility = match info.sphere_eligibility {
            0 => Eligible,
            1 => HasTextureCoordinates,
            2 => NoFit,
            3 => RadialError,
            4 => IncompleteSurface,
            _ => return Err(Error("Unrecognized sphere eligibility value".to_owned())),
        };
        let analysis = Analysis {
            original_triangles: info.original_triangles,
            has_uv: info.has_uv != 0,
            fit: (info.fit_valid != 0).then_some(SphereFit {
                center: info.sphere_center,
                radius: info.sphere_radius,
                max_relative_error: info.max_relative_error,
                rms_relative_error: info.rms_relative_error,
            }),
            sphere_eligibility,
            max_face_inward_relative_error: (info.max_face_inward_error_valid != 0)
                .then_some(info.max_face_inward_relative_error),
        };
        Ok(Self { handle, analysis })
    }
}

fn check(status: u32) -> Result<(), Error> {
    if status == 0 {
        return Ok(());
    }
    // The native library gives this thread a live, null-terminated error buffer.
    let message = unsafe { CStr::from_ptr(lodestar_last_error()) };
    Err(Error(message.to_string_lossy().into_owned()))
}

#[derive(Debug)]
struct AssetHandle(NonNull<NativeAsset>);

impl Drop for AssetHandle {
    fn drop(&mut self) {
        unsafe { lodestar_asset_destroy(self.0.as_ptr()) };
    }
}

#[derive(Debug)]
struct MeshHandle(NonNull<NativeMesh>);

impl Drop for MeshHandle {
    fn drop(&mut self) {
        unsafe { lodestar_mesh_destroy(self.0.as_ptr()) };
    }
}

#[derive(Debug)]
struct CollisionMeshHandle(NonNull<NativeCollisionMesh>);

impl Drop for CollisionMeshHandle {
    fn drop(&mut self) {
        unsafe { lodestar_collision_mesh_destroy(self.0.as_ptr()) };
    }
}

#[derive(Debug)]
struct ConvexCompoundHandle(NonNull<NativeConvexCompound>);

impl Drop for ConvexCompoundHandle {
    fn drop(&mut self) {
        unsafe { lodestar_convex_compound_destroy(self.0.as_ptr()) };
    }
}

#[repr(C)]
struct NativeAsset {
    _opaque: [u8; 0],
}

#[repr(C)]
struct NativeMesh {
    _opaque: [u8; 0],
}

#[repr(C)]
struct NativeCollisionMesh {
    _opaque: [u8; 0],
}

#[repr(C)]
struct NativeConvexCompound {
    _opaque: [u8; 0],
}

#[repr(C)]
struct NativeAssetInfo {
    original_triangles: usize,
    has_uv: u32,
    fit_valid: u32,
    sphere_eligibility: u32,
    max_face_inward_error_valid: u32,
    sphere_center: [f32; 3],
    sphere_radius: f32,
    max_relative_error: f32,
    rms_relative_error: f32,
    max_face_inward_relative_error: f32,
}

#[repr(C)]
#[derive(Debug)]
struct NativeMeshView {
    vertices: *const Vertex,
    vertex_count: usize,
    indices: *const u32,
    index_count: usize,
    has_uv: u32,
}

#[repr(C)]
#[derive(Debug)]
struct NativeCollisionMeshView {
    positions: *const [f32; 3],
    position_count: usize,
    indices: *const u32,
    index_count: usize,
    skipped_degenerate_triangles: usize,
}

impl NativeCollisionMeshView {
    /// The native owner must keep these buffers valid for the lifetime of this view.
    unsafe fn borrow(&self) -> CollisionMeshView<'_> {
        CollisionMeshView {
            positions: if self.position_count == 0 {
                &[]
            } else {
                unsafe { slice::from_raw_parts(self.positions, self.position_count) }
            },
            indices: if self.index_count == 0 {
                &[]
            } else {
                unsafe { slice::from_raw_parts(self.indices, self.index_count) }
            },
            skipped_degenerate_triangles: self.skipped_degenerate_triangles,
        }
    }
}

#[repr(C)]
struct NativeConvexHullInfo {
    input_vertices: usize,
    max_deviation: f64,
    met_target: u32,
}

#[repr(C)]
struct NativeCoherentCollisionInfo {
    surface_error_bound: f64,
    exact_solid_conversion: u32,
}

#[repr(C)]
struct NativeMeshSizePlan {
    visible_minimum: [f64; 3],
    visible_maximum: [f64; 3],
    container_minimum: [f64; 3],
    container_maximum: [f64; 3],
    center: [f64; 3],
    visible_size: [f64; 3],
    container_size: [f64; 3],
    visible_fraction: [f64; 3],
    requires_container: u32,
    requires_hull_physics: u32,
}

#[repr(C)]
#[derive(Debug)]
struct NativeMeshMetadata {
    triangle_materials: *const u32,
    triangle_material_count: usize,
    has_additional_uv: u32,
}

#[repr(C)]
#[derive(Debug)]
struct NativeMeshProvenance {
    vertex_position_ids: *const u64,
    vertex_position_id_count: usize,
    triangle_surface_ids: *const u32,
    triangle_surface_id_count: usize,
    has_unrepresented_vertex_attributes: u32,
}

#[repr(C)]
struct NativeMeshInput {
    mesh: NativeMeshView,
    metadata: NativeMeshMetadata,
    provenance: NativeMeshProvenance,
}

extern "C" {
    fn lodestar_fit_surface(
        parameters: *const [f64; 2],
        positions: *const [f64; 3],
        count: usize,
        result: *mut PolynomialSurface,
    ) -> u32;
    fn lodestar_polynomial_surface_patch(
        surface: *const PolynomialSurface,
        domain: *const [f64; 2],
        result: *mut BezierTriangle,
    ) -> u32;
    fn lodestar_differentiate_surface(
        patch: *const BezierTriangle,
        u: f64,
        v: f64,
        result: *mut SurfaceDifferential,
    ) -> u32;
    fn lodestar_calculate_surface_lod(
        patches: *const BezierTriangle,
        patch_count: usize,
        world_units_per_pixel: f64,
        maximum_pixel_error: f64,
        relative_tolerance: f64,
        maximum_depth: u32,
        result: *mut SurfaceLodBudget,
    ) -> u32;
    fn lodestar_tessellate_surface(
        patch: *const BezierTriangle,
        world_units_per_pixel: f64,
        maximum_pixel_error: f64,
        maximum_depth: u32,
        result: *mut *mut NativeMesh,
        maximum_deviation_bound: *mut f64,
    ) -> u32;
    fn lodestar_last_error() -> *const c_char;
    fn lodestar_asset_load_in_space(
        path: *const c_char,
        space: u32,
        result: *mut *mut NativeAsset,
    ) -> u32;
    fn lodestar_asset_analyze_mesh(
        input: *const NativeMeshInput,
        result: *mut *mut NativeAsset,
    ) -> u32;
    fn lodestar_asset_sample(result: *mut *mut NativeAsset) -> u32;
    fn lodestar_asset_destroy(asset: *mut NativeAsset);
    fn lodestar_asset_get_info(asset: *const NativeAsset, result: *mut NativeAssetInfo) -> u32;
    fn lodestar_asset_inspect_topology(
        asset: *const NativeAsset,
        result: *mut TopologyReport,
    ) -> u32;
    fn lodestar_asset_inspect_boundary_budget(
        asset: *const NativeAsset,
        result: *mut BoundaryBudgetReport,
    ) -> u32;
    fn lodestar_asset_generate(
        asset: *const NativeAsset,
        mode: u32,
        quality: i32,
        result: *mut *mut NativeMesh,
    ) -> u32;
    fn lodestar_asset_generate_planar(
        asset: *const NativeAsset,
        target_triangles: usize,
        result: *mut *mut NativeMesh,
    ) -> u32;
    fn lodestar_asset_generate_seam(
        asset: *const NativeAsset,
        target_triangles: usize,
        result: *mut *mut NativeMesh,
    ) -> u32;
    fn lodestar_asset_generate_collision(
        asset: *const NativeAsset,
        result: *mut *mut NativeCollisionMesh,
    ) -> u32;
    fn lodestar_simplify_convex_hull(
        positions: *const [f32; 3],
        position_count: usize,
        indices: *const u32,
        index_count: usize,
        target_vertices: usize,
        max_deviation: f64,
        preserve_vertical_support: u32,
        result: *mut *mut NativeCollisionMesh,
        info: *mut NativeConvexHullInfo,
    ) -> u32;
    fn lodestar_build_convex_hull(
        positions: *const [f32; 3],
        position_count: usize,
        result: *mut *mut NativeCollisionMesh,
    ) -> u32;
    fn lodestar_build_coherent_collision(
        positions: *const [f32; 3], position_count: usize, indices: *const u32, index_count: usize,
        max_hulls: usize, max_surface_error: f64, result: *mut *mut NativeConvexCompound,
        info: *mut NativeCoherentCollisionInfo,
    ) -> u32;
    fn lodestar_plan_mesh_size(
        positions: *const [f32; 3], position_count: usize, minimum_dimension: f64,
        triangle_physics_minimum: f64, result: *mut NativeMeshSizePlan,
    ) -> u32;
    fn lodestar_decompose_collision_hulls(
        positions: *const [f32; 3],
        position_count: usize,
        indices: *const u32,
        index_count: usize,
        max_hulls: usize,
        result: *mut *mut NativeConvexCompound,
    ) -> u32;
    fn lodestar_convex_compound_get_hull_count(
        compound: *const NativeConvexCompound,
        result: *mut usize,
    ) -> u32;
    fn lodestar_convex_compound_get_hull_view(
        compound: *const NativeConvexCompound,
        hull_index: usize,
        result: *mut NativeCollisionMeshView,
    ) -> u32;
    fn lodestar_convex_compound_destroy(compound: *mut NativeConvexCompound);
    fn lodestar_collision_mesh_get_view(
        mesh: *const NativeCollisionMesh,
        result: *mut NativeCollisionMeshView,
    ) -> u32;
    fn lodestar_collision_mesh_destroy(mesh: *mut NativeCollisionMesh);
    fn lodestar_mesh_get_view(mesh: *const NativeMesh, result: *mut NativeMeshView) -> u32;
    fn lodestar_mesh_get_metadata(mesh: *const NativeMesh, result: *mut NativeMeshMetadata) -> u32;
    fn lodestar_mesh_get_provenance(
        mesh: *const NativeMesh,
        result: *mut NativeMeshProvenance,
    ) -> u32;
    fn lodestar_mesh_destroy(mesh: *mut NativeMesh);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_surface_fitting_recovers_a_reusable_quadratic() -> Result<(), Error> {
        let mut parameters = Vec::new();
        let mut positions = Vec::new();
        for i in 0..=4 {
            for j in 0..=4 - i {
                let u = f64::from(i) / 4.0;
                let v = f64::from(j) / 4.0;
                parameters.push([u, v]);
                positions.push([u, v, 0.75 * u * u]);
            }
        }
        let surface = fit_surface(&parameters, &positions)?;
        assert_eq!(surface.degree, 2);
        assert!(surface.maximum_vertex_residual < 1e-12);
        let patch = surface.patch([[0., 0.], [1., 0.], [0., 1.]])?;
        let d = differentiate_surface(&patch, 0.2, 0.3)?;
        assert!((d.position[2] - 0.03).abs() < 1e-12);
        assert!((d.duu[2] - 1.5).abs() < 1e-12);
        assert!(fit_surface(&parameters[..2], &positions).is_err());
        assert!(fit_surface(&[[0., 0.]; 3], &[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]).is_err());
        assert!(surface.patch([[0., 0.]; 3]).is_err());
        let tessellation = tessellate_surface(&patch, 0.04, 0.5, 8)?;
        assert!(tessellation.mesh.triangle_count() > 0);
        assert!(tessellation.maximum_deviation_bound <= 0.02);
        Ok(())
    }

    fn calculus_patch(curvature: f64) -> BezierTriangle {
        let powers = [
            (0, 0),
            (3, 0),
            (0, 3),
            (1, 0),
            (2, 0),
            (0, 1),
            (0, 2),
            (2, 1),
            (1, 2),
            (1, 1),
        ];
        BezierTriangle {
            controls: powers.map(|(i, j)| {
                let u = f64::from(i);
                [u / 3.0, f64::from(j) / 3.0, curvature * u * (u - 1.0) / 6.0]
            }),
        }
    }

    #[test]
    fn surface_calculus_crosses_the_native_abi() -> Result<(), Error> {
        let patch = calculus_patch(0.75);
        let d = differentiate_surface(&patch, 0.2, 0.3)?;
        assert!((d.position[2] - 0.03).abs() < 1e-12);
        assert!((d.du[2] - 0.3).abs() < 1e-12);
        assert!((d.duu[2] - 1.5).abs() < 1e-12);
        let options = SurfaceLodOptions {
            world_units_per_pixel: 0.01,
            maximum_pixel_error: 0.5,
            relative_tolerance: 1e-5,
            maximum_depth: 8,
        };
        let flat = calculate_surface_lod(&[calculus_patch(0.0)], options)?;
        assert!(flat.converged());
        assert!((flat.area - 0.5).abs() < 1e-12);
        assert!(flat.absolute_curvature < 1e-12);
        assert_eq!(flat.suggested_triangles, 1);
        let near = calculate_surface_lod(&[patch], options)?;
        let far = calculate_surface_lod(
            &[patch],
            SurfaceLodOptions {
                world_units_per_pixel: 0.02,
                ..options
            },
        )?;
        assert!(near.converged() && far.converged());
        assert!(
            (near.triangle_density_integral - 2.0 * far.triangle_density_integral).abs() < 1e-10
        );
        assert!(far.suggested_triangles < near.suggested_triangles);
        assert_eq!(
            near.interior_vertex_density_integral,
            near.triangle_density_integral / 2.0
        );
        assert!(calculate_surface_lod(&[], options).is_err());
        assert!(differentiate_surface(&patch, 0.8, 0.8).is_err());
        let unfinished = calculate_surface_lod(
            &[patch],
            SurfaceLodOptions {
                relative_tolerance: 1e-10,
                maximum_depth: 0,
                ..options
            },
        )?;
        assert!(!unfinished.converged());
        let near_mesh = tessellate_surface(&patch, 0.01, 0.5, 8)?;
        let far_mesh = tessellate_surface(&patch, 0.04, 0.5, 8)?;
        assert!(far_mesh.mesh.vertices().len() < near_mesh.mesh.vertices().len());
        assert!(near_mesh.maximum_deviation_bound <= 0.005);
        assert!(far_mesh.maximum_deviation_bound <= 0.02);
        assert!(near_mesh
            .mesh
            .indices()
            .iter()
            .all(|&i| (i as usize) < near_mesh.mesh.vertices().len()));
        assert!(tessellate_surface(&patch, 0.01, 0.5, 0).is_err());
        let mut failed_mesh = NonNull::<NativeMesh>::dangling().as_ptr();
        let mut failed_bound = f64::NAN;
        let status = unsafe {
            lodestar_tessellate_surface(&patch, 0.01, -0.5, 8, &mut failed_mesh, &mut failed_bound)
        };
        assert_ne!(status, 0);
        assert!(failed_mesh.is_null());
        assert_eq!(failed_bound, 0.0);
        for vertex in near_mesh.mesh.vertices() {
            assert!((vertex.position[2] - 0.75 * vertex.position[0].powi(2)).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn binding_round_trip() -> Result<(), Error> {
        let asset = Asset::sample()?;
        let analysis = asset.analysis();
        assert_eq!(analysis.sphere_eligibility, SphereEligibility::Eligible);
        assert!(!analysis.has_uv);
        assert!(analysis.fit.unwrap().max_relative_error < 0.01);
        assert!(analysis.max_face_inward_relative_error.unwrap() < 0.1);

        let original = asset.generate(Mode::Original, 4)?;
        let baseline = asset.generate(Mode::Baseline, 4)?;
        let sphere = asset.generate(Mode::Sphere, 4)?;
        let planar = asset.generate(Mode::Planar, 4)?;
        let planar_again = asset.generate_planar(2)?;
        drop(asset);

        assert_eq!(original.triangle_count(), analysis.original_triangles);
        assert_eq!(planar.indices(), original.indices());
        assert_eq!(planar_again.vertices(), original.vertices());
        assert!(planar.triangle_materials().is_empty());
        assert!(!planar.has_additional_uv());
        assert!(baseline.triangle_count() < analysis.original_triangles);
        assert!(sphere.triangle_count() < analysis.original_triangles);
        for mesh in [original, baseline, sphere] {
            assert!(mesh.triangle_count() > 0);
            assert!(mesh
                .indices()
                .iter()
                .all(|&i| (i as usize) < mesh.vertices().len()));
        }
        assert!(Asset::load("").is_err());
        Ok(())
    }

    #[test]
    fn planar_scene_preserves_materials_and_coordinates() -> Result<(), Error> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/assets/planar-scene.dae");
        let asset = Asset::load_in_space(path, ImportSpace::Scene)?;
        let topology = asset.inspect_topology()?;
        assert!(topology.is_valid());
        assert_eq!(topology.geometric_edges, 16);
        assert_eq!(topology.boundary_edges, 8);
        assert_eq!(topology.indexed_boundary_edges, 8);
        assert_eq!(topology.shared_edges, 8);
        assert_eq!(topology.recoverable_split_edges, 0);
        let budget = asset.inspect_boundary_budget()?;
        assert!(budget.is_valid());
        assert_eq!(budget.source_triangles, 8);
        assert_eq!(budget.certified_charts, 2);
        assert_eq!(budget.unsupported_charts, 0);
        assert_eq!(budget.protected_boundary_edges, 8);
        assert_eq!(budget.boundary_loops, 2);
        assert_eq!(budget.interior_vertices, 2);
        assert_eq!(budget.lower_bound_triangles, 4);
        let original = asset.generate(Mode::Original, 12)?;
        let planar = asset.generate_planar(4)?;
        drop(asset);

        assert_eq!(original.triangle_count(), 8);
        assert_eq!(planar.triangle_count(), 4);
        assert!(planar.has_uv());
        assert!(!planar.has_additional_uv());
        assert_eq!(planar.triangle_materials().len(), 4);
        for &material in original.triangle_materials() {
            assert_eq!(
                planar
                    .triangle_materials()
                    .iter()
                    .filter(|&&id| id == material)
                    .count(),
                2
            );
        }
        assert!(planar.vertices().iter().all(|vertex| {
            (5.0..=11.0).contains(&vertex.position[0]) && vertex.position[2] == 3.0
        }));
        Ok(())
    }

    #[test]
    fn collision_preserves_source_geometry_through_visual_lod() -> Result<(), Error> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/assets/planar-scene.dae");
        let asset = Asset::load_in_space(path, ImportSpace::Scene)?;
        let original = asset.generate(Mode::Original, 12)?;
        let collision = asset.generate_collision()?;
        assert_eq!(asset.generate_planar(4)?.triangle_count(), 4);
        drop(asset);

        assert_eq!(collision.triangle_count(), 8);
        assert_eq!(collision.skipped_degenerate_triangles(), 0);
        for (&collider_index, &source_index) in collision.indices().iter().zip(original.indices()) {
            let position = collision.positions()[collider_index as usize];
            assert_eq!(
                position,
                original.vertices()[source_index as usize].position
            );
            assert!((5.0..=11.0).contains(&position[0]));
            assert_eq!(position[2], 3.0);
        }
        Ok(())
    }

    #[test]
    fn coherent_open_floor_and_micro_size_plan_use_native_math() -> Result<(), Error> {
        let points = [[0., 0., 0.], [0., 0., 1.], [1., 0., 1.], [1., 0., 0.]];
        let shape = build_coherent_collision_shape(&points, &[0, 1, 2, 0, 2, 3], 256, 0.01)?;
        assert!(!shape.exact_solid_conversion);
        assert!(shape.surface_error_bound <= 0.01);
        assert!(!shape.compound.is_empty());
        for hull in shape.compound.hulls() {
            assert!(hull.positions().iter().all(|p| p[1] <= 0.));
        }
        let plan = plan_mesh_size(&[[0., 0., 0.], [0.002, 0.004, 0.006]], 0.01, 0.5)?;
        assert!(plan.requires_container && plan.requires_hull_physics);
        assert_eq!(plan.container_size, [0.01; 3]);
        assert!(!plan_mesh_size(&[[0.; 3], [0.01; 3]], 0.01, 0.5)?.requires_container);
        assert!(build_convex_hull(&points).is_err());
        Ok(())
    }

    #[test]
    fn convex_hull_simplification_returns_an_owned_mesh() -> Result<(), Error> {
        let result = {
            let positions = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.25, 0.25, 0.0],
            ];
            let indices = [0, 2, 4, 2, 1, 4, 1, 0, 4, 0, 1, 3, 0, 3, 2, 1, 2, 3];
            simplify_convex_hull(
                &positions,
                &indices,
                ConvexHullOptions {
                    target_vertices: 4,
                    ..ConvexHullOptions::default()
                },
            )?
        };
        assert_eq!(result.input_vertices, 5);
        assert_eq!(result.max_deviation, 0.0);
        assert!(result.met_target);
        assert_eq!(result.mesh.positions().len(), 4);
        assert_eq!(result.mesh.triangle_count(), 4);
        assert!(!result.mesh.positions().contains(&[0.25, 0.25, 0.0]));
        assert!(result.mesh.indices().iter().all(|&index| index < 4));
        Ok(())
    }

    #[test]
    fn convex_hull_simplification_rejects_an_open_piece() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let indices = [0, 2, 1, 0, 1, 3, 0, 3, 2];
        assert!(simplify_convex_hull(&positions, &indices, ConvexHullOptions::default()).is_err());
    }

    #[test]
    fn convex_decomposition_owns_distinct_hull_buffers() -> Result<(), Error> {
        let compound = {
            let tetrahedron = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ];
            let triangles = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
            let positions = [tetrahedron, tetrahedron.map(|[x, y, z]| [x + 3.0, y, z])].concat();
            let indices = [triangles, triangles.map(|vertex| vertex + 4)].concat();
            assert!(decompose_collision_hulls(&positions, &indices, 1).is_err());
            decompose_collision_hulls(&positions, &indices, 2)?
        };
        assert_eq!(compound.len(), 2);
        assert!(!compound.is_empty());
        assert!(compound.hull(2).is_none());
        let mut minima = Vec::new();
        for (index, hull) in compound.hulls().enumerate() {
            assert_eq!(hull.positions().len(), 4);
            assert_eq!(hull.triangle_count(), 4);
            assert!(hull.indices().iter().all(|&vertex| vertex < 4));
            assert_eq!(compound.hull(index).unwrap().positions(), hull.positions());
            minima.push(
                hull.positions()
                    .iter()
                    .map(|p| p[0])
                    .reduce(f32::min)
                    .unwrap(),
            );
        }
        minima.sort_by(f32::total_cmp);
        assert_eq!(minima, [0.0, 3.0]);
        Ok(())
    }

    #[test]
    fn supplied_position_ids_enable_safe_planar_reduction() -> Result<(), Error> {
        let vertices = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
        .map(|[x, y]| Vertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [x / 2.0, y / 2.0],
        });
        let input = MeshInput {
            vertices: &vertices,
            indices: &[0, 1, 4, 1, 2, 4, 2, 3, 5, 3, 0, 5],
            has_uv: true,
            triangle_materials: &[17, 17, 17, 17],
            triangle_surface_ids: &[7, 7, 7, 7],
            ..MeshInput::default()
        };
        let unknown = Asset::from_mesh(input)?;
        assert_eq!(unknown.inspect_topology()?.recoverable_split_edges, 0);
        assert_eq!(unknown.generate_planar(2)?.indices(), input.indices);

        let proven = Asset::from_mesh(MeshInput {
            vertex_position_ids: &[1, 2, 3, 4, 5, 5],
            ..input
        })?;
        assert_eq!(proven.inspect_topology()?.recoverable_split_edges, 2);
        let reduced = proven.generate_planar(2)?;
        drop(proven);
        assert_eq!(reduced.triangle_count(), 2);
        assert_eq!(reduced.vertex_position_ids(), &[1, 2, 3, 4]);
        assert_eq!(reduced.triangle_surface_ids(), &[7, 7]);
        assert_eq!(reduced.triangle_materials(), &[17, 17]);
        assert!(Asset::from_mesh(MeshInput {
            vertex_position_ids: &[1, 1, 3, 4, 5, 5],
            ..input
        })
        .is_err());
        Ok(())
    }

    #[test]
    fn seam_generation_preserves_both_sides_attributes() -> Result<(), Error> {
        let positions = [
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ];
        let vertices: [Vertex; 8] = std::array::from_fn(|i| {
            let [x, y] = positions[i];
            let right = i >= 4;
            Vertex {
                position: [x, y, 0.0],
                normal: if right {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0]
                },
                uv: if right { [x + 4.0, 1.0 - y] } else { [x, y] },
            }
        });
        let input = MeshInput {
            vertices: &vertices,
            indices: &[0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6],
            has_uv: true,
            triangle_materials: &[3, 3, 4, 4],
            triangle_surface_ids: &[9, 9, 10, 10],
            vertex_position_ids: &[1, 2, 3, 4, 5, 2, 3, 4],
            ..MeshInput::default()
        };
        let asset = Asset::from_mesh(input)?;
        let reduced = asset.generate_seam(2)?;
        assert_eq!(reduced.triangle_count(), 2);
        assert_eq!(reduced.triangle_materials(), &[3, 4]);
        assert_eq!(reduced.triangle_surface_ids(), &[9, 10]);
        assert_eq!(reduced.vertex_position_ids(), &[1, 2, 4, 5, 2, 4]);
        assert_eq!(asset.generate(Mode::Seam, 4)?.indices(), reduced.indices());
        assert_eq!(asset.generate_seam(4)?.indices(), input.indices);
        assert_eq!(
            reduced.vertices(),
            &[
                vertices[0],
                vertices[1],
                vertices[3],
                vertices[4],
                vertices[5],
                vertices[7]
            ],
        );

        let mut altered = vertices;
        altered[6].uv[1] += 0.125;
        let protected = Asset::from_mesh(MeshInput {
            vertices: &altered,
            ..input
        })?;
        let unchanged = protected.generate_seam(2)?;
        assert_eq!(unchanged.indices(), input.indices);
        assert_eq!(unchanged.vertices(), &altered);
        Ok(())
    }
}
