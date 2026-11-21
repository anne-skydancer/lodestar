//! Rust bindings for Lodestar's native mesh conversion library.
//!
//! Build the native `lodestar_c` target with CMake before compiling this crate.

use std::ffi::{c_char, CStr, CString};
use std::fmt;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::slice;

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

    fn generate_using(
        &self,
        generate: impl FnOnce(*const NativeAsset, *mut *mut NativeMesh) -> u32,
    ) -> Result<Mesh, Error> {
        let mut handle = ptr::null_mut();
        check(generate(self.handle.0.as_ptr(), &mut handle))?;
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
        // Success initializes the views. Keep their buffers alive in the returned mesh.
        Ok(Mesh {
            _handle: handle,
            view: unsafe { view.assume_init() },
            metadata: unsafe { metadata.assume_init() },
            provenance: unsafe { provenance.assume_init() },
        })
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

#[repr(C)]
struct NativeAsset {
    _opaque: [u8; 0],
}

#[repr(C)]
struct NativeMesh {
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
