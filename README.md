# Lodestar

Lodestar is a C++20 and Rust prototype for exploring geometric levels of detail (LOD) on Second Life-style meshes. It compares imported triangles with meshoptimizer, analytic sphere, planar, and exact seam representations. You can compare silhouettes and triangle counts at different quality levels. The original mesh is the default for every asset.

## Build and run

Install CMake 3.24 or later, a C++20 toolchain (Visual Studio with the **Desktop development with C++** workload on Windows), and the Vulkan SDK with `glslc`. The first configure uses Git to fetch its dependencies from GitHub. The viewer needs a Vulkan-capable graphics driver with swapchain support.

Use the Visual Studio generator to run these PowerShell commands from the repository root.

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release --parallel 4
.\build\Release\inverseglod.exe
```

For a single-configuration build, configure with Ninja.

```sh
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-ninja --parallel 4
./build-ninja/inverseglod
```

For Ninja on Windows, use a Visual Studio developer shell and run `.\build-ninja\inverseglod.exe`. On Linux, install the zlib development package and the GLFW build dependencies for your display backend. For an X11 build, add `-DGLFW_BUILD_WAYLAND=OFF` at configure time. Shaders are built into the executable, so it can run from any working directory.

With no arguments, Lodestar opens a built-in UV-free sphere. To open your own mesh, pass a `.dae`, `.gltf`, or `.glb` file as the first argument, or drop it into the window. Lodestar applies scene-node transforms and centers and scales the mesh for viewing. It renders normal-based lighting with depth testing. If a dropped file fails to load, the previous mesh stays visible and the window title shows the error.

## Compare representations

| Input | Action |
| --- | --- |
| `1` | View the original triangles. |
| `2` | View the meshoptimizer baseline. |
| `3` | View the analytic sphere when eligible. Otherwise, Lodestar shows the original and reports the reason. |
| `4` | View the planar simplification. The triangle count can remain above the requested target when constraints prevent further reduction. |
| `5` | View the exact seam simplification. It shortens eligible straight boundary chains while preserving their geometry and attributes. |
| `Up` / `Down` or `+` / `-` | Adjust the LOD quality from 1 to 12. |
| `Left` / `Right`, `W` / `S` | Orbit horizontally or vertically. |
| `Page Up` / `Page Down` | Zoom in or out. |
| `Esc` | Close the viewer. |

The window title shows the active representation, its triangle count, the original count, and the sphere-fit error when available. The meshoptimizer baseline can miss the requested target because of its error budget. The built-in sphere lets you try the viewer without downloading an asset.

Run `--inspect` for a terminal comparison without opening a window.

```powershell
.\build\Release\inverseglod.exe --inspect
.\build\Release\inverseglod.exe --inspect "C:\path\to\model.glb" --quality 6
```

Inspection reports vertex and triangle counts, indexed and geometric topology, a fixed-boundary triangle bound, sphere fit and eligibility, and the planar and seam minimums. For seam mode it includes the number of removable boundary positions. Timings cover import and analysis, mesh generation, planar and seam preparation, and analytic tessellation when available. They exclude rendering and GPU uploads. Use identical assets and quality levels when comparing runs.

In the Windows Release build at quality 4, the built-in sphere had 3,968 source triangles, 318 baseline triangles, and 960 analytic triangles. The maximum inward deviation of the source triangles was about 0.24% of the fitted radius. Quality controls the baseline target and analytic tessellation density independently.

## Benchmark Second Life import samples

Run `lodestar_benchmark` to compare geometry and CPU generation costs and write CSV to standard output. The benchmark builds alongside the libraries with or without the Vulkan viewer. The Python helper downloads the public test files to a local directory and checks their hashes.

```powershell
$corpus = Join-Path $env:TEMP 'lodestar-corpus'
py tools/fetch_benchmark_assets.py $corpus
$meshes = @('duck.dae', 'TestHall_MESH.dae', 'simplebot.dae',
            'wholey sphere.dae', 'BarramundiFish.glb', 'Icosphere.dae') |
    ForEach-Object { Join-Path $corpus $_ }
.\build\Release\lodestar_benchmark.exe --sample --repeats 15 --samples 4096 --resolution 192 `
    --quality 1 --quality 3 --quality 4 --quality 8 --quality 12 @meshes > build/lodestar-benchmark.csv
```

For an eligible sphere, the benchmark requests a meshoptimizer baseline at the actual analytic triangle count. `matched` means the baseline is within 5% of the target. The benchmark records `unmatched` when the baseline misses the target, `above_source` when the target exceeds the original count, and `unpaired` when the sphere falls back. On fallback it verifies that the generated mesh preserves the original indices, UV flag, and all vertex components exactly. The viewer and `--inspect` use independent quality curves for their interactive comparison.

Surface error uses deterministic area-weighted samples from both meshes and nearest points on the opposite triangle surface. Distances are relative to the largest axis-aligned extent of the source mesh. The sampled maximum can miss a local deviation. In each orthographic view, silhouette error divides the differing pixels by the pixels covered by either mesh. The benchmark reports the mean and maximum across 13 views. Timings are CPU wall-clock medians after one warmup. Import and analysis include the sphere fit. Generation timings start with the analyzed asset and exclude measurement, GPU upload, and rendering.

### Second Life reference files

These results are from a Windows Release build on an AMD Ryzen 9 9950X3D, with 15 repetitions, 4,096 surface samples per direction, and 192 × 192 silhouette masks. The four COLLADA fixtures come from the Second Life viewer import test plan. Barramundi Fish, a CC0 model by Microsoft, is the glTF upload example in the Second Life 2025.05 release notes. [Second Life Wiki mesh import test](https://wiki.secondlife.com/wiki/Mesh_Import_test) [Second Life 2025.05 release notes](https://releasenotes.secondlife.com/viewer/7.2.0.16729091892.html) [Barramundi Fish credits and license](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BarramundiFish/README.md)

| Sample, quality 4 | Source triangles | Baseline triangles | Baseline surface RMS | Baseline mean silhouette error | Sphere fallback reason |
| --- | ---: | ---: | ---: | ---: | --- |
| Duck | 4,212 | 508 | 0.975% | 3.418% | UVs |
| TestHall | 8,602 | 8,271 | 0.051% | 1.057% | UVs |
| Simplebot | 10,312 | 10,312 | 0.000% | 0.000% | Radial error |
| Holey sphere | 1,008 | 992 | 0.009% | 0.021% | Incomplete coverage |
| Barramundi Fish | 3,864 | 310 | 0.253% | 3.540% | UVs |

The holey sphere passed the vertex fit threshold with a 0.898% maximum radial residual. The coverage check detected the openings and kept the original mesh. Simplebot was evaluated as static geometry.

The Duck file identifies Sony Computer Entertainment Inc. as the copyright holder and lists the SCEA Shared Source License, Version 1.0. The TestHall, Simplebot, and holey sphere packages are published as test downloads and contain no explicit asset license. [Khronos COLLADA Duck credits](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Duck/README.md)

### TestHall authored LOD

The same archive provides `TestHall_MESH.dae` and its authored medium LOD, `TestHall_LOD2.dae`. Both declare the same units, up axis, and node transforms. The authored file's vertex positions carry a baked X displacement. The explicit translation of +13.59409 m is the median X difference across 793 corresponding point pairs near the dominant offset and agrees with the difference between the meshes' minimum X bounds. The benchmark retains both files' scene coordinates and applies the requested translation to the authored mesh without rescaling or automatic alignment. [Second Life Wiki TestHall procedure](https://wiki.secondlife.com/wiki/Mesh_Import_test#Models_with_more_than_8_unique_faces_can_be_imported)

```powershell
.\build\Release\lodestar_benchmark.exe --repeats 9 --samples 32768 --resolution 384 `
    --authored-lod (Join-Path $corpus 'TestHall_LOD2.dae') --authored-translate 13.59409 0 0 `
    (Join-Path $corpus 'TestHall_MESH.dae') > build/testhall-exact-seam-comparison.csv
```

The authored mesh imports as 4,769 triangles, which becomes the requested budget for meshoptimizer, planar, and seam generation. All fidelity measurements use the same high-detail reference. The CSV records actual counts and budget status, topology and fixed-boundary counts, and median times for authored import, boundary analysis, preparation, and generation. The seam row records its own minimum and the number of positions its prepared edits can remove.

| TestHall representation | Triangles | Budget result | Surface RMS | Surface max, sampled | Mean silhouette error | Worst silhouette error |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| High-detail reference | 8,602 | Source | — | — | — | — |
| Authored medium | 4,769 | Target | 0.124% | 2.308% | 0.139% | 0.645% |
| Meshoptimizer | 8,271 | Unmatched | 0.056% | 1.987% | 1.207% | 3.745% |
| Planar | 8,602 | Original fallback | 0.000% | 0.000% | 0.000% | 0.000% |
| Exact seam | 8,602 | Original fallback | 0.000% | 0.000% | 0.000% | 0.000% |

The authored LOD had lower silhouette error at a substantially lower triangle count. Meshoptimizer had lower sampled surface RMS. Planar found no closed indexed vertex fan in the imported TestHall geometry, so its minimum remained 8,602 and it returned the source exactly. Median meshoptimizer generation took 0.673 ms. Planar preparation took 4.314 ms, and replay took 0.009 ms.

#### TestHall topology

Assimp imports TestHall as 64 mesh partitions across 12 scene nodes and 15 material IDs. In the scene-space benchmark, the high-detail mesh has 15,336 geometric edges. Of these, 4,866 have one incident triangle, 4,282 are shared by index, and 6,188 have two incident triangles that use split indices. Counting boundaries by index gives 17,242 one-sided edges. Among split edges, 5,597 differ in normals, 2,553 in UVs, and 1,173 in material. Counts overlap. Zero have fully compatible corner attributes or qualify for recovery.

#### Fixed-boundary triangle budget

The budget check groups triangles into connected charts across shared index edges with two consistently wound faces in the same material and surface. Chart boundaries retain every edge segment. For a certified orientable manifold chart, `F = B + 2I − 2χ`, where `F` is the triangle count, `B` the number of boundary edge segments, `I` the number of interior vertices, and `χ` the Euler characteristic. Holding the boundary segments and chart topology fixed gives a lower bound of `B − 2χ` because `I` is nonnegative. A closed chart contributes at least four triangles. The report also gives a bound that counts unsupported charts at their unchanged triangle counts.

TestHall has 4,321 charts. The check certifies 4,254 of them, with 16,860 protected boundary segments, 4,255 boundary loops, zero interior vertices, and a summed Euler characteristic of 4,253. The resulting lower bound is 8,354 triangles. Disconnected vertex fans leave 67 charts, containing 248 source triangles, unsupported. Keeping those charts unchanged raises the bound to 8,602, the full source count. The median boundary analysis time was 1.249 ms.

The authored target of 4,769 is below even the certified bound. A broader interior simplifier would face the same limit under fixed boundary segments and chart topology. Reaching the authored budget requires changing those constraints. The CSV reports `below_bound` for this result, `below_fallback_bound` for a target below the bound with unsupported charts retained, and `not_ruled_out` if the bound leaves the target possible. Invalid mesh structure is reported as `invalid`. Geometry and attribute constraints still determine which targets are attainable.

The two-panel control has eight source triangles and a four-triangle bound. Its authored target of four reports `not_ruled_out`, and planar mode reaches four while preserving the panel boundaries. [Two-panel control](tests/assets/planar-scene.dae)

#### Exact seam reductions

Seam mode can remove intermediate vertices from exact straight boundary chains. Each side must admit a coplanar retriangulation with the same affine normal and UV interpolation, checked for exact equality on the stored float values. Along a seam shared by two charts, both sides shorten the same geometric chain in one edit while keeping their separate material and surface assignments. An eligible exterior boundary can shorten on one side. Prepared edits use disjoint source vertex neighborhoods and retain a point if it touches an unsplit edge. The fixed-boundary budget still describes the source's original edge segments. Seam mode can go below that bound when it proves that a shorter chain represents the same boundary.

An independent exact arithmetic check on the stored scene-space vertices found 77 exactly collinear boundary vertex occurrences in TestHall's certified charts. Of these, 54 also had affine normals and zero had affine UVs. The smallest maximum-component UV deviation among the 54 was about 5.20 × 10⁻⁷. Two were within 10⁻⁶ and 47 within 10⁻⁴. With the same chart topology and no interior vertices, each removed boundary occurrence saves one triangle. Reaching the authored target would take 3,833 deletions. Across all charts, there are 81 exactly straight boundary occurrences before the attribute checks.

Seam mode found zero removable positions and returned all 8,602 source triangles exactly. Median preparation took 8.941 ms, and replay took 0.010 ms.

The imported split-seam control shortened both sides of a seam with different materials, normals, and UVs. A single coupled removal reduced four triangles to two with zero sampled surface displacement and silhouette error. The original fixed-boundary comparison still had a four-triangle bound. Median preparation and replay took 0.0048 ms and 0.0001 ms. Separate controls with a UV or normal break at the middle kept all four triangles.

### Matched sphere controls

The controls are the built-in sphere and an independent CC0 icosphere by The People's Internet. The fetch helper copies the published icosphere's original triangle positions and winding into indexed COLLADA with no UVs. [Icosphere source and license](https://commons.wikimedia.org/wiki/File:Sphere_(Icosphere).stl)

| Control | Triangles in each result | Surface RMS, baseline → analytic | Mean silhouette error, baseline → analytic | Generation median, baseline → analytic |
| --- | ---: | ---: | ---: | ---: |
| Built-in sphere, quality 3 | 528 | 0.492% → 0.441% | 1.242% → 1.163% | 0.359 → 0.003 ms |
| Built-in sphere, quality 4 | 960 | 0.243% → 0.222% | 0.638% → 0.544% | 0.337 → 0.006 ms |
| Icosphere, quality 3 | 528 | 0.541% → 0.376% | 1.121% → 0.944% | 0.069 → 0.003 ms |
| Icosphere, quality 4 | 960 | 0.234% → 0.163% | 0.282% → 0.416% | 0.068 → 0.006 ms |

The analytic controls had lower sampled surface RMS at both matched budgets. The 960-triangle icosphere had higher mean silhouette error in the analytic mode. At 16,384 surface samples per direction and 384 × 384 masks, all four matched comparisons showed the same direction of change for both metrics. The 960-triangle icosphere's mean silhouette error increased from 0.279% to 0.388% in that run.

At quality 1, both baselines missed the 48-triangle target. At quality 12, the analytic budget exceeded both original counts. The five Second Life inputs exercised fallback behavior. Texture fidelity, skinning, viewer-generated Second Life LOD, and rendering performance remain unmeasured by this benchmark.

## Use the libraries

The viewer shares its geometry code with the C++ library and Rust bindings. Configure with `-DLODESTAR_BUILD_VIEWER=OFF` to build the libraries and benchmark without Vulkan or a display server.

### C++

Add the Lodestar checkout as a CMake subdirectory and link the `Lodestar::core` target. The public interface is in [`include/lodestar/lodestar.hpp`](include/lodestar/lodestar.hpp).

```cmake
add_subdirectory(path/to/lodestar lodestar-build)
target_link_libraries(my_app PRIVATE Lodestar::core)
```

```cpp
#include <lodestar/lodestar.hpp>

auto asset = lodestar::loadAsset("model.glb");
auto mesh = lodestar::generateMesh(asset, lodestar::Mode::Sphere, 4);
```

Use `lodestar::sampleAsset()` for the built-in sphere or `lodestar::analyze()` for an in-memory mesh. File imports apply scene transforms, then center and scale the mesh by default. Use `loadAsset(path, lodestar::ImportSpace::Scene)` to retain the importer’s coordinate system and units for aligned comparisons. The asset exposes the source mesh, fit metrics, and sphere eligibility. An ineligible sphere request returns the source mesh.

Select `lodestar::Mode::Planar` for a quality-based planar mesh. For repeated triangle budgets, call `lodestar::analyzePlanarMesh()` once and use `lodestar::generatePlanarMesh()` with each budget. `minimumTriangles()` reports the lowest count available from the prepared edits.

Select `lodestar::Mode::Seam` for seam reduction. Use `lodestar::analyzeSeamMesh()` and `lodestar::generateSeamMesh()` for repeated triangle budgets, or `lodestar::simplifySeamMesh()` for a single budget. `removablePositions()` counts the geometric boundary positions for which the prepared edits can update every incident side together.

Call `lodestar::inspectTopology(mesh)` to compare indexed edges with exact endpoint position matches. For a caller-owned mesh, populate `vertexPositionIds` in vertex order from known authored vertex identities. Equal nonzero IDs identify one geometric vertex. Use `triangleSurfaceIds` in triangle order when the mesh contains separate surfaces or instances. Recovery requires matching normal, UV and material data. Each pair of endpoints must share an index or the same authored position ID, and exactly two triangles in one surface must traverse the edge in opposite directions.

Call `lodestar::inspectBoundaryBudget(mesh)` to check the current indexed charts, including material and surface boundaries. The diagnostic keeps split indices in place even if the input supplies authored position IDs.

### Rust

Build the shared C ABI with `cmake --build build --target lodestar_c --config Release`. Add the crate as a local path dependency in your `Cargo.toml`.

```toml
[dependencies]
lodestar = { path = "../lodestar/rust" }
```

Set `LODESTAR_LIB_DIR` to the directory containing the native library. On Windows, include that directory on `PATH` at runtime. These PowerShell commands use the Visual Studio build shown above.

```powershell
$env:LODESTAR_LIB_DIR = (Resolve-Path .\build\Release).Path
$env:Path = "$env:LODESTAR_LIB_DIR;$env:Path"
cargo test --manifest-path rust/Cargo.toml
```

On Linux, make the corresponding directory available via `LD_LIBRARY_PATH`. The Rust API owns the native buffers and exposes read-only vertex and index slices.

```rust
use lodestar::{Asset, Mode};

fn main() -> Result<(), lodestar::Error> {
    let asset = Asset::load("model.glb")?;
    let mesh = asset.generate(Mode::Sphere, 4)?;
    println!("{:?}: {} triangles", asset.analysis().sphere_eligibility, mesh.triangle_count());
    Ok(())
}
```

For aligned imports, call `Asset::load_in_space()` with `ImportSpace::Scene`. `Asset::inspect_topology()` reports edge and seam counts, and `Asset::inspect_boundary_budget()` reports the fixed-boundary triangle bound. To supply an authored mesh, pass borrowed `vertices` and `indices` to `Asset::from_mesh(MeshInput { … })` with `vertex_position_ids`, `triangle_surface_ids`, and `triangle_materials` where available. Use `..MeshInput::default()` for omitted fields. Set the UV flags to match the input and set `has_unrepresented_vertex_attributes` if its source carries other per-vertex data. Zero position IDs mean unknown. Generated meshes expose the retained IDs through `Mesh::vertex_position_ids()`, `Mesh::triangle_surface_ids()`, and `Mesh::triangle_materials()`.

`Mode::Planar` and `Mode::Seam` use quality levels. `Asset::generate_planar()` and `Asset::generate_seam()` each take a triangle budget and reuse their prepared analysis on subsequent calls.

The C ABI exposes in-memory mesh creation through `lodestar_asset_analyze_mesh()`, topology and budget inspection through `lodestar_asset_inspect_topology()` and `lodestar_asset_inspect_boundary_budget()`, and generated mesh IDs through `lodestar_mesh_get_provenance()`. Use `lodestar_asset_generate_seam()` to generate a seam mesh at an explicit triangle budget. [C header](include/lodestar/lodestar_c.h)

Run the regression tests with `ctest --test-dir build -C Release --output-on-failure`. The planar test reduces a textured 32-triangle grid to 14 triangles and checks preserved boundaries, holes, material seams, winding, and vertex attributes. A split fan reduces from four triangles to two with supplied authored vertex identities and keeps four when those identities are unknown. Seam tests cover coupled edits and protected attribute breaks. The suite also checks fixed-boundary certification, sphere eligibility, exact fallback, and benchmark metrics.

## Geometry modes and scope

Lodestar fits a sphere to the mesh vertices by least squares and reports the RMS and maximum radial residuals relative to the fitted radius. The analytic mode requires a valid fit, a maximum vertex residual under 1%, no UV coordinates, a consistently wound surface enclosing the fitted center, and a maximum inward deviation from the sphere of at most 10% of the radius across source triangle interiors and edges. It then tessellates the fitted sphere on the CPU at the requested quality.

Planar mode removes interior vertices from eligible coplanar triangle fans and retains their boundary segments. Edits require compatible winding, material IDs, and affine normal and UV interpolation within tolerance. Seam mode uses a separate exact check to shorten eligible boundary chains. Both modes preserve unsupported regions and can stay above the target. Imports with unrepresented vertex attributes such as extra UV components, colors, tangents, skin weights or morph targets keep their source triangles.

Topology diagnostics group edges by exact endpoint positions and report existing index sharing, boundaries, split edges, attribute seams, winding conflicts and ambiguities. A split can have matching stored attributes even when its authored connection is unknown. Assimp imports keep these unproven splits intact, which preserves intentional slits or coincident surfaces. Planar recovery uses caller-provided authored vertex identities on in-memory meshes.

Assimp imports positions, normals, and UV0.xy into the mesh buffers. The sphere gate checks for UVs in all channels. Imported triangle material IDs are local to each source asset. The viewer uses a single material and normal-based lighting. The meshoptimizer comparison can change the shape. General inverse-GLOD reconstruction, analytic patches with certified error bounds, a persistent LOD cache, adaptive screen-space tessellation, and Second Life viewer integration remain research goals.
