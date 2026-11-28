# Lodestar

Lodestar is a C++20 and Rust prototype for exploring geometric levels of detail (LOD) on Second Life-style meshes. It compares imported triangles with meshoptimizer, analytic sphere, planar, and exact seam representations. You can compare silhouettes and triangle counts at different quality levels. The original mesh is the default for every asset.

The SL upload prototype automatically computes visual LODs, refines collision surfaces, and attempts collision hull generation against the authoritative High surface. The [upload contract and current integration limits](docs/sl-upload-contract.md) distinguish automatic preparation from the Physics tab's hull-only Analyze button.

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
| `6` | View the static physics mesh created from the imported source. |
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

## Generate a conforming physics mesh

On file open or drop, the viewer creates static collision triangles from the same source positions and triangle winding it renders in Original mode. This works with externally authored COLLADA and glTF files. A visual LOD change reuses the source collider, so physical openings and walkable surfaces stay in the same place as the camera moves. All nonzero-area source triangles participate, including decorative surfaces.

Export those triangles as a COLLADA physics file in the source scene's coordinate system and scale.

```powershell
.\build\Release\inverseglod.exe --export-physics build/model-physics.dae C:\path\to\model.dae
```

The export flattens the imported scene into one static mesh. The linked guide describes a per-object workflow using the uploader's Physics tab and the rezzed object's **Prim** shape type. Use it as background and verify the actual settings, openings, and walkable surfaces in the target viewer at the final in-world scale. [Second Life mesh upload reference](https://community.secondlife.com/knowledgebase/english/uploading-a-mesh-model-r274/)

### Simplify a convex collision piece

Pass one closed convex piece to the hull simplifier. With zero allowed deviation, it can remove redundant vertices on flat faces and edges while preserving the occupied volume. A positive deviation allows a smaller hull inside the original and keeps the top and bottom contact footprints by default. The result reports the maximum deviation and whether it met the vertex target.

```powershell
.\build\Release\inverseglod.exe --export-physics build/piece-hull.dae --simplify-hull 64 0.005 C:\path\to\convex-piece.glb
```

Keep each convex piece separate when building compound collision. A doorway can use separate posts and a lintel. A staircase can use separate treads. The simplifier checks its input for a closed convex surface. Use the existing source triangle collider for concave objects such as TestHall and validate any compound collider with clearance and support probes.

The C++ API is `lodestar::simplifyConvexHull`. Rust exposes `lodestar::simplify_convex_hull`, and the C ABI exposes `lodestar_simplify_convex_hull`. All three take a vertex target and a maximum deviation in source units. The COLLADA export command imports in metre scene space.

### Convert a solid into collision hulls

Hull conversion keeps closed convex shells as separate pieces. It also partitions closed, axis-aligned concave solids into boxes that cover the same occupied space. An integrated doorway or staircase can therefore be converted from a single source mesh. Open surfaces and unsupported concave geometry can use the source triangle collider.

```powershell
.\build\Release\inverseglod.exe --export-physics build/door-hulls.dae --convert-to-hulls 256 C:\path\to\closed-doorway.glb
```

The number is a maximum piece count. Exceeding it stops conversion. The output has one COLLADA node per hull. Add `--simplify-hull vertices max-deviation-m` to apply the bounded inward simplifier to each piece. C++ exposes `lodestar::decomposeCollisionHulls` and Rust exposes `lodestar::decompose_collision_hulls`. The C API returns an owned compound with a view into each hull.

The TestHall Firestorm package uses its mesh-derived triangle collider. For supported closed solids, the [Firestorm integration](integrations/firestorm/README.md) routes the uploader's Analyze button through Lodestar and sends the resulting pieces through Firestorm's convex-physics upload path.

### TestHall authored physics repair

The published TestHall fixture includes an authored collider with a false blockage near `Wing2Break`. After applying the archived physics file's X offset of −13.13098 m for measurement, a 1.69 m tall, 0.22 m radius capsule overlaps the authored collider by 2.87 cm along the test route. Replacing `Wing2Break_PHYS` with Lodestar's source triangles gives the same route 2.15 cm of clearance after reimport. The nearby wall still blocks passage, a second route stays open, and the measured floor height stays at 7.91017 m. [Second Life TestHall fixture](https://wiki.secondlife.com/wiki/Mesh_Import_test#Models_with_more_than_8_unique_faces_can_be_imported)

```powershell
$corpus = Join-Path $env:TEMP 'lodestar-corpus'
py -B tools/fetch_benchmark_assets.py $corpus
py -B tools/check_testhall_physics.py $corpus build/testhall-physics-litmus `
    --library build/Release/lodestar_c.dll --lodestar build/Release/inverseglod.exe
```

The check writes `testhall-repaired-physics.dae` and `report.json`. It replaces 272 triangles with 1,104 and retains the original 686 triangles across the other 11 collider parts. These are offline geometric checks on a static mesh. Second Life avatar movement and physics cost require in-world verification.

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

### TestHall diffuse bake proof

`tools/bake_planar_patch.py` exports a rectangular patch as two triangles with a new UV map and an embedded PNG. The TestHall proof uses four scene-space faces from `Wing2Break`, selected by their zero-based imported indices. `DirtyConcrete-material` binds `CrackedConcrete__DIFF.tga` through UV set 0. The texture archive contains the test images and no separate asset license. The proof uses local copies. [Second Life Wiki texture test and download](https://wiki.secondlife.com/wiki/Mesh_Import_test#Textures)

Use the corpus directory from the benchmark above and a Release build of the C library.

```powershell
py -m pip install numpy Pillow
$textureZip = Join-Path $corpus 'TestHall_textures.zip'
Invoke-WebRequest -Uri 'https://github.com/user-attachments/files/18456930/TestHall_textures.zip' -OutFile $textureZip
$textures = Join-Path $corpus 'TestHall_textures'
Expand-Archive -LiteralPath $textureZip -DestinationPath $textures -Force
$proof = 'build/testhall-texture-bake'
New-Item -ItemType Directory -Path $proof -Force | Out-Null
py tools/bake_planar_patch.py (Join-Path $corpus 'TestHall_MESH.dae') (Join-Path $textures 'CrackedConcrete__DIFF.tga') `
    (Join-Path $proof 'testhall-patch.glb') --library build/Release/lodestar_c.dll `
    --faces 5052,5053,5060,5061 --resolution 1024 --preview (Join-Path $proof 'comparison.png') `
    --report (Join-Path $proof 'report.json')
```

The exporter uses bilinear sampling with repeated source UVs and clamps the baked texture at its edges. The unlit GLB approximates the patch's diffuse appearance under this sampling model. The report compares 65,536 offset samples, separate from the bake grid. Errors are absolute differences in 8-bit sRGB channel values. The preview shows the sampled source, the bake, and a magnified error image.

| Atlas | GLB bytes | Mean error | 95th percentile | Maximum error |
| --- | ---: | ---: | ---: | ---: |
| 512 × 512 | 224,712 | 4.460 | 12.669 | 49.394 |
| 1,024 × 1,024 | 779,572 | 0.681 | 2.777 | 25.719 |

Both resolutions reduce the patch from four triangles to two. In a separate geometry comparison, the reimported GLB had zero sampled surface displacement and zero silhouette error across 13 views at 384 × 384. The test covers the selected patch's opaque diffuse color. Full-scene materials and shader lighting are outside its scope.

### Wing2Break multi-patch atlas

`tools/bake_section_atlas.py` processes the entire named `Wing2Break` node. It resolves the seven opaque diffuse textures from the COLLADA material bindings. It finds rectangular patches, retains boundary points shared with neighboring geometry, and packs the accepted patches into one texture with two-texel gutters. Triangles outside those patches keep their original geometry, normals, and texture mapping.

Use the corpus and extracted texture paths from the preceding example.

```powershell
py -B tools/bake_section_atlas.py (Join-Path $corpus 'TestHall_MESH.dae') $textures `
    build/testhall-section-atlas --library build/Release/lodestar_c.dll --node Wing2Break `
    --texel-scale 0.75 --max-tile 512 --max-atlas 2048 --supersamples 2 --view-size 512
```

The run found 26 candidate patches and baked 11. It replaced 44 triangles with 26 and carried forward the other 1,060 source triangles. Seven patches went from four triangles to two. Four kept a shared boundary point and went from four triangles to three. The output directory contains `source.glb`, `lod.glb`, `atlas.png`, `comparison.png`, and `report.json`.

| Measure | Source GLB | Atlas GLB |
| --- | ---: | ---: |
| Triangles | 1,104 | 1,086 |
| Indexed geometry payload | 78,240 bytes | 77,620 bytes |
| Referenced diffuse images | 7 | 8 |
| Estimated base-level RGBA8 texture storage | 7,340,032 bytes | 8,989,168 bytes |
| GLB size | 3,687,060 bytes | 3,962,144 bytes |

All seven original images remain in use on untouched triangles. The added 516 × 799 atlas increases the GLB size and estimated texture storage despite the triangle reduction. The RGBA8 estimate counts each referenced image once at four bytes per texel and covers base levels only. GLB sizes include compressed PNGs, geometry, and container data.

The comparison uses six shared orthographic cameras at 512 × 512, the same directional Lambert light and ambient term, and bilinear texture sampling in linear light. Under this model, both meshes cover the same 342,765 pixels. Whole-node mean, 95th-percentile, and maximum errors are 0.136, 1, and 41 in 8-bit sRGB channel units. On the 27,119 visible baked-patch pixels the same errors are 1.719, 7, and 41. The other compared pixels match exactly. This checks static diffuse color under the common software lighting model. The COLLADA Phong specular response and external renderer lighting are outside the comparison.

After GLB reimport, a separate comparison of 16,384 surface samples per direction found a maximum sampled displacement of 2.9 × 10⁻¹⁶ relative to the source extent. Silhouette error was zero across 13 views at 384 × 384.

For a lower-memory configuration, use `--texel-scale 0.25 --max-tile 256` and a separate output directory. On the same section, this produced a 260 × 291 atlas, a 3,737,636-byte GLB, and an estimated 7,642,672 bytes of base-level RGBA8 textures. The triangle count and six measured silhouettes were identical to the default run. The edited regions had a mean error of 4.18 and a 95th-percentile error of 14 in 8-bit sRGB channel units. On this run, atlas baking and packing took about 90 ms at the lower resolution and 261 ms at the default resolution.

### Wing2Break across UV seams

Add `--cross-uv` to group adjacent triangles with compatible planes and shading normals into simple polygons. The baker samples the original diffuse texture for each source triangle, including when one region uses several textures. It retriangulates each accepted polygon and removes exactly straight boundary points when every incident region can update that point together. Points referenced outside the section stay pinned. The pass retains regions with unsupported boundaries, including holes.

```powershell
py -B tools/bake_section_atlas.py (Join-Path $corpus 'TestHall_MESH.dae') $textures `
    build/testhall-cross-uv --library build/Release/lodestar_c.dll --node Wing2Break --cross-uv
```

The pass accepted 36 of 54 candidate regions, combined 50 internal UV seams, and reduced 1,104 triangles to 1,010 (8.51%). Both atlas sizes retain all 912 triangles outside the baked regions. All six comparison silhouettes matched, and the separate geometry benchmark measured zero silhouette error across its 13 views. At 16,384 surface samples per direction, the maximum sampled displacement was 2.9 × 10⁻¹⁶ relative to the source extent.

| Measure | Source | Default atlas | Compact atlas |
| --- | ---: | ---: | ---: |
| Estimated base-level RGBA8 texture storage | 7.00 MiB | 17.01 MiB | 8.84 MiB |
| Mean RGB error on edited pixels, out of 255 | — | 2.02 | 4.11 |
| 95th-percentile RGB error on edited pixels, out of 255 | — | 8 | 14 |

Use `--texel-scale 0.25 --max-tile 256` for the compact atlas. Both configurations keep the seven source images in use. The color comparison uses the same fixed diffuse lighting as the rectangle experiment.

### Wing2Break triangle targets

Run the triangle budgets without loading images, then check the exported geometry against the original node and test both openings and the floor in the surrounding TestHall scene.

```powershell
$source = Join-Path $corpus 'TestHall_MESH.dae'
$library = 'build/Release/lodestar_c.dll'
$trial = 'build/wing2break-trials/geometry-993'
py -B tools/bake_section_atlas.py $source $textures $trial --library $library --cross-uv `
    --geometry-only --target-triangles 993 --max-boundary-deviation 1e-7
py -B tools/check_wing2break_geometry.py $source $trial --library $library `
    --benchmark build/Release/lodestar_benchmark.exe
```

The first profile produces 993 triangles. Use `--target-triangles 883 --max-boundary-deviation 5e-5 --normal-tolerance 5e-4 --plane-relative-tolerance 1e-4` for the wider profile, which produces 882. The two candidates preserve all 13 measured silhouettes, the measured clearances through both openings, the wall contact, and the floor height. Maximum sampled surface errors are 4.19 × 10⁻⁹ and 6.81 × 10⁻⁶ relative to the source extent. The wider profile moved a boundary by at most 43.2 µm in the planner.

Bake the 993 profile into a separate directory for the textured comparison.

```powershell
py -B tools/bake_section_atlas.py $source $textures build/testhall-target-993-replace --library $library `
    --cross-uv --target-triangles 993 --max-boundary-deviation 1e-7 `
    --crop-repeats --texel-scale 0.10 --max-tile 256 --retire-material 0
```

The original image bound to material 0 is replaced by the atlas. In the wider profile, geometry edits already include all of that material's faces. The `--crop-repeats` option also writes `source-cropped.glb`, so both variants can be measured against the same base-level texture crop. Cropping preserves the bilinear taps used by retained faces. Another option, `--reuse-texels`, places every bake tile inside unused texels. The 993 profile fits all 37 tiles with `--crop-repeats --reuse-texels --texel-scale 0.10 --max-tile 128`.

```powershell
py -m pip install --target build/render-benchmark-wgpu --only-binary=:all: wgpu==0.33.0
$env:PYTHONPATH = (Resolve-Path 'build/render-benchmark-wgpu').Path
py -B tools/benchmark_textured_render.py build/testhall-target-993-replace/source-cropped.glb `
    build/testhall-target-993-replace/lod.glb --resolution 512 `
    --output build/testhall-target-993-replace/render-cropped-512.json
```

On an NVIDIA RTX 5070 Ti, the 993-triangle replacement lowered paired GPU time by 4.1% at 512 × 512 and 6.1% at 1024 × 1024 against the cropped source. The 882-triangle replacement lowered it by 11.6% and 13.1%. Estimated RGBA8 base texture storage went from 5.61 MiB to 5.46 and 5.10 MiB. Mean color error on edited pixels was 7.19 and 7.82 out of 255 under the shared diffuse light. The 993-triangle reuse result held storage at 5.61 MiB and lowered edited-pixel mean error to 5.53. Its GPU timing difference was within the uncertainty against the cropped source. These GPU timestamps cover prerecorded offscreen clears and draws. Second Life render performance remains unmeasured.

Applying the wider profile to each supported TestHall section reduced the whole asset from 8,602 to 7,336 triangles, 2,567 above the authored medium LOD. All 13 geometry silhouettes matched. The four gates stayed at 4,560 triangles combined, compared with 2,265 in the authored medium. The remaining unsupported `Wing2` import retained its 218 triangles.

### Reproduce the TestHall CPU baseline

Build Release as above and fetch the fixture files with `tools/fetch_benchmark_assets.py`. Install the CPU tool dependencies in a local environment. The regression runner uses the native tools without opening a window or timing a GPU.

```powershell
py -m venv build/cpu-python
$python = 'build/cpu-python/Scripts/python.exe'
& $python -m pip install -r tools/requirements-cpu.txt
$native = 'build/Release'
& $python -B tools/check_testhall_regressions.py $corpus build/testhall-regressions `
    --library "$native/lodestar_c.dll" --lodestar "$native/inverseglod.exe" `
    --benchmark "$native/lodestar_benchmark.exe"
```

This checks the 993- and 882-triangle Wing2Break variants, the authored physics repair, and the 7,336-triangle full-scene result against the 4,769-triangle reference. Add `--textures $textures` to check the Wing2Break diffuse bakes. The full-scene command is `tools/check_testhall_full.py`. The regression exits with an error on a baseline mismatch and writes `scorecard.json` on success.

### TestHall nonplanar gate trial

The gate pass proposes meshoptimizer reductions on geometric components. It retains each component's manifold topology, boundary-loop count, and pinned contacts. The CPU check compares reimported geometry with the aligned authored medium LOD before baking. A candidate must match or improve the authored surface and silhouette measurements and leave source-clear pixels with a three-pixel margin open in the gate's front projection. The bake projects diffuse color from the same component and checks mean and 95th-percentile color error against the authored reference.

```powershell
& $python -B tools/check_testhall_gates.py $corpus build/testhall-gates `
    --library "$native/lodestar_c.dll" --benchmark "$native/lodestar_benchmark.exe" `
    --lodestar "$native/inverseglod.exe" --holdout (Join-Path $corpus 'BarramundiFish.glb') `
    --textures $textures --atlas-only --max-tile 12 --supersamples 2 --screen-components `
    --relative-error 0.002 0.005 0.01 0.025
```

Use `--gate gate` for a single-gate pilot. The full run writes the part reports, a combined textured gate comparison, a geometry-only full-scene comparison, and a source-derived collider. The isolated GPU benchmark can use the exported `source.glb`, `lod.glb` and `authored.glb` for a gate or `combined-gates`, using the same `tools/benchmark_textured_render.py` command shown above.

| Trial | Source triangles | Candidate triangles | Authored target |
| --- | ---: | ---: | ---: |
| `gate` | 1,140 | 651 | 651 |
| `gate1` | 1,140 | 686 | 538 |
| `gate2` | 1,140 | 686 | 538 |
| `gate3` | 1,140 | 656 | 538 |
| Full TestHall | 8,602 | 5,455 | 4,769 |

All four gate candidates passed the measured surface, silhouette, opening, and diffuse color checks. The full-scene geometry comparison had a worst silhouette IoU error of 0.105%, compared with 0.645% for the authored LOD. Barramundi Fish went from 3,864 to 1,924 triangles in the geometry holdout, with a worst silhouette IoU error of 0.675%. Its textured PBR appearance remains unmeasured.

Combined gate mean RGB error was 4.07 of 255, compared with 8.86 for the authored gates under the same diffuse light. The original gates share one 1.00 MiB image. Their four baked images total 2.26 MiB in base-level RGBA8. At 512 × 512 the combined candidate used 1.6% less GPU time than the source and 11.6% more than the authored LOD in matched offscreen timing on an RTX 5070 Ti.

A single authored mesh supplies the visual source and its collider. The viewer builds the collider from the original triangles and keeps it fixed as visual LODs change. The CPU collision fixture checks doorway clearance, floor support, and tread and riser contacts on a four-step staircase. The gate trial also checks that the exported source collider round-trips with the original positions and winding. Avatar step handling and Second Life traversal still require runtime validation.

### Textured TestHall at 4,769 triangles

```powershell
& $python -B tools/check_testhall_target.py $corpus $textures build/testhall-4769-qualified `
    --library "$native/lodestar_c.dll" --benchmark "$native/lodestar_benchmark.exe" `
    --lodestar "$native/inverseglod.exe"
```

The target runner preserves the four qualified gate results, the original Wing3 façade and textures, and the Hall1 planar LOD. It uses bounded nonplanar reductions in other sections and keeps selected original components to obtain exactly 4,769 triangles. The bake reuses original UVs when samples across a face agree, and puts incompatible texture mappings into atlases. The sharper profile uses 64-pixel caps for nonplanar tiles and 256-pixel caps for planar charts.

At 512 × 512 in the matched diffuse views, the target has mean RGB error of 1.99 out of 255 and a 95th-percentile error of 12. The aligned authored medium LOD has errors of 4.03 and 24. Estimated base-level RGBA8 storage is 23.92 MiB for this profile and 15.00 MiB for the source. `comparison.png` and `authored-comparison.png` show both comparisons with the same framing.

The runner exports `lod.glb`, `lod.gltf`, and `lod.dae`, reimports their textures and geometry, and generates collision from each mesh. Keep the adjacent `.bin` with `lod.gltf` and the `lod-textures` directory with `lod.dae`. The images use PNG. This workflow covers static opaque diffuse meshes with UV0 and matching repeat or clamp modes on both texture axes.

The collision checks replay supported capsule walks through TestHall's doorway and arch routes. The full-resolution `source-collision.dae` derives from the original mesh. A four-tread render mesh supplies the separate stair fixture. Its generated collider supports three 0.17 m rises, and the CPU tests reject an excessive riser, a low lintel, a missing tread, and a gap. The runner records the geometry, appearance, and traversal checks in `scorecard.json`.

#### Second Life upload target

Start with an authored glTF or COLLADA mesh. Generate a matched High, Medium, Low, and Lowest chain and derive collision automatically from the source. Keep object names, transforms, and material slots stable across the levels. The reference material describes `_LOD2`, `_LOD1`, `_LOD0`, and `_PHYS` naming conventions. Use those conventions as a starting point. Determine compatibility by testing name matching, material bindings, and physics handling in the target Firestorm release. [Second Life upload background](https://community.secondlife.com/knowledgebase/english/uploading-a-mesh-model-r274/) · [Reference mesh import tests](https://wiki.secondlife.com/wiki/Mesh_Import_test#Name_based_matching)

Build the named TestHall package around the qualified 4,769-triangle Medium with the same corpus, textures, and native tools.

```powershell
& $python -B tools/build_testhall_firestorm.py $corpus $textures build/testhall-firestorm-qualified `
    --library "$native/lodestar_c.dll" --benchmark "$native/lodestar_benchmark.exe" `
    --lodestar "$native/inverseglod.exe"
```

The output contains `TestHall`, `TestHall_LOD2`, `TestHall_LOD1`, `TestHall_LOD0`, and `TestHall_PHYS` in COLLADA, GLB, and glTF. Keep the shared `textures` directory with the COLLADA files and the hashed `.bin` companions with glTF. Each level carries the same named parts. The visual levels use the material slots and images bound by High. Lower levels retain a previous part when its geometry or texture projection exceeds the measured limits. `scorecard.json` records the actual counts, surface and silhouette measurements, diffuse errors, texture memory, supported walks, and per-part triangle checks after modeled viewer position quantization.

TestHall keeps Medium's Hall1 and Wing3 in both lower tiers. Measured texture errors at the intended viewing sizes exceed the authored Medium's errors when both parts use the more aggressive reductions.

The native source collider is retained in `measurements/source-collision.dae`. Firestorm rejects some geometrically valid but very thin physics triangles. The upload collider repairs these with local diagonal flips and short edge collapses, then certifies both directions of surface deviation within 3 cm including position packing. The scorecard lists the repairs and any residual faces removed. Named part bounds, passage clearance, floor support, and wall contact are checked separately.

In Firestorm 7.2.5.81788, open **Upload Model** with `TestHall.dae` or `TestHall.glb`. Select **Load from file** for each lower level and **From file** for the matching `TestHall_PHYS`, keeping the triangle collider as supplied. Enable the texture preview and review all four visual levels and the physics overlay. In a test region where you can rez meshes, set **Physics Shape Type** to **Prim** to exercise triangle collision. The local check models the upload format and probes collision. First-load timing, viewer download and physics costs, JPEG 2000 texture quality, and avatar traversal require the signed viewer and a test-region session. [Pinned Firestorm uploader source](https://github.com/FirestormViewer/phoenix-firestorm/blob/4ac201fdc56127de27de34567286885f4e75d194/indra/newview/llmodelpreview.cpp)

The signed Firestorm 7.2.5.81788 preview loaded all five COLLADA files and showed green checks for all visual tiers. The preview counts of 4,788, 4,071, and 3,990 include 19, 22, and 23 empty material slots, respectively. Firestorm serializes those slots as `NoGeometry`. The Physics tab showed 8,165 triangles and `Hulls: N/A` with the mesh-derived collider loaded from file. Avatar traversal and streaming still require a rezzed test. [Firestorm mesh serialization](https://github.com/FirestormViewer/phoenix-firestorm/blob/4ac201fdc56127de27de34567286885f4e75d194/indra/llprimitive/llmodel.cpp#L913)

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

### Calculate LODs from a continuous surface

The surface calculus API accepts an explicit cubic triangular Bézier patch. It evaluates the analytic first and second derivatives, forms the first and second fundamental forms, and obtains the principal curvatures. Adaptive Gaussian quadrature evaluates the surface integrals

\[
A=\iint_D\|S_u\times S_v\|\,du\,dv,\qquad
K=\iint_D\max(|k_1|,|k_2|)\|S_u\times S_v\|\,du\,dv.
\]

Given world units per pixel and an allowed pixel error, the world-space tolerance is their product, \(\varepsilon\). A local equilateral triangle interpolation model proposes a continuous triangle budget \(2K/(3\sqrt{3}\varepsilon)\). Its interior vertex density is half the triangle density. Boundaries and topology affect the actual vertex count. The numerical report includes quadrature error estimates and convergence status.

`tessellateSurface` uses that budget to propose a subdivision resolution. It then restricts the Bézier controls to every generated triangle and checks their deviation from linear interpolation. The Bernstein convex-hull property bounds the corresponding surface-point error in real arithmetic. The implementation adds a double-precision arithmetic allowance and the measured float32 position rounding. It refines until the checked bound meets the tolerance or fails at the depth limit. The arithmetic allowance uses standard floating-point error scaling and has no interval-arithmetic certificate. Normals follow the analytic derivatives.

The native executable can calculate and generate a patch LOD without opening a window. The example contains ten XYZ control points in the order documented in [surface_calculus.hpp](include/lodestar/surface_calculus.hpp).

```powershell
.\build\Release\inverseglod.exe --surface-lod tests/assets/parabolic-cylinder.patch --pixel-scale 0.01 --pixel-error 0.5
```

For this patch, changing world units per pixel gives the following actual generated meshes at a maximum error of half a pixel.

| World units per pixel | Vertices | Triangles | Checked deviation bound |
| --- | ---: | ---: | ---: |
| 0.01 | 45 | 64 | 0.003907 |
| 0.04 | 15 | 16 | 0.015626 |
| 0.16 | 6 | 4 | 0.062501 |
| 0.64 | 3 | 1 | 0.250001 |

In C++, call `differentiateSurface(patch, u, v)`, `calculateSurfaceLod(patches, worldUnitsPerPixel, maximumPixelError)`, and `tessellateSurface(patch, worldUnitsPerPixel, maximumPixelError)`. The tessellation owns an ordinary indexed `Mesh`. Rust exposes `BezierTriangle`, `differentiate_surface`, `calculate_surface_lod` with `SurfaceLodOptions`, and `tessellate_surface`. The C ABI exposes the same operations with owned mesh output and a deviation report.

The direct patch tessellator accepts an authoritative regular patch and generates one untextured patch at a time. The fitting workflow below uses continuous surfaces to plan legacy mesh LODs and checks their attributes against High. The curvature density is a planning approximation. The direct tessellator's final counts come from its checked subdivision. Direct patch stitching and automatic fitting inside Firestorm's visual LOD generator require further integration.

The tests compare a plane and a parabolic cylinder with closed-form area and curvature integrals, verify analytic derivatives, check scale and translation behavior, and exercise actual vertex reduction, normals, float32 interpolation bounds, convergence failure, and C/Rust ownership.

### Fit real meshes and generate all visual LODs

`tools/fit_mesh_lods.py` loads static opaque diffuse meshes with UV0 from COLLADA, glTF, or GLB. It partitions position/UV-identical connections by material and named part, retaining discontinuous normals as per-corner reference fields. Each supported manifold UV chart receives a fitted polynomial description. Actual triangle-area and interior-edge curvature integrals supply the sampling budgets, including geometric creases that a smooth fit can miss. These are planning budgets, not acceptance certificates. The scorecard stores the coefficients, UV-domain transform, source faces and measured correspondence bounds. The original High mesh remains authoritative.

```powershell
py -B tools/fit_mesh_lods.py model.glb build/fitted-lods --library build/Release/lodestar_c.dll
# COLLADA also needs its diffuse image directory.
py -B tools/fit_mesh_lods.py model.dae build/fitted-lods --library build/Release/lodestar_c.dll --textures path/to/textures
```

The default view sizes are 1,024, 512, 192, and 96 pixels at half a pixel of geometric error. `--pixels` accepts four descending view sizes. `--pixel-error` and `--normal-angle` control the geometric and shading limits. The default normal limit is one degree. World units per pixel use the source scene's largest extent divided by the requested view size, an orthographic object-size convention. Perspective camera integration requires a caller-supplied conservative pixel scale.

High's budget is calculated and its source triangles are retained. Medium, Low, and Lowest use integrated density to propose automatic counts. Interior edits use original vertices; coordinated seam edits may remove exactly straight UV knots only when every incident chart accepts the same geometric edge. UV domains, material boundaries, protected contacts and named-model extents remain. A complete overlay of source and candidate UV triangulations bounds surface displacement and shading-normal deviation, including sided source normals. Unsupported charts retain their source faces. Bounds use double arithmetic with roundoff allowances and have no interval-arithmetic certificate.

An imported doubly curved checker-textured panel generated 1,152, 518, 272, and 178 triangles across the four levels with Lodestar's cavity remesher. The three reduced levels had maximum correspondence bounds of 0.000868, 0.002604, and 0.005035 source units, with shading-normal bounds below 0.34 degrees. The 128-pixel comparisons had identical silhouette masks. Fitting, exports, round trips, rendering, and physics checks took approximately 9.5 seconds on the development machine.

The previous indexed-chart fitter retained all 8,602 TestHall triangles. Position/UV connectivity now finds 1,051 charts and fits 1,047; four non-disk charts retain their source faces. Coordinated boundary coarsening and complete per-corner shading checks produce 8,602 / 8,602 / 8,553 / 8,550 triangles. UV chart domains and material boundaries remain protected. This is only a 0.6% visual reduction, not a meaningful solution to the authored-budget goal.

The package contains named `Lodestar`, `Lodestar_LOD2`, `Lodestar_LOD1`, and `Lodestar_LOD0` files in GLB, glTF, and COLLADA. Reimport checks verify named-part counts, images and sampler modes. Input and output hashes accompany decisions in `scorecard.json`. Native column-pivoted QR fits chart descriptions. The analytic Bézier differentiation/integration APIs remain available in C++/C/Rust; the current mesh preparation worker computes triangle-area and edge-curvature integrals in Python. Python also handles chart partitioning, full UV correspondence, cavity and seam edits, position-only collision simplification, selection and export. These operations use no external simplifier and still need native ports. The Firestorm bridge imports the calculated files and analyzes the selected collision companion. See [runtime setup](integrations/firestorm/README.md).

With the Lodestar cavity remesher, the doubly curved checker control generated 1,152, 518, 272 and 178 triangles. Its High GLB remained byte-for-byte identical and native Firestorm packing produced no zero-area visual triangles. A closed curved solid generated 1,152, 408, 312 and 280 triangles, with a 72-triangle refined collider. Native source-space hull conversion produced one exact 38-vertex hull for that solid. These are separate controls, with retained UVs and material boundaries. Chart orchestration and cavity retriangulation currently run in Python and require a native port for complete C++/Rust workflow parity.

C++ exposes `fitSurface(parameters, positions)` and `surfacePatch(model, triangularDomain)` in `surface_calculus.hpp`. Rust exposes `fit_surface` and `PolynomialSurface::patch`. The C ABI equivalents are `lodestar_fit_surface` and `lodestar_polynomial_surface_patch`. These return fitted coefficients and vertex residuals. Vertex residuals alone are insufficient to establish agreement with every High triangle, so the file-based workflow additionally bounds the complete chart correspondence before accepting generated geometry.

### Rebuild physics from High

The fitting runner also exports `Lodestar_source_PHYS.dae` from the authoritative High surface and repairs collision triangles using exact planar retriangulation, coplanar diagonal flips, and conforming strip refinement. Rational predicates check planarity, straight boundary knots, and added boundary samples. Strip refinement also subdivides incident faces along shared edges, including at real creases. Positive-area source geometry and original corners are retained. Short-edge collapse and residual-face omission are disabled in this workflow. Native convex conversion is probed for supported closed solids.

The normal `Lodestar_PHYS.dae` companion is written only when every triangle passes both pre-packing and modeled post-packing checks. Otherwise the runner writes `Lodestar_unresolved_PHYS.dae` for diagnosis, marks the scorecard `physics_blocked`, and exits with status 2. It refuses to leave an older normal companion in place during a blocked rerun. Float32 export displacement and U16 packing limits are distinct from ideal surface equality.

To compare an existing collider, add `--existing-physics old_PHYS.dae --benchmark build/Release/lodestar_benchmark.exe`. An optional `--existing-translation x y z` explicitly aligns an offset authored collider for comparison.

For TestHall, exact repair reduced 8,602 source triangles to 7,947 through 169 planar regions and six diagonal flips. Source-to-rebuild sampled surface displacement was at floating-point roundoff, with a maximum relative error of 1.36 × 10⁻¹⁶ and zero measured silhouette error. The original authored collider had a sampled maximum relative error of 0.0172 after explicit alignment. The rebuilt doorway route had the same 2.15 cm clearance as High, while the old collider overlapped the capsule by 2.87 cm. The solid wall and the separate open route retained High's collision behavior.

Firestorm's modeled triangle rejections decreased from 461 to 334. The scorecard lists unresolved triangles for each named model and records `all_parts_have_supported_representation` as false for this run. A completed generation report therefore requires separate uploader-readiness review. The finite U16 position grid also imposes a measurable precision floor. Exact surface preservation before packing does not guarantee a zero-error simulator collider. More general constrained remeshing or a supported convex representation is required for the remaining rejections. These files require signed-viewer and in-world validation.

Conforming strip refinement subsequently reduced the 334 remaining TestHall rejections to 198, with 8,575 collision triangles. Firestorm's native parser and asset write/read round trip confirmed 198 rejections at both stages and no mathematical zero-area triangles. This exact-surface collider remains blocked. It is distinct from the bounded collision representation below.

### Select simplified collision independently of visual attributes

Preparation now prefers the lowest generated visual level with a supported hull set. When none qualifies, Lodestar tries position-only edge quotients, starting with small fractions of that level's surface allowance. Shared model contacts are edited in one transaction. Material/UV/shading splits impose no collision constraints. Each named model retains its bounds, horizontal support heights and uploader checks. The vertex-to-representative correspondence bounds the entire original triangle stream, including faces mapped into surviving edges. The selected candidate's visual, simplification and float-export bounds are composed against High. Generic self-intersection certification remains incomplete.

The TestHall package in `build/testhall-selected-physics` contains a **4,913-triangle collider**, a 42.9% reduction from High, selected from Lowest. Closed-shell orientation and nonzero volume are protected; tetrahedral pieces cannot collapse into overlapping faces. Exact incident-edge bisection handles the remaining long-triangle uploader failures. Native Firestorm import and U16 write/read checks report **zero rejected and zero zero-area triangles**. Known supported walking routes, the blocking wall and all four guarded gate-aperture checks pass. The High GLB remains byte-identical to the previous reference. The surface displacement bound is approximately **0.10772 m before uploader packing**; this is a bounded collider, not an exact copy of High. Simulator testing remains outstanding.

With the revised mesh-integral and seam workflow, the closed curved control generates **1,152 / 214 / 106 / 74 visual triangles**. Lowest does not support current native hull conversion, so the selector uses Low, which produces one 38-vertex hull. It checks all generated levels instead of resetting collision to High.

```powershell
python -B tools/check_testhall_preparation.py build/testhall-selected-physics `
    --library build/Release/lodestar_c.dll --native-audit path/to/firestorm-native-audit.exe `
    --reference-high build/testhall-physics-refinement/Lodestar.glb
```

The Firestorm patch imports this selected companion before hull analysis. It no longer unconditionally derives collision from High. Broad material/attribute coverage and native C++/Rust parity for the worker's orchestration remain unfinished. TestHall's visual counts are still **8,602 / 8,602 / 8,553 / 8,550**; meaningful visual simplification remains unresolved.

### Coherent collision from open surfaces

Collision coherence applies to the assembled shape; each source element need not be watertight. `buildCoherentCollisionShape` first attempts exact solid conversion, then partitions the surface into closed convex pieces within a supplied surface allowance. It checks source facet half-spaces, axis-constant contacts, inward-shell cavity guards and bidirectional surface displacement. Planar patches receive minimal inward thickness derived from float precision and the U16 grid. Piece and work limits cause explicit failure. A bounded surface cover does not certify exact occupied volume or every opening.

```powershell
.\build\Release\inverseglod.exe --export-physics build/coherent-hulls.dae --coherent-hulls C:\path\to\model.glb
```

The CLI and Firestorm backend automatically try allowances of maximum extent / 384, / 192, / 96 and / 48. Firestorm restores metric distances through nonuniform model normalization, converts required thin instances, and retains valid triangle physics for other models. Its Log reports the additional hull bound against selected physics and the exact-solid flag.

TestHall's four thin gates produced **164 / 161 / 193 / 178 hulls** in the native Firestorm backend. Native packed-asset round trips retained positive volume in every piece and preserved the High payload. All four guarded opening checks and the known supported walking routes passed offline. The additional hull surface bound is about **0.02636 m**, on top of the selected collider's bound against High and subsequent packing error. The previous GUI build remained blocked by thin triangle physics; this new backend still needs GUI and simulator validation.

C++ exposes `buildConvexHull`, `buildCoherentCollisionShape` and `planMeshSize`; Rust exposes `build_convex_hull`, `build_coherent_collision_shape` and `plan_mesh_size`. The corresponding C ABI functions are `lodestar_build_convex_hull`, `lodestar_build_coherent_collision` and `lodestar_plan_mesh_size`. The coherent result distinguishes exact solid conversion from a bounded surface cover. A point-cloud envelope alone makes no source-conformity or opening-preservation claim.

### Size proposals below 0.01 m

```powershell
.\build\Release\inverseglod.exe --plan-size C:\path\to\small-model.glb
```

The native planner proposes container dimensions of at least 0.01 m per axis, with the source center and visible fractions. For a 2 × 4 × 6 mm object it proposes a 10 mm cube with fractions 0.2 / 0.4 / 0.6. The reversible map is `(source_position - center) / container_size`; the source surface is unchanged.

Automatic preparation now applies **physical container frames**, with a closed box collider and no added render geometry. It writes applied-frame details to `size-proposals.json` and embeds a `LODESTAR physical_container` extra in every visual and physics COLLADA model. The patched importer uses that common frame for normalization and instance placement. The visible surface retains its authored dimensions, UVs and materials; normals follow the inverse-transpose rule. Existing padding frames survive cleanup without recreating degenerate faces.

The calculation includes import scale: a smaller scale enlarges the source-space physical frame enough to keep the uploaded prim legal. Changing that control automatically reruns preparation when needed. Native checks reject invalid frame metadata, missing frames and packed visual degeneracy. Tiny solids, flat surfaces, translated and thin meshes, existing padding, mixed-size scenes, and reduced import scale pass native packed-asset checks. These COLLADA files require the patched Lodestar importer; ordinary importers ignore the extension. GUI and in-world validation are still pending.

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

Call `lodestar::buildCollisionMesh(mesh)` with an imported or caller-owned indexed triangle mesh. It returns owned positions and indices in the supplied coordinate space and reports skipped zero-area triangles.

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

Call `Asset::generate_collision()` on a loaded asset or one created with `Asset::from_mesh()`. Its `CollisionMesh` owns the collider and exposes read-only position and index slices. The C ABI equivalents are `lodestar_asset_generate_collision()` and `lodestar_collision_mesh_get_view()`.

The C ABI exposes in-memory mesh creation through `lodestar_asset_analyze_mesh()`, topology and budget inspection through `lodestar_asset_inspect_topology()` and `lodestar_asset_inspect_boundary_budget()`, and generated mesh IDs through `lodestar_mesh_get_provenance()`. Use `lodestar_asset_generate_seam()` to generate a seam mesh at an explicit triangle budget. [C header](include/lodestar/lodestar_c.h)

Run the regression tests with `ctest --test-dir build -C Release --output-on-failure`. The planar test reduces a textured 32-triangle grid to 14 triangles and checks preserved boundaries, holes, material seams, winding, and vertex attributes. A split fan reduces from four triangles to two with supplied authored vertex identities and keeps four when those identities are unknown. Seam tests cover coupled edits and protected attribute breaks. The suite also checks fixed-boundary certification, sphere eligibility, exact fallback, and benchmark metrics.

## Geometry modes and scope

Lodestar fits a sphere to the mesh vertices by least squares and reports the RMS and maximum radial residuals relative to the fitted radius. The analytic mode requires a valid fit, a maximum vertex residual under 1%, no UV coordinates, a consistently wound surface enclosing the fitted center, and a maximum inward deviation from the sphere of at most 10% of the radius across source triangle interiors and edges. It then tessellates the fitted sphere on the CPU at the requested quality.

Planar mode removes interior vertices from eligible coplanar triangle fans and retains their boundary segments. Edits require compatible winding, material IDs, and affine normal and UV interpolation within tolerance. Seam mode uses a separate exact check to shorten eligible boundary chains. Both modes preserve unsupported regions and can stay above the target. Imports with unrepresented vertex attributes such as extra UV components, colors, tangents, skin weights or morph targets keep their source triangles.

Topology diagnostics group edges by exact endpoint positions and report existing index sharing, boundaries, split edges, attribute seams, winding conflicts and ambiguities. A split can have matching stored attributes even when its authored connection is unknown. Assimp imports keep these unproven splits intact, which preserves intentional slits or coincident surfaces. Planar recovery uses caller-provided authored vertex identities on in-memory meshes.

Assimp imports positions, normals, and UV0.xy into the mesh buffers. The sphere gate checks for UVs in all channels. Imported triangle material IDs are local to each source asset. The viewer uses a single material and normal-based lighting. The meshoptimizer comparison can change the shape. General inverse-GLOD reconstruction, analytic patches with certified error bounds, a persistent LOD cache, adaptive screen-space tessellation, and Second Life viewer integration remain research goals.

## Licence

Lodestar is licensed under the MIT License (`SPDX-License-Identifier: MIT`). See [LICENSE](LICENSE). Third-party dependencies keep their own licences, and test assets keep the terms described above.
