# Lodestar

Lodestar is a C++20 Vulkan prototype for exploring geometric levels of detail (LOD) on Second Life-style meshes. It compares imported triangles with meshoptimizer, analytic sphere, planar, and exact seam representations. You can compare silhouettes and triangle counts at different quality levels. The original mesh is the default for every asset.

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

The export flattens the imported scene into one static mesh. For Second Life's per-object physics workflow, export the corresponding object separately, select the file in the uploader's Physics tab, and set the rezzed object's **Physics Shape Type** to **Prim** when using that custom shape. Check openings and walkable surfaces at their final in-world scale. [Second Life mesh upload and physics settings](https://community.secondlife.com/knowledgebase/english/uploading-a-mesh-model-r274/)

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

Start with an authored glTF or COLLADA mesh. Generate a matched High, Medium, Low, and Lowest chain and derive collision automatically from the source. Each exported level needs stable object names, transforms, and material slots. Second Life's documented matching uses `_LOD2`, `_LOD1`, `_LOD0`, and `_PHYS` for the accompanying models. Validate those bindings in the target Firestorm uploader. [Second Life upload guide](https://community.secondlife.com/knowledgebase/english/uploading-a-mesh-model-r274/) · [Mesh import tests](https://wiki.secondlife.com/wiki/Mesh_Import_test#Name_based_matching)

The current 4,769-triangle export is a qualified medium-LOD candidate for local evaluation. It combines TestHall's parts into one visual mesh. An uploader package still needs matching per-object levels and material bindings. Its acceptance test should cover the first recognizable shape during loading, the silhouette and visible detail around LOD transitions, the viewer's reported download cost, and walking through doors and up stairs with the derived physics.

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
