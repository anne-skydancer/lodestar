# Lodestar preparation in Firestorm

The Windows x64 patch targets Firestorm 7.2.5.81788 at commit
`4ac201fdc56127de27de34567286885f4e75d194`. It routes **Physics → Analyze**
through Lodestar. Firestorm previews and serializes the resulting pieces using
its native convex physics format.

The development patch also makes Lodestar the automatic preparation path for
new High files. It calculates all three lower LODs, performs conforming collision
refinement, imports the calculated files, and attempts hull generation from
the selected collision companion. Lower LOD source controls display **Lodestar Auto**. **Prepare with
Lodestar** reruns the workflow. The stock first-load meshoptimizer generation
job is suppressed. Explicit legacy generator options remain available for
comparison.

This path uses a local Python worker around Lodestar's C++ fitting and calculus
API. Install the Python dependencies in `tools/requirements-cpu.txt`, place
`lodestar_c.dll` beside the viewer, and configure the runtime:

```powershell
python tools/configure_firestorm.py path/to/viewer-runtime
```

The configuration selects `pythonw.exe` on Windows to avoid a console window.
The worker is launched with structured arguments, writes an atomic LLSD result
and a hashed scorecard in a unique temporary folder, and performs no upload.
The viewer imports the generated visual levels and attempts hull conversion.
If hull conversion is unsupported, it validates the refined triangle companion.
Readiness remains blocked during preparation and on unresolved geometry or
physics failures. Missing worker dependencies produce a visible failure.

The native worker launcher, LLSD handoff and packed visual imports have been
replayed successfully. The selected 4,913-triangle TestHall collider passed
native triangle checks in the GUI but remained blocked by four thin instances
requiring hulls. The subsequent coherent-hull backend passes native asset
write/read checks for those instances. Its GUI sequence and simulator behavior
still require validation.

Build the Lodestar C ABI and run the bridge regression.

```powershell
cmake -S . -B build/firestorm-library -DLODESTAR_BUILD_VIEWER=OFF
cmake --build build/firestorm-library --config Release --target lodestar_c lodestar_firestorm_decomp_test
ctest --test-dir build/firestorm-library -C Release -R '^lodestar_firestorm_decomp$' --output-on-failure
```

Apply `firestorm-7.2.5.81788.patch` from the Firestorm checkout root with
`git apply`. In the configured Firestorm build, set the CMake cache variable
`LODESTAR_SOURCE_DIR` to the absolute path of the Lodestar checkout. Rebuild
the `firestorm-bin` target and include the 64-bit `lodestar_c.dll` beside the
viewer executable in the runtime folder.

Automatic preparation prefers the lowest generated level with supported
hulls, then tries bounded position-only polygonal candidates without UV or
material constraints. It imports the selected `_PHYS` companion through
**From file** before Analyze. It never resets that selection to High. Changing
the displayed visual level leaves collision fixed. The Log reports the basis
and surface displacement bound against High, before uploader packing.
Select **Analyze** to convert the selected physics
source to convex pieces. Lodestar determines the vertices required by each
hull from its geometry and removes redundant coplanar samples. The uploader
shows **Vertices per hull: Automatic**. The **Log** tab reports the hull count,
total vertices, and largest hull, or the reason conversion failed.
Analysis preserves the existing physics if any model fails to convert.
Failures now appear in a modal alert and in the Physics tab's persistent result
text, as well as the Log. A failed conversion explicitly reports that no
replacement collider was applied. This prevents an unsupported model from
appearing to have completed analysis successfully.

The Physics tab's Analyze button performs convex conversion. The automatic
High preparation and **Prepare with Lodestar** operation run the complete
preparation sequence. The [SL upload contract](../../docs/sl-upload-contract.md)
records the remaining supported-input, library-parity and validation limits.

The hull control sets a maximum number of pieces. Each hull must fit the
upload format's 256-vertex limit. Lodestar splits patches as necessary and
fails if its work, precision or piece limits cannot be met. The bridge regression
checks automatic selection of eight vertices for boxes and six for an
octahedron. It also exercises explicit library limits, preserves input positions
and indices after successful and failed calls, and retries after failure.
Visual shading and the full uploader control sequence require separate viewer
checks.

The Physics tab reports the 0.5 m warning for thin instances using triangle
physics. A successful analysis replaces that triangle collider with hulls and
clears the warning. The check uses the dimensions at the selected import scale.

Lodestar first attempts exact closed-solid conversion, then a bounded coherent
surface cover. Input elements may be open. Output pieces are closed convex
volumes, with minimal inward thickness for planar patches. Source facet and
contact-plane checks constrain completion; inward shells guard cavities.
This cover has surface-error semantics and does not certify exact occupied
volume. Unsupported cases retain the prior collider and report failure.

The backend restores metre distances through the model's nonuniform
normalization and import scale. It automatically tries maximum source extent
divided by 384, 192, 96 and 48, selecting the first success. The Log reports the
additional surface bound against the selected physics source and whether the
conversion was an exact solid. Combine this bound with the worker's bound
against High; uploader packing adds further error.

Automatic preparation converts the required thin models and retains valid
triangle physics for the others. On TestHall, the native backend produced
164 / 161 / 193 / 178 hulls for gate1 / gate2 / gate3 / gate. Native U16
serialization preserved positive volume in every decoded piece and left
the corresponding High payload unchanged. Offline guarded openings and
supported walks passed. These are not GUI or simulator results.

For dimensions below 0.01 m at the selected import scale, preparation now applies
a **physical container frame** and closed box collider. No render geometry is
added. Every visual LOD keeps its authored positions, UVs and materials. A
`LODESTAR` geometry extra records the common physical frame; the native importer
normalizes the model into it and uses its matching instance transform. The
packed asset carries ordinary SL position domains and convex physics. The
runtime surface check rejects collapsed packed triangles before readiness.

`size-proposals.json` records applied frames, source and scaled dimensions, and
visible fractions. Existing padding frames are recognized after degenerate
source faces are removed. Changing import scale recalculates required frames
from the source; the control is locked during preparation. Invalid or missing
metadata blocks import. Stock COLLADA importers ignore this extension, so use
the patched viewer for these files. Tiny solids, flat surfaces, translated and
thin meshes, existing padding, mixed-size scenes, and reduced import scale pass
native import and packed-asset checks. GUI and simulator checks are pending.

The native regression source is `tests/physical_container_audit.cpp`. Build it
against the same Firestorm headers and native libraries as the viewer, with
`lodestar_c.dll` beside the executable. It accepts a generated package folder,
a JSON result path, an optional import scale, and an optional `true` to exercise
the alternate normal compatibility setting. The preparation and validation
driver uses a fresh output folder:

```powershell
python -B tools/check_physical_containers.py --library build/Release/lodestar_c.dll `
    --native-audit path/to/physical-container-audit.exe --output build/container-check
```

It prepares seven controls, checks 32 native visual-LOD round trips, exercises
both normal compatibility settings for framed models, and confirms rejection
of invalid versions, inverted bounds, excluded source geometry and duplicate
frame metadata. The result records measured displacement and explicitly leaves
GUI and simulator validation pending.
