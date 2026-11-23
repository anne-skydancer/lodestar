# SL upload preparation

Lodestar must read the creator's mesh and derive its representations from that
mesh. The imported High surface is the authoritative reference. Fitting is a
way to describe that reference and calculate suitable vertices. It does not
authorize replacing the reference with the fitted shape.

The intended upload workflow has three automatic stages:

1. Calculate the visual LODs, including their vertex and triangle counts.
   Differentiate the fitted surfaces for slope and curvature, and integrate
   over their domains to calculate surface area and sampling density. Check
   each proposed tessellation against the complete source surface and its
   shading and texture fields.
2. Analyze and correct mesh defects. Preserve the creator's surface, openings,
   creases, UV mapping, shading, and material boundaries. Additional triangles
   are acceptable when needed for valid collision tessellation. Density alone
   does not justify deleting meaningful geometry.
3. Select collision independently of visual attributes. Prefer the lowest
   Lodestar-generated level that supplies a supported hull set. Otherwise,
   generate position-only polygonal collision, simplify shared model seams in
   one transaction, and validate the result. High remains the reference for
   measuring displacement, openings and support surfaces, rather than a
   mandatory collision tessellation. Changing the displayed visual LOD must
   not change the selected collider.

The default workflow must not require hand-selected LOD counts or vertices per
hull. C++, C, Rust, the standalone app, and the viewer integration should use
the same native preparation operations and expose consistent reports.

## Acceptance conditions

- Generated triangles have positive area and valid winding.
- The actual uploader's normalization and packed representation are checked.
  A positive-area triangle can still fail Firestorm's physics test.
- Shared boundaries remain conforming. Refinement must not introduce unmatched
  edge subdivisions or hide physical cracks with shading.
- High geometry and attributes remain authoritative. Export rounding and
  uploader position quantization have explicit, separately measured bounds.
- Lower visual levels may use the authorized half-pixel surface and one-degree
  shading-normal bounds. UV domains, original diffuse images and sampler modes,
  material boundaries and named-model assignments remain protected. Shared
  straight UV knots may be removed only through coordinated seam edits.
- Collision drops UV and shading constraints. Its automatic displacement
  ceiling follows the selected level's geometric allowance; several fractions
  are tested and the smallest valid representation within that ceiling is
  selected. Report the composed bound against High. This is bounded
  simplification, not an exact-surface claim. Preserve horizontal support
  heights, named-model bounds, closed-shell orientation and nonzero volume;
  validate clearance separately. Never replace a closed shell with coincident
  opposite faces, even if every individual triangle passes uploader tests.
- A failed repair retains the reference and reports its cause. An unresolved
  diagnostic must never be presented as a validated upload companion.
- Collision must form a coherent representation; each input element need not
  be a closed solid. Output hulls must be closed convex volumes. Retain
  openings, support planes and shared contacts. Bounded surface completion may
  give an open sheet the thickness required to survive packing, within the
  reported surface allowance. Report whether the result is an exact solid
  conversion or a surface cover; a surface-distance bound alone does not
  certify occupied volume, mass, every clearance or self-intersection freedom.
- Detect dimensions below 0.01 m and calculate a legal physical container while
  retaining the visible source coordinates. A physics box alone does not set
  visual normalization bounds. Preserve existing declared padding frames when
  removing zero-area source faces. Use its coordinate frame consistently in
  every LOD and the collider; do not introduce degenerate padding triangles.
- Test fixtures and native serialization checks precede viewer preview checks.
  Simulator collision behavior requires separate in-world validation.

## Current implementation

The file-based fitting runner computes all four visual LODs, performs strict
collision refinement, probes supported hull conversion, and records unresolved
cases. A failed triangle preflight produces `Lodestar_unresolved_PHYS.dae`, marks
the scorecard `physics_blocked`, and exits with status 2.

The development Firestorm integration launches preparation when High is
selected. It imports Lodestar's calculated levels sequentially, labels them
`Lodestar Auto`, imports the selected collision companion, and attempts native
hull generation from that companion. It suppresses
the stock first-load meshoptimizer job. The preparation worker uses Lodestar's
own UV cavity retriangulation and coordinated boundary coarsening. Native
polynomial fitting describes charts; actual mesh area and edge-curvature
integrals supply budgets in the worker. Complete High correspondence checks
accept or reject proposals. It never calls meshoptimizer to generate levels.

The local Python worker is an explicit development dependency. Chart
orchestration, seam coarsening, mesh-integral budgets, collision simplification
and candidate selection still need native C++ and Rust APIs.
Exact hull conversion supports closed convex shells and closed axis-aligned
concave solids. The native coherent collision operation additionally partitions
open and other surface meshes into convex patches. It checks bidirectional
surface displacement, source facet half-spaces, axis-constant contact planes,
and inward-shell cavity guards. It splits patches that cannot meet those
checks. Open planar patches receive minimal inward thickness derived from
float precision and the U16 grid, subject to the surface allowance. Work,
precision and piece limits can still reject a source.

The viewer restores metric distances through each model's nonuniform
normalization before hull construction. It automatically tries allowances of
maximum extent / 384, / 192, / 96 and / 48, using the first successful result.
These are additional bounds against the selected collision source. Automatic
preparation requires hulls for instances below 0.5 m on any axis; other valid
models retain triangle physics. It commits required hull conversions together.
Manual Analyze processes the selected physics source. The Log reports each
model's additional bound and exact-solid flag.

Native Firestorm serialization checks on TestHall's four thin gates produced
164, 161, 193 and 178 hulls. Decoded pieces retained positive volume and the
High payload was unchanged. Offline guarded openings and supported routes
passed. These checks do not establish simulator behavior. The previous GUI
build was blocked by thin triangle physics; the coherent-hull build still
needs a GUI and in-world test.

The C++, C and Rust libraries expose coherent collision and size planning.
The worker applies physical container frames through the patched uploader.
It retains all visual streams in authored coordinates, stores the physical
frame in each generated COLLADA geometry's `LODESTAR` extra, and supplies a
positive-volume box collider. The native importer normalizes each LOD into
that frame and compensates its instance transform, preserving visible scale
and placement without render padding. Normals use the inverse-transpose rule.
Existing declared padding frames are retained after zero-area face removal.

The minimum source frame dimension is 0.01 m divided by import scale. Changing
that scale reruns preparation when a physical frame is present or required.
Import scale is locked while preparation runs. Missing frame metadata,
invalid bounds, and packed visual degeneracy block readiness. Size reporting
remains in `size-proposals.json`, with applied-frame flags and scaled sizes.
The frame extension requires the patched Lodestar importer; stock COLLADA
importers do not apply it. Native packed-asset tests cover tiny, flat,
translated, thin, pre-padded and mixed-size meshes, plus changed import scale.
GUI and simulator validation remain pending. Broad material/attribute coverage, general solid inference,
self-intersection certification, native worker parity and meaningful TestHall
visual reduction also remain incomplete.
