# The certified LOD metric and record v1

`lodestar_common` measures how far a creator LOD is from High and stores the answer as a small record that a
viewer can consume. High stays authoritative: nothing here replaces it. The record only says how far a LOD is
from High, with a proof on one side and a witness on the other.

## Libraries

| Target | Role | Depends on |
|---|---|---|
| `lodestar_common` | Mesh types, topology, the sampled metrics (test oracles), the certified metric, the record | glm (Assimp only for the optional importer) |
| `lodestar_simplifier` | LOD generation: meshoptimizer, planar, seam, surface fit and tessellation | common, meshoptimizer |
| `lodestar_physics` | Collision meshes, convex hulls, decomposition | common, meshoptimizer |
| `lodestar_render` | The evaluation and preview renderer. Never a live viewer renderer | common, Vulkan, GLFW |

No specialised library depends on another. `lodestar_core` is an interface umbrella over common, simplifier
and physics for the C ABI, the apps and the tests.

## Numbers

Every error has two numbers per component:

- **U**, a certified upper bound: the true error is at most U.
- **W**, a witnessed lower bound: the true error is at least W, because it was measured at a point.

Errors are stored in normalised units (a fraction of High's largest bounding-box extent). The record holds the
extents, so a world error is `value * largest extent`. The viewer applies its own pixel threshold (tau); the
library has none.

| Component | Meaning | U | W |
|---|---|---|---|
| Shape G | Two-sided Hausdorff distance to High | Branch and bound: piece radius rho, `d(c) + rho`, and the convexity bound over a nearby target triangle; the worst piece splits in four until `U <= 1.05 W + 2 grid steps`, or the work cap | The largest distance found at a corner or centroid |
| Texture T | Distance between points with the same texture coordinate | Exact on the UV overlay cells (`correspondenceBound`, port of `tools/lodestar_surface_fit.py`); infinite where High does not cover the LOD's UVs or the overlay limit is hit | Corner values on the overlay, nearest of the High faces that contain the point |
| Shading N | Distance to the nearest point whose normal is within 0.5 degrees | Through the UV overlay when the normals match there, or through normal cones when the spread of the LOD's normals is small enough to afford; otherwise infinite | Measured at samples against a superset of the matching triangles |
| Axes A, R | Per axis offset and piece radius | `sqrt(sum s_i^2 A_i^2) + sqrt(sum s_i^2 R_i^2)` bounds the shape error under object scale s | |

A LOD that is bit-identical to High gets the copy flag and exact zeros.

### Rigged meshes

The record stores the bind-pose bound and the weight terms apart, so the viewer can apply its own joint scale:

```
error over every joint rotation <= sigmaMax * shape.upper + riggedWeight * 2 * riggedRadius
```

`riggedWeight` is an upper bound of half the L1 distance of the skin weights between a point and its partner
(interval arithmetic over the pieces and the target triangles within reach). `riggedRadius` is the largest
distance from a point to its joints' pivots. `riggedUpper(lod, sigmaMax)` evaluates the formula.

`poseWitness(high, lod, poses, jointCount)` skins both meshes with each pose (skinning matrices in) and returns
the largest distance found between the posed surfaces. Any single sampled pose that deforms is a valid witness.
The poses must keep each joint's scale at most the sigmaMax used with the upper bound.

## Record v1

A fixed little-endian record of 238 bytes with the format version in the first byte: format and metric versions,
rigged flag, LOD count, High's triangle count and XXH64 block hash, High's extents, and three LOD slots (Medium,
Low, Lowest) of triangles, hash, flags, U and W for G, T and N, A and R, and the rigged terms. Flags: measured,
copy of High, shape work cap hit, texture certified, shading certified, no UV, rigged terms valid.

`deserialise` rejects another format or metric version, a wrong size and NaN. The record is meant to be stored
per mesh and discarded when the metric version changes.

## Use

```cpp
#include <lodestar/metric.hpp>
std::vector<const lodestar::Mesh*> lods{&medium, &low, &lowest};
lodestar::metric::Record r = lodestar::metric::measureAsset(high, lods);
auto bytes = lodestar::metric::serialise(r);
```

## Tests

`lodestar_metric_test` checks U against the sampled maximum on heightfield and sphere pairs (a narrow spike,
non-uniform object scales, 40 random cases), zero error for identical LODs, the texture overlay, shading, the
rigged terms against sampled poses, the XXH64 vectors and the record round trip.
`correspondence_oracle_test.py` runs the Python `correspondence_bound` on shared-UV fixtures and compares the C++
port to 1e-9 (it needs numpy).

`lodestar_metric_bench` times `measureAsset` per mesh on a corpus (`tools/fetch_benchmark_assets.py`) and a
synthetic size ladder.
