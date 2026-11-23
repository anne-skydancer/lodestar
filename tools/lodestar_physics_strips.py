"""Conforming, planar collision refinement without moving source corners.

Firestorm's aspect-dependent area test can reject a perfectly valid long panel.
Split a convex pair into strips, and split every incident boundary face with it.
Only exactly coplanar pairs are eligible. Dyadic interpolation allows rational
predicates to certify the added points against the stored source geometry.
"""

from fractions import Fraction

import numpy as np

from lodestar_firestorm_collision import _exact_collinear, _exact_coplanar, _geometry
from lodestar_firestorm_degenerate import physics_degenerate_mask, packed_physics_degenerate_mask
from lodestar_nonplanar import _edge_uses
from bake_planar_patch import cross2


def _bad(triangles):
    return physics_degenerate_mask(triangles) | packed_physics_degenerate_mask(triangles)


def _convex_quad(quad, normal):
    # Project away the dominant normal component and certify every turn.
    axes = [axis for axis in range(3) if axis != int(np.abs(normal).argmax())]
    projected = [[Fraction(float(point[axis])) for axis in axes] for point in quad]
    turns = []
    for index in range(4):
        a, b, c = [projected[(index + offset) % 4] for offset in range(3)]
        turns.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(turn > 0 for turn in turns) or all(turn < 0 for turn in turns)


def _subdivide_neighbor(triangle, indices, knots):
    split_edges = [(i, a, b) for i, (a, b) in enumerate(zip(indices, np.roll(indices, -1)))
                   if (min(a, b), max(a, b)) in knots]
    if len(split_edges) == 1:
        i, a, b = split_edges[0]
        chain = knots[min(a, b), max(a, b)]
        if a > b:
            chain = chain[::-1]
        opposite = triangle[(i + 2) % 3]
        return np.stack((np.broadcast_to(opposite, chain[:-1].shape), chain[:-1], chain[1:]), axis=1)
    boundary = []
    for a, b in zip(indices, np.roll(indices, -1)):
        edge = min(a, b), max(a, b)
        chain = knots.get(edge)
        if chain is None:
            boundary.append(triangle[list(indices).index(a)])
        else:
            boundary.extend(chain[:-1] if a < b else chain[:0:-1])
    # Interior dyadic point, with the original winding. Each subtriangle stays
    # on its own source facet, including when a neighbor meets a real crease.
    center = triangle[0] * .5 + triangle[1] * .25 + triangle[2] * .25
    boundary = np.asarray(boundary)
    return np.stack((np.broadcast_to(center, boundary.shape), boundary,
                     np.roll(boundary, -1, axis=0)), axis=1)


def exact_strip_repair(source, *, maximum_surface_displacement=0.0):
    """Return a conforming refinement and explicit float-export error bounds.

    Every accepted edit strictly decreases the combined preview/packed reject
    count. Unsupported facets remain in place, with no positive-area omission.
    The returned double-precision surface equals the original surface. Casting
    its added vertices to the file's float32 format has a separately measured
    displacement bound, never silently treated as mathematical equality.
    """
    source = np.asarray(source, dtype=np.float64)
    if (source.ndim != 3 or source.shape[1:] != (3, 3) or not len(source) or
            not np.isfinite(source).all()):
        raise ValueError("Strip refinement requires finite nonempty triangles")
    result = source.copy()
    if not np.isfinite(maximum_surface_displacement) or maximum_surface_displacement < 0:
        raise ValueError("Strip displacement must be finite and nonnegative")
    initial = int(_bad(source).sum())
    edits = neighbor_splits = 0
    surface_change = 0.0
    while True:
        points, faces, rejected = _geometry(result)
        bad = rejected(faces)
        if not bad.any():
            break
        edges = _edge_uses(faces)
        accepted = None
        span = np.ptp(result, axis=(0, 1))
        span[span < 1e-5] = 1
        for first in np.flatnonzero(bad):
            for a, b, c in (faces[first], np.roll(faces[first], 1), np.roll(faces[first], 2)):
                uses = edges[min(a, b), max(a, b)]
                if len(uses) != 2:
                    continue
                second, tail, head = next(use for use in uses if use[0] != first)
                if (tail, head) != (b, a):
                    continue
                d = next(vertex for vertex in faces[second] if vertex not in (a, b))
                planar = d != c and _exact_coplanar(*points[[a, b, c]], points[d])
                if d == c or (not planar and maximum_surface_displacement == 0):
                    continue
                ids = np.array((c, a, d, b))
                quad = points[ids]
                normal = np.cross(quad[1] - quad[0], quad[3] - quad[0])
                if np.linalg.norm(normal) == 0:
                    continue
                if not _convex_quad(quad, normal):
                    continue
                lengths = np.linalg.norm((np.roll(quad, -1, axis=0) - quad) / span, axis=1)
                if lengths[0] + lengths[2] > lengths[1] + lengths[3]:
                    ids = np.roll(ids, 1)
                    quad = points[ids]
                for power in range(1, 11):
                    parameter = np.arange(2 ** power + 1)[:, None] / 2 ** power
                    left = quad[0] + parameter * (quad[3] - quad[0])
                    right = quad[1] + parameter * (quad[2] - quad[1])
                    strips = np.concatenate((
                        np.stack((left[:-1], right[:-1], right[1:]), axis=1),
                        np.stack((left[:-1], right[1:], left[1:]), axis=1)))
                    # All endpoints remain, so the source normalization domain
                    # also applies to the replacement-only feasibility check.
                    trial = np.concatenate((result, strips))
                    if _bad(trial)[-len(strips):].any():
                        continue
                    knots = {}
                    for start, end, chain in ((ids[0], ids[3], left), (ids[1], ids[2], right)):
                        knots[min(start, end), max(start, end)] = chain if start < end else chain[::-1]
                    adjacent = {use[0] for edge in knots for use in edges[edge]} - {first, second}
                    patches = [_subdivide_neighbor(points[faces[index]], faces[index], knots)
                               for index in sorted(adjacent)]
                    replaced = {first, second} | adjacent
                    keep = np.array([index not in replaced for index in range(len(result))])
                    candidate = np.concatenate((result[keep], strips, *patches))
                    candidate_bad = _bad(candidate)
                    if int(candidate_bad.sum()) >= int(bad.sum()):
                        continue
                    # Do not create new rejected facets in the replacement.
                    if candidate_bad[np.count_nonzero(keep):].any():
                        continue
                    # These exact predicates are the final geometric gate.
                    if any(not _exact_collinear(quad[0], point, quad[3]) for point in left):
                        continue
                    if any(not _exact_collinear(quad[1], point, quad[2]) for point in right):
                        continue
                    if any(not _exact_coplanar(*points[faces[index]], point)
                           for index, patch in zip(sorted(adjacent), patches) for point in patch.reshape(-1, 3)):
                        continue
                    change = 0.0
                    if not planar:
                        # Project both triangulations onto the same convex
                        # polygon. Their affine difference is bounded over
                        # every cell of the complete intersection overlay.
                        from lodestar_surface_fit import correspondence_bound
                        combined = np.concatenate((result[[first, second]], strips))
                        local_points, local_faces, _ = _geometry(combined)
                        axes = [axis for axis in range(3) if axis != int(np.abs(normal).argmax())]
                        uv = local_points[:, axes].copy()
                        uv -= uv.min(axis=0)
                        uv /= np.ptp(uv, axis=0)
                        triangle = uv[local_faces[0]]
                        if cross2(triangle[1] - triangle[0], triangle[2] - triangle[0]) < 0:
                            uv[:, 1] = 1 - uv[:, 1]
                        try:
                            change = correspondence_bound(uv, local_points,
                                np.tile([0., 0., 1.], (len(local_points), 1)),
                                local_faces[2:], local_faces[:2])['surface_bound']
                        except (ValueError, np.linalg.LinAlgError):
                            continue
                        if surface_change + change > maximum_surface_displacement:
                            continue
                    accepted = candidate, len(adjacent), change
                    break
                if accepted is not None:
                    break
            if accepted is not None:
                break
        if accepted is None:
            break
        result, count, change = accepted
        surface_change += change
        edits += 1
        neighbor_splits += count
    export_shift = float(np.linalg.norm(result.astype(np.float32).astype(np.float64) - result, axis=2).max())
    return result, {
        "source_rejections": initial, "remaining_rejections": int(_bad(result).sum()),
        "strip_pairs": edits, "incident_faces_refined": neighbor_splits,
        "added_triangles": len(result) - len(source), "omitted_positive_area_triangles": 0,
        "source_corner_displacement": 0.0, "ideal_surface_change": surface_change,
        "maximum_surface_displacement": maximum_surface_displacement,
        "float32_export_displacement_bound": export_shift,
        "conforming_shared_edges": True}


def exact_edge_refinement(source):
    """Bisect an edge and every incident face when it clears uploader failures.

    A near-threshold long triangle can pass after bisection even when no convex
    planar quad is available. All new triangles remain on their own source
    facets. No accepted replacement may introduce a new rejected triangle.
    """
    result = np.asarray(source, dtype=np.float64).copy()
    initial = int(_bad(result).sum())
    edits = 0
    while True:
        points, faces, rejected = _geometry(result)
        bad = rejected(faces)
        if not bad.any():
            break
        edges = _edge_uses(faces)
        accepted = None
        span = np.ptp(result, axis=(0, 1))
        span[span < 1e-5] = 1
        for index in np.flatnonzero(bad):
            triangle = faces[index]
            options = sorted(zip(triangle, np.roll(triangle, -1)),
                key=lambda pair: -float(np.linalg.norm((points[pair[0]] - points[pair[1]]) / span)))
            for a, b in options:
                midpoint = points[a] * .5 + points[b] * .5
                if not _exact_collinear(points[a], midpoint, points[b]):
                    continue
                uses = edges[min(a, b), max(a, b)]
                selected = {entry[0] for entry in uses}
                replacement = []
                for face, first, last in uses:
                    third = next(v for v in faces[face] if v not in (first, last))
                    replacement.extend(((points[first], midpoint, points[third]),
                                        (midpoint, points[last], points[third])))
                keep = np.array([i not in selected for i in range(len(result))])
                candidate = np.concatenate((result[keep], replacement))
                candidate_bad = _bad(candidate)
                if candidate_bad[np.count_nonzero(keep):].any() or candidate_bad.sum() >= bad.sum():
                    continue
                accepted = candidate
                break
            if accepted is not None:
                break
        if accepted is None:
            break
        result = accepted
        edits += 1
    return result, {'edge_bisections': edits, 'source_rejections': initial,
                    'remaining_rejections': int(_bad(result).sum()), 'ideal_surface_change': 0.0,
                    'float32_export_displacement_bound': float(np.linalg.norm(
                        result.astype(np.float32).astype(float) - result, axis=2).max())}
