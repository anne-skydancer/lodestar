"""Repair source-derived triangle physics for Firestorm while bounding surface change."""

import numpy as np
from fractions import Fraction

from check_testhall_physics import triangle_key
from lodestar_collision_probes import _point_triangle_distance_sq
from lodestar_firestorm_degenerate import (
    _decoded_triangles,
    _physics_degenerate,
    _quantize,
    _zero_area,
    decoded_upload_positions,
    normalize_triangles,
    packed_physics_degenerate_mask,
    physics_degenerate_mask,
)
from lodestar_nonplanar import _directed_sampled_distance, _edge_uses


def _geometry(positions):
    points, inverse = np.unique(positions.reshape(-1, 3), axis=0, return_inverse=True)
    normalized = normalize_triangles(positions)
    packed, low, span = _quantize(normalized)
    normalized_points = np.empty(points.shape, dtype=np.float32)
    packed_points = np.empty(points.shape, dtype=np.int64)
    normalized_points[inverse] = normalized.reshape(-1, 3)
    packed_points[inverse] = packed.reshape(-1, 3)
    unpacked_points = _decoded_triangles(packed_points, low, span)

    def rejected(faces):
        return (_physics_degenerate(normalized_points[faces]) |
                _zero_area(packed_points[faces]) |
                _physics_degenerate(unpacked_points[faces]))

    return points, inverse.reshape(-1, 3), rejected


def _normals(triangles):
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def _exact_coplanar(a, b, c, d):
    a, b, c, d = ([Fraction(float(value)) for value in point] for point in (a, b, c, d))
    ab, ac, ad = ([end[i] - a[i] for i in range(3)] for end in (b, c, d))
    normal = [ab[1]*ac[2] - ab[2]*ac[1], ab[2]*ac[0] - ab[0]*ac[2], ab[0]*ac[1] - ab[1]*ac[0]]
    return sum(normal[i] * ad[i] for i in range(3)) == 0


def _exact_collinear(a, b, c):
    a, b, c = ([Fraction(float(value)) for value in point] for point in (a, b, c))
    first, second = ([end[i] - a[i] for i in range(3)] for end in (b, c))
    return (first[1]*second[2] == first[2]*second[1] and
            first[2]*second[0] == first[0]*second[2] and
            first[0]*second[1] == first[1]*second[0])


def _flip_edges(points, faces, rejected, *, exact=False):
    bad = rejected(faces)
    count = 0
    while bad.any():
        edges = _edge_uses(faces)
        for first in np.flatnonzero(bad):
            a, b, c = faces[first]
            for a, b, c in ((a, b, c), (b, c, a), (c, a, b)):
                uses = edges[min(a, b), max(a, b)]
                if len(uses) != 2:
                    continue
                second, tail, head = next(use for use in uses if use[0] != first)
                if (tail, head) != (b, a):
                    continue
                d = next(vertex for vertex in faces[second] if vertex not in (a, b))
                if d == c or (min(c, d), max(c, d)) in edges:
                    continue
                normal = np.cross(points[b] - points[a], points[c] - points[a])
                length = np.linalg.norm(normal)
                if length == 0 or abs(normal @ (points[d] - points[a])) > length * 4e-6:
                    continue
                if exact and not _exact_coplanar(points[a], points[b], points[c], points[d]):
                    continue
                replacement = np.array(((c, d, b), (d, c, a)))
                if np.any(_normals(points[replacement]) @ normal <= 0):
                    continue
                replacement_bad = rejected(replacement)
                if np.count_nonzero(replacement_bad) < 1 + int(bad[second]):
                    faces[[first, second]] = replacement
                    bad[[first, second]] = replacement_bad
                    count += 1
                    break
            else:
                continue
            break
        else:
            break
    return count


def _short_edge_collapses(points, faces, rejected, fixed_positions, max_edge):
    bad = rejected(faces)
    low, high = points.min(axis=0), points.max(axis=0)
    fixed = np.any(((points == low) | (points == high)) & (low != high), axis=1)
    fixed |= np.array([tuple(point) in fixed_positions for point in points])
    count = 0
    while bad.any():
        edges = _edge_uses(faces)
        adjacency = [set() for _ in points]
        boundary, unsupported = set(), set()
        for (a, b), uses in edges.items():
            adjacency[a].add(b)
            adjacency[b].add(a)
            if len(uses) == 1:
                boundary.update((a, b))
            elif len(uses) != 2 or uses[0][1:] == uses[1][1:]:
                unsupported.update((a, b))

        collapse = None
        for face in faces[bad]:
            # Add a stable tie-break for equal-length edges.
            options = sorted((np.linalg.norm(points[a] - points[b]), a, b)
                             for a, b in zip(face, np.roll(face, -1)))
            for length, a, b in options:
                if length > max_edge or a in unsupported or b in unsupported:
                    continue
                edge_count = len(edges[min(a, b), max(a, b)])
                if edge_count == 2 and a in boundary and b in boundary:
                    continue
                if len(adjacency[a] & adjacency[b]) != edge_count:
                    continue
                for origin, dest in ((a, b), (b, a)):
                    if fixed[origin] or (origin in boundary and dest not in boundary):
                        continue
                    incident = (faces == origin).any(axis=1)
                    removed = incident & (faces == dest).any(axis=1)
                    changed = incident & ~removed
                    if not changed.any():
                        continue
                    replacement = faces[changed].copy()
                    replacement[replacement == origin] = dest
                    replacement_bad = rejected(replacement)
                    if replacement_bad.sum() >= bad[incident].sum():
                        continue
                    old_normals = _normals(points[faces[changed]])
                    new_normals = _normals(points[replacement])
                    if np.any((old_normals * new_normals).sum(axis=1) <= 0):
                        continue
                    collapse = removed, changed, replacement, replacement_bad
                    break
                if collapse is not None:
                    break
            if collapse is not None:
                break
        if collapse is None:
            break
        removed, changed, replacement, replacement_bad = collapse
        faces[changed] = replacement
        faces = faces[~removed]
        bad[changed] = replacement_bad
        bad = bad[~removed]
        count += 1
    return faces, bad, count


def _changed_faces(source, target):
    keys = set(map(triangle_key, target))
    return source[[triangle_key(triangle) not in keys for triangle in source]]


def _surface_bound(source, target, tolerance):
    """Bound every changed point using the convexity of distance to a triangle."""
    changed = _changed_faces(source, target)
    if not len(changed):
        return 0.0, 0.0
    sampled = float(_directed_sampled_distance(changed, target))
    if sampled > tolerance:
        raise ValueError(f"Collider sampled surface error {sampled:g} m exceeds {tolerance:g} m")

    corners = tuple(np.moveaxis(target, 1, 0))
    normals = _normals(target)
    low, high = target.min(axis=1), target.max(axis=1)
    stack = [(triangle, 0) for triangle in changed]
    maximum = 0.0
    while stack:
        triangle, depth = stack.pop()
        near = ((low <= triangle.max(axis=0) + tolerance).all(axis=1) &
                (high >= triangle.min(axis=0) - tolerance).all(axis=1))
        if not near.any():
            raise ValueError("No retained collider surface is near a changed face")
        distances = _point_triangle_distance_sq(triangle[:, None, :],
                                                tuple(corner[None, near] for corner in corners),
                                                normals[None, near])
        bound = float(np.sqrt(distances.max(axis=0).min()))
        if bound <= tolerance:
            maximum = max(maximum, bound)
            continue
        if depth == 16 or distances.min(axis=1).max() > tolerance ** 2:
            raise ValueError(f"Cannot bound collider surface change by {tolerance:g} m")
        lengths_squared = ((triangle - np.roll(triangle, -1, axis=0)) ** 2).sum(axis=1)
        a, b, c = np.roll(triangle, -int(lengths_squared.argmax()), axis=0)
        middle = (a + b) * 0.5
        stack.extend(((np.array((a, middle, c)), depth + 1),
                      (np.array((middle, b, c)), depth + 1)))
    return sampled, maximum


def build_firestorm_collider(positions, *, fixed_positions=(), max_surface_change=0.03):
    """Return source-derived collider faces and a report, or fail the preflight.

    Repair skinny triangles with local retriangulation and short edge collapses,
    then omit only remaining uploader rejections. Keep the source model bounds
    and check both surface directions. The caller checks scene-level support.
    """
    source = np.asarray(positions, dtype=np.float64)
    if (source.ndim != 3 or source.shape[1:] != (3, 3) or not len(source) or
            not np.isfinite(source).all() or not np.isfinite(max_surface_change) or max_surface_change <= 0):
        raise ValueError("Expected source collision triangles and a positive finite surface limit")
    points, faces, rejected = _geometry(source)
    original_bad = int(rejected(faces).sum())
    flips = _flip_edges(points, faces, rejected)
    faces, bad, collapses = _short_edge_collapses(points, faces, rejected, set(map(tuple, fixed_positions)),
                                                 min(0.025, max_surface_change))
    result = points[faces[~bad]]
    if (not len(result) or not np.array_equal(result.min(axis=(0, 1)), source.min(axis=(0, 1))) or
            not np.array_equal(result.max(axis=(0, 1)), source.max(axis=(0, 1)))):
        raise ValueError("Collider repair changed the source model bounds")
    if physics_degenerate_mask(result).any() or packed_physics_degenerate_mask(result).any():
        raise ValueError("Collider repair introduced a degenerate triangle")
    # Barycentric interpolation bounds each packed triangle's displacement by
    # its largest vertex shift. Reserve that amount in both surface directions.
    decoded = decoded_upload_positions(result, source)
    packing_shift = float(np.linalg.norm(decoded - result, axis=2).max())
    geometric_limit = max_surface_change - packing_shift
    if geometric_limit <= 0:
        raise ValueError("Collider asset position quantization exceeds the surface limit")
    source_sampled, source_bound = _surface_bound(source, result, geometric_limit)
    collider_sampled, collider_bound = _surface_bound(result, source, geometric_limit)
    return result, {
        "source_triangles": len(source),
        "collider_triangles": len(result),
        "source_uploader_or_packed_degenerates": original_bad,
        "diagonal_flips": flips,
        "short_edge_collapses": collapses,
        "omitted_after_local_repair": int(bad.sum()),
        "maximum_surface_change_m": max_surface_change,
        "source_to_collider_sampled_error_m": source_sampled,
        "collider_to_source_sampled_error_m": collider_sampled,
        "source_to_collider_upper_bound_m": source_bound,
        "collider_to_source_upper_bound_m": collider_bound,
        "maximum_packing_shift_m": packing_shift,
        "source_to_packed_collider_upper_bound_m": source_bound + packing_shift,
        "packed_collider_to_source_upper_bound_m": collider_bound + packing_shift,
    }
