"""Coarsen shared chart seams atomically, retaining their complete UV domains."""

from collections import Counter, defaultdict, deque
from fractions import Fraction

import numpy as np

from bake_planar_patch import boundary_loop, cross2
from lodestar_patch_layout import triangulate_boundary
from lodestar_surface_fit import correspondence_bound


def straight_uv(a, b, c):
    """Exact collinearity of stored UVs, with the middle point between endpoints."""
    if np.any(b < np.minimum(a, c)) or np.any(b > np.maximum(a, c)):
        return False
    ax, ay = map(Fraction, map(float, a))
    bx, by = map(Fraction, map(float, b))
    cx, cy = map(Fraction, map(float, c))
    return (bx - ax) * (cy - ay) == (by - ay) * (cx - ax)


def boundary_proposal(chart, faces, vertex):
    boundary = boundary_loop(faces)
    if vertex not in boundary:
        return None
    index = boundary.index(vertex)
    first, last = boundary[index - 1], boundary[(index + 1) % len(boundary)]
    if not straight_uv(*chart.source_uv[[first, vertex, last]]):
        return None
    selected = np.flatnonzero(np.any(faces == vertex, axis=1))
    old = faces[selected]
    ring = np.asarray([v for v in boundary_loop(old) if v != vertex], dtype=np.intp)
    if len(ring) < 3:
        return None
    local = triangulate_boundary(chart.uv[ring])
    if local is None:
        return None
    replacement = ring[local]
    if len(replacement) >= len(old):
        return None
    old_normal = np.cross(chart.positions[old[:, 1]] - chart.positions[old[:, 0]],
                          chart.positions[old[:, 2]] - chart.positions[old[:, 0]])
    new_normal = np.cross(chart.positions[replacement[:, 1]] - chart.positions[replacement[:, 0]],
                          chart.positions[replacement[:, 2]] - chart.positions[replacement[:, 0]])
    if np.any(new_normal @ old_normal.T <= 0):
        return None
    return selected, replacement, tuple(sorted((tuple(chart.positions[first]), tuple(chart.positions[last]))))


def coarsen_boundaries(scene, groups, fitted, current, floors, surface_limit, normal_limit, pins):
    """A seam edit commits only when every incident chart accepts the same edge.

    UV coordinates and material assignments are retained. Removing an exactly
    straight UV knot preserves its UV domain, while the 3D seam is allowed only
    the explicitly bounded displacement from High. Unsupported incident charts
    and intentional contact pins veto the entire edit.
    """
    owners = defaultdict(set)
    for index, selected in enumerate(groups):
        for point in set(map(tuple, scene.positions[selected].reshape(-1, 3))):
            owners[point].add(index)
    vertices = {}
    for index, chart in fitted.items():
        lookup = defaultdict(list)
        for vertex, point in enumerate(chart.positions):
            lookup[tuple(point)].append(vertex)
        vertices[index] = lookup
    pending = deque(sorted(owners))
    queued = set(pending)
    changed = set()
    rejected = Counter()
    removed = attempts = 0
    while pending and attempts < 50000:
        point = pending.popleft()
        queued.discard(point)
        attempts += 1
        if point in pins or not owners[point].issubset(fitted):
            rejected['pinned_or_unsupported_incident_chart'] += 1
            continue
        proposals = {}
        seam = None
        for index in sorted(owners[point]):
            chart = fitted[index]
            local = vertices[index][point]
            if len(local) != 1:
                rejected['multiple_uv_corners_at_point'] += 1
                break
            try:
                proposal = boundary_proposal(chart, current[index], local[0])
            except ValueError:
                rejected['invalid_cavity'] += 1
                break
            if proposal is None:
                rejected['no_uv_preserving_boundary_edit'] += 1
                break
            selected, replacement, endpoints = proposal
            if (seam is not None and seam != endpoints) or (
                    len(current[index]) - len(selected) + len(replacement) < floors[index]):
                rejected['seam_mismatch_or_calculated_floor'] += 1
                break
            seam = endpoints
            try:
                bound = correspondence_bound(chart.uv, chart.positions, chart.normals, replacement, chart.faces,
                    source_normals=chart.normals_for(replacement), target_normals=chart.corner_normals)
            except (ValueError, np.linalg.LinAlgError):
                rejected['overlay_failure'] += 1
                break
            if bound['surface_bound'] > surface_limit or bound['normal_angle_bound_degrees'] > normal_limit:
                rejected['surface_limit' if bound['surface_bound'] > surface_limit else 'normal_limit'] += 1
                break
            proposals[index] = selected, replacement
        else:
            for index, (selected, replacement) in proposals.items():
                keep = np.ones(len(current[index]), dtype=bool)
                keep[selected] = False
                current[index] = np.concatenate((current[index][keep], replacement))
                changed.add(index)
            removed += 1
            for endpoint in seam:
                if endpoint not in queued:
                    pending.append(endpoint)
                    queued.add(endpoint)
    return {"removed_seam_vertices": removed, "changed_charts": sorted(changed),
            "attempts": attempts, "work_limit_reached": bool(pending), "rejection_reasons": dict(rejected)}
