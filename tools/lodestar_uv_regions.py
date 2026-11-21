"""Plan texture transfer across UV seams within compatible planar regions."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from bake_planar_patch import boundary_loop, cross2
from lodestar_patch_layout import triangulate_boundary


NORMAL_TOLERANCE = 1e-6
PLANE_RELATIVE_TOLERANCE = 1e-6
AREA_TOLERANCE = 1e-10
MIN_STRAIGHT_DOT = np.cos(np.deg2rad(5.0))


@dataclass
class RegionLayout:
    faces: tuple
    projected: np.ndarray
    normal: np.ndarray
    plane_error: float
    texture_boundary: tuple
    boundary: np.ndarray
    boundary_uv: np.ndarray
    straight_knots: dict
    boundary_error: float = 0.0
    geometric_normal_error: float = 0.0
    shading_normal_error: float = 0.0


def planar_regions(faces, geometric, lengths, shading, shading_lengths, normal_tolerance):
    usable = (lengths > 0) & np.all(shading_lengths > 0, axis=1)
    usable &= np.all(np.abs(shading - shading[:, :1]) <= normal_tolerance, axis=(1, 2))
    edges = defaultdict(list)
    for face, corners in enumerate(faces.tolist()):
        for a, b in zip(corners, corners[1:] + corners[:1]):
            edges[min(a, b), max(a, b)].append((a, b, face))

    adjacency = [[] for _ in faces]
    for occurrences in edges.values():
        if len(occurrences) != 2:
            continue
        (a, b, first), (c, d, second) = occurrences
        if a != d or b != c or not usable[first] or not usable[second]:
            continue
        if (np.max(np.abs(geometric[first] - geometric[second])) <= normal_tolerance and
                np.max(np.abs(shading[first, 0] - shading[second, 0])) <= normal_tolerance):
            adjacency[first].append(second)
            adjacency[second].append(first)

    visited = set()
    for seed in range(len(faces)):
        if seed in visited:
            continue
        connected = [seed]
        visited.add(seed)
        for face in connected:
            for neighbor in adjacency[face]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    connected.append(neighbor)
        if len(connected) >= 3:
            yield tuple(sorted(connected))


def simple_polygon(points):
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        for j in range(i + 2, len(points) - (i == 0)):
            c, d = points[j], points[(j + 1) % len(points)]
            if (np.any(np.maximum(a, b) < np.minimum(c, d) - AREA_TOLERANCE) or
                    np.any(np.maximum(c, d) < np.minimum(a, b) - AREA_TOLERANCE)):
                continue
            ab = cross2(b - a, np.stack((c - a, d - a)))
            cd = cross2(d - c, np.stack((a - c, b - c)))
            # Touching nonadjacent edges would make the region ambiguous.
            if (ab.min() <= AREA_TOLERANCE and ab.max() >= -AREA_TOLERANCE and
                    cd.min() <= AREA_TOLERANCE and cd.max() >= -AREA_TOLERANCE):
                return False
    return True


def signed_areas(points):
    return cross2(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])


def near_straight(a, b, tolerance):
    dot = float(a @ b)
    if dot >= 0:
        return False
    cross = np.cross(a, b)
    if np.all(cross == 0):
        return True
    if tolerance == 0 or -dot < MIN_STRAIGHT_DOT * np.linalg.norm(a) * np.linalg.norm(b):
        return False
    return np.linalg.norm(cross) <= tolerance * np.linalg.norm(b - a)


def describe_region(group, section, vertices, faces, lengths, geometric, shading, incident, outside,
                    max_boundary_deviation, normal_tolerance, plane_relative_tolerance):
    selected = list(group)
    indexed = faces[selected]
    try:
        boundary = np.asarray(boundary_loop(indexed), dtype=np.intp)
    except ValueError:
        return None
    members = set(group)
    interior = set(indexed.ravel()) - set(boundary)
    if any(index in outside or incident[index] - members for index in interior):
        return None

    positions = section.positions[selected]
    largest = int(lengths[selected].argmax())
    axis = geometric[selected[largest]]
    normal = shading[selected[largest], 0]
    geometric_normal_error = float(np.max(np.abs(geometric[selected] - axis)))
    shading_normal_error = float(np.max(np.abs(shading[selected] - normal)))
    if max(geometric_normal_error, shading_normal_error) > normal_tolerance:
        return None
    extent = float(np.ptp(positions, axis=(0, 1)).max())
    delta = positions - positions[largest, 0]
    error = float(np.max(np.abs(delta @ axis)))
    if error > extent * plane_relative_tolerance:
        return None
    tangent = delta[largest, 1]
    tangent = tangent / np.linalg.norm(tangent)
    basis = np.column_stack((tangent, np.cross(axis, tangent)))
    projected = delta @ basis
    base = projected.min(axis=(0, 1))
    span = projected.max(axis=(0, 1)) - base
    if np.any(span <= 0):
        return None
    projected = (projected - base) / span
    positions_on_boundary = vertices[boundary]
    boundary_uv = ((positions_on_boundary - positions[largest, 0]) @ basis - base) / span
    area = cross2(boundary_uv, np.roll(boundary_uv, -1, axis=0)).sum()
    areas = signed_areas(projected)
    if (not np.all(areas > 0) or abs(areas.sum() - area) > AREA_TOLERANCE or
            not simple_polygon(boundary_uv)):
        return None

    boundary_edges = list(zip(boundary, np.roll(boundary, -1)))
    edge_refs = {}
    for face, corners in enumerate(indexed):
        for first in range(3):
            last = (first + 1) % 3
            edge_refs[corners[first], corners[last]] = (face, first, last)
    texture_boundary = tuple(edge_refs[edge] for edge in boundary_edges)
    knots = {}
    for i, center in enumerate(boundary):
        if center in outside:
            continue
        first, last = boundary[i - 1], boundary[(i + 1) % len(boundary)]
        a, b = vertices[first] - vertices[center], vertices[last] - vertices[center]
        if near_straight(a, b, max_boundary_deviation):
            knots[center] = (first, center, last)
    if knots or interior:
        return RegionLayout(group, projected, normal, error, texture_boundary, boundary, boundary_uv, knots,
                            geometric_normal_error=geometric_normal_error, shading_normal_error=shading_normal_error)
    return None


def coupled_knots(layouts, incident):
    owners = {face: key for key, layout in layouts.items() for face in layout.faces}
    knots = {index for layout in layouts.values() for index in layout.straight_knots}
    removable = set()
    for index in knots:
        touching = {owners.get(face) for face in incident[index]}
        if None in touching or len(touching) > 2:
            continue
        chains = [layouts[key].straight_knots.get(index) for key in touching]
        if chains[0] is not None and (len(chains) == 1 or chains[1] == chains[0][::-1]):
            removable.add(index)
    return removable


def retained_boundary(layout, removable):
    return [i for i, index in enumerate(layout.boundary) if index not in removable]


def boundary_chords(layout, keep, vertices):
    """Measure the entire old subchain against each replacement chord."""
    size = len(layout.boundary)
    for first, last in zip(keep, keep[1:] + [keep[0] + size]):
        if last - first == 1:
            continue
        chain = layout.boundary[np.arange(first, last + 1) % size]
        offsets = vertices[chain] - vertices[chain[0]]
        chord = offsets[-1]
        along = offsets @ chord / (chord @ chord)
        distances = np.linalg.norm(offsets[1:-1] - along[1:-1, None] * chord, axis=1)
        yield chain[1:-1], distances, bool(np.all(np.diff(along) > 0))


def bound_chords(layouts, vertices, removable, tolerance):
    """Restore shared knots together when consecutive deletions exceed the bound."""
    while removable:
        restore = set()
        for layout in layouts.values():
            keep = retained_boundary(layout, removable)
            if len(keep) < 3 or not simple_polygon(layout.boundary_uv[keep]):
                for index in layout.boundary:
                    if index in removable:
                        first, center, last = layout.straight_knots[index]
                        if not near_straight(vertices[first] - vertices[center],
                                             vertices[last] - vertices[center], 0.0):
                            restore.add(index)
                continue
            for knots, distances, monotonic in boundary_chords(layout, keep, vertices):
                index = int(distances.argmax())
                if distances[index] > tolerance or not monotonic:
                    restore.add(knots[index])
        if not restore:
            break
        removable.difference_update(restore)


def select_target(plans, removable, reduction):
    if reduction <= 0:
        return []
    owners = defaultdict(list)
    neighbors = [set() for _ in plans]
    for index, (layout, _, _, _) in enumerate(plans):
        for knot in layout.boundary:
            if knot in removable:
                owners[knot].append(index)
    for matches in owners.values():
        if len(matches) == 2:
            first, second = matches
            neighbors[first].add(second)
            neighbors[second].add(first)

    groups, seen = [], set()
    for seed in range(len(plans)):
        if seed in seen:
            continue
        group = [seed]
        seen.add(seed)
        for index in group:
            pending = neighbors[index] - seen
            group.extend(sorted(pending))
            seen.update(pending)
        gains = sum(len(plans[index][0].faces) - len(plans[index][3]) for index in group)
        deviation = max(plans[index][0].boundary_error for index in group)
        groups.append((deviation, gains, group))

    # Prefer less geometric change, and keep paired boundaries as one edit.
    # Track reachable reductions so a combination that meets the target exactly
    # can win over a group that overshoots it.
    groups.sort(key=lambda item: (item[0], -item[1], item[2]))
    if sum(gains for _, gains, _ in groups) <= reduction:
        return plans
    maximum = reduction + max(gains for _, gains, _ in groups) - 1
    mask = (1 << (maximum + 1)) - 1
    reachable = 1
    target = 1 << reduction
    source_group = {}
    for index, (_, gains, _) in enumerate(groups):
        new = ((reachable << gains) & mask) & ~reachable
        reachable |= new
        while new:
            bit = new & -new
            source_group[bit.bit_length() - 1] = index
            new ^= bit
        if reachable & target:
            break
    eligible = reachable >> reduction
    best = reduction + (eligible & -eligible).bit_length() - 1
    selected = set()
    while best:
        _, gains, group = groups[source_group[best]]
        selected.update(group)
        best -= gains
    return [plan for index, plan in enumerate(plans) if index in selected]


def plan_uv_regions(section, *, outside_positions=(), max_boundary_deviation=0.0, target_triangles=None,
                    normal_tolerance=NORMAL_TOLERANCE, plane_relative_tolerance=PLANE_RELATIVE_TOLERANCE):
    """Retriangulate flat regions, keeping contacts or editing both sides together.

    A positive boundary tolerance permits near-straight knots within that distance
    of the replacement edge in scene units. The normal tolerance bounds component
    changes between unit normals. Plane tolerance is relative to each chart's extent.
    A target is an upper triangle count.
    """
    if not np.isfinite(max_boundary_deviation) or max_boundary_deviation < 0:
        raise ValueError("Boundary deviation must be finite and nonnegative")
    if not np.isfinite(normal_tolerance) or normal_tolerance < 0:
        raise ValueError("Normal tolerance must be finite and nonnegative")
    if not np.isfinite(plane_relative_tolerance) or plane_relative_tolerance < 0:
        raise ValueError("Relative plane tolerance must be finite and nonnegative")
    if target_triangles is not None and (not isinstance(target_triangles, (int, np.integer)) or target_triangles < 0):
        raise ValueError("Target triangle count must be a nonnegative integer")
    if target_triangles is not None:
        target_triangles = int(target_triangles)
    positions = section.positions
    vertices, mapping = np.unique(positions.reshape(-1, 3), axis=0, return_inverse=True)
    faces = mapping.reshape(-1, 3)
    outside_positions = {tuple(position) for position in outside_positions}
    outside = {index for index, point in enumerate(vertices) if tuple(point) in outside_positions}
    incident = defaultdict(set)
    for face, corners in enumerate(faces):
        for vertex in corners:
            incident[vertex].add(face)

    cross = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    geometric = cross / np.where(lengths > 0, lengths, 1)[:, None]
    shading_lengths = np.linalg.norm(section.normals, axis=2)
    shading = section.normals / np.where(shading_lengths > 0, shading_lengths, 1)[..., None]
    layouts = {}
    for group in planar_regions(faces, geometric, lengths, shading, shading_lengths, normal_tolerance):
        layout = describe_region(group, section, vertices, faces, lengths, geometric, shading, incident, outside,
                                 max_boundary_deviation, normal_tolerance, plane_relative_tolerance)
        if layout is not None:
            layouts[group[0]] = layout
    candidates = [layout.faces for layout in layouts.values()]

    while layouts:
        removable = coupled_knots(layouts, incident)
        if max_boundary_deviation > 0:
            bound_chords(layouts, vertices, removable, max_boundary_deviation)
        accepted, blocked = [], []
        for key, layout in layouts.items():
            keep = retained_boundary(layout, removable)
            projected = layout.boundary_uv[keep]
            triangles = triangulate_boundary(projected) if len(projected) >= 3 else None
            if max_boundary_deviation > 0:
                if not simple_polygon(projected):
                    blocked.append(key)
                    continue
                area = cross2(projected, np.roll(projected, -1, axis=0)).sum()
            else:
                area = signed_areas(layout.projected).sum()
            if (triangles is None or len(triangles) >= len(layout.faces) or
                    abs(signed_areas(projected[triangles]).sum() - area) > AREA_TOLERANCE):
                blocked.append(key)
                continue
            retained = vertices[layout.boundary[keep]]
            if max_boundary_deviation > 0:
                layout.boundary_error = max((float(distances.max())
                                             for _, distances, _ in boundary_chords(layout, keep, vertices)),
                                            default=0.0)
            accepted.append((layout, retained, projected * [1, -1] + [0, 1], triangles))
        if not blocked:
            if target_triangles is not None:
                accepted = select_target(accepted, removable, len(faces) - target_triangles)
            return candidates, accepted
        for key in blocked:
            del layouts[key]
    return candidates, []
