"""Generate and validate geometry-only nonplanar simplification candidates."""

from collections import defaultdict
import ctypes as ct
from dataclasses import dataclass
import heapq
from pathlib import Path

import numpy as np

from lodestar_collision_probes import _point_triangle_distance_sq


_SAMPLE_WEIGHTS = np.asarray([(i, j, 4 - i - j) for i in range(5) for j in range(5 - i)], dtype=float) / 4


@dataclass(frozen=True)
class NonplanarResult:
    positions: np.ndarray
    source_components: np.ndarray
    target_components: np.ndarray
    components: list[dict]


def _edge_uses(faces):
    edges = defaultdict(list)
    for index, (a, b, c) in enumerate(faces.tolist()):
        for start, end in ((a, b), (b, c), (c, a)):
            edges[min(start, end), max(start, end)].append((index, start, end))
    return edges


def _face_components(count, edges):
    neighbors = [[] for _ in range(count)]
    for occurrences in edges.values():
        anchor = occurrences[0][0]
        for other, _, _ in occurrences[1:]:
            neighbors[anchor].append(other)
            neighbors[other].append(anchor)
    seen = set()
    result = []
    for seed in range(count):
        if seed in seen:
            continue
        group = [seed]
        seen.add(seed)
        for face in group:
            for neighbor in neighbors[face]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    group.append(neighbor)
        result.append(np.asarray(sorted(group), dtype=np.intp))
    return result


def _topology(points, faces):
    if len(faces) == 0:
        return None
    triangles = points[faces]
    if np.any(np.all(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) == 0, axis=1)):
        return None
    rows = faces.tolist()
    if len({tuple(sorted(face)) for face in rows}) != len(rows):
        return None
    edges = _edge_uses(faces)
    boundary = {}
    incoming = set()
    for occurrences in edges.values():
        _, start, end = occurrences[0]
        if len(occurrences) == 1:
            if start in boundary or end in incoming:
                return None
            boundary[start] = end
            incoming.add(end)
        elif len(occurrences) != 2 or occurrences[1][1:] != (end, start):
            return None
    if set(boundary) != incoming or len(_face_components(len(faces), edges)) != 1:
        return None

    # Each vertex link must form one path on a boundary or one cycle inside.
    links = defaultdict(lambda: defaultdict(list))
    for a, b, c in rows:
        for vertex, previous, following in ((a, b, c), (b, c, a), (c, a, b)):
            links[vertex][previous].append(following)
            links[vertex][following].append(previous)
    for vertex, graph in links.items():
        degrees = [len(neighbors) for neighbors in graph.values()]
        if max(degrees) > 2 or degrees.count(1) != (2 if vertex in boundary else 0):
            return None
        first = next(iter(graph))
        visited = {first}
        pending = [first]
        for current in pending:
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        if len(visited) != len(graph):
            return None

    loops = 0
    visited = set()
    for start in boundary:
        if start not in visited:
            loops += 1
            while start not in visited:
                visited.add(start)
                start = boundary[start]
    return len(links) - len(edges) + len(rows), loops, frozenset(boundary.items())


def _bind(library):
    native = ct.CDLL(str(Path(library).resolve()))
    vertices = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
    faces = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags="C_CONTIGUOUS")
    locks = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    native.lodestar_last_error.argtypes = []
    native.lodestar_last_error.restype = ct.c_char_p
    try:
        simplify = native.lodestar_simplify_geometry
    except AttributeError as error:
        raise ValueError("The Lodestar C library needs to be rebuilt with nonplanar geometry support") from error
    simplify.argtypes = [vertices, ct.c_size_t, faces, ct.c_size_t, locks, ct.c_size_t, ct.c_float,
                         ct.c_uint32, faces, ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_float)]
    simplify.restype = ct.c_uint32

    def run(points, triangles, pinned, target, limit, lock_boundary):
        output = np.empty_like(triangles)
        count = ct.c_size_t()
        error = ct.c_float()
        status = simplify(points, len(points), triangles, triangles.size, pinned, target, limit, lock_boundary,
                          output, ct.byref(count), ct.byref(error))
        if status:
            raise ValueError(native.lodestar_last_error().decode("utf-8", errors="replace"))
        return output[:count.value // 3].copy(), float(error.value)

    return run


def _directed_sampled_distance(source, target):
    samples = np.unique((_SAMPLE_WEIGHTS @ source).reshape(-1, 3), axis=0)
    corners = tuple(target[None, :, i] for i in range(3))
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    largest = 0.0
    for offset in range(0, len(samples), 64):
        distances = _point_triangle_distance_sq(samples[offset:offset + 64, None, :], corners, normal)
        largest = max(largest, float(distances.min(axis=1).max()))
    return np.sqrt(largest)


def _sampled_distance(before, after):
    return max(_directed_sampled_distance(before, after), _directed_sampled_distance(after, before))


def _options(run, points, faces, pinned, extent, limit, lock_boundary, sampled_limit):
    topology = _topology(points, faces)
    base = [(faces, 0.0)]
    if topology is None or len(faces) <= 1:
        return base, topology

    # Constrain the native error using the whole mesh's scale while translating
    # each component to preserve precision in the float position buffer.
    normalized = np.ascontiguousarray((points - points[0]) / extent, dtype=np.float32)
    trials = np.unique(np.linspace(1, len(faces) - 1, min(32, len(faces) - 1), dtype=np.intp))
    seen = {faces.tobytes()}
    pinned_ids = np.flatnonzero(pinned)
    best = {}
    for target in trials:
        result, error = run(normalized, faces, pinned, int(target), limit, lock_boundary)
        key = result.tobytes()
        if len(result) >= len(faces) or key in seen:
            continue
        seen.add(key)
        if not np.isfinite(error) or error > limit + 1e-7 or not np.isin(pinned_ids, result).all():
            continue
        signature = _topology(points, result)
        if signature is None or signature[:2] != topology[:2] or (lock_boundary and signature[2] != topology[2]):
            continue
        if sampled_limit is not None and _sampled_distance(points[faces], points[result]) > extent * sampled_limit:
            continue
        current = best.get(len(result))
        if current is None or current[1] > error:
            best[len(result)] = (result, error)

    frontier = []
    lowest = np.inf
    for count, entry in sorted(best.items()):
        if entry[1] < lowest:
            frontier.append(entry)
            lowest = entry[1]
    return [*base, *reversed(frontier)], topology


def _allocate_budget(options, source_size, target):
    chosen = [0] * len(options)
    queue = []

    def offer(component, current):
        if current + 1 == len(options[component]):
            return
        before, old_error = options[component][current]
        after, new_error = options[component][current + 1]
        saved = len(before) - len(after)
        heapq.heappush(queue, ((new_error - old_error) / saved, new_error, component, saved))

    for index in range(len(options)):
        offer(index, 0)
    while source_size > target and queue:
        _, _, index, saved = heapq.heappop(queue)
        chosen[index] += 1
        source_size -= saved
        offer(index, chosen[index])
    return chosen


def simplify_nonplanar(library, positions, target_triangles, *, max_relative_error=0.02,
                       outside_positions=(), lock_boundary=False, max_sampled_relative_error=None):
    """Simplify position-connected components and retain rejected components.

    The error limit is meshoptimizer's estimate relative to the largest input
    axis extent. An optional second limit screens sampled distances in both
    directions within each component. The output positions only use input
    vertices. Measure surface and silhouette quality independently before use.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if (positions.ndim != 3 or positions.shape[1:] != (3, 3) or not len(positions) or
            not np.isfinite(positions).all() or not isinstance(target_triangles, (int, np.integer)) or
            target_triangles < 1 or not 0 <= max_relative_error <= 1):
        raise ValueError("Expected finite triangle positions and a positive triangle budget with a relative error in [0, 1]")
    if max_sampled_relative_error is not None and not 0 <= max_sampled_relative_error <= 1:
        raise ValueError("Expected a maximum sampled relative error in [0, 1]")
    points, mapping = np.unique(positions.reshape(-1, 3), axis=0, return_inverse=True)
    faces = mapping.reshape(-1, 3)
    extent = np.ptp(points, axis=0).max()
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Expected a finite nonzero geometry extent")

    groups = _face_components(len(faces), _edge_uses(faces))
    group_vertices = [np.unique(faces[group]) for group in groups]
    references = np.bincount(np.concatenate(group_vertices), minlength=len(points))
    outside = {tuple(position) for position in outside_positions}
    pinned = np.array([ref > 1 or tuple(point) in outside for point, ref in zip(points, references)], dtype=np.uint8)
    solver = _bind(library)
    options = []
    topologies = []
    source_groups = np.empty(len(faces), dtype=np.intp)
    for identifier, (group, vertices) in enumerate(zip(groups, group_vertices)):
        source_groups[group] = identifier
        local = np.ascontiguousarray(np.searchsorted(vertices, faces[group]), dtype=np.uint32)
        candidates, topology = _options(solver, points[vertices], local, pinned[vertices], extent,
                                        max_relative_error, lock_boundary, max_sampled_relative_error)
        options.append(candidates)
        topologies.append(topology)

    selected = _allocate_budget(options, len(faces), target_triangles)
    chunks = []
    reports = []
    target_groups = []
    for index, choice in enumerate(selected):
        triangles, estimate = options[index][choice]
        topology = topologies[index]
        chunks.append(points[group_vertices[index][triangles]])
        target_groups.extend([index] * len(triangles))
        reports.append({
            "source_triangles": len(groups[index]), "candidate_triangles": len(triangles),
            "estimated_relative_error": estimate, "supported": topology is not None,
            "euler_characteristic": topology[0] if topology is not None else None,
            "boundary_loops": topology[1] if topology is not None else None,
        })
        if max_sampled_relative_error is not None:
            reports[-1]["sampled_relative_error"] = float(_sampled_distance(
                points[group_vertices[index][options[index][0][0]]], chunks[-1]) / extent) if choice else 0.0
    if not any(selected):
        return NonplanarResult(positions.copy(), source_groups, source_groups.copy(), reports)
    return NonplanarResult(np.concatenate(chunks), source_groups, np.asarray(target_groups), reports)
