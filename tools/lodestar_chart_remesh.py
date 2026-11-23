"""Lodestar chart remeshing with protected boundaries and High correspondence.

Remove interior vertex fans and triangulate their UV cavities. The integral
budget supplies the target. Every accepted cavity is checked against High's
position and normal fields before it becomes part of a lower LOD.
"""

import heapq

import numpy as np

from bake_planar_patch import boundary_loop, cross2
from lodestar_patch_layout import triangulate_boundary


def remesh_chart(chart, starting_faces, target, surface_limit, normal_limit, locks, bounder):
    faces = {index: tuple(map(int, face)) for index, face in enumerate(starting_faces)}
    next_face = len(faces)
    incident = [set() for _ in chart.positions]
    for index, face in faces.items():
        for vertex in face:
            incident[vertex].add(index)
    protected = set(boundary_loop(np.asarray(starting_faces))) | set(np.flatnonzero(locks))
    versions = np.zeros(len(incident), dtype=np.int64)
    heap = []
    tested = removed = 0

    def proposal(vertex):
        if vertex in protected or not 3 <= len(incident[vertex]) <= 128:
            return None
        old_ids = sorted(incident[vertex])
        old = np.asarray([faces[index] for index in old_ids])
        try:
            ring = np.asarray(boundary_loop(old), dtype=np.intp)
        except ValueError:
            return None
        if vertex in ring:
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
        # A cheap ordering estimate. Acceptance uses the complete UV overlay.
        uv = chart.uv[replacement]
        first, second = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        delta = chart.uv[vertex] - uv[:, 0]
        denominator = cross2(first, second)
        u, v = cross2(delta, second) / denominator, cross2(first, delta) / denominator
        inside = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
        if not inside.any():
            return None
        index = int(np.flatnonzero(inside)[0])
        weights = np.array((1 - u[index] - v[index], u[index], v[index]))
        position_error = np.linalg.norm(weights @ chart.positions[replacement[index]] - chart.positions[vertex])
        normal_error = np.linalg.norm(weights @ chart.normals[replacement[index]] - chart.normals[vertex])
        cost = float(position_error / max(surface_limit, 1e-30) + normal_error / max(np.deg2rad(normal_limit), 1e-12))
        return cost, old_ids, replacement, ring

    def enqueue(vertex):
        candidate = proposal(vertex)
        if candidate is not None:
            heapq.heappush(heap, (candidate[0], vertex, int(versions[vertex])))

    for vertex in range(len(incident)):
        enqueue(vertex)
    while heap and len(faces) > target and tested < 200000:
        _, vertex, version = heapq.heappop(heap)
        if version != versions[vertex]:
            continue
        candidate = proposal(vertex)
        if candidate is None:
            continue
        _, old_ids, replacement, ring = candidate
        if len(faces) - len(old_ids) + len(replacement) < target:
            continue
        tested += 1
        try:
            # Check each new triangle's complete domain against the original
            # High triangulation. Previously accepted triangles stay valid.
            bound = bounder(chart.uv, chart.positions, chart.normals, replacement, chart.faces)
        except ValueError:
            continue
        if bound["surface_bound"] > surface_limit or bound["normal_angle_bound_degrees"] > normal_limit:
            continue
        for index in old_ids:
            for corner in faces[index]:
                incident[corner].remove(index)
            del faces[index]
        for triangle in replacement:
            faces[next_face] = tuple(map(int, triangle))
            for corner in triangle:
                incident[corner].add(next_face)
            next_face += 1
        removed += 1
        versions[vertex] += 1
        for corner in ring:
            versions[corner] += 1
            enqueue(int(corner))
    return np.asarray(list(faces.values()), dtype=np.intp), {
        "remesher": "lodestar_uv_cavity", "removed_interior_vertices": removed,
        "tested_cavities": tested, "work_limit_reached": tested >= 200000}
