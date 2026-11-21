"""Retriangulate rectangles without dropping boundary points used by neighboring faces."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from bake_planar_patch import boundary_loop, cross2, rectangular_domain


@dataclass
class PatchLayout:
    faces: tuple
    projected: np.ndarray
    normal: np.ndarray
    plane_error: float
    boundary: tuple
    chains: dict
    rectangle: np.ndarray

    def domain_coordinates(self, points):
        a = self.rectangle[0]
        u, v = self.rectangle[1] - a, self.rectangle[3] - a
        delta = points - a
        return np.column_stack((delta @ u / (u @ u), delta @ v / (v @ v)))


def describe_patch(section, faces):
    positions, normals = section.positions[list(faces)], section.normals[list(faces)]
    rectangle, projected, normal, error = rectangular_domain(positions, normals)
    unique, inverse = np.unique(positions.reshape(-1, 3), axis=0, return_inverse=True)
    perimeter = tuple(tuple(unique[index]) for index in boundary_loop(inverse.reshape(-1, 3)))
    corners = {tuple(vertex) for vertex in rectangle}
    start = next(i for i, point in enumerate(perimeter) if point in corners)
    perimeter = perimeter[start:] + perimeter[:start]
    chains = {}
    chain = [perimeter[0]]
    for point in (*perimeter[1:], perimeter[0]):
        chain.append(point)
        if point in corners:
            # Only remove exactly straight points from the stored scene-space boundary.
            first, last = np.asarray(chain[0]), np.asarray(chain[-1])
            if all(np.all(np.cross(last - first, np.asarray(knot) - first) == 0) for knot in chain[1:-1]):
                chains.update((knot, tuple(chain)) for knot in chain[1:-1])
            chain = [point]
    return PatchLayout(faces, projected, normal, error, perimeter, chains, rectangle)


def triangulate_boundary(points):
    active = list(range(len(points)))
    result = []
    tolerance = 1e-10
    while len(active) > 3:
        for i, center in enumerate(active):
            first, last = active[i - 1], active[(i + 1) % len(active)]
            a, b, c = points[[first, center, last]]
            if cross2(b - a, c - a) <= tolerance:
                continue
            rest = points[[index for index in active if index not in (first, center, last)]]
            inside = ((cross2(b - a, rest - a) >= -tolerance) &
                      (cross2(c - b, rest - b) >= -tolerance) &
                      (cross2(a - c, rest - c) >= -tolerance))
            if inside.any():
                continue
            result.append((first, center, last))
            del active[i]
            break
        else:
            return None
    a, b, c = points[active]
    if cross2(b - a, c - a) <= tolerance:
        return None
    result.append(tuple(active))
    return np.asarray(result, dtype=np.intp)


def protect_contacts(section, patches):
    """Return reductions with outside references retained or matching chains edited together."""
    incident = defaultdict(set)
    for face, positions in enumerate(section.positions):
        for position in positions:
            incident[tuple(position)].add(face)

    layouts = {i: describe_patch(section, tuple(patch)) for i, patch in enumerate(patches)}
    while layouts:
        owners = {face: i for i, layout in layouts.items() for face in layout.faces}
        reductions = []
        blocked = []
        for i, layout in layouts.items():
            def remove(point):
                chain = layout.chains.get(point)
                if chain is None:
                    return False
                touching = {owners.get(face) for face in incident[point]} - {i}
                if not touching:
                    return True
                if len(touching) != 1 or None in touching:
                    return False
                other = layouts[next(iter(touching))]
                return other.chains.get(point) == chain[::-1]

            # An interior point used by outside faces would need constrained interior triangulation.
            interior = {tuple(vertex) for vertex in section.positions[list(layout.faces)].reshape(-1, 3)}
            interior.difference_update(layout.boundary)
            if any(incident[point] - set(layout.faces) for point in interior):
                blocked.append(i)
                continue
            positions = np.array([point for point in layout.boundary if not remove(point)])
            domain = layout.domain_coordinates(positions)
            triangles = triangulate_boundary(domain)
            if triangles is None or len(triangles) >= len(layout.faces):
                blocked.append(i)
                continue
            reductions.append((layout, positions, domain * [1, -1] + [0, 1], triangles))
        if not blocked:
            return reductions
        for i in blocked:
            del layouts[i]
    return []
