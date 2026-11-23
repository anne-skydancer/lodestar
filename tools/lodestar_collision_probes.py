"""Geometric clearance probes for static triangle collision meshes."""

import math

import numpy as np


def _dot(first, second):
    return np.einsum("...i,...i->...", first, second)


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)


def _point_segment_distance_sq(point, start, end):
    direction = end - start
    offset = point - start
    parameter = np.clip(_ratio(_dot(offset, direction), _dot(direction, direction)), 0, 1)
    difference = offset - parameter[..., None] * direction
    return _dot(difference, difference)


def _inside_triangle_projection(point, corners, normal):
    # The edge tests are unaffected by the point's offset along the triangle normal.
    return np.logical_and.reduce([
        _dot(np.cross(end - start, point - start), normal) >= 0
        for start, end in zip(corners, (*corners[1:], corners[0]))
    ])


def _point_triangle_distance_sq(point, corners, normal):
    edge_distance = np.minimum.reduce([
        _point_segment_distance_sq(point, start, end)
        for start, end in zip(corners, (*corners[1:], corners[0]))
    ])
    area_sq = _dot(normal, normal)
    normal_offset = _dot(point - corners[0], normal)
    face_distance = _ratio(normal_offset ** 2, area_sq)
    use_face = (area_sq > 0) & _inside_triangle_projection(point, corners, normal)
    return np.where(use_face, np.minimum(edge_distance, face_distance), edge_distance)


def _edge_intersects_triangle(start, end, corners, normal):
    start_offset = _dot(start - corners[0], normal)
    end_offset = _dot(end - corners[0], normal)
    difference = start_offset - end_offset
    parameter = _ratio(start_offset, difference)
    crossing = (difference != 0) & (parameter >= 0) & (parameter <= 1)
    intersection = start + np.where(crossing, parameter, 0)[..., None] * (end - start)
    return crossing & _inside_triangle_projection(intersection, corners, normal)


def _edge_interior_distance_sq(start_a, end_a, start_b, end_b):
    # Endpoint-to-edge distances are covered by the point-to-triangle checks.
    # The additional candidate is where the closest points lie inside both edges.
    axis_a = end_a - start_a
    axis_b = end_b - start_b
    offset = start_b - start_a
    normal = np.cross(axis_a, axis_b)
    normal_sq = _dot(normal, normal)
    parameter_a = _ratio(_dot(np.cross(offset, axis_b), normal), normal_sq)
    parameter_b = _ratio(_dot(np.cross(offset, axis_a), normal), normal_sq)
    inside = (normal_sq > 0) & (parameter_a >= 0) & (parameter_a <= 1) & (parameter_b >= 0) & (parameter_b <= 1)
    distance = _ratio(_dot(offset, normal) ** 2, normal_sq)
    return np.where(inside, distance, np.inf)


def _distance_to_triangles_sq(patch, triangles):
    corners = tuple(np.moveaxis(triangles, 1, 0))
    patch_normal = np.cross(patch[1] - patch[0], patch[2] - patch[0])
    normals = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    distance = np.full(len(triangles), np.inf)
    crosses = np.zeros(len(triangles), dtype=bool)
    for corner in range(3):
        next_corner = (corner + 1) % 3
        distance = np.minimum(distance, _point_triangle_distance_sq(patch[corner], corners, normals))
        distance = np.minimum(distance, _point_triangle_distance_sq(corners[corner], patch, patch_normal))
        crosses |= _edge_intersects_triangle(patch[corner], patch[next_corner], corners, normals)
        crosses |= _edge_intersects_triangle(corners[corner], corners[next_corner], patch, patch_normal)
        for other_corner in range(3):
            distance = np.minimum(distance, _edge_interior_distance_sq(
                patch[corner], patch[next_corner], corners[other_corner], corners[(other_corner + 1) % 3]
            ))
    return np.where(crosses, 0.0, distance)


def sweep_clearance(triangles, start, end, radius, height):
    """Return minimum surface distance to an upright swept capsule axis minus its radius.

    Coordinates are Y-up. Start and end locate the capsule's feet and height includes
    both end caps. A positive result is clearance, zero is contact, and a negative
    result is overlap with a triangle surface. An empty triangle mesh returns infinity.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or start.shape != (3,) or end.shape != (3,):
        raise ValueError("Expected triangles shaped (N, 3, 3) and XYZ start and end positions")
    if not (np.isfinite(triangles).all() and np.isfinite(start).all() and np.isfinite(end).all()):
        raise ValueError("Collision probe positions must be finite")
    if not math.isfinite(radius) or not math.isfinite(height) or radius < 0 or height < 2 * radius:
        raise ValueError("Capsule height must be at least twice its nonnegative radius")
    if not len(triangles):
        return math.inf

    lower = np.array([0, radius, 0])
    upper = np.array([0, height - radius, 0])
    strip = (start + lower, end + lower, end + upper, start + upper)
    # The swept axis is a parallelogram, including point and segment degeneracies.
    distance_sq = min(
        _distance_to_triangles_sq((strip[0], strip[1], strip[2]), triangles).min(),
        _distance_to_triangles_sq((strip[0], strip[2], strip[3]), triangles).min(),
    )
    return math.sqrt(distance_sq) - radius
