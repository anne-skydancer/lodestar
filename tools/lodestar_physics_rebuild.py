"""Rebuild exact High-derived physics and diagnose uploader representation limits."""

import ctypes as ct
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lodestar_firestorm_collision import _exact_collinear, _exact_coplanar, _flip_edges, _geometry
from lodestar_firestorm_degenerate import physics_degenerate_mask, packed_physics_degenerate_mask
from lodestar_uv_regions import plan_uv_regions
from lodestar_physics_strips import exact_strip_repair, exact_edge_refinement


class _Vertex(ct.Structure):
    _fields_ = [("position", ct.c_float * 3)]


class _View(ct.Structure):
    _fields_ = [("positions", ct.POINTER(_Vertex)), ("position_count", ct.c_size_t),
                ("indices", ct.POINTER(ct.c_uint32)), ("index_count", ct.c_size_t),
                ("skipped_degenerate_triangles", ct.c_size_t)]


class _CoverInfo(ct.Structure):
    _fields_ = [('surface_error_bound', ct.c_double), ('exact_solid_conversion', ct.c_uint32)]


class HullProbe:
    def __init__(self, library):
        self.api = ct.CDLL(str(Path(library).resolve()))
        signatures = {
            "lodestar_last_error": ([], ct.c_char_p),
            "lodestar_decompose_collision_hulls": ([ct.POINTER(_Vertex), ct.c_size_t,
                ct.POINTER(ct.c_uint32), ct.c_size_t, ct.c_size_t, ct.POINTER(ct.c_void_p)], ct.c_uint32),
            "lodestar_convex_compound_get_hull_count": ([ct.c_void_p, ct.POINTER(ct.c_size_t)], ct.c_uint32),
            "lodestar_convex_compound_get_hull_view": ([ct.c_void_p, ct.c_size_t, ct.POINTER(_View)], ct.c_uint32),
            "lodestar_convex_compound_destroy": ([ct.c_void_p], None)}
        for name, (arguments, result) in signatures.items():
            getattr(self.api, name).argtypes = arguments
            getattr(self.api, name).restype = result

    def envelope(self, points):
        """Construct a native point hull; callers must certify conformity."""
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError('A convex envelope needs a 3D point array')
        self.api.lodestar_build_convex_hull.argtypes = [ct.POINTER(_Vertex), ct.c_size_t, ct.POINTER(ct.c_void_p)]
        self.api.lodestar_build_convex_hull.restype = ct.c_uint32
        self.api.lodestar_collision_mesh_get_view.argtypes = [ct.c_void_p, ct.POINTER(_View)]
        self.api.lodestar_collision_mesh_get_view.restype = ct.c_uint32
        self.api.lodestar_collision_mesh_destroy.argtypes = [ct.c_void_p]
        self.api.lodestar_collision_mesh_destroy.restype = None
        handle = ct.c_void_p()
        try:
            status = self.api.lodestar_build_convex_hull(points.ctypes.data_as(ct.POINTER(_Vertex)), len(points), ct.byref(handle))
            if status:
                raise ValueError(self.api.lodestar_last_error().decode('utf-8', errors='replace'))
            view = _View()
            if self.api.lodestar_collision_mesh_get_view(handle, ct.byref(view)):
                raise ValueError('Cannot inspect a point envelope')
            vertices = np.ctypeslib.as_array(ct.cast(view.positions, ct.POINTER(ct.c_float)),
                                            shape=(view.position_count * 3,)).reshape(-1, 3).copy().astype(float)
            indices = np.ctypeslib.as_array(view.indices, shape=(view.index_count,)).reshape(-1, 3).copy()
            return vertices[indices]
        finally:
            if handle:
                self.api.lodestar_collision_mesh_destroy(handle)

    def coherent(self, triangles, maximum_error, maximum_hulls=256):
        points, indices = np.unique(np.asarray(triangles, dtype=np.float32).reshape(-1, 3), axis=0, return_inverse=True)
        points = np.ascontiguousarray(points)
        indices = np.ascontiguousarray(indices, dtype=np.uint32)
        convert = self.api.lodestar_build_coherent_collision
        convert.argtypes = [ct.POINTER(_Vertex), ct.c_size_t, ct.POINTER(ct.c_uint32), ct.c_size_t,
                            ct.c_size_t, ct.c_double, ct.POINTER(ct.c_void_p), ct.POINTER(_CoverInfo)]
        convert.restype = ct.c_uint32
        handle = ct.c_void_p()
        info = _CoverInfo()
        try:
            status = convert(points.ctypes.data_as(ct.POINTER(_Vertex)), len(points),
                indices.ctypes.data_as(ct.POINTER(ct.c_uint32)), len(indices), maximum_hulls, maximum_error,
                ct.byref(handle), ct.byref(info))
            if status:
                raise ValueError(self.api.lodestar_last_error().decode('utf-8', errors='replace'))
            count = ct.c_size_t()
            if self.api.lodestar_convex_compound_get_hull_count(handle, ct.byref(count)):
                raise ValueError('Cannot inspect coherent hull count')
            output = []
            for index in range(count.value):
                view = _View()
                if self.api.lodestar_convex_compound_get_hull_view(handle, index, ct.byref(view)):
                    raise ValueError('Cannot inspect a coherent hull')
                vertices = np.ctypeslib.as_array(ct.cast(view.positions, ct.POINTER(ct.c_float)),
                    shape=(view.position_count * 3,)).reshape(-1, 3).copy().astype(float)
                faces = np.ctypeslib.as_array(view.indices, shape=(view.index_count,)).reshape(-1, 3).copy()
                output.append(vertices[faces])
            return output, {'hulls': count.value, 'surface_error_bound': info.surface_error_bound,
                            'exact_solid_conversion': bool(info.exact_solid_conversion)}
        finally:
            if handle:
                self.api.lodestar_convex_compound_destroy(handle)

    def analyze(self, triangles):
        positions, indices = np.unique(np.asarray(triangles, dtype=np.float32).reshape(-1, 3),
                                       axis=0, return_inverse=True)
        positions = np.ascontiguousarray(positions)
        indices = np.ascontiguousarray(indices, dtype=np.uint32)
        handle = ct.c_void_p()
        try:
            status = self.api.lodestar_decompose_collision_hulls(
                positions.ctypes.data_as(ct.POINTER(_Vertex)), len(positions),
                indices.ctypes.data_as(ct.POINTER(ct.c_uint32)), len(indices), 256, ct.byref(handle))
            if status:
                return {"supported": False, "reason": self.api.lodestar_last_error().decode("utf-8", errors="replace")}
            count = ct.c_size_t()
            if self.api.lodestar_convex_compound_get_hull_count(handle, ct.byref(count)):
                raise ValueError("Cannot inspect the generated collision compound")
            vertices = []
            for index in range(count.value):
                view = _View()
                if self.api.lodestar_convex_compound_get_hull_view(handle, index, ct.byref(view)):
                    raise ValueError("Cannot inspect a generated collision hull")
                vertices.append(view.position_count)
            return {"supported": all(4 <= value <= 256 for value in vertices),
                    "hulls": count.value, "vertices_per_hull": vertices,
                    "geometry": "exact supported solid before uploader position quantization"}
        finally:
            if handle:
                self.api.lodestar_convex_compound_destroy(handle)


def exact_planar_repair(source, outside_positions=()):
    """Retriangulate planar regions across visual attribute seams for collision.

    Floating-point grouping proposes regions. Rational predicates certify their
    planarity and all removed boundary knots before a proposal is used.
    """
    edges = source[:, 1:] - source[:, :1]
    normals = np.cross(edges[:, 0], edges[:, 1])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    shading = np.broadcast_to(normals[:, None], source.shape).copy()
    section = SimpleNamespace(positions=source, normals=shading)
    vertices = np.unique(source.reshape(-1, 3), axis=0)
    rejected = physics_degenerate_mask(source) | packed_physics_degenerate_mask(source)
    while True:
        _, plans = plan_uv_regions(section, outside_positions=outside_positions, max_boundary_deviation=0,
                                   normal_tolerance=1e-10, plane_relative_tolerance=1e-12)
        keep = np.ones(len(source), dtype=bool)
        replacements, blocked = [], []
        for layout, points, _, faces in plans:
            selected = list(layout.faces)
            reference = source[selected[0]]
            planar = all(_exact_coplanar(*reference, point) for point in source[selected].reshape(-1, 3))
            retained = set(map(tuple, points))
            straight = all(tuple(vertices[center]) in retained or _exact_collinear(*vertices[list(chain)])
                           for center, chain in layout.straight_knots.items())
            replacement = points[faces]
            # Normalization uses the complete named High model, including its bounds.
            trial = np.concatenate((source[~np.isin(np.arange(len(source)), selected)], replacement))
            new_bad = (physics_degenerate_mask(trial)[-len(replacement):] |
                       packed_physics_degenerate_mask(trial)[-len(replacement):])
            if not planar or not straight or int(new_bad.sum()) > int(rejected[selected].sum()):
                blocked.extend(selected)
                continue
            keep[selected] = False
            replacements.append(replacement)
        if not blocked:
            result = np.concatenate((source[keep], *replacements)) if replacements else source
            return result, len(replacements)
        # A rejected neighboring region must pin its shared boundary knots again.
        shading[blocked] = 0


def rebuild_physics(parts, library, *, maximum_surface_displacement=0.0):
    """Retain High's complete surface with certified planar collision repairs.

    Reports unresolved uploader triangles instead of dropping High geometry or
    moving boundary points. Convex conversion is probed as an alternative format
    for supported solids. The exact triangle result remains available in all cases.
    """
    probe = HullProbe(library)
    output, reports = {}, {}
    owners = defaultdict(set)
    for name, positions in parts.items():
        for point in set(map(tuple, np.asarray(positions).reshape(-1, 3))):
            owners[point].add(name)
    for name, source in parts.items():
        source = np.asarray(source, dtype=np.float64)
        if source.ndim != 3 or source.shape[1:] != (3, 3) or not len(source) or not np.isfinite(source).all():
            raise ValueError("Physics rebuilding requires finite High triangles")
        area = np.linalg.norm(np.cross(source[:, 1] - source[:, 0], source[:, 2] - source[:, 0]), axis=1)
        source_zero = area == 0
        valid = source[~source_zero]
        if not len(valid):
            raise ValueError(f"High part {name} has no nondegenerate physical surface")
        before = int(np.count_nonzero(physics_degenerate_mask(valid) | packed_physics_degenerate_mask(valid)))
        outside = {point for point in set(map(tuple, valid.reshape(-1, 3))) if len(owners[point]) > 1}
        planar, planar_count = exact_planar_repair(valid, outside)
        points, faces, rejected = _geometry(planar)
        flips = _flip_edges(points, faces, rejected, exact=True)
        candidate, strip_report = exact_strip_repair(points[faces],
            maximum_surface_displacement=maximum_surface_displacement)
        candidate, edge_report = exact_edge_refinement(candidate)
        bad = physics_degenerate_mask(candidate) | packed_physics_degenerate_mask(candidate)
        hulls = probe.analyze(valid)
        pitch = np.ptp(valid, axis=(0, 1)) / 65535
        output[name] = candidate
        reports[name] = {
            "source_triangles": len(source), "triangles": len(candidate),
            "source_zero_area_triangles": int(source_zero.sum()), "exact_diagonal_flips": flips,
            "exact_planar_regions": planar_count,
            "exact_strip_refinement": strip_report,
            "exact_edge_refinement": edge_report,
            "source_uploader_rejections": before, "unresolved_triangle_indices": np.flatnonzero(bad).tolist(),
            "triangle_upload_ready": not bool(bad.any()), "convex_conversion": hulls,
            "surface_change_before_packing": strip_report['ideal_surface_change'],
            "short_edge_collapses": 0, "omitted_positive_area_triangles": 0,
            "u16_position_step_per_axis": pitch.tolist(),
            "u16_position_error_upper_bound": float(np.linalg.norm(pitch)),
            "thin_instance_threshold_requires_runtime_scale": 0.5,
            "recommended_representation": "convex" if hulls["supported"] else ("triangles" if not bad.any() else "unresolved")}
    return output, {"parts": reports, "triangle_upload_ready": all(
        part["triangle_upload_ready"] for part in reports.values()), "all_parts_have_supported_representation": all(
        part["recommended_representation"] != "unresolved" for part in reports.values()),
        "reference": "complete input surface", "maximum_surface_change_before_packing": max(
            part['surface_change_before_packing'] for part in reports.values()),
        "quantization": "finite uploader position precision prevents a universal zero-error upload guarantee"}
