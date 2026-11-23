"""Fit UV-chart surfaces and validate LODs against the retained High mesh."""

from collections import defaultdict
import ctypes as ct
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from bake_planar_patch import boundary_loop, cross2
from lodestar_nonplanar import _topology
from lodestar_patch_layout import triangulate_boundary
from lodestar_uv_regions import simple_polygon
from lodestar_chart_remesh import remesh_chart


POWERS = np.array([(3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (1, 2, 0),
                   (2, 0, 1), (1, 0, 2), (0, 2, 1), (0, 1, 2), (1, 1, 1)])
NODES = POWERS / 3
BERNSTEIN = np.prod(NODES[:, None, :] ** POWERS[None], axis=2) * [1, 1, 1, 3, 3, 3, 3, 3, 3, 6]


class _Patch(ct.Structure):
    _fields_ = [("controls", (ct.c_double * 3) * 10)]


class _Budget(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in (
        "area", "absolute_curvature", "estimated_area_error", "estimated_curvature_error", "world_error",
        "triangle_density_integral", "interior_vertex_density_integral")]
    _fields_ += [("suggested_triangles", ct.c_size_t), ("evaluations", ct.c_size_t), ("converged", ct.c_uint32)]


class _Polynomial(ct.Structure):
    _fields_ = [("coefficients", (ct.c_double * 3) * 10),
                ("maximum_vertex_residual", ct.c_double), ("degree", ct.c_uint32)]


class SurfaceCalculator:
    def __init__(self, library):
        self.failed_integrals = set()
        self.api = ct.CDLL(str(Path(library).resolve()))
        self.api.lodestar_last_error.argtypes = []
        self.api.lodestar_last_error.restype = ct.c_char_p
        self.api.lodestar_calculate_surface_lod.argtypes = [
            ct.POINTER(_Patch), ct.c_size_t, ct.c_double, ct.c_double, ct.c_double, ct.c_uint32,
            ct.POINTER(_Budget)]
        self.api.lodestar_calculate_surface_lod.restype = ct.c_uint32
        try:
            self.api.lodestar_fit_surface.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                                                     ct.c_size_t, ct.POINTER(_Polynomial)]
            self.api.lodestar_fit_surface.restype = ct.c_uint32
            self.api.lodestar_polynomial_surface_patch.argtypes = [ct.POINTER(_Polynomial), ct.POINTER(ct.c_double),
                                                                  ct.POINTER(_Patch)]
            self.api.lodestar_polynomial_surface_patch.restype = ct.c_uint32
        except AttributeError as error:
            raise ValueError("Rebuild the Lodestar C library with native surface-fitting support") from error

    def fit(self, parameters, positions):
        parameters = np.ascontiguousarray(parameters, dtype=np.float64)
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        if parameters.shape != (len(positions), 2) or positions.shape != (len(positions), 3):
            raise ValueError("Surface fitting needs paired 2D parameters and 3D positions")
        model = _Polynomial()
        status = self.api.lodestar_fit_surface(parameters.ctypes.data_as(ct.POINTER(ct.c_double)),
            positions.ctypes.data_as(ct.POINTER(ct.c_double)), len(positions), ct.byref(model))
        if status:
            raise ValueError(self.api.lodestar_last_error().decode("utf-8", errors="replace"))
        return np.ctypeslib.as_array(model.coefficients).copy(), int(model.degree)

    def patch(self, coefficients, domain):
        coefficients = np.asarray(coefficients, dtype=np.float64)
        domain = np.ascontiguousarray(domain, dtype=np.float64)
        if coefficients.shape != (10, 3) or domain.shape != (3, 2):
            raise ValueError("Patch restriction needs ten coefficients and a triangular parameter domain")
        model = _Polynomial()
        np.ctypeslib.as_array(model.coefficients)[:] = coefficients
        output = _Patch()
        status = self.api.lodestar_polynomial_surface_patch(ct.byref(model),
            domain.ctypes.data_as(ct.POINTER(ct.c_double)), ct.byref(output))
        if status:
            raise ValueError(self.api.lodestar_last_error().decode("utf-8", errors="replace"))
        return np.ctypeslib.as_array(output.controls).copy()

    def calculate(self, controls, pixel_scale, pixel_error):
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        if controls.ndim != 3 or controls.shape[1:] != (10, 3) or not len(controls):
            raise ValueError("Expected one or more cubic triangular patch control nets")
        key = controls.tobytes()
        if key in self.failed_integrals:
            raise ValueError("Surface integral did not converge")
        report = _Budget()
        status = self.api.lodestar_calculate_surface_lod(
            controls.ctypes.data_as(ct.POINTER(_Patch)), len(controls), pixel_scale, pixel_error,
            1e-5, 8, ct.byref(report))
        if status:
            raise ValueError(self.api.lodestar_last_error().decode("utf-8", errors="replace"))
        if not report.converged:
            self.failed_integrals.add(key)
        return {name: getattr(report, name) for name, _ in report._fields_}

    def calculate_chart(self, chart, pixel_scale, pixel_error):
        """Integrate the actual piecewise-linear mesh, including crease curvature.

        Face derivatives are constant. Area is their exact analytic integral;
        absolute curvature is a measure on interior edges, length times the
        angle between adjacent face normals. A smooth fitted polynomial can
        miss these creases, and is retained as a description, not an authority.
        The resulting sampling budget is a heuristic; full overlay validation
        supplies the acceptance bound.
        """
        points = chart.positions[chart.faces]
        cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        normals = cross / lengths[:, None]
        uses = defaultdict(list)
        for index, face in enumerate(chart.faces.tolist()):
            for a, b in zip(face, face[1:] + face[:1]):
                uses[min(a, b), max(a, b)].append(index)
        curvature = 0.0
        for (a, b), faces in uses.items():
            if len(faces) == 2:
                first, second = normals[faces]
                angle = np.arctan2(np.linalg.norm(np.cross(first, second)), np.clip(first @ second, -1., 1.))
                curvature += float(np.linalg.norm(chart.positions[a] - chart.positions[b]) * angle)
        error = pixel_scale * pixel_error
        density = 2 * curvature / (3 * np.sqrt(3) * error)
        return {"area": float(lengths.sum() * .5), "absolute_curvature": curvature,
                "world_error": error, "triangle_density_integral": density,
                "interior_vertex_density_integral": density * .5,
                "suggested_triangles": max(1, int(np.ceil(density))), "converged": 1,
                "method": "piecewise_linear_surface_and_edge_curvature_integrals",
                "budget_is_acceptance_certificate": False}


def polynomial(uv):
    u, v = np.moveaxis(np.asarray(uv), -1, 0)
    return np.stack((np.ones_like(u), u, v, u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v), axis=-1)


def patch_controls(coefficients, domain):
    samples = polynomial(NODES @ domain) @ coefficients
    return np.linalg.solve(BERNSTEIN, samples)


def fit_bound(coefficients, uv, positions, patcher=patch_controls):
    """Bernstein bound on the fit against every High triangle at matching UVs."""
    maximum = 0.0
    for domain, corners in zip(uv, positions):
        differences = patcher(coefficients, domain) - NODES @ corners
        maximum = max(maximum, float(np.linalg.norm(differences, axis=1).max()))
    return maximum


def charts(scene):
    """Join position/UV-identical edges within one material and instance.

    Shading discontinuities remain in the per-corner reference field. They
    must be checked by the remesher, rather than splitting geometric topology.
    """
    nodes = np.full(len(scene.positions), -1, dtype=np.int64)
    for node, section in enumerate(scene.sections.values()):
        nodes[section.face_indices] = node
    if np.any(nodes < 0):
        raise ValueError("Named parts must account for every source triangle")
    uses = defaultdict(list)
    rows = np.concatenate((scene.positions, scene.uvs), axis=-1).reshape(-1, 5)
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    for face, corners in enumerate(inverse.reshape(-1, 3).tolist()):
        for a, b in zip(corners, corners[1:] + corners[:1]):
            uses[min(a, b), max(a, b)].append((face, a, b))
    adjacency = [[] for _ in scene.positions]
    for edge in uses.values():
        if len(edge) != 2:
            continue
        (a, start, end), (b, tail, head) = edge
        if ((start, end) == (head, tail) and scene.materials[a] == scene.materials[b] and nodes[a] == nodes[b]):
            adjacency[a].append(b)
            adjacency[b].append(a)
    visited = set()
    result = []
    for seed in range(len(scene.positions)):
        if seed in visited:
            continue
        connected = [seed]
        visited.add(seed)
        for face in connected:
            for neighbor in adjacency[face]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    connected.append(neighbor)
        result.append(np.array(sorted(connected), dtype=np.intp))
    return result


def _intersection(first, second):
    polygon = list(first)
    for a, b in zip(second, np.roll(second, -1, axis=0)):
        if not polygon:
            break
        output = []
        previous = polygon[-1]
        old_distance = cross2(b - a, previous - a)
        for current in polygon:
            distance = cross2(b - a, current - a)
            if (distance >= 0) != (old_distance >= 0):
                t = old_distance / (old_distance - distance)
                output.append(previous + t * (current - previous))
            if distance >= 0:
                output.append(current)
            previous, old_distance = current, distance
        polygon = output
    return np.asarray(polygon)


def _weights(points, triangle):
    offsets = np.linalg.solve((triangle[1:] - triangle[0]).T, (points - triangle[0]).T).T
    return np.column_stack((1 - offsets.sum(axis=1), offsets))


def correspondence_bound(uv, positions, normals, source, target, *, maximum_pairs=500000,
                         source_normals=None, target_normals=None):
    """Bound piecewise-linear position/normal differences on the complete UV overlay.

    Each overlay cell is convex and both interpolants are affine there. The
    norm of their difference reaches its maximum at a polygon vertex. UVs
    themselves are the common domain, preserving the original texture map.
    """
    low = uv[target].min(axis=1)
    high = uv[target].max(axis=1)
    source_field = normals[source] if source_normals is None else np.asarray(source_normals)
    target_field = normals[target] if target_normals is None else np.asarray(target_normals)
    if source_field.shape != (len(source), 3, 3) or target_field.shape != (len(target), 3, 3):
        raise ValueError("Correspondence needs one normal at each triangle corner")
    constant_normals = bool(np.all(source_field == source_field[0, 0]) and
                            np.all(target_field == source_field[0, 0]))
    surface_error = normal_angle = 0.0
    pairs = 0
    for source_index, face in enumerate(source):
        domain = uv[face]
        candidates = np.flatnonzero(np.all(high >= domain.min(axis=0), axis=1) &
                                    np.all(low <= domain.max(axis=0), axis=1))
        pairs += len(candidates)
        if pairs > maximum_pairs:
            raise ValueError("UV correspondence overlay exceeds its work limit")
        covered_area = 0.0
        for index in candidates:
            other = target[index]
            polygon = _intersection(domain, uv[other])
            if len(polygon) < 3:
                continue
            area = float(np.abs(cross2(polygon, np.roll(polygon, -1, axis=0)).sum()) * 0.5)
            if area == 0:
                continue
            covered_area += area
            before, after = _weights(polygon, domain), _weights(polygon, uv[other])
            surface_error = max(surface_error, float(np.linalg.norm(
                before @ positions[face] - after @ positions[other], axis=1).max()))
            if not constant_normals:
                before_normals = before @ source_field[source_index]
                after_normals = after @ target_field[index]
                reference = before_normals.mean(axis=0)
                length = np.linalg.norm(reference)
                if length == 0:
                    raise ValueError("A source normal field is singular in an overlay cell")
                normal_floor = float(np.min(before_normals @ (reference / length)))
                if normal_floor <= 0:
                    raise ValueError("A source normal field crosses a hemisphere in an overlay cell")
                difference = float(np.linalg.norm(before_normals - after_normals, axis=1).max())
                difference += 1e-12 * float(np.linalg.norm(source_field[source_index], axis=1).max())
                angle = np.rad2deg(2 * np.arcsin(min(1.0, difference / normal_floor)))
                normal_angle = max(normal_angle, float(angle))
        expected = float(cross2(domain[1] - domain[0], domain[2] - domain[0]) * 0.5)
        if abs(covered_area - expected) > max(expected * 1e-8, 1e-13):
            raise ValueError("The candidate does not cover the complete source UV domain")
    # Floating-point allowances accompany the exact real-arithmetic overlay bound.
    surface_error += 1e-10 * float(np.ptp(positions, axis=0).max())
    return {"surface_bound": surface_error, "normal_angle_bound_degrees": normal_angle, "overlay_pairs": pairs}


@dataclass
class FittedChart:
    source_faces: np.ndarray
    global_vertices: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    source_uv: np.ndarray
    uv: np.ndarray
    faces: np.ndarray
    controls: np.ndarray
    report: dict
    corner_normals: np.ndarray

    @cached_property
    def normal_lookup(self):
        originals = {tuple(face): i for i, face in enumerate(self.faces.tolist())}
        incident = defaultdict(list)
        for i, face in enumerate(self.faces):
            for corner, vertex in enumerate(face):
                incident[int(vertex)].append((i, corner))
        return originals, incident, {}

    def normals_for(self, faces):
        """Retain unchanged face normals and choose sided normals for new faces."""
        originals, incident, cache = self.normal_lookup
        result = []
        for face in faces:
            key = tuple(face)
            if key in cache:
                result.append(cache[key])
                continue
            original = originals.get(tuple(face))
            if original is not None:
                result.append(self.corner_normals[original])
                continue
            center = self.uv[face].mean(axis=0)
            values = []
            for vertex in face:
                probe = self.uv[vertex] * (1 - 1e-6) + center * 1e-6
                choices = incident[int(vertex)]
                best = max(choices, key=lambda entry: float(_weights(
                    probe[None], self.uv[self.faces[entry[0]]]).min()))
                values.append(self.corner_normals[best[0], best[1]])
            cache[key] = np.asarray(values)
            result.append(cache[key])
        return np.asarray(result)


def fit_chart(scene, selected, fitter=None, patcher=None):
    if not len(selected):
        raise ValueError("Cannot fit an empty chart")
    rows = np.concatenate((scene.positions[selected], scene.uvs[selected]), axis=-1).reshape(-1, 5)
    _, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    identities = scene.corner_indices[selected].ravel()[first]
    positions = scene.positions[selected].reshape(-1, 3)[first]
    normals = scene.normals[selected].reshape(-1, 3)[first]
    source_uv = scene.uvs[selected].astype(np.float32).reshape(-1, 2)[first].astype(float)
    faces = inverse.reshape(-1, 3)
    topology = _topology(positions, faces)
    if topology is None or topology[:2] != (1, 1):
        raise ValueError("Chart must be a manifold disk")
    span = np.ptp(source_uv, axis=0)
    if np.any(span <= 0):
        raise ValueError("Chart has no regular UV parameterization")
    uv = (source_uv - source_uv.min(axis=0)) / span
    areas = cross2(uv[faces[:, 1]] - uv[faces[:, 0]], uv[faces[:, 2]] - uv[faces[:, 0]])
    reflected = bool(np.all(areas < 0))
    if reflected:
        uv[:, 1] = 1 - uv[:, 1]
        areas = -areas
    if not np.all(areas > 1e-14):
        raise ValueError("Chart UVs fold or contain a collapsed triangle")
    boundary = np.array(boundary_loop(faces), dtype=np.intp)
    if len(boundary) > 256 or not simple_polygon(uv[boundary]):
        raise ValueError("Chart boundary is unsupported or exceeds the fitting work limit")
    polygon_area = float(cross2(uv[boundary], np.roll(uv[boundary], -1, axis=0)).sum())
    if abs(float(areas.sum()) - polygon_area) > 1e-9:
        raise ValueError("The UV parameterization overlaps or leaves a hole")
    domains = triangulate_boundary(uv[boundary])
    if domains is None:
        raise ValueError("Cannot triangulate the protected chart boundary")
    # Lowest-degree fit adequate to the stored positions. The retained High
    # triangles remain authoritative even when the budget model has residuals.
    offset = positions[0]
    extent = float(np.ptp(positions, axis=0).max())
    if fitter is None:
        coefficients = np.zeros((10, 3))
        degree = 0
        for proposed_degree, width in ((1, 3), (2, 6), (3, 10)):
            if len(positions) < width:
                break
            values, _, rank, singular = np.linalg.lstsq(polynomial(uv)[:, :width], positions - offset, rcond=1e-10)
            if rank != width or singular[0] / singular[-1] > 1e10:
                break
            coefficients[:] = 0
            coefficients[:width] = values
            degree = proposed_degree
            if np.max(np.linalg.norm(polynomial(uv) @ coefficients - (positions - offset), axis=1)) <= extent * 1e-7:
                break
        coefficients[0] += offset
    else:
        coefficients, degree = fitter(uv, positions)
    if degree == 0:
        raise ValueError("Chart does not admit a stable polynomial fit")
    patcher = patch_controls if patcher is None else patcher
    controls = np.stack([patcher(coefficients, uv[boundary[domain]]) for domain in domains])
    deviation = fit_bound(coefficients, uv[faces], positions[faces], patcher) + 1e-10 * extent
    if degree == 1:
        # The fitted model is analytically affine. Integrate it in a rigid plane
        # frame so tiny conversion roundoff cannot invent normal curvature.
        tangent_length = np.linalg.norm(coefficients[1])
        normal = np.cross(coefficients[1], coefficients[2])
        normal_length = np.linalg.norm(normal)
        if tangent_length == 0 or normal_length <= 1e-14 * tangent_length * np.linalg.norm(coefficients[2]):
            raise ValueError("Fitted affine surface is singular")
        tangent = coefficients[1] / tangent_length
        normal /= normal_length
        basis = np.column_stack((tangent, np.cross(normal, tangent), normal))
        controls = (controls - offset) @ basis
        controls[:, :, 2] = 0
    return FittedChart(selected, identities, positions, normals, source_uv, uv, faces, controls,
                       {"source_triangles": len(selected), "degree": degree,
                        "model_fit_deviation_bound": deviation, "patches": len(controls),
                        "boundary_vertices": len(boundary), "authoritative_high_retained": True,
                        "source_face_indices": selected.tolist(), "material": int(scene.materials[selected[0]]),
                        "uv_origin": source_uv.min(axis=0).tolist(), "uv_span": span.tolist(),
                        "uv_reflected": reflected, "polynomial_coefficients": coefficients.tolist(),
                        "fitting_backend": "native QR" if fitter is not None else "NumPy reference"},
                       scene.normals[selected].copy())


def generate_chart(chart, calculator, pixel_scale, pixel_error, maximum_normal_angle, locks, previous=None):
    budget = (calculator.calculate_chart(chart, pixel_scale, pixel_error)
              if hasattr(calculator, "calculate_chart") else calculator.calculate(chart.controls, pixel_scale, pixel_error))
    if not budget["converged"]:
        raise ValueError("Surface integral did not converge")
    before = chart.faces if previous is None else previous
    target = min(len(before), int(budget["suggested_triangles"]))
    report = {"calculation": budget, "requested_triangles": target, "triangles": len(before), "changed": False}
    if target >= len(before):
        return before, report
    def local_bound(uv, positions, normals, source, target):
        return correspondence_bound(uv, positions, normals, source, target,
                                    source_normals=chart.normals_for(source), target_normals=chart.corner_normals)
    result, details = remesh_chart(chart, before, target, pixel_scale * pixel_error,
                                   maximum_normal_angle, locks, local_bound)
    report.update(details)
    if len(result) >= len(before):
        return before, report
    topology = _topology(chart.positions, result)
    if topology is None or topology[:2] != _topology(chart.positions, chart.faces)[:2]:
        raise ValueError("Candidate changed chart topology")
    areas = cross2(chart.uv[result[:, 1]] - chart.uv[result[:, 0]], chart.uv[result[:, 2]] - chart.uv[result[:, 0]])
    if np.any(areas <= 0) or not np.isin(np.flatnonzero(locks), result).all():
        raise ValueError("Candidate reversed a UV face or removed a protected contact")
    bound = correspondence_bound(chart.uv, chart.positions, chart.normals, chart.faces, result,
                                 source_normals=chart.corner_normals, target_normals=chart.normals_for(result))
    if bound["surface_bound"] > pixel_scale * pixel_error or bound["normal_angle_bound_degrees"] > maximum_normal_angle:
        raise ValueError("The complete remesh exceeds High correspondence limits")
    report.update(bound, triangles=len(result), changed=True)
    return result, report
