"""Plan the TestHall visual levels below the qualified Medium mesh."""

from dataclasses import dataclass

import numpy as np

from check_testhall_gates import GATES, aperture_report
from lodestar_firestorm_degenerate import normalize_triangles, visual_degenerate_mask
from lodestar_nonplanar import NonplanarResult, _SAMPLE_WEIGHTS, simplify_nonplanar
from lodestar_nonplanar_bake import _closest_points, _face_normals
from lodestar_section_compare import _project_pair, _rasterize
from lodestar_testhall_budget import _choose_restorations
from lodestar_walkability import testhall_traversal


_VIEW_DIRECTIONS = tuple((x, y, z) for x in range(2) for y in range(-1, 2) for z in range(-1, 2)
                         if x or y > 0 or (y == 0 and z > 0))


@dataclass(frozen=True)
class TestHallVisualLevel:
    sections: dict[str, np.ndarray]
    replacements: dict[str, NonplanarResult]
    report: dict


@dataclass(frozen=True)
class _Profile:
    resolution: int
    max_relative_error: float
    max_surface_pixels: float
    max_silhouette_iou_error: float
    max_projection_distance: float


_PROFILES = {
    "low": _Profile(192, 0.03, 1.0, 0.02, 0.12),
    "lowest": _Profile(96, 0.07, 1.0, 0.03, 0.24),
}


def _extrema_pins(positions):
    points = positions.reshape(-1, 3)
    # The viewer normalizes each model. Keep a source vertex at each end of
    # each axis so a collapse cannot move a node's origin or change its scale.
    return frozenset(tuple(points[indices[axis]]) for indices in (points.argmin(axis=0), points.argmax(axis=0))
                     for axis in range(3))


def _bounds(positions):
    return np.stack((positions.min(axis=(0, 1)), positions.max(axis=(0, 1))))


def _degenerate_triangles(positions):
    # Reject near-collinear faces in normalized float space and zero-area
    # faces in the released viewer's packed mesh representation.
    normalized = normalize_triangles(positions)
    edges = np.roll(normalized, -1, axis=1) - normalized
    max_edge_length_sq = np.sum(edges ** 2, axis=2).max(axis=1)
    double_area = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
    limit = 4 * np.finfo(np.float32).eps * max_edge_length_sq
    return int(np.count_nonzero((double_area <= limit) | visual_degenerate_mask(positions)))


def _oriented_projection_distance(source, candidate, maximum):
    # Sample with the texture baker's geometric normals and closest points.
    # A nearby opposite-facing triangle cannot provide surface attributes.
    source_normals = _face_normals(source)
    target_normals = _face_normals(candidate.positions)
    distance = 0.0
    for triangle, normal, label in zip(candidate.positions, target_normals, candidate.target_components):
        neighbors = (candidate.source_components == label) & (source_normals @ normal > 0)
        if not neighbors.any():
            return np.inf
        _, _, errors = _closest_points(_SAMPLE_WEIGHTS @ triangle, source[neighbors], source_normals[neighbors])
        distance = max(distance, float(errors.max()))
        if distance > maximum:
            break
    return distance


def _masks(positions, frame, resolution):
    result = []
    for direction in _VIEW_DIRECTIONS:
        _, projected = _project_pair(positions, positions, direction, resolution, frame)
        result.append(_rasterize(projected, resolution)[0] >= 0)
    return np.stack(result)


def _neighbor_pixels(masks):
    padded = np.pad(masks, ((0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
    return windows.any(axis=(-1, -2))


def _silhouette(reference, candidate):
    union = (reference | candidate).sum(axis=(1, 2))
    differing = (reference ^ candidate).sum(axis=(1, 2))
    error = np.divide(differing, union, out=np.zeros(len(union)), where=union > 0)
    outside = (reference & ~_neighbor_pixels(candidate)) | (candidate & ~_neighbor_pixels(reference))
    return {"mean_iou_error": float(error.mean()), "max_iou_error": float(error.max()),
            "pixels_outside_one_pixel_envelope": int(outside.sum())}


def _qualify(scene, sections, previous, profile, masks):
    composed = {name: np.logical_or.reduce([masks(positions) for positions in meshes.values()])
                for name, meshes in (("high", {name: s.positions for name, s in scene.sections.items()}),
                                     ("previous", previous), ("candidate", sections))}
    high = _silhouette(composed["high"], composed["candidate"])
    adjacent = _silhouette(composed["previous"], composed["candidate"])
    inherited_error = _silhouette(composed["high"], composed["previous"])["pixels_outside_one_pixel_envelope"]
    geometry_passed = (max(high["max_iou_error"], adjacent["max_iou_error"]) <= profile.max_silhouette_iou_error and
                       high["pixels_outside_one_pixel_envelope"] <= inherited_error and
                       adjacent["pixels_outside_one_pixel_envelope"] == 0)
    walkability = None
    if geometry_passed:
        walkability = testhall_traversal(np.concatenate(list(sections.values())), check_supported_walk=True)
    return {"silhouette": {"views": len(_VIEW_DIRECTIONS), "resolution": profile.resolution,
                           "maximum_iou_error": profile.max_silhouette_iou_error,
                           "against_high": high, "against_previous": adjacent,
                           "high_to_previous_outside_envelope": inherited_error},
            "walkability": walkability, "passed": walkability is not None and walkability["passed"]}


def _plan_level(library, scene, previous, previous_replacements, target, profile):
    max_distance = float(np.ptp(scene.positions, axis=(0, 1)).max()) * profile.max_surface_pixels / profile.resolution
    cache = {}
    # All projections use the same frame. Cache models while trying restorations.
    frame = np.unique(scene.positions.reshape(-1, 3), axis=0)[:, None]

    def masks(positions):
        key = id(positions)
        if key not in cache:
            cache[key] = _masks(positions, frame, profile.resolution)
        return cache[key]

    previous_masks = {name: masks(positions) for name, positions in previous.items()}
    proposed, decisions, restoration_choices = {}, {}, []
    for name, source in scene.sections.items():
        before = previous[name]
        neighbors = np.logical_or.reduce([mask for node, mask in previous_masks.items() if node != name],
                                         initial=False)
        pins = source.outside_positions | _extrema_pins(source.positions)
        extent = float(np.ptp(source.positions, axis=(0, 1)).max())
        decision = {"previous_triangles": len(before), "trials": []}
        decisions[name] = decision
        for fraction in (1.0, 0.65):
            result = simplify_nonplanar(library, source.positions, 1,
                                        outside_positions=pins,
                                        max_relative_error=profile.max_relative_error * fraction,
                                        max_sampled_relative_error=min(1.0, max_distance * fraction / extent))
            positions = result.positions
            trial = {"triangles": len(positions),
                     "maximum_allowed_sampled_surface_distance_m": max_distance * fraction}
            decision["trials"].append(trial)
            if len(positions) >= len(before):
                trial["rejected"] = "triangle count"
                continue
            if not np.array_equal(_bounds(positions), _bounds(source.positions)):
                trial["rejected"] = "model bounds"
                continue
            trial["degenerate_triangles"] = _degenerate_triangles(positions)
            if trial["degenerate_triangles"]:
                trial["rejected"] = "normalized triangle degeneracy"
                continue
            if name in GATES:
                apertures = aperture_report(source.positions, positions)
                trial["aperture"] = apertures
                if apertures["newly_blocked_guarded_pixels"]:
                    trial["rejected"] = "gate opening"
                    continue

            candidate_masks = _masks(positions, frame, profile.resolution)
            adjacent = _silhouette(previous_masks[name] | neighbors, candidate_masks | neighbors)
            trial["scene_silhouette_against_previous"] = adjacent
            if adjacent["pixels_outside_one_pixel_envelope"]:
                trial["rejected"] = "silhouette envelope"
                continue
            with_candidate = [positions if node == name else geometry for node, geometry in previous.items()]
            if not testhall_traversal(np.concatenate(with_candidate))["passed"]:
                trial["rejected"] = "opening, wall or floor probe"
                continue
            projection = _oriented_projection_distance(source.positions, result, profile.max_projection_distance)
            trial["sampled_texture_projection_distance_m"] = projection if np.isfinite(projection) else None
            if projection > profile.max_projection_distance:
                trial["rejected"] = "oriented texture projection"
                continue

            error = max(part["sampled_relative_error"] for part in result.components) * extent
            trial["maximum_sampled_surface_distance_m"] = error
            proposed[name] = result
            cache[id(positions)] = candidate_masks
            area = np.linalg.norm(np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0]), axis=1)
            restoration_choices.append((name, len(before) - len(positions), float(area.sum() * error ** 2)))
            break

    minimum = sum(len(proposed[name].positions) if name in proposed else len(mesh)
                  for name, mesh in previous.items())
    retained = _choose_restorations(restoration_choices, max(0, target - minimum))

    def assemble():
        selected = {name: result for name, result in proposed.items() if name not in retained}
        replacements = {**previous_replacements, **selected}
        sections = {name: selected[name].positions if name in selected else mesh for name, mesh in previous.items()}
        return sections, replacements, _qualify(scene, sections, previous, profile, masks)

    sections, replacements, quality = assemble()
    # Local guards admit a candidate, then the complete level must still pass
    # because geometry from separate nodes can interact on the same route.
    for name, _, _ in sorted(restoration_choices, key=lambda item: (-item[2], item[0])):
        if quality["passed"]:
            break
        if name not in retained:
            retained.add(name)
            decisions[name]["retained_for_scene_checks"] = True
            sections, replacements, quality = assemble()

    for name, positions in sections.items():
        decisions[name]["triangles"] = len(positions)
        decisions[name]["retained_previous"] = positions is previous[name]
        decisions[name]["changed_bounds_m"] = float(np.abs(_bounds(positions) - _bounds(previous[name])).max())
        decisions[name]["degenerate_triangles"] = _degenerate_triangles(positions)
    degenerate = sum(decision["degenerate_triangles"] for decision in decisions.values())
    quality["passed"] &= degenerate == 0
    count = sum(map(len, sections.values()))
    previous_count = sum(map(len, previous.values()))
    return TestHallVisualLevel(sections, replacements,
                               {"triangles": count, "previous_triangles": previous_count, "target_triangles": target,
                                "met_target": count <= target, "reduced": count < previous_count,
                                "triangle_gap": max(0, count - target),
                                "degenerate_triangles": degenerate,
                                "bounds": "carry previous nodes or pin the original High extrema",
                                "maximum_allowed_sampled_surface_distance_m": max_distance,
                                "maximum_allowed_texture_projection_distance_m": profile.max_projection_distance,
                                "sections": decisions, **quality})


def plan_testhall_levels(library, scene, medium_replacements, *, medium_sections=None,
                         low_target=3400, lowest_target=2400):
    """Build independent proposals from High, bounded against each previous level.

    Explicit Medium positions are needed for a section produced by a planar
    pass, since this geometry has no NonplanarResult. A lower-level replacement
    retains the High source component IDs for direct texture projection. The
    target counts are aspirational and the report records the gap.
    """
    medium_sections = medium_sections or {}
    if (medium_sections.keys() | medium_replacements.keys()) - scene.sections.keys():
        raise ValueError("The Medium mesh references unknown TestHall sections")
    previous = {}
    for name, source in scene.sections.items():
        replacement = medium_replacements.get(name)
        positions = medium_sections.get(name, source.positions if replacement is None else replacement.positions)
        if (not len(positions) or len(positions) > len(source.positions) or
                (replacement is not None and not np.array_equal(positions, replacement.positions))):
            raise ValueError(f"Medium geometry does not match its source or bake provenance for {name}")
        previous[name] = positions
    if (not all(isinstance(target, (int, np.integer)) for target in (low_target, lowest_target)) or
            not 1 <= lowest_target < low_target < sum(map(len, previous.values()))):
        raise ValueError("Expected strictly descending positive TestHall triangle targets below Medium")

    replacements = dict(medium_replacements)
    levels = {}
    for name, target in (("low", int(low_target)), ("lowest", int(lowest_target))):
        level = _plan_level(library, scene, previous, replacements, target, _PROFILES[name])
        if level.report["degenerate_triangles"]:
            raise ValueError(f"TestHall {name} has {level.report['degenerate_triangles']} degenerate triangles")
        if not level.report["passed"]:
            raise ValueError(f"TestHall {name} could not preserve the tested shape and routes")
        levels[name] = level
        previous, replacements = level.sections, level.replacements
    return levels
