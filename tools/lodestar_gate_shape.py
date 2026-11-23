"""Compare exterior shape, negative space, and visible depth at one LOD size."""

from dataclasses import dataclass
from math import ceil
from numbers import Real

import numpy as np
from PIL import Image, ImageDraw

from lodestar_section_compare import BACKGROUND, _project_pair, _rasterize


_DIRECTIONS = tuple((x, y, z) for x in range(2) for y in range(-1, 2) for z in range(-1, 2)
                    if x or y > 0 or (y == 0 and z > 0))


@dataclass
class GateShapeResult:
    metrics: dict
    image: Image.Image | None


def _points(value):
    triangles = np.asarray(value, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("Expected finite corner positions shaped (triangles, 3, 3)")
    return triangles


def _nearby(mask, radius, *, outside=False):
    if radius == 0:
        return mask.copy()
    # Rectangular dilation is the horizontal and vertical one-dimensional passes.
    window = 2 * radius + 1
    horizontal = np.lib.stride_tricks.sliding_window_view(
        np.pad(mask, ((0, 0), (radius, radius)), constant_values=outside), window, axis=1).any(axis=-1)
    return np.lib.stride_tricks.sliding_window_view(
        np.pad(horizontal, ((radius, radius), (0, 0)), constant_values=outside), window, axis=0).any(axis=-1)


def _components(mask):
    """Return the four-connected foreground components of a binary mask."""
    remaining = mask.copy()
    height, width = mask.shape
    for y, x in zip(*np.nonzero(remaining)):
        if not remaining[y, x]:
            continue
        queue = [(y, x)]
        remaining[y, x] = False
        for row, col in queue:
            for v, u in ((row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)):
                if 0 <= v < height and 0 <= u < width and remaining[v, u]:
                    remaining[v, u] = False
                    queue.append((v, u))
        yield tuple(np.array(queue, dtype=np.intp).T)


def _touches_border(rows, cols, shape):
    return bool(rows.min() == 0 or cols.min() == 0 or rows.max() == shape[0] - 1 or cols.max() == shape[1] - 1)


def _apertures(reference, candidate, margin, minimum_samples, supersample):
    """Distinguish bounded holes and open-bottom gaps from exterior background."""
    ref_rows, ref_cols = np.nonzero(reference)
    if not len(ref_rows):
        raise ValueError("The reference has no visible front projection at this screen size")
    frame_rows, frame_cols = np.nonzero(reference | candidate)
    # Padding the union avoids creating an artificial exterior connection if
    # the candidate extends past the reference at an open-bottom arch.
    dy, dx = max(frame_rows.min() - 1, 0), max(frame_cols.min() - 1, 0)
    region = np.s_[dy:frame_rows.max() + 2, dx:frame_cols.max() + 2]
    before, after = reference[region], candidate[region]
    guarded = ~_nearby(reference, margin, outside=True)[region]
    exterior = np.zeros_like(before)
    protected = np.zeros_like(before)
    apertures = []
    sample_area = 1 / supersample ** 2
    candidate_exterior = np.zeros_like(after)
    for rows, cols in _components(~after):
        if _touches_border(rows, cols, after.shape):
            candidate_exterior[rows, cols] = True

    def record(rows, cols, kind):
        interior = guarded[rows, cols]
        rr, cc = rows[interior], cols[interior]
        if len(rr) < minimum_samples:
            return
        protected[rr, cc] = True
        blocked = int(after[rr, cc].sum())
        apertures.append({
            "kind": kind,
            "bounds_xyxy": [int(pos) / supersample for pos in
                            (cols.min() + dx, rows.min() + dy, cols.max() + dx + 1, rows.max() + dy + 1)],
            "reference_open_pixels": len(rows) * sample_area,
            "guarded_open_pixels": len(rr) * sample_area,
            "newly_blocked_guarded_pixels": blocked * sample_area,
            "guarded_retention": float(1 - blocked / len(rr)),
            "exterior_connected_guarded_pixels": (
                int(candidate_exterior[rr, cc].sum()) * sample_area if kind == "open_between_sides" else None),
        })

    for rows, cols in _components(~before):
        if _touches_border(rows, cols, before.shape):
            exterior[rows, cols] = True
        else:
            record(rows, cols, "enclosed")

    # Open-bottom arches are connected to the outside. Inside the row's two
    # occupied sides their negative space is distinct from the image exterior.
    left = np.logical_or.accumulate(before, axis=1)
    right = np.logical_or.accumulate(before[:, ::-1], axis=1)[:, ::-1]
    for rows, cols in _components(exterior & left & right):
        record(rows, cols, "open_between_sides")

    mask = np.zeros_like(reference)
    mask[region] = protected
    blocked = int((mask & candidate).sum())
    guarded_count = int(mask.sum())
    return {
        "guard_margin_px": margin / supersample,
        "minimum_guarded_aperture_pixels": minimum_samples * sample_area,
        "major_aperture_count": len(apertures),
        "guarded_open_pixels": guarded_count * sample_area,
        "newly_blocked_guarded_pixels": blocked * sample_area,
        "guarded_retention": 1 - blocked / guarded_count if guarded_count else None,
        "sealed_opening_count": sum(a["exterior_connected_guarded_pixels"] == 0 and a["guarded_retention"] > 0
                                    for a in apertures),
        "apertures": apertures,
    }, mask


def _pixel_scale(frame, direction, resolution):
    # Use the same orthographic frame and 5% border as _project_pair.
    camera = np.asarray(direction, dtype=np.float64)
    camera /= np.linalg.norm(camera)
    up = [0, 1, 0] if abs(camera[1]) < 0.99 else [0, 0, -1]
    right = np.cross(up, camera)
    right /= np.linalg.norm(right)
    screen_up = np.cross(camera, right)
    projected = (frame.reshape(-1, 3) - frame[0, 0]) @ np.column_stack((right, screen_up))
    extent = np.ptp(projected, axis=0).max()
    if not np.isfinite(extent) or extent == 0:
        raise ValueError("The camera frame needs a finite projected extent")
    return float(0.9 * resolution / extent)


def _depth_summary(error, sample_area):
    return {
        "pixels": len(error) * sample_area,
        "median_error_px": float(np.median(error)) if len(error) else None,
        "p95_error_px": float(np.percentile(error, 95)) if len(error) else None,
        "over_one_pixel": int(np.count_nonzero(error > 1.0 + 1e-12)) * sample_area,
    }


def _front_facing(triangles):
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    # _project_pair points its depth axis at the camera and its screen Y down.
    return (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) < 0


def _view(source, candidate, frame, direction, resolution, supersample, cull_backfaces):
    grid = resolution * supersample
    before, after = _project_pair(source, candidate, direction, grid, frame)
    if cull_backfaces:
        before, after = before[_front_facing(before)], after[_front_facing(after)]
    first, second = (_rasterize(positions, grid) for positions in (before, after))
    reference, proposed = first[0] >= 0, second[0] >= 0
    union = int((reference | proposed).sum())
    shared = reference & proposed
    inside = ~_nearby(~shared, supersample, outside=True)
    shared_count = int(shared.sum())
    scale = _pixel_scale(frame, direction, resolution)
    depth = np.zeros_like(reference, dtype=np.float64)
    for (visible, weights), projected, sign in ((first, before, 1), (second, after, -1)):
        values = projected[visible[shared], :, 2]
        depth[shared] += sign * np.einsum("ij,ij->i", values, weights[shared])
    depth = np.abs(depth) * scale
    coverage = [mask.reshape(resolution, supersample, resolution, supersample).mean(axis=(1, 3))
                for mask in (reference, proposed)]
    coverage_union = np.maximum(*coverage).sum()
    sample_area = 1 / supersample ** 2
    return {
        "direction": list(direction),
        "pixels_per_world_unit": scale,
        "reference_pixels": int(reference.sum()) * sample_area,
        "candidate_pixels": int(proposed.sum()) * sample_area,
        "shared_pixels": shared_count * sample_area,
        "silhouette_iou": shared_count / union if union else None,
        "pixel_coverage_iou": float(np.minimum(*coverage).sum() / coverage_union) if coverage_union else None,
        "missing_pixels": int((reference & ~proposed).sum()) * sample_area,
        "added_pixels": int((proposed & ~reference).sum()) * sample_area,
        "outside_one_pixel_envelope": int((reference & ~_nearby(proposed, supersample)).sum() +
                                          (proposed & ~_nearby(reference, supersample)).sum()) * sample_area,
        "visible_depth": _depth_summary(depth[shared], sample_area),
        "interior_visible_depth": _depth_summary(depth[inside], sample_area),
    }, reference, proposed


def _preview(views, resolution):
    tile, gap, header = min(resolution, 128), 6, 20
    image = Image.new("RGB", (3 * tile + 4 * gap, (tile + header + gap) * len(views) + gap), BACKGROUND)
    draw = ImageDraw.Draw(image)

    def panel(y, column, mask, color, accents=()):
        pixels = np.empty(mask.shape + (3,), dtype=np.uint8)
        pixels[:] = BACKGROUND
        pixels[mask] = color
        for changed, highlight in accents:
            pixels[changed] = highlight
        image.paste(Image.fromarray(pixels).resize((tile, tile), Image.Resampling.BOX),
                    (gap + column * (tile + gap), y + header))

    for i, (label, before, after, apertures) in enumerate(views):
        y = gap + i * (tile + header + gap)
        draw.text((gap, y), label, fill="white")
        panel(y, 0, before, (208, 215, 225))
        panel(y, 1, after, (208, 215, 225))
        accents = [(before & ~after, (242, 106, 98)), (after & ~before, (243, 192, 82))]
        if apertures is not None:
            accents.extend(((apertures & ~after, (79, 170, 191)), (apertures & after, (235, 83, 171))))
        panel(y, 2, before & after, (95, 107, 125), accents)
    return image


def compare_gate_shape(source, candidate, *, resolution=96, frame_positions=None, front_axis=None,
                       aperture_margin_px=1, minimum_aperture_pixels=4, supersample=4, cull_backfaces=False,
                       opposite_views=False, include_image=False):
    """Measure one model against High using fixed orthographic framing.

    Use the original full-scene triangle positions as frame_positions for a
    gate's intended screen size, and the original gate for a close-up. Distances
    in pixels use each view's world-to-screen scale. Interior depth excludes
    one pixel at common coverage boundaries; the full shared depth is also
    reported. Apertures require the requested number of open pixels after the
    source edge guard. Supersampling avoids aliasing the outline and small
    openings; pixel areas and depths are normalized to the stated resolution.
    Set cull_backfaces to model opaque preview geometry and opposite_views
    to include the other hemisphere. A null metric indicates no visible samples.
    """
    if (not isinstance(supersample, (int, np.integer)) or not 1 <= supersample <= 8 or
            not isinstance(resolution, (int, np.integer)) or not 16 <= resolution or resolution * supersample > 2048 or
            not isinstance(aperture_margin_px, Real) or not 0 <= aperture_margin_px <= 8 or
            not isinstance(minimum_aperture_pixels, Real) or not np.isfinite(minimum_aperture_pixels) or
            minimum_aperture_pixels <= 0 or minimum_aperture_pixels > resolution ** 2):
        raise ValueError("Invalid raster size or aperture guard")
    resolution, supersample = int(resolution), int(supersample)
    margin = ceil(aperture_margin_px * supersample)
    minimum = ceil(minimum_aperture_pixels * supersample ** 2)
    source, candidate = _points(source), _points(candidate)
    frame = source if frame_positions is None else _points(frame_positions)
    if front_axis is None:
        front_axis = int(np.ptp(source, axis=(0, 1)).argmin())
    if not isinstance(front_axis, (int, np.integer)) or not 0 <= front_axis < 3:
        raise ValueError("The front axis must be 0, 1, or 2")
    front_direction = tuple(int(i == front_axis) for i in range(3))
    views, panels = [], []
    directions = _DIRECTIONS
    if opposite_views:
        directions += tuple(tuple(-coordinate for coordinate in view) for view in _DIRECTIONS)
    for direction in directions:
        metrics, before, after = _view(source, candidate, frame, direction, resolution, supersample, cull_backfaces)
        apertures = None
        if direction == front_direction:
            front, apertures = _apertures(before, after, margin, minimum, supersample)
        if include_image:
            label = f"{direction} | source / candidate / difference"
            panels.append((label, before, after, apertures))
        views.append(metrics)

    iou = [view["silhouette_iou"] for view in views if view["silhouette_iou"] is not None]
    coverage = [view["pixel_coverage_iou"] for view in views if view["pixel_coverage_iou"] is not None]
    visible_depths = [view["visible_depth"]["p95_error_px"] for view in views
                      if view["visible_depth"]["p95_error_px"] is not None]
    depths = [view["interior_visible_depth"]["p95_error_px"] for view in views
              if view["interior_visible_depth"]["p95_error_px"] is not None]
    metrics = {
        "resolution": int(resolution),
        "supersample": supersample,
        "backface_culling": bool(cull_backfaces),
        "front_axis": int(front_axis),
        "reference_triangles": int(len(source)),
        "candidate_triangles": int(len(candidate)),
        "silhouette_mean_iou": float(np.mean(iou)) if iou else None,
        "silhouette_worst_iou": min(iou) if iou else None,
        "pixel_coverage_mean_iou": float(np.mean(coverage)) if coverage else None,
        "pixel_coverage_worst_iou": min(coverage) if coverage else None,
        "silhouette_pixels_outside_one_pixel_envelope": sum(v["outside_one_pixel_envelope"] for v in views),
        "worst_visible_depth_p95_px": max(visible_depths) if visible_depths else None,
        "worst_interior_visible_depth_p95_px": max(depths) if depths else None,
        "front_apertures": front,
        "views": views,
    }
    return GateShapeResult(metrics, _preview(panels, resolution) if include_image else None)
