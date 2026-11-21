"""Compare section appearance with shared orthographic cameras and diffuse lighting."""

from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from PIL import Image, ImageDraw

from bake_planar_patch import linear_image, rgb8, sample


VIEWS = (
    ("Front", (0, 0, 1)),
    ("Back", (0, 0, -1)),
    ("Left", (-1, 0, 0)),
    ("Right", (1, 0, 0)),
    ("Top", (0, 1, 0)),
    ("Oblique", (1, 0.65, 1)),
)
LIGHT = np.array([0.35, 0.75, 1.0])
LIGHT /= np.linalg.norm(LIGHT)
AMBIENT = 0.35
BACKGROUND = (24, 28, 35)
MASK_DIFFERENCE = (238, 108, 164)


@dataclass
class ComparisonResult:
    metrics: dict
    image: Image.Image


@dataclass
class _Section:
    positions: np.ndarray
    uvs: np.ndarray
    normals: np.ndarray
    materials: np.ndarray
    textures: dict[int, tuple[np.ndarray, bool]]
    edited: np.ndarray | None


def _section(positions, uvs, normals, materials, textures, image_cache, edited):
    positions = np.asarray(positions, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    materials = np.asarray(materials)
    if (positions.ndim != 3 or positions.shape[1:] != (3, 3) or len(positions) == 0 or
            uvs.shape != positions.shape[:2] + (2,) or normals.shape != positions.shape or
            materials.shape != (len(positions),) or materials.dtype.kind not in "iu" or
            not all(np.isfinite(values).all() for values in (positions, uvs, normals))):
        raise ValueError("Section needs finite per-corner positions, UVs and normals and integer face material IDs")
    if edited is not None:
        edited = np.asarray(edited)
        if edited.shape != materials.shape or edited.dtype.kind != "b":
            raise ValueError("The edited mask needs one boolean per triangle")
    converted = {}
    for material in np.unique(materials):
        if material not in textures:
            raise ValueError(f"Section material {material} needs a diffuse texture")
        image, repeat = textures[material]
        if not isinstance(image, Image.Image) or image.mode != "RGB" or min(image.size) <= 0:
            raise ValueError(f"Section material {material} needs a nonempty RGB image")
        if not isinstance(repeat, (bool, np.bool_)):
            raise ValueError(f"Section material {material} needs a boolean repeat setting")
        identity = id(image)
        if identity not in image_cache:
            image_cache[identity] = linear_image(image)
        converted[material] = image_cache[identity], repeat
    return _Section(positions, uvs, normals, materials, converted, edited)


def _project_pair(source, target, direction, resolution):
    toward_camera = np.asarray(direction, dtype=np.float64)
    toward_camera /= np.linalg.norm(toward_camera)
    reference_up = [0, 1, 0] if abs(toward_camera[1]) < 0.99 else [0, 0, -1]
    right = np.cross(reference_up, toward_camera)
    right /= np.linalg.norm(right)
    up = np.cross(toward_camera, right)
    basis = np.column_stack((right, -up, toward_camera))

    origin = source[0, 0]
    source = (source - origin) @ basis
    target = (target - origin) @ basis
    minimum = np.minimum(source.min(axis=(0, 1)), target.min(axis=(0, 1)))[:2]
    maximum = np.maximum(source.max(axis=(0, 1)), target.max(axis=(0, 1)))[:2]
    extent = (maximum - minimum).max()
    if not np.isfinite(extent):
        raise ValueError("Section exceeds a finite orthographic projection")
    if extent == 0:
        return source, target

    center = minimum + (maximum - minimum) / 2
    for triangles in (source, target):
        triangles[:, :, :2] = (triangles[:, :, :2] - center) / extent * (resolution * 0.9) + resolution / 2
    return source, target


def _edge_values(a, b, x, y, orientation):
    dx, dy = (b[:2] - a[:2]) * orientation
    values = dx * (y - a[1]) - dy * (x - a[0])
    top_left = dy < 0 or (dy == 0 and dx > 0)
    return values, top_left


def _rasterize(triangles, resolution):
    depth = np.full((resolution, resolution), -np.inf)
    visible_face = np.full(depth.shape, -1, dtype=np.intp)
    weights = np.empty(depth.shape + (3,), dtype=np.float64)
    pixels = np.arange(resolution) + 0.5
    edge_epsilon = 8 * np.finfo(np.float64).eps * resolution ** 2

    for face, triangle in enumerate(triangles):
        a, b, c = triangle
        ab, ac = b[:2] - a[:2], c[:2] - a[:2]
        signed_area = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(signed_area) <= edge_epsilon:
            continue
        orientation = 1 if signed_area > 0 else -1
        left, top = np.maximum(0, np.floor(triangle.min(axis=0)[:2])).astype(int)
        right, bottom = np.minimum(resolution, np.ceil(triangle.max(axis=0)[:2])).astype(int)
        if left >= right or top >= bottom:
            continue
        x, y = pixels[None, left:right], pixels[top:bottom, None]
        edges = []
        inside = np.ones((bottom - top, right - left), dtype=bool)
        for start, end in ((b, c), (c, a), (a, b)):
            values, top_left = _edge_values(start, end, x, y, orientation)
            inside &= (values > edge_epsilon) | (top_left & (np.abs(values) <= edge_epsilon))
            edges.append(values)
        rows, columns = np.nonzero(inside)
        if not len(rows):
            continue
        barycentric = np.column_stack([values[rows, columns] for values in edges]) / abs(signed_area)
        candidate_depth = barycentric @ triangle[:, 2]
        rows, columns = rows + top, columns + left
        closer = candidate_depth > depth[rows, columns]
        rows, columns = rows[closer], columns[closer]
        depth[rows, columns] = candidate_depth[closer]
        visible_face[rows, columns] = face
        weights[rows, columns] = barycentric[closer]
    return visible_face, weights


def _render(section, projection, resolution):
    visible_face, weights = _rasterize(projection, resolution)
    mask = visible_face >= 0
    pixels = np.empty(mask.shape + (3,), dtype=np.uint8)
    pixels[:] = BACKGROUND
    faces, active_weights = visible_face[mask], weights[mask]
    uv = np.einsum("ij,ijk->ik", active_weights, section.uvs[faces])
    normals = np.einsum("ij,ijk->ik", active_weights, section.normals[faces])
    lengths = np.linalg.norm(normals, axis=1)
    cosine = np.divide(normals @ LIGHT, lengths, out=np.zeros(len(normals)), where=lengths > 0)
    illumination = AMBIENT + (1 - AMBIENT) * np.clip(cosine, 0, 1)
    material = section.materials[faces]
    colors = np.empty((len(faces), 3))
    for identifier in np.unique(material):
        selected = material == identifier
        texture, repeat = section.textures[identifier]
        colors[selected] = sample(texture, uv[selected], repeat=repeat)
    pixels[mask] = rgb8(colors * illumination[:, None])
    visible_edits = np.zeros_like(mask)
    if section.edited is not None:
        visible_edits[mask] = section.edited[faces]
    return mask, pixels, visible_edits


def _error_stats(histogram):
    samples = int(histogram.sum())
    if samples == 0:
        return {"rgb_samples": 0, "mean_abs_error": None, "p95_abs_error": None, "max_abs_error": None}
    cumulative = np.cumsum(histogram)
    rank = 0.95 * (samples - 1)
    lower, upper = np.searchsorted(cumulative, [floor(rank), ceil(rank)], side="right")
    p95 = float(lower + (rank - floor(rank)) * (upper - lower))
    return {"rgb_samples": samples, "mean_abs_error": float(histogram @ np.arange(256) / samples),
            "p95_abs_error": p95, "max_abs_error": int(np.flatnonzero(histogram)[-1])}


def _edited_error_stats(histogram):
    stats = _error_stats(histogram)
    pixels = stats.pop("rgb_samples") // 3
    return {"edited_pixels": pixels, **{f"edited_{key}": value for key, value in stats.items()}}


def _silhouette_iou(intersection, union):
    return intersection / union if union else None


def compare_section(source_positions, source_uvs, source_normals, source_materials, source_textures,
                    target_positions, target_uvs, target_normals, target_materials, target_textures,
                    resolution=512, *, source_edited=None, target_edited=None):
    """Compare visible diffuse RGB and silhouettes in shared scene coordinates.

    Pass per-triangle corner arrays and one integer material ID per triangle.
    Diffuse textures map a material ID to a Pillow RGB image and a repeat flag.
    Both UV arrays use top-origin V. RGB errors are measured on shared covered
    pixels after linear-light shading and conversion to 8-bit sRGB.
    Optional boolean masks mark edited triangles. Edited RGB errors are measured
    where both surfaces are visible and either visible triangle is marked.
    """
    if not isinstance(resolution, (int, np.integer)) or not 16 <= resolution <= 2048:
        raise ValueError("Use a comparison resolution from 16 to 2048 pixels")
    image_cache = {}
    source = _section(source_positions, source_uvs, source_normals, source_materials, source_textures,
                      image_cache, source_edited)
    target = _section(target_positions, target_uvs, target_normals, target_materials, target_textures,
                      image_cache, target_edited)

    gap, header, label, tile = 8, 24, 19, min(resolution, 320)
    preview = Image.new("RGB", (3 * tile + 4 * gap, header + len(VIEWS) * (tile + label + gap) + gap), "#20232a")
    draw = ImageDraw.Draw(preview)
    offsets = [gap + j * (tile + gap) for j in range(3)]
    for x, title in zip(offsets, ("Source diffuse", "Atlas diffuse", "RGB error (4x), mask change in pink")):
        draw.text((x, 6), title, fill="white")

    histogram = np.zeros(256, dtype=np.int64)
    edited_histogram = np.zeros_like(histogram)
    counts = {"source_pixels": 0, "target_pixels": 0, "compared_pixels": 0, "union_pixels": 0}
    views = []
    for i, (name, direction) in enumerate(VIEWS):
        source_projection, target_projection = _project_pair(source.positions, target.positions, direction, resolution)
        source_mask, source_rgb, source_edits = _render(source, source_projection, resolution)
        target_mask, target_rgb, target_edits = _render(target, target_projection, resolution)
        intersection = source_mask & target_mask
        view_counts = {"source_pixels": int(source_mask.sum()), "target_pixels": int(target_mask.sum()),
                       "compared_pixels": int(intersection.sum()), "union_pixels": int((source_mask | target_mask).sum())}
        for key, count in view_counts.items():
            counts[key] += count

        errors = np.abs(source_rgb[intersection].astype(np.int16) - target_rgb[intersection])
        view_histogram = np.bincount(errors.ravel(), minlength=256)
        histogram += view_histogram
        edited_errors = errors[(source_edits | target_edits)[intersection]]
        edited_view_histogram = np.bincount(edited_errors.ravel(), minlength=256)
        edited_histogram += edited_view_histogram
        views.append({"name": name, "direction": list(direction), **view_counts,
                      "silhouette_iou": _silhouette_iou(view_counts["compared_pixels"], view_counts["union_pixels"]),
                      **_error_stats(view_histogram), **_edited_error_stats(edited_view_histogram)})

        difference = np.empty_like(source_rgb)
        difference[:] = BACKGROUND
        difference[intersection] = np.minimum(errors * 4, 255)
        difference[source_mask ^ target_mask] = MASK_DIFFERENCE
        y = header + i * (tile + label + gap)
        draw.text((gap, y + 2), name, fill="white")
        for x, pixels in zip(offsets, (source_rgb, target_rgb, difference)):
            image = Image.fromarray(pixels)
            preview.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (x, y + label))

    metrics = {"resolution": int(resolution), **counts,
               "silhouette_iou": _silhouette_iou(counts["compared_pixels"], counts["union_pixels"]),
               **_error_stats(histogram), **_edited_error_stats(edited_histogram),
               "light_direction": LIGHT.tolist(), "ambient": AMBIENT, "views": views}
    return ComparisonResult(metrics, preview)
