"""Export a simplified rectangular patch with its diffuse texture baked into a GLB."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from lodestar_textured_glb import write_textured_glb


@dataclass
class BakeResult:
    positions: np.ndarray
    triangles: np.ndarray
    uvs: np.ndarray
    normal: np.ndarray
    texture: Image.Image
    preview: Image.Image
    metrics: dict


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def boundary_loop(faces):
    edges = defaultdict(list)
    for index, face in enumerate(faces.tolist()):
        for a, b in zip(face, face[1:] + face[:1]):
            edges[tuple(sorted((a, b)))].append((a, b, index))
    following = {}
    incoming = set()
    neighbors = [set() for _ in faces]
    for occurrences in edges.values():
        a, b, first = occurrences[0]
        if len(occurrences) == 1:
            if a == b or a in following or b in incoming:
                raise ValueError("Patch boundary branches")
            following[a] = b
            incoming.add(b)
        elif len(occurrences) == 2 and occurrences[1][:2] == (b, a):
            second = occurrences[1][2]
            neighbors[first].add(second)
            neighbors[second].add(first)
        else:
            raise ValueError("Patch has inconsistent winding or a nonmanifold edge")

    connected = {0}
    pending = [0]
    while pending:
        new_neighbors = neighbors[pending.pop()] - connected
        connected.update(new_neighbors)
        pending.extend(new_neighbors)
    if len(connected) != len(faces) or not following or following.keys() != incoming:
        raise ValueError("Patch needs a single connected surface with a closed boundary")
    start = min(following)
    loop = [start]
    while following[loop[-1]] != start:
        loop.append(following[loop[-1]])
    if len(loop) != len(following):
        raise ValueError("Patch contains multiple boundary loops")
    return loop


def rectangular_domain(positions, normals):
    vertices, indices = np.unique(positions.reshape(-1, 3), axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)
    loop = boundary_loop(faces)
    extent = float(np.ptp(vertices, axis=0).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Patch needs finite, nonzero extent")
    tolerance = extent * 1e-7

    cross = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    largest = int(lengths.argmax())
    if not np.isfinite(lengths).all() or lengths[largest] <= 0:
        raise ValueError("Patch contains degenerate triangles")
    axis = cross[largest] / lengths[largest]
    if np.any(cross @ axis <= 0):
        raise ValueError("Patch contains degenerate or inconsistently wound triangles")
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(lengths == 0) or not np.isfinite(lengths).all():
        raise ValueError("Patch needs finite, nonzero shading normals")
    unit_normals = normals / lengths
    normal = unit_normals[0, 0]
    if not np.allclose(unit_normals, normal, rtol=0, atol=1e-6):
        raise ValueError("Patch needs a single shading normal")

    # A straight boundary point adds no shape. Keep the original coordinates at the corners.
    corners = []
    for i, index in enumerate(loop):
        a = vertices[loop[i - 1]]
        b = vertices[loop[(i + 1) % len(loop)]]
        edge = b - a
        delta = vertices[index] - a
        edge_length = np.linalg.norm(edge)
        if edge_length == 0:
            raise ValueError("Patch has a degenerate boundary")
        if np.linalg.norm(np.cross(edge, delta)) > tolerance * edge_length or not 0 < delta @ edge < edge @ edge:
            corners.append(index)
    if len(corners) != 4:
        raise ValueError("Patch boundary must form a rectangle")
    rectangle = vertices[corners]
    origin = rectangle[0]
    u, v = rectangle[1] - origin, rectangle[3] - origin
    u_length, v_length = np.linalg.norm(u), np.linalg.norm(v)
    if (min(u_length, v_length) <= tolerance or
            abs(u @ v) > tolerance * max(u_length, v_length) or
            np.linalg.norm(rectangle[2] - origin - u - v) > tolerance):
        raise ValueError("Patch boundary must form a rectangle")

    offsets = positions - origin
    max_plane_error = float(np.max(np.abs(offsets @ axis)))
    if max_plane_error > tolerance:
        raise ValueError("Patch must be planar")
    projected = np.stack(((offsets @ u) / (u @ u), (offsets @ v) / (v @ v)), axis=-1)
    areas = cross2(projected[:, 1] - projected[:, 0], projected[:, 2] - projected[:, 0])
    if (np.any(areas <= 0) or np.any(projected < -1e-6) or np.any(projected > 1 + 1e-6) or
            not np.isclose(areas.sum(), 2.0, rtol=0, atol=2e-6)):
        raise ValueError("Source triangles must cover the rectangle once")
    return rectangle, projected, normal, max_plane_error


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)


def linear_image(image):
    return srgb_to_linear(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)


def sample(texture, uv, *, repeat):
    height, width, _ = texture.shape
    coordinates = (uv % 1 if repeat else np.clip(uv, 0, 1)) * [width, height] - 0.5
    base = np.floor(coordinates).astype(np.intp)
    fraction = coordinates - base
    x0, y0 = base.T
    x1, y1 = x0 + 1, y0 + 1
    if repeat:
        x0, x1, y0, y1 = x0 % width, x1 % width, y0 % height, y1 % height
    else:
        x0, x1 = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
        y0, y1 = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    fx, fy = fraction[:, 0, None], fraction[:, 1, None]
    top = texture[y0, x0] * (1 - fx) + texture[y0, x1] * fx
    bottom = texture[y1, x0] * (1 - fx) + texture[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def source_colors(points, faces, source_uvs, texture, *, materials=None, boundary=()):
    uv = np.empty_like(points)
    owners = np.full(len(points), -1, dtype=np.intp)
    for index, ((a, b, c), (ua, ub, uc)) in enumerate(zip(faces, source_uvs)):
        candidates = np.flatnonzero(owners < 0)
        if not candidates.size:
            break
        delta = points[candidates] - a
        e1, e2 = b - a, c - a
        area = cross2(e1, e2)
        wb = cross2(delta, e2) / area
        wc = cross2(e1, delta) / area
        inside = (wb >= -1e-7) & (wc >= -1e-7) & (wb + wc <= 1 + 1e-7)
        matched = candidates[inside]
        uv[matched] = ua + wb[inside, None] * (ub - ua) + wc[inside, None] * (uc - ua)
        owners[matched] = index
    pending = np.flatnonzero(owners < 0)
    if pending.size:
        if not boundary:
            raise ValueError("Source triangles leave uncovered points in the patch")
        # Extend the nearest boundary color into the unused part of the tile. The
        # extension also gives bilinear samples near polygon edges valid neighbors.
        missing = points[pending]
        distances = np.full(len(pending), np.inf)
        for face, first, last in boundary:
            a, b = faces[face, [first, last]]
            ua, ub = source_uvs[face, [first, last]]
            edge = b - a
            weight = np.clip((missing - a) @ edge / (edge @ edge), 0.0, 1.0)
            closest = a + weight[:, None] * edge
            distance = np.sum((closest - missing) ** 2, axis=1)
            better = distance < distances
            uv[pending[better]] = ua + weight[better, None] * (ub - ua)
            owners[pending[better]] = face
            distances[better] = distance[better]
    uv[:, 1] = 1 - uv[:, 1]
    if materials is None:
        return sample(texture, uv, repeat=True)
    colors = np.empty((len(uv), 3))
    sample_materials = materials[owners]
    for material in np.unique(sample_materials):
        selected = sample_materials == material
        colors[selected] = sample(texture[material], uv[selected], repeat=True)
    return colors


def grid(width, height, x_offset, y_offset, first_row=0, last_row=None):
    x = (np.arange(width) + x_offset) / width
    y = (np.arange(first_row, height if last_row is None else last_row) + y_offset) / height
    xx, yy = np.meshgrid(x, y)
    return np.stack((xx.ravel(), yy.ravel()), axis=-1)


def rgb8(linear):
    return np.rint(linear_to_srgb(linear) * 255).astype(np.uint8)


def bake_texture(projected, source_uvs, source, width, height, supersamples, *, materials=None, boundary=()):
    baked = np.empty((height, width, 3), dtype=np.uint8)
    for first in range(0, height, 32):
        last = min(first + 32, height)
        colors = np.zeros(((last - first) * width, 3))
        for y in range(supersamples):
            for x in range(supersamples):
                points = grid(width, height, (x + 0.5) / supersamples, (y + 0.5) / supersamples, first, last)
                points[:, 1] = 1 - points[:, 1]
                colors += source_colors(points, projected, source_uvs, source, materials=materials, boundary=boundary)
        baked[first:last] = rgb8(colors / supersamples ** 2).reshape(last - first, width, 3)
    return baked


def bake_patch(positions, source_uvs, normals, texture, resolution=256, supersamples=2):
    """Bake opaque diffuse color from bottom-origin UV0 onto a two-triangle rectangle."""
    start = perf_counter()
    positions = np.asarray(positions, dtype=np.float64)
    source_uvs = np.asarray(source_uvs, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if (positions.ndim != 3 or positions.shape[1:] != (3, 3) or positions.shape[0] < 3 or
            source_uvs.shape != positions.shape[:2] + (2,) or normals.shape != positions.shape or
            not all(np.isfinite(array).all() for array in (positions, source_uvs, normals)) or
            np.max(np.abs(positions)) > np.finfo(np.float32).max):
        raise ValueError("Patch needs at least three triangles with finite position, UV and normal data")
    if not 16 <= resolution <= 2048 or not 1 <= supersamples <= 4:
        raise ValueError("Use a resolution from 16 to 2048 and 1 to 4 samples per texel axis")
    if texture.convert("RGBA").getextrema()[3] != (255, 255):
        raise ValueError("This bake requires an opaque source texture")

    rectangle, projected, normal, plane_error = rectangular_domain(positions, normals)
    source = linear_image(texture)
    baked = bake_texture(projected, source_uvs, source, resolution, resolution, supersamples)
    bake_ms = (perf_counter() - start) * 1000

    # Validate at a separate, offset grid rather than at the locations sampled by the bake.
    check_size = 256
    target_uv = grid(check_size, check_size, 0.371, 0.619)
    source_xy = target_uv * [1, -1] + [0, 1]
    reference = linear_to_srgb(source_colors(source_xy, projected, source_uvs, source)) * 255
    rebaked = linear_to_srgb(sample(srgb_to_linear(baked.astype(np.float32) / 255), target_uv, repeat=False)) * 255
    error = np.abs(reference - rebaked)
    metrics = {
        "source_triangles": len(positions),
        "target_triangles": 2,
        "resolution": resolution,
        "supersamples_per_texel": supersamples ** 2,
        "max_source_plane_deviation": plane_error,
        "validation_samples": len(target_uv),
        "mean_abs_error": float(error.mean()),
        "p95_abs_error": float(np.percentile(error, 95)),
        "max_abs_error": float(error.max()),
        "prepare_and_bake_ms": bake_ms,
    }
    preview = comparison_image(reference, rebaked, error, check_size)
    return BakeResult(
        rectangle.astype("<f4"), np.array([[0, 1, 2], [0, 2, 3]], dtype="<u2"),
        np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype="<f4"), normal.astype("<f4"),
        Image.fromarray(baked), preview, metrics,
    )


def comparison_image(reference, rebaked, error, size):
    header = 24
    comparison = Image.new("RGB", (size * 3, size + header), "#20232a")
    draw = ImageDraw.Draw(comparison)
    for i, (title, pixels) in enumerate(zip(
        ("Source diffuse", "Baked diffuse", "Absolute error (4x)"),
        (reference, rebaked, np.clip(error * 4, 0, 255)),
    )):
        draw.text((i * size + 6, 6), title, fill="white")
        image = Image.fromarray(np.rint(pixels).astype(np.uint8).reshape(size, size, 3))
        comparison.paste(image, (i * size, header))
    return comparison


def write_glb(path, result):
    """Write positions, UVs, shading normals, and the baked PNG in one glTF 2.0 binary."""
    primitive = (result.positions, np.broadcast_to(result.normal, result.positions.shape), result.uvs,
                 result.triangles, 0)
    write_textured_glb(path, [primitive], {0: (result.texture, False)}, unlit=True)


def face_indices(value):
    try:
        indices = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated, zero-based triangle indices") from error
    if min(indices) < 0 or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("Use unique non-negative triangle indices")
    return indices


def main():
    from lodestar_patch_source import load_patch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source mesh file")
    parser.add_argument("diffuse", type=Path, help="Opaque source diffuse image")
    parser.add_argument("glb", type=Path, help="Output GLB path")
    parser.add_argument("--library", type=Path, required=True, help="Compiled Lodestar C library")
    parser.add_argument("--faces", type=face_indices, required=True, help="Comma-separated triangle indices")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--supersamples", type=int, default=2, help="Samples per texel axis")
    parser.add_argument("--preview", type=Path, help="Optional side-by-side PNG comparing diffuse color")
    parser.add_argument("--report", type=Path, help="Optional JSON report")
    args = parser.parse_args()

    try:
        inputs = {path.resolve() for path in (args.source, args.diffuse, args.library)}
        outputs = [path.resolve() for path in (args.glb, args.preview, args.report) if path is not None]
        if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
            raise ValueError("Input and output paths must be distinct")
        positions, uv, normals = load_patch(args.library, args.source, args.faces)
        with Image.open(args.diffuse) as diffuse:
            result = bake_patch(positions, uv, normals, diffuse, args.resolution, args.supersamples)
        write_glb(args.glb, result)
        report = json.dumps({**result.metrics, "source_face_indices": args.faces}, indent=2, allow_nan=False)
        if args.preview:
            result.preview.save(args.preview, format="PNG")
        if args.report:
            args.report.write_text(report + "\n", encoding="utf-8")
        print(report)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Lodestar texture bake: {error}\n")


if __name__ == "__main__":
    main()
