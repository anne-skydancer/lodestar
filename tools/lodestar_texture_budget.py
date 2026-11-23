"""Reduce the base-level texture footprint of baked sections."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from PIL import Image


def _crop_period(values, length):
    # Include the two bilinear taps and a guard texel on either end. Work in
    # the unwrapped lattice so a face that crosses a repeat is handled whole.
    coordinates = values * length - 0.5
    if np.abs(coordinates).max() > 2**50:
        return None
    first = np.floor(coordinates.min(axis=1)).astype(np.int64) - 1
    last = np.floor(coordinates.max(axis=1)).astype(np.int64) + 2
    counts = last - first + 1
    if np.any(counts >= length):
        return None

    used = np.zeros(length, dtype=bool)
    for start, count in zip(first % length, counts):
        stop = start + count
        used[start:stop] = True
        if stop > length:
            used[:stop - length] = True
    occupied = np.flatnonzero(used)
    following = np.r_[occupied[1:], occupied[0] + length]
    gaps = following - occupied - 1
    cut = int(gaps.argmax())
    if gaps[cut] == 0:
        return None

    start = int(following[cut] % length)
    size = int(length - gaps[cut])
    origins = start + ((first - start) // length) * length
    remapped = (values * length - origins[:, None]) / size
    return np.arange(start, start + size) % length, remapped


def crop_repeat_textures(uvs, materials, textures):
    """Return per-corner UVs, texture bindings, and a list of exact crops.

    A crop keeps every texel that an interpolated face UV could sample and
    retains repeat mode along both axes. Shared source images get one common
    crop across all their material bindings. Clamped images stay untouched.
    This applies to base-level bilinear sampling without a mip chain.
    """
    uvs = np.asarray(uvs, dtype=np.float64)
    materials = np.asarray(materials)
    if (materials.ndim != 1 or uvs.shape != (len(materials), 3, 2) or
            not np.isfinite(uvs).all()):
        raise ValueError("Texture cropping requires finite per-corner UVs and one material per face")

    bindings = defaultdict(list)
    for material in np.unique(materials):
        image, repeat = textures[material]
        bindings[id(image)].append((int(material), image, repeat))

    result_uvs = uvs.copy()
    result_textures = dict(textures)
    changes = []
    for group in bindings.values():
        if any(not repeat for _, _, repeat in group):
            continue
        identifiers = [material for material, _, _ in group]
        image = group[0][1]
        faces = np.isin(materials, identifiers)
        original_uvs = uvs[faces]
        crops = [_crop_period(original_uvs[:, :, axis], length) for axis, length in enumerate(image.size)]
        if all(crop is None for crop in crops):
            continue

        columns, rows = [crop[0] if crop is not None else np.arange(length)
                         for crop, length in zip(crops, image.size)]
        cropped = Image.fromarray(np.asarray(image)[np.ix_(rows, columns)])
        for axis, crop in enumerate(crops):
            if crop is not None:
                result_uvs[faces, :, axis] = crop[1]
        for material in identifiers:
            result_textures[material] = cropped, True
        changes.append({"materials": identifiers, "source_size": list(image.size), "size": list(cropped.size)})
    return result_uvs, result_textures, changes


@dataclass
class _RetiredFace:
    faces: tuple[int, ...]
    projected: np.ndarray
    normal: np.ndarray
    plane_error: float = 0.0
    texture_boundary: tuple = ((0, 0, 1), (0, 1, 2), (0, 2, 0))


def _face_uv_chart(uv, positions):
    origin = uv.min(axis=0)
    extent = uv.max(axis=0) - origin
    if np.all(extent > 0):
        projected = (uv - origin) / extent
        direction = np.linalg.det(projected[1:] - projected[:1])
        if abs(direction) > 1e-8:
            if direction < 0:
                projected[:, 1] = 1 - projected[:, 1]
            return projected

    # Degenerate texture coordinates still define a sampled color field over
    # the physical triangle. Use the position projection to provide a domain.
    normal = np.cross(positions[1] - positions[0], positions[2] - positions[0])
    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] == 0:
        raise ValueError("Cannot retexture a degenerate triangle")
    projected = positions[:, [(axis + 1) % 3, (axis + 2) % 3]].copy()
    if normal[axis] < 0:
        projected = projected[:, ::-1]
    return (projected - projected.min(axis=0)) / np.ptp(projected, axis=0)


def retirement_plans(section, plans, material_ids):
    """Move retained faces of the chosen materials into the replacement atlas.

    Supplemental plans keep the exact original positions, normals and triangle
    winding. A separate chart per retained triangle allows arbitrary source
    UV repeats. The full material image can be dropped when none of its other
    bindings reference it.
    """
    chosen = set(material_ids)
    known = set(section.triangle_materials.tolist())
    if chosen - known:
        raise ValueError(f"Cannot retire absent material IDs {sorted(chosen - known)}")
    changed = {face for layout, _, _, _ in plans for face in layout.faces}
    replacement = []
    triangle = np.array([[0, 1, 2]], dtype=np.intp)
    for face in np.flatnonzero(np.isin(section.triangle_materials, list(chosen))):
        if face in changed:
            continue
        positions = section.positions[face]
        projected = _face_uv_chart(section.source_uvs[face], positions)
        layout = _RetiredFace((int(face),), projected[None, ...], section.normals[face])
        target_uv = projected * [1, -1] + [0, 1]
        replacement.append((layout, positions, target_uv, triangle))
    return replacement


def _occupied_texels(uvs, size):
    width, height = size
    occupied = np.zeros((height, width), dtype=bool)
    for triangle in uvs * [width, height] - 0.5:
        if np.abs(triangle).max() > 2**50 or np.ptp(triangle[:, 1]) > 16 * height:
            return np.ones_like(occupied)
        rows = np.arange(np.floor(triangle[:, 1].min()) - 2,
                         np.ceil(triangle[:, 1].max()) + 3, dtype=np.int64)
        limits = rows[:, None] + [-2, 2]
        left = np.full(len(rows), np.inf)
        right = np.full(len(rows), -np.inf)
        # The extreme x of a triangle clipped to a horizontal slab lies at
        # a source vertex or an intersection of a triangle edge with the slab.
        for x, y in triangle:
            in_slab = (limits[:, 0] <= y) & (y <= limits[:, 1])
            left[in_slab] = np.minimum(left[in_slab], x)
            right[in_slab] = np.maximum(right[in_slab], x)
        for a, b in zip(triangle, np.roll(triangle, 1, axis=0)):
            if a[1] == b[1]:
                continue
            t = (limits - a[1]) / (b[1] - a[1])
            x = a[0] + t * (b[0] - a[0])
            valid = (0 <= t) & (t <= 1)
            left = np.minimum(left, np.where(valid, x, np.inf).min(axis=1))
            right = np.maximum(right, np.where(valid, x, -np.inf).max(axis=1))
        covered = np.isfinite(left)
        for row, low, high in zip(rows[covered] % height, left[covered], right[covered]):
            # One guard texel beyond the bilinear footprint, with wrapping.
            first, last = int(np.floor(low)) - 2, int(np.ceil(high)) + 2
            count = last - first + 1
            if count >= width:
                occupied[row] = True
            else:
                column = first % width
                stop = column + count
                occupied[row, column:stop] = True
                if stop > width:
                    occupied[row, :stop - width] = True
        if occupied.all():
            break
    return occupied


def _pack_in_unused_texels(tiles, occupied, gutter):
    dimensions = [(h + 2 * gutter, w + 2 * gutter) for h, w in (tile.pixels.shape[:2] for tile in tiles)]
    keys = (lambda hw: (hw[0] * hw[1],),
            lambda hw: (max(hw), min(hw)),
            lambda hw: (min(hw), max(hw)))
    for key in keys:
        order = sorted(range(len(tiles)), key=lambda index: key(dimensions[index]), reverse=True)
        pages = {material: mask.copy() for material, mask in occupied.items()}

        def integral(mask):
            return np.pad(mask.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))

        summed = {material: integral(mask) for material, mask in pages.items()}
        placed = [None] * len(tiles)
        for tile in order:
            height, width = dimensions[tile]
            choices = []
            for material, page in summed.items():
                for rotated, (h, w) in enumerate(((height, width), (width, height))):
                    if h >= page.shape[0] or w >= page.shape[1]:
                        continue
                    available = page[h:, w:] - page[:-h, w:] - page[h:, :-w] + page[:-h, :-w]
                    free = np.flatnonzero(available == 0)
                    if free.size:
                        y, x = map(int, np.unravel_index(free[0], available.shape))
                        choices.append((len(free), y * h + x * w, material, y, x, h, w, bool(rotated)))
            if not choices:
                break
            _, _, material, y, x, h, w, rotated = min(choices)
            pages[material][y:y + h, x:x + w] = True
            summed[material] = integral(pages[material])
            placed[tile] = material, x, y, rotated
        else:
            return placed
    raise ValueError("The bake tiles do not fit in unused texels of the retained repeating textures")


def reuse_atlas_texels(uvs, materials, textures, tiles, atlas_id, *, gutter=2):
    """Replace the extra atlas with padded tiles in proven unused texel rectangles.

    Faces must be ordered as the retained original triangles followed by each
    tile's triangles in order. UVs use top-origin V and base-level bilinear
    filtering. The retained UVs and their sampled texels stay intact.
    """
    uvs = np.asarray(uvs, dtype=np.float64)
    materials = np.asarray(materials)
    if (materials.ndim != 1 or uvs.shape != (len(materials), 3, 2) or
            not np.isfinite(uvs).all() or gutter < 1):
        raise ValueError("Atlas reuse requires finite per-corner UVs and a positive gutter")
    first = len(materials) - sum(len(tile.triangles) for tile in tiles)
    if (first < 1 or first >= len(materials) or np.any(materials[:first] == atlas_id) or
            np.any(materials[first:] != atlas_id)):
        raise ValueError("Atlas reuse needs retained triangles followed by all bake tiles")

    groups = defaultdict(list)
    for material in np.unique(materials[:first]):
        image, repeat = textures[material]
        groups[id(image)].append((int(material), image, repeat))
    pages = {group[0][0]: group for group in groups.values() if all(repeat for _, _, repeat in group)}
    occupied = {}
    for material, group in pages.items():
        selected = np.isin(materials[:first], [key for key, _, _ in group])
        mask = _occupied_texels(uvs[:first][selected], group[0][1].size)
        if not mask.all():
            occupied[material] = mask
    placement = _pack_in_unused_texels(tiles, occupied, gutter)

    remapped_uvs = uvs.copy()
    remapped_materials = materials.copy()
    remapped_textures = {material: binding for material, binding in textures.items() if material != atlas_id}
    changed = {}
    placements = []
    for tile, (material, x, y, rotated) in zip(tiles, placement):
        if material not in changed:
            changed[material] = np.array(textures[material][0])
        page = changed[material]
        pixels, tile_uvs = tile.pixels, tile.uvs
        if rotated:
            pixels = np.rot90(pixels)
            tile_uvs = np.column_stack((tile_uvs[:, 1], 1 - tile_uvs[:, 0]))
        height, width = pixels.shape[:2]
        padded = np.pad(pixels, ((gutter, gutter), (gutter, gutter), (0, 0)), mode="edge")
        page[y:y + height + 2 * gutter, x:x + width + 2 * gutter] = padded
        last = first + len(tile.triangles)
        mapped = (tile_uvs * [width, height] + [x + gutter, y + gutter]) / [page.shape[1], page.shape[0]]
        remapped_uvs[first:last] = mapped[tile.triangles]
        remapped_materials[first:last] = material
        placements.append({"material": material, "offset": [x, y], "size": [width, height], "rotated": rotated})
        first = last
    for material, pixels in changed.items():
        image = Image.fromarray(pixels)
        for identifier, _, _ in pages[material]:
            remapped_textures[identifier] = image, True
    return remapped_uvs, remapped_materials, remapped_textures, placements
