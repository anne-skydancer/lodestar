"""Share diffuse material bindings across the named parts of static mesh LODs."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from PIL import Image

from lodestar_testhall_textures import TexturedScene, _replacement_corners


@dataclass(frozen=True)
class _Faces:
    level: str
    indices: np.ndarray
    texels: np.ndarray


@dataclass
class _Region:
    image: Image.Image
    repeat: bool
    bounds: tuple
    faces: list


def textured_sections(scene, replacements=None):
    """Get named source slices and apply any named textured replacements."""
    replacements = {} if replacements is None else replacements
    if not scene.sections or set(replacements) - scene.sections.keys():
        raise ValueError("Textured LOD replacements must reference named source sections")
    result = {}
    for name, section in scene.sections.items():
        if name in replacements:
            positions, uvs, normals, materials, textures = replacements[name]
            corners = _replacement_corners(positions, uvs, normals)
        else:
            positions, uvs, normals, materials = (array[section.face_indices] for array in scene.arrays[:4])
            textures, corners = scene.textures, scene.corner_indices[section.face_indices]
        bindings = {int(label): textures[label] for label in np.unique(materials)}
        result[name] = TexturedScene(positions, uvs, normals, materials, bindings, corners)
    return result


def combine_lod_parts(parts):
    """Collect prepared named parts for rendering or writing a single mesh."""
    textures, corners, material_chunks = {}, [], []
    digests, keys = {}, {}
    available = max(max(scene.textures) for scene in parts.values()) + 1
    offset = 0
    for scene in parts.values():
        materials = scene.materials.astype(np.int64, copy=True)
        for source, (image, repeat) in scene.textures.items():
            label = int(source)
            binding = _image_key(image, digests), repeat
            if label in keys and keys[label] != binding:
                label = available
                available += 1
                materials[scene.materials == source] = label
            textures[label] = (image, repeat)
            keys[label] = binding
        material_chunks.append(materials)
        corners.append(scene.corner_indices + offset)
        offset = int(corners[-1].max()) + 1
    attributes = [np.concatenate([scene.arrays[i] for scene in parts.values()]) for i in range(3)]
    return TexturedScene(*attributes, np.concatenate(material_chunks), textures, np.concatenate(corners))


def _image_key(image, digests):
    if id(image) not in digests:
        digests[id(image)] = (image.size, sha256(image.tobytes()).digest())
    return digests[id(image)]


def _bindings(parts, digests):
    bindings = {}
    for level, scene in parts.items():
        positions, uvs, normals, materials = scene.arrays[:4]
        if (positions.ndim != 3 or positions.shape[1:] != (3, 3) or not len(positions) or
                normals.shape != positions.shape or uvs.shape != positions.shape[:2] + (2,) or
                materials.shape != (len(positions),) or materials.dtype.kind not in "iu" or
                not all(np.isfinite(array).all() for array in (positions, uvs, normals))):
            raise ValueError(f"{level} needs finite textured triangle corners and integer material IDs")
        for label in np.unique(materials):
            if label not in scene.textures:
                raise ValueError(f"{level} material {label} has no diffuse binding")
            image, repeat = scene.textures[label]
            if (not isinstance(image, Image.Image) or image.mode != "RGB" or min(image.size) < 1 or
                    not isinstance(repeat, (bool, np.bool_))):
                raise ValueError(f"{level} material {label} needs a nonempty RGB image and repeat flag")
            key = _image_key(image, digests), bool(repeat)
            if key not in bindings:
                bindings[key] = (image, bool(repeat), [])
            indices = np.flatnonzero(materials == label)
            selected = uvs[indices].astype(np.float64)
            # Shift each repeating triangle by whole periods, keeping affine
            # interpolation across triangle edges and texture wrap boundaries.
            if repeat:
                selected -= np.floor(selected.min(axis=1))[:, None, :]
            bindings[key][2].append(_Faces(level, indices, selected * image.size))
    return bindings


def _face_bounds(texels):
    minimum = np.floor(texels.min(axis=1))
    maximum = np.maximum(np.ceil(texels.max(axis=1)), minimum + 1)
    if np.max(np.abs([minimum, maximum])) >= 2**53:
        raise ValueError("Texture coordinates exceed the supported texel range")
    return np.column_stack((minimum, maximum)).astype(np.int64)


def _size(bounds, gutter):
    left, top, right, bottom = bounds
    return right - left + 2 * gutter, bottom - top + 2 * gutter


def _area(bounds, gutter):
    width, height = _size(bounds, gutter)
    return width * height


def _windows(faces):
    windows = {}
    for group in faces:
        for index, bounds in enumerate(_face_bounds(group.texels)):
            windows.setdefault(tuple(map(int, bounds)), []).append((group, index))
    return windows


def _regions(image, repeat, faces, limit, gutter):
    # Crop each chart to the texels its faces use. Coalesce overlapping crops
    # when the union costs less atlas space than storing two padded crops.
    windows = _windows(faces)
    regions = []
    for bounds, users in sorted(windows.items(), key=lambda item: (-_area(item[0], gutter), item[0])):
        if max(_size(bounds, gutter)) > limit:
            raise ValueError("A texture chart exceeds the shared atlas dimensions at its original density")
        left, top, right, bottom = bounds
        for region in regions:
            a, b, c, d = region.bounds
            if max(left, a) >= min(right, c) or max(top, b) >= min(bottom, d):
                continue
            union = min(left, a), min(top, b), max(right, c), max(bottom, d)
            if (max(_size(union, gutter)) <= limit and
                    _area(union, gutter) <= _area(bounds, gutter) + _area(region.bounds, gutter)):
                region.bounds = union
                region.faces.extend(users)
                break
        else:
            regions.append(_Region(image, repeat, bounds, users.copy()))
    return regions


def _power_of_two(size):
    return 1 << (size - 1).bit_length()


def _shelves(dimensions, order, width, limit):
    rows, offsets, height = [], [None] * len(dimensions), 0
    for index in order:
        tile_width, tile_height = dimensions[index]
        for row in rows:
            x, y, row_height = row
            if x + tile_width <= width and tile_height <= row_height:
                row[0] += tile_width
                offsets[index] = (x, y)
                break
        else:
            if height + tile_height > limit:
                return None
            offsets[index] = (0, height)
            rows.append([tile_width, height, tile_height])
            height += tile_height
    return _power_of_two(max(row[0] for row in rows)), _power_of_two(height), offsets


def _free_rectangles(dimensions, order, width, height, split_long):
    free, offsets = [(0, 0, width, height)], [None] * len(dimensions)
    for index in order:
        tile_width, tile_height = dimensions[index]
        candidates = []
        for slot, (_, _, w, h) in enumerate(free):
            dx, dy = w - tile_width, h - tile_height
            if dx >= 0 and dy >= 0:
                candidates.append((min(dx, dy), max(dx, dy), slot))
        if not candidates:
            return None
        _, _, slot = min(candidates)
        x, y, w, h = free.pop(slot)
        offsets[index] = x, y
        dx, dy = w - tile_width, h - tile_height
        if (dx > dy) == split_long:
            remaining = [(x + tile_width, y, dx, h), (x, y + tile_height, tile_width, dy)]
        else:
            remaining = [(x + tile_width, y, dx, tile_height), (x, y + tile_height, w, dy)]
        free.extend(rect for rect in remaining if rect[2] and rect[3])
    return width, height, offsets


def _pack(regions, limit, gutter):
    sizes = [_size(region.bounds, gutter) for region in regions]
    orders = [sorted(range(len(sizes)), key=key) for key in (
        lambda index: (-sizes[index][1], -sizes[index][0], index),
        lambda index: (-sizes[index][0], -sizes[index][1], index),
        lambda index: (-sizes[index][0] * sizes[index][1], -max(sizes[index]), index),
    )]
    exponents = [range(_power_of_two(max(size[axis] for size in sizes)).bit_length() - 1, limit.bit_length())
                 for axis in (0, 1)]
    layouts = [_shelves(sizes, orders[0], 1 << exponent, limit) for exponent in exponents[0]]

    def footprint(layout):
        return layout[0] * layout[1], max(layout[:2]), layout[0]

    best = min((layout for layout in layouts if layout is not None), default=None, key=footprint)
    minimum_area = sum(w * h for w, h in sizes)
    shapes = [(1 << x, 1 << y) for x in exponents[0] for y in exponents[1] if 1 << (x + y) >= minimum_area]
    for width, height in sorted(shapes, key=footprint):
        if best is not None and footprint((width, height)) >= footprint(best):
            break
        for order in orders:
            for split_long in (True, False):
                packed = _free_rectangles(sizes, order, width, height, split_long)
                if packed is not None:
                    return packed
    if best is None:
        raise ValueError("The shared texture charts do not fit one atlas at their original density")
    return best


def _pixels(image, repeat, bounds, gutter):
    left, top, right, bottom = bounds
    x = np.arange(left - gutter, right + gutter)
    y = np.arange(top - gutter, bottom + gutter)
    if repeat:
        x, y = x % image.width, y % image.height
    else:
        x, y = x.clip(0, image.width - 1), y.clip(0, image.height - 1)
    return x, y


def _retained_bindings(bindings, high, limit, slots, gutter):
    direct, mapped = [], []
    for key, (_, repeat, faces) in bindings.items():
        if not repeat or not any(group.level == high for group in faces):
            mapped.append(key)
        else:
            direct.append(key)
    if len(direct) + bool(mapped) > slots:
        candidates = []
        for key in direct:
            image, repeat, faces = bindings[key]
            windows = _windows(faces)
            if all(max(_size(bounds, gutter)) <= limit for bounds in windows):
                regions = _regions(image, repeat, faces, limit, gutter)
                candidates.append((sum(_area(region.bounds, gutter) for region in regions), key))
        count = len(direct) - slots + 1
        selected = [key for _, key in sorted(candidates)[:count]]
        if len(selected) != count:
            raise ValueError("The material slot budget cannot preserve the full repeating UV spans")
        mapped.extend(selected)
        direct = [key for key in direct if key not in selected]
    return direct, mapped


def _anchor(bindings, direct, high, limit, gutter):
    """Pick an existing small High triangle without emptying a reference slot."""
    choices = []
    for key in direct:
        high_groups = [group for group in bindings[key][2] if group.level == high]
        if sum(len(group.indices) for group in high_groups) < 2:
            continue
        for group in high_groups:
            bounds = _face_bounds(group.texels)
            dimensions = bounds[:, 2:] - bounds[:, :2] + 2 * gutter
            eligible = np.flatnonzero(np.all(dimensions <= limit, axis=1))
            if len(eligible):
                areas = np.prod(dimensions[eligible], axis=1)
                index = int(areas.argmin())
                choices.append((int(areas[index]), key, group, int(eligible[index])))
    if not choices:
        raise ValueError("The LODs need a shared texture slot with an existing High triangle")
    _, key, group, index = min(choices, key=lambda choice: choice[:2])
    return key, _Faces(high, group.indices[index:index + 1], group.texels[index:index + 1])


def _make_atlas(regions, limit, gutter, uvs, materials, label):
    width, height, offsets = _pack(regions, limit, gutter)
    atlas = np.zeros((height, width, 3), dtype=np.uint8)
    images = {}
    for region, (x, y) in zip(regions, offsets):
        image = region.image
        if id(image) not in images:
            images[id(image)] = np.asarray(image)
        source_x, source_y = _pixels(image, region.repeat, region.bounds, gutter)
        atlas[y:y + len(source_y), x:x + len(source_x)] = images[id(image)][np.ix_(source_y, source_x)]
        left, top = region.bounds[:2]
        translation = np.array([x + gutter - left, y + gutter - top])
        for group, index in region.faces:
            face = group.indices[index]
            uvs[group.level][face] = (group.texels[index] + translation) / [width, height]
            materials[group.level][face] = label
    return Image.fromarray(atlas)


def _share_part(parts, high, digests, first_label, limit, slots, gutter):
    bindings = _bindings(parts, digests)
    direct, mapped = _retained_bindings(bindings, high, limit, slots, gutter)
    uvs = {level: part.uvs.astype(np.float64, copy=True) for level, part in parts.items()}
    materials = {level: np.full(len(part.positions), -1, dtype=np.int64) for level, part in parts.items()}
    textures, regions = {}, []
    for label, key in enumerate(direct, first_label):
        image, repeat, faces = bindings[key]
        if max(image.size) > limit or any(_power_of_two(size) != size for size in image.size):
            raise ValueError("A retained repeating texture needs upload-sized power-of-two dimensions")
        textures[label] = (image, repeat)
        for group in faces:
            materials[group.level][group.indices] = label
    for key in mapped:
        image, repeat, faces = bindings[key]
        regions.extend(_regions(image, repeat, faces, limit, gutter))

    atlas = None
    anchor_index = None
    if regions:
        if not any(group.level == high for key in mapped for group in bindings[key][2]):
            key, anchor = _anchor(bindings, direct, high, limit, gutter)
            image, repeat, _ = bindings[key]
            regions.extend(_regions(image, repeat, [anchor], limit, gutter))
            anchor_index = int(anchor.indices[0])
        label = first_label + len(direct)
        atlas = _make_atlas(regions, limit, gutter, uvs, materials, label)
        textures[label] = atlas, False

    prepared = {}
    for level, part in parts.items():
        assigned = np.unique(materials[level])
        if not set(assigned).issubset(textures) or len(assigned) > slots:
            raise ValueError(f"{level} did not receive valid shared material assignments")
        bound = {int(label): textures[label] for label in assigned}
        corners = _replacement_corners(part.positions, uvs[level], part.normals)
        prepared[level] = TexturedScene(part.positions, uvs[level], part.normals, materials[level], bound, corners)
    if any(set(part.textures) - prepared[high].textures.keys() for part in prepared.values()):
        raise ValueError("Every lower-LOD material must be used on real High geometry")
    report = {"atlas_size": list(atlas.size) if atlas else None, "atlas_regions": len(regions),
              "original_repeat_bindings": len(direct), "high_atlas_anchor_face": anchor_index,
              "materials": {level: sorted(part.textures) for level, part in prepared.items()}}
    return prepared, report, len(textures)


def share_lod_materials(levels, *, max_size=2048, material_limit=8, gutter=4):
    """Preserve texels while sharing named part material bindings among LODs.

    The first level is High. Its material bindings are the reference for the
    other LODs. Existing periodic surfaces keep their samplers when possible.
    Other charts use identical RGB data with texel-aligned UV remapping into
    one common power-of-two atlas per part. Every High slot has actual faces.
    """
    if (not levels or not next(iter(levels.values())) or type(max_size) is not int or
            max_size < 16 or _power_of_two(max_size) != max_size or type(gutter) is not int or
            not 1 <= gutter < max_size // 2 or type(material_limit) is not int or material_limit < 1):
        raise ValueError("Sharing LOD materials needs named parts, power-of-two limits and positive slot and gutter sizes")
    high = next(iter(levels))
    names = sorted(levels[high])
    if any(parts.keys() != levels[high].keys() for parts in levels.values()):
        raise ValueError("Every LOD must have the same named parts")

    prepared, report, digests = {level: {} for level in levels}, {}, {}
    next_label = 0
    for name in names:
        try:
            parts, information, count = _share_part({level: scenes[name] for level, scenes in levels.items()},
                                                     high, digests, next_label, max_size, material_limit, gutter)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        for level, part in parts.items():
            prepared[level][name] = part
        report[name] = information
        next_label += count
    images = {_image_key(image, digests) for part in prepared[high].values() for image, _ in part.textures.values()}
    return prepared, {"max_texture_size": max_size, "max_material_slots": material_limit,
                      "atlas_gutter_pixels": gutter, "unique_images": len(images),
                      "base_rgba8_bytes": sum(width * height * 4 for (width, height), _ in images),
                      "parts": report}
