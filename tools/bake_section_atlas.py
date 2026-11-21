"""Simplify certified patches in a COLLADA section and bake them into a shared GLB atlas."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from bake_planar_patch import bake_texture, linear_image
from lodestar_patch_layout import protect_contacts
from lodestar_textured_glb import write_textured_glb


GUTTER = 2


@dataclass
class Tile:
    faces: tuple
    materials: tuple[int, ...]
    positions: np.ndarray
    uvs: np.ndarray
    triangles: np.ndarray
    normal: np.ndarray
    pixels: np.ndarray
    plane_error: float
    uv_seams: int


def tile_size(projected, uv, image_size, scale, limit):
    domain_edges = projected[:, 1:] - projected[:, :1]
    texture_edges = (uv[:, 1:] - uv[:, :1]) * np.asarray(image_size).reshape(-1, 1, 2)
    gradients = np.linalg.solve(domain_edges, texture_edges)
    demand = np.linalg.norm(gradients, axis=2).max(axis=0)
    width, height = np.clip(np.ceil(demand * scale), 16, limit).astype(int)
    return int(width), int(height)


def pack_atlas(tiles, limit):
    dimensions = [(tile.pixels.shape[1] + 2 * GUTTER, tile.pixels.shape[0] + 2 * GUTTER) for tile in tiles]
    order = sorted(range(len(tiles)), key=lambda i: (-dimensions[i][1], -dimensions[i][0], i))
    minimum_width = max(width for width, _ in dimensions)
    widths = sorted({minimum_width, limit} | {2**n for n in range(4, 13) if minimum_width <= 2**n <= limit})
    best = None
    for width in widths:
        x = y = row_height = used_width = 0
        offsets = [None] * len(tiles)
        for i in order:
            w, h = dimensions[i]
            if x + w > width:
                x, y, row_height = 0, y + row_height, 0
            offsets[i] = (x, y)
            x += w
            used_width = max(used_width, x)
            row_height = max(row_height, h)
        height = y + row_height
        if used_width <= limit and height <= limit:
            layout = (used_width * height, max(used_width, height), used_width, height, offsets)
            if best is None or layout[:4] < best[:4]:
                best = layout
    if best is None:
        raise ValueError("The baked patches do not fit within the requested atlas limit")

    _, _, width, height, offsets = best
    atlas = np.zeros((height, width, 3), dtype=np.uint8)
    for tile, (x, y) in zip(tiles, offsets):
        pixels = np.pad(tile.pixels, ((GUTTER, GUTTER), (GUTTER, GUTTER), (0, 0)), mode="edge")
        h, w = pixels.shape[:2]
        atlas[y:y + h, x:x + w] = pixels
    return Image.fromarray(atlas), offsets


def read_textures(materials, folder):
    root = folder.resolve()
    by_file = {}
    textures = {}
    for material_id, material in sorted(materials.items()):
        path = (root / material.texture_filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError("A diffuse binding points outside the texture directory")
        if path not in by_file:
            with Image.open(path) as image:
                if image.convert("RGBA").getextrema()[3] != (255, 255):
                    raise ValueError(f"{path.name} needs an opaque diffuse texture")
                by_file[path] = image.convert("RGB")
        textures[int(material_id)] = (by_file[path], True)
    return textures


def internal_uv_seams(positions, uvs):
    edges = {}
    for face, uv in zip(positions, uvs):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            first, last = tuple(face[a]), tuple(face[b])
            uv_a, uv_b = tuple(uv[a]), tuple(uv[b])
            if first > last:
                first, last = last, first
                uv_a, uv_b = uv_b, uv_a
            edges.setdefault((first, last), []).append((uv_a, uv_b))
    return sum(len(values) == 2 and values[0] != values[1] for values in edges.values())


def build_tiles(section, plans, textures, scale, limit, supersamples):
    prepared = []
    source_textures = {material: linear_image(image) for material, (image, _) in textures.items()}
    for layout, positions, target_uv, triangles in plans:
        faces = list(layout.faces)
        face_materials = section.triangle_materials[faces]
        materials = tuple(int(material) for material in np.unique(face_materials))
        uv = section.source_uvs[faces]
        sizes = [textures[int(material)][0].size for material in face_materials]
        width, height = tile_size(layout.projected, uv, sizes, scale, limit)
        pixels = bake_texture(layout.projected, uv, source_textures, width, height, supersamples,
                              materials=face_materials, boundary=getattr(layout, "texture_boundary", ()))
        uv_seams = internal_uv_seams(section.positions[faces], uv)
        prepared.append(Tile(layout.faces, materials, positions, target_uv, triangles, layout.normal,
                             pixels, layout.plane_error, uv_seams))
    return prepared


def replace_patches(section, source_uv, patches, atlas_material):
    removed = np.zeros(len(section.positions), dtype=bool)
    positions, uv, normals, materials, indices = [], [], [], [], []
    next_vertex = int(section.corner_indices.max()) + 1
    for faces, points, patch_uv, triangles, normal in patches:
        removed[list(faces)] = True
        positions.append(points[triangles])
        normals.append(np.broadcast_to(normal, (len(triangles), 3, 3)))
        uv.append(patch_uv[triangles])
        materials.append(np.full(len(triangles), atlas_material))
        indices.append(next_vertex + triangles)
        next_vertex += len(points)
    keep = ~removed
    return (
        np.concatenate([section.positions[keep], *positions]),
        np.concatenate([source_uv[keep], *uv]),
        np.concatenate([section.normals[keep], *normals]),
        np.concatenate([section.triangle_materials[keep], *materials]),
        np.concatenate([section.corner_indices[keep], *indices]),
    )


def build_target(section, source_uv, textures, tiles, atlas, offsets):
    atlas_material = max(textures) + 1
    patches = []
    for tile, (x, y) in zip(tiles, offsets):
        height, width = tile.pixels.shape[:2]
        atlas_uv = (tile.uvs * [width, height] + [x + GUTTER, y + GUTTER]) / atlas.size
        patches.append((tile.faces, tile.positions, atlas_uv, tile.triangles, tile.normal))
    *attributes, indices = replace_patches(section, source_uv, patches, atlas_material)
    target_textures = {int(material): textures[int(material)] for material in np.unique(attributes[3])
                       if material != atlas_material}
    target_textures[atlas_material] = (atlas, False)
    return (*attributes, target_textures, indices)


def indexed_primitives(positions, uvs, normals, materials, corner_indices):
    primitives = []
    for material in np.unique(materials):
        keep = materials == material
        corners = np.concatenate((positions[keep], normals[keep], uvs[keep]), axis=-1).reshape(-1, 8)
        _, first, indices = np.unique(corner_indices[keep].ravel(), return_index=True, return_inverse=True)
        vertices = corners[first]
        primitives.append((vertices[:, :3], vertices[:, 3:6], vertices[:, 6:], indices, int(material)))
    return primitives


def texture_images(textures):
    return {id(image): image for image, _ in textures.values()}.values()


def texture_bytes(textures):
    return sum(image.width * image.height * 4 for image in texture_images(textures))


def geometry_bytes(primitives):
    return sum(len(positions) * 32 + len(indices) * (2 if len(positions) <= 65535 else 4)
               for positions, _, _, indices, _ in primitives)


def comparison_order(section, edited):
    # GLB primitives are emitted in material order, with source order inside each primitive.
    order = np.argsort(section[3], kind="stable")
    return (*[array[order] for array in section[:4]], section[4]), edited[order]


def output_paths(args):
    source_path, lod_path = args.output / "source.glb", args.output / "lod.glb"
    reserved = [source_path, lod_path]
    if args.crop_repeats:
        reserved.append(args.output / "source-cropped.glb")
    if args.source.resolve() in {path.resolve() for path in reserved}:
        raise ValueError("An output GLB would overwrite the source mesh")
    args.output.mkdir(parents=True, exist_ok=True)
    return source_path, lod_path


def target_result(requested, actual):
    return {"target_triangles": actual, "requested_triangles": requested,
            "triangle_target_met": None if requested is None else actual <= requested}


def write_report(path, report):
    document = json.dumps(report, indent=2, allow_nan=False)
    path.write_text(document + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key not in {"patches", "comparison"}}
    print(json.dumps(summary, indent=2))


def write_geometry_trial(args, section, source_uv, plans, report):
    patches = [(layout.faces, points, uv, triangles, layout.normal) for layout, points, uv, triangles in plans]
    atlas_id = max(section.materials) + 1
    target = replace_patches(section, source_uv, patches, atlas_id)
    original = indexed_primitives(section.positions, source_uv, section.normals,
                                  section.triangle_materials, section.corner_indices)
    candidate = indexed_primitives(*target)
    source_path, lod_path = output_paths(args)
    write_textured_glb(source_path, original, None)
    write_textured_glb(lod_path, candidate, None)
    report.update(target_result(args.target_triangles, len(target[0])))
    report["source_geometry_bytes"] = geometry_bytes(original)
    report["target_geometry_bytes"] = geometry_bytes(candidate)
    write_report(args.output / "report.json", report)


def run(args):
    from lodestar_section_compare import compare_section
    from lodestar_section_source import discover_patches, load_section

    start = perf_counter()
    section = load_section(args.library, args.source, args.node)
    source_uv = section.source_uvs * [1, -1] + [0, 1]
    import_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    if args.cross_uv:
        from lodestar_uv_regions import plan_uv_regions
        patches, plans = plan_uv_regions(section, outside_positions=section.outside_positions,
                                        max_boundary_deviation=args.max_boundary_deviation,
                                        target_triangles=args.target_triangles,
                                        normal_tolerance=args.normal_tolerance,
                                        plane_relative_tolerance=args.plane_relative_tolerance)
    else:
        patches = discover_patches(section)
        plans = protect_contacts(section, patches)
    candidates = [face for patch in patches for face in patch]
    if candidates and (len(set(candidates)) != len(candidates) or min(candidates) < 0 or
                       max(candidates) >= len(section.positions)):
        raise ValueError("The patch plan contains invalid or overlapping source faces")
    all_faces = [face for layout, _, _, _ in plans for face in layout.faces]
    discovery_ms = (perf_counter() - start) * 1000
    report = {
        "section": section.node_name,
        "geometry_id": section.geometry_id,
        "layout": "cross_uv_regions" if args.cross_uv else "rectangles",
        "geometry_only": args.geometry_only,
        "source_triangles": len(section.positions),
        "candidate_patches": len(patches),
        "planned_patches": len(plans),
        "max_boundary_deviation": args.max_boundary_deviation,
        "normal_tolerance": args.normal_tolerance,
        "plane_relative_tolerance": args.plane_relative_tolerance,
        "maximum_patch_plane_deviation": max((layout.plane_error for layout, *_ in plans), default=0.0),
        "maximum_boundary_deviation": max((getattr(layout, "boundary_error", 0.0) for layout, *_ in plans), default=0.0),
        "maximum_shading_normal_delta": max((getattr(layout, "shading_normal_error", 0.0)
                                              for layout, *_ in plans), default=0.0),
        "import_ms": import_ms,
        "discovery_ms": discovery_ms,
    }
    if args.geometry_only:
        write_geometry_trial(args, section, source_uv, plans, report)
        return
    if not plans:
        raise ValueError("The section has no reductions with boundary contacts preserved")
    supplemental = []
    if args.retire_material:
        from lodestar_texture_budget import retirement_plans
        supplemental = retirement_plans(section, plans, args.retire_material)
        all_faces.extend(face for layout, _, _, _ in supplemental for face in layout.faces)

    start = perf_counter()
    textures = read_textures(section.materials, args.textures)
    texture_load_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    tiles = build_tiles(section, [*plans, *supplemental], textures, args.texel_scale, args.max_tile, args.supersamples)
    atlas, offsets = pack_atlas(tiles, args.max_atlas)
    target = build_target(section, source_uv, textures, tiles, atlas, offsets)
    texture_crops = []
    crop_only = None
    placements = []
    if args.crop_repeats:
        from lodestar_texture_budget import crop_repeat_textures
        remapped, cropped_textures, texture_crops = crop_repeat_textures(target[1], target[3], target[4])
        target = (target[0], remapped, target[2], target[3], cropped_textures, target[5])
    if args.reuse_texels:
        from lodestar_texture_budget import reuse_atlas_texels
        remapped, remapped_materials, packed_textures, placements = reuse_atlas_texels(
            target[1], target[3], target[4], tiles, max(textures) + 1
        )
        target = (target[0], remapped, target[2], remapped_materials, packed_textures, target[5])
    bake_ms = (perf_counter() - start) * 1000

    source = section.positions, source_uv, section.normals, section.triangle_materials, textures
    # Compare the float32 UVs and vertices that the GLB actually stores.
    source = (*[array.astype("<f4") for array in source[:3]], *source[3:])
    target = (*[array.astype("<f4") for array in target[:3]], *target[3:])
    source_primitives = indexed_primitives(*source[:4], section.corner_indices)
    target_primitives = indexed_primitives(*target[:4], target[5])
    source_edited = np.zeros(len(source[0]), dtype=bool)
    source_edited[all_faces] = True
    ordered_source, source_edited = comparison_order(source, source_edited)
    replaced = np.arange(len(target[0])) >= len(section.positions) - len(all_faces)
    ordered_target, target_edited = comparison_order(target, replaced)
    comparison = compare_section(*ordered_source, *ordered_target, resolution=args.view_size,
                                 source_edited=source_edited, target_edited=target_edited)

    source_path, lod_path = output_paths(args)
    write_textured_glb(source_path, source_primitives, textures)
    write_textured_glb(lod_path, target_primitives, target[4])
    if args.crop_repeats:
        remapped, cropped_textures, changes = crop_repeat_textures(source_uv, section.triangle_materials, textures)
        cropped_source = indexed_primitives(section.positions, remapped, section.normals,
                                           section.triangle_materials, section.corner_indices)
        cropped_path = args.output / "source-cropped.glb"
        write_textured_glb(cropped_path, cropped_source, cropped_textures)
        crop_only = {"base_rgba8_bytes": texture_bytes(cropped_textures), "glb_bytes": cropped_path.stat().st_size,
                     "texture_crops": changes}
    comparison.image.save(args.output / "comparison.png")
    if not placements:
        atlas.save(args.output / "atlas.png")
    report.update({
        **target_result(args.target_triangles, len(target[0])),
        "baked_patches": len(tiles),
        "retirement_tiles": len(supplemental),
        "retired_materials": sorted(set(args.retire_material) - set(map(int, np.unique(target[3])))),
        "baked_internal_uv_seams": sum(tile.uv_seams for tile in tiles),
        "baked_source_triangles": len(all_faces),
        "unchanged_triangles": len(source[0]) - len(all_faces),
        "source_glb_bytes": source_path.stat().st_size,
        "target_glb_bytes": lod_path.stat().st_size,
        "source_base_rgba8_bytes": texture_bytes(textures),
        "target_base_rgba8_bytes": texture_bytes(target[4]),
        "source_texture_count": len(texture_images(textures)),
        "target_texture_count": len(texture_images(target[4])),
        "source_geometry_bytes": geometry_bytes(source_primitives),
        "target_geometry_bytes": geometry_bytes(target_primitives),
        "atlas_size": None if placements else list(atlas.size),
        "reused_texture_placements": placements,
        "atlas_gutter": GUTTER,
        "atlas_occupied_texels": sum(tile.pixels.shape[0] * tile.pixels.shape[1] for tile in tiles),
        "texture_load_ms": texture_load_ms,
        "texture_crops": texture_crops,
        "crop_only_source": crop_only,
        "bake_and_pack_ms": bake_ms,
        "patches": [
            {"source_face_indices": section.face_indices[list(tile.faces)].tolist(),
             "target_triangles": len(tile.triangles),
             "source_material": tile.materials[0] if len(tile.materials) == 1 else None,
             "source_materials": list(tile.materials), "internal_uv_seams": tile.uv_seams,
             "size": list(tile.pixels.shape[1::-1]), "offset": placements[i]["offset"] if placements else list(offset)}
            for i, (tile, offset) in enumerate(zip(tiles, offsets))
        ],
        "comparison": comparison.metrics,
    })
    write_report(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source COLLADA model")
    parser.add_argument("textures", type=Path, help="Directory containing the source diffuse images")
    parser.add_argument("output", type=Path, help="Directory for source and LOD GLBs and comparison files")
    parser.add_argument("--library", type=Path, required=True, help="Compiled Lodestar C library")
    parser.add_argument("--node", default="Wing2Break")
    parser.add_argument("--cross-uv", action="store_true", help="Bake connected planar regions across UV splits")
    parser.add_argument("--max-boundary-deviation", type=float, default=0.0,
                        help="Allowed straightening distance in scene units with --cross-uv")
    parser.add_argument("--normal-tolerance", type=float, default=1e-6,
                        help="Unit-normal component tolerance for compatible faces with --cross-uv")
    parser.add_argument("--plane-relative-tolerance", type=float, default=1e-6,
                        help="Plane residual limit relative to each region extent with --cross-uv")
    parser.add_argument("--target-triangles", type=int, help="Requested section budget with --cross-uv")
    parser.add_argument("--geometry-only", action="store_true", help="Export the geometry trial without loading images")
    parser.add_argument("--crop-repeats", action="store_true", help="Trim unused texels from retained repeating images")
    parser.add_argument("--reuse-texels", action="store_true", help="Fit every bake tile in unused retained texels")
    parser.add_argument("--retire-material", type=int, action="append", default=[],
                        help="Bake all remaining faces of this source material to free its image; can repeat")
    parser.add_argument("--texel-scale", type=float, default=0.75, help="Scale for source texel density per patch")
    parser.add_argument("--max-tile", type=int, default=512, help="Maximum width or height of a patch texture")
    parser.add_argument("--max-atlas", type=int, default=2048, help="Maximum width or height of the shared atlas")
    parser.add_argument("--supersamples", type=int, default=2, help="Samples per texel axis")
    parser.add_argument("--view-size", type=int, default=512, help="Width and height of each comparison view")
    args = parser.parse_args()
    if (not math.isfinite(args.max_boundary_deviation) or args.max_boundary_deviation < 0 or
            (args.target_triangles is not None and args.target_triangles <= 0) or
            not 0 < args.normal_tolerance <= 0.01 or not 0 < args.plane_relative_tolerance <= 0.01 or
            (not args.cross_uv and (args.max_boundary_deviation or args.target_triangles is not None or
             args.normal_tolerance != 1e-6 or args.plane_relative_tolerance != 1e-6))):
        parser.error("Use --cross-uv with a nonnegative boundary deviation, a positive triangle target, "
                     "and normal and plane tolerances up to 0.01")
    if (not 0 < args.texel_scale <= 4 or not 16 <= args.max_tile <= 2048 or
            not args.max_tile + 2 * GUTTER <= args.max_atlas <= 4096 or
            not 1 <= args.supersamples <= 4 or not 64 <= args.view_size <= 1024):
        parser.error("Use a positive texel scale up to 4, a tile size of 16–2048, an atlas size up to 4096 "
                     "that holds a padded tile, 1–4 samples per texel axis, and a view size of 64–1024")
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Lodestar section atlas: {error}\n")


if __name__ == "__main__":
    main()
