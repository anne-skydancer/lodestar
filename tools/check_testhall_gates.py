"""Test bounded nonplanar gate LODs against the authored TestHall medium LOD."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from bake_section_atlas import indexed_primitives, read_textures, texture_bytes
from check_testhall_full import AUTHORED_TRANSLATE_X, benchmark_pair, export_geometry, read_geometry, run as run_full
from check_testhall_physics import load_scene, triangle_key
from lodestar_nonplanar import simplify_nonplanar
from lodestar_section_compare import _project_pair, _rasterize, compare_section
from lodestar_section_source import load_section
from lodestar_textured_glb import write_textured_glb


GATES = ("gate", "gate1", "gate2", "gate3")
GEOMETRY_METRICS = ("surface_rms_relative", "surface_max_relative",
                    "silhouette_mean_iou_error", "silhouette_max_iou_error")


def scene_arrays(positions):
    edge_a = positions[:, 1] - positions[:, 0]
    edge_b = positions[:, 2] - positions[:, 0]
    cross = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = np.divide(cross, lengths, out=np.zeros_like(cross), where=lengths != 0)
    uv = np.zeros((len(positions), 3, 2))
    return positions, uv, np.broadcast_to(normals[:, None, :], positions.shape)


def authored_gate(library, path, name):
    section = load_section(library, path, name + "_LOD2", allow_position_ulp=True)
    return (section.positions + [AUTHORED_TRANSLATE_X, 0, 0], section.source_uvs * [1, -1] + [0, 1],
            section.normals, section.triangle_materials, section.materials)


def aperture_report(source, candidate):
    resolution = 384
    axis = int(np.ptp(source, axis=(0, 1)).argmin())
    direction = np.eye(3)[axis]
    before, after = _project_pair(source, candidate, direction, resolution)
    masks = [_rasterize(triangles, resolution)[0] >= 0 for triangles in (before, after)]
    occupied_y, occupied_x = np.nonzero(masks[0])
    if not len(occupied_x):
        raise ValueError("The reference gate has no visible pixels in its front projection")
    # Keep three pixels clear around each measured gap to exclude rasterization
    # noise and subpixel shifts at a silhouette edge.
    roi = np.s_[occupied_y.min():occupied_y.max() + 1, occupied_x.min():occupied_x.max() + 1]
    occupied = np.pad(masks[0], 3, constant_values=True)
    guarded_open = ~np.lib.stride_tricks.sliding_window_view(occupied, (7, 7)).any(axis=(-1, -2))
    return {"view_axis": axis, "resolution": resolution, "open_pixels_with_three_pixel_margin": int(guarded_open[roi].sum()),
            "newly_blocked_guarded_pixels": int((guarded_open[roi] & masks[1][roi]).sum())}


def plan_gate(args, section, reference, output):
    source_path = output / "source-geometry.glb"
    authored_path = output / "authored-geometry.glb"
    export_geometry(source_path, section.positions, section.source_uvs * [1, -1] + [0, 1], section.normals)
    export_geometry(authored_path, *reference[:3])
    authored = benchmark_pair(args.benchmark, source_path, authored_path, output / "authored-vs-source.csv")
    expected = len(reference[0])
    target = args.gate_target or expected
    report = dict(node=section.node_name, source_triangles=len(section.positions), authored_triangles=expected,
                  requested_triangles=target, authored_metrics=authored, trials=[])
    trials = []
    accepted = None
    profiles = [(error, False) for error in args.relative_error]
    if args.screen_components:
        profiles.extend((error, True) for error in args.relative_error)
    for index, (error, screen) in enumerate(profiles):
        limit = {"max_sampled_relative_error": authored["surface_max_relative"]} if screen else {}
        result = simplify_nonplanar(args.library, section.positions, target, max_relative_error=error,
                                    outside_positions=section.outside_positions, lock_boundary=args.lock_boundary, **limit)
        positions = result.positions
        if any(np.array_equal(positions, previous) for previous in trials):
            continue
        trials.append(positions)
        trial_path = output / f"trial-{index}.glb"
        export_geometry(trial_path, *scene_arrays(positions))
        exported = read_geometry(args.library, trial_path)[0]
        if len(exported) != len(positions):
            raise ValueError(f"{section.node_name} changed triangle count on reimport")
        metrics = benchmark_pair(args.benchmark, source_path, trial_path, output / f"trial-{index}-vs-source.csv")
        apertures = aperture_report(section.positions, exported)
        quality = all(metrics[key] <= authored[key] + 1e-9 for key in GEOMETRY_METRICS)
        geometry_ok = quality and apertures["newly_blocked_guarded_pixels"] == 0
        trial = dict(relative_error_limit=error, component_surface_limit=limit.get("max_sampled_relative_error"),
                     triangles=len(exported), target_met=len(exported) <= target,
                     geometry_ok=geometry_ok, metrics=metrics, aperture=apertures, components=result.components)
        report["trials"].append(trial)
        if geometry_ok and len(positions) < len(section.positions):
            if accepted is None or len(positions) < len(accepted.positions):
                accepted = result
            if len(positions) <= target:
                break
    return accepted, report


def write_textured(path, arrays, indices=None):
    positions, uv, normals, materials, textures = arrays
    attributes = tuple(array.astype("<f4") for array in (positions, uv, normals))
    if indices is None:
        corners = np.column_stack([array.reshape(-1, array.shape[-1]) for array in attributes])
        _, inverse = np.unique(corners, axis=0, return_inverse=True)
        indices = inverse.reshape(-1, 3)
    primitives = indexed_primitives(*attributes, materials, indices)
    write_textured_glb(path, primitives, textures)


def write_bake(args, section, result, reference, output):
    from lodestar_nonplanar_bake import bake_nonplanar

    textures = read_textures(section.materials, args.textures)
    source = (section.positions, section.source_uvs * [1, -1] + [0, 1], section.normals,
              section.triangle_materials, textures)
    positions = result.positions
    uv, normals, materials, baked_textures, report = bake_nonplanar(
        section, positions, textures, max_projection_distance=args.projection_distance,
        source_components=result.source_components, target_components=result.target_components,
        texel_scale=args.texel_scale, max_tile=args.max_tile, supersamples=args.supersamples,
        retain_exact=not args.atlas_only)
    target = positions, uv, normals, materials, baked_textures
    write_textured(output / "source.glb", source, section.corner_indices)
    write_textured(output / "lod.glb", target)
    comparison = compare_section(*source, *target, resolution=args.view_size)
    comparison.image.save(output / "comparison.png")
    authored_textures = read_textures(reference[4], args.textures)
    authored_arrays = (*reference[:4], authored_textures)
    write_textured(output / "authored.glb", authored_arrays)
    authored = compare_section(*source, *authored_arrays, resolution=args.view_size)
    report.update(source_bytes=texture_bytes(textures), candidate_bytes=texture_bytes(baked_textures),
                  color=comparison.metrics, authored_color=authored.metrics,
                  source_glb_bytes=(output / "source.glb").stat().st_size,
                  candidate_glb_bytes=(output / "lod.glb").stat().st_size)
    report["appearance_ok"] = all(comparison.metrics[key] <= authored.metrics[key] + 1e-9
                                  for key in ("mean_abs_error", "p95_abs_error"))
    return source, target, authored_arrays, report


def merge_textured(sections):
    # Identical images in different instances share one stored image, including
    # the texture originally shared across the whole TestHall gate family.
    images, bindings, textures, chunks = {}, {}, {}, []
    for positions, uv, normals, labels, section_textures in sections:
        materials = np.empty(len(labels), dtype=np.intp)
        for label in np.unique(labels):
            image, repeat = section_textures[label]
            image_key = (image.mode, image.size, image.tobytes())
            binding = (image_key, repeat)
            if binding not in bindings:
                index = len(textures)
                bindings[binding] = index
                textures[index] = images.setdefault(image_key, image), repeat
            materials[labels == label] = bindings[binding]
        chunks.append((positions, uv, normals, materials))
    return (*[np.concatenate(arrays) for arrays in zip(*chunks)], textures)


def write_combined_gates(args, gates):
    output = args.output / "combined-gates"
    output.mkdir(exist_ok=True)
    source, target, authored = [merge_textured(sections) for sections in zip(*gates)]
    for name, arrays in (("source", source), ("lod", target), ("authored", authored)):
        write_textured(output / f"{name}.glb", arrays)
    comparison = compare_section(*source, *target, resolution=args.view_size)
    comparison.image.save(output / "comparison.png")
    authored_color = compare_section(*source, *authored, resolution=args.view_size)
    return {"source_triangles": len(source[0]), "candidate_triangles": len(target[0]),
            "source_rgba8_bytes": texture_bytes(source[4]), "candidate_rgba8_bytes": texture_bytes(target[4]),
            "candidate_color_error": comparison.metrics["mean_abs_error"],
            "authored_color_error": authored_color.metrics["mean_abs_error"]}


def check_source_collision(args, source):
    from collections import Counter

    output = args.output / "source-collision.dae"
    subprocess.run([str(args.lodestar.resolve()), "--export-physics", str(output), str(source)],
                   check=True, capture_output=True, text=True)
    visual, _ = load_scene(args.library, source)
    collider, _ = load_scene(args.library, output)
    if Counter(map(triangle_key, collider)) != Counter(map(triangle_key, visual)):
        raise ValueError("Derived collision does not match the original source triangles")
    return {"triangles": len(collider), "matches_source": True, "file": output.name}


def check_holdout(args):
    path = args.holdout
    arrays = read_geometry(args.library, path)
    target = len(arrays[0]) // 2
    result = simplify_nonplanar(args.library, arrays[0], target, max_relative_error=args.relative_error[0])
    output = args.output / "holdout"
    output.mkdir(exist_ok=True)
    source, candidate = output / "source.glb", output / "lod.glb"
    export_geometry(source, *arrays)
    export_geometry(candidate, *scene_arrays(result.positions))
    return {"mesh": path.name, "requested_triangles": target, "components": result.components,
            "metrics": benchmark_pair(args.benchmark, source, candidate, output / "candidate-vs-source.csv")}


def run(args):
    args.output.mkdir(exist_ok=True, parents=True)
    source = args.corpus / "TestHall_MESH.dae"
    authored = args.corpus / "TestHall_LOD2.dae"
    accepted, gates, textures_by_gate = {}, [], []
    for name in args.gate or GATES:
        output = args.output / name
        output.mkdir(exist_ok=True)
        section = load_section(args.library, source, name)
        reference = authored_gate(args.library, authored, name)
        result, report = plan_gate(args, section, reference, output)
        report["accepted_triangles"] = len(result.positions) if result is not None else len(section.positions)
        if result is not None:
            if not args.textures:
                accepted[name] = scene_arrays(result.positions)
            else:
                try:
                    original, arrays, authored_arrays, report["bake"] = write_bake(args, section, result, reference, output)
                    if report["bake"]["appearance_ok"]:
                        accepted[name] = arrays[:3]
                        textures_by_gate.append((original, arrays, authored_arrays))
                except ValueError as error:
                    report["bake_error"] = str(error)
        report["selected_triangles"] = len(accepted[name][0]) if name in accepted else len(section.positions)
        report["triangle_target_met"] = report["selected_triangles"] <= report["requested_triangles"]
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        gates.append({key: value for key, value in report.items() if key not in ("trials", "bake")})
        print(f"{name}: {report['source_triangles']} → {report['selected_triangles']} selected, budget {report['requested_triangles']}")

    report = {"gates": gates, "authored_translate_x_m": AUTHORED_TRANSLATE_X}
    if args.lodestar:
        report["source_collision"] = check_source_collision(args, source)
    if args.holdout:
        report["holdout"] = check_holdout(args)
    if not args.gate and len(textures_by_gate) == len(GATES):
        report["combined_gates"] = write_combined_gates(args, textures_by_gate)
    if not args.gate:
        full = run_full(args.corpus, args.output / "full", args.library, benchmark=args.benchmark,
                        replace_section=lambda section: accepted.get(section.node_name))
        report["full"] = {key: full[key] for key in ("source_triangles", "candidate_triangles", "authored_medium_triangles",
                                                     "above_authored_triangles", "candidate_metrics", "authored_metrics")}
    (args.output / "scorecard.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", type=Path, required=True, help="Compiled Lodestar C ABI library")
    parser.add_argument("--benchmark", type=Path, required=True, help="Compiled Lodestar geometry benchmark")
    parser.add_argument("--lodestar", type=Path, help="Verify a collider derived from the original render mesh")
    parser.add_argument("--gate", choices=GATES, action="append", help="Run selected gate(s) before full assembly")
    parser.add_argument("--gate-target", type=int, help="Override the medium LOD budget for each tested gate")
    parser.add_argument("--relative-error", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--lock-boundary", action="store_true")
    parser.add_argument("--screen-components", action="store_true", help="Screen component proposals against authored surface error")
    parser.add_argument("--holdout", type=Path, help="Unrelated mesh for geometry validation")
    parser.add_argument("--textures", type=Path, help="Bake geometry-qualified candidates using these diffuse images")
    parser.add_argument("--projection-distance", type=float, default=0.04)
    parser.add_argument("--texel-scale", type=float, default=0.5)
    parser.add_argument("--max-tile", type=int, default=64)
    parser.add_argument("--supersamples", type=int, default=1)
    parser.add_argument("--atlas-only", action="store_true", help="Retire the source image and bake every selected face")
    parser.add_argument("--view-size", type=int, default=512)
    args = parser.parse_args()
    if args.gate_target is not None and args.gate_target <= 0:
        parser.error("The gate triangle budget must be positive")
    try:
        run(args)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"TestHall gate check: {error.stderr or error}\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"TestHall gate check: {error}\n")


if __name__ == "__main__":
    main()
