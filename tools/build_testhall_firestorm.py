"""Build the named TestHall LOD chain and preflight it against Firestorm 7.2.5."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from bake_section_atlas import indexed_primitives
from check_testhall_full import AUTHORED_TRANSLATE_X, benchmark_pair
from check_testhall_gates import GATES, GEOMETRY_METRICS
from check_testhall_physics import load_scene, triangle_key
from check_testhall_target import bake_named_sections, baseline_sections, check_walkability
from lodestar_firestorm_degenerate import (decoded_upload_positions, packed_physics_degenerate_mask,
                                          physics_degenerate_mask, visual_degenerate_mask)
from lodestar_gltf_export import write_gltf_from_glb
from lodestar_lod_materials import combine_lod_parts, share_lod_materials, textured_sections
from lodestar_section_compare import compare_section
from lodestar_testhall_budget import plan_budget
from lodestar_testhall_levels import plan_testhall_levels
from lodestar_testhall_textures import load_textured_scene
from lodestar_textured_glb import write_textured_glb
from lodestar_visual_collada import write_lod_collada, write_lod_collision_collada
from lodestar_walkability import testhall_traversal


SUFFIXES = {"high": "", "medium": "_LOD2", "low": "_LOD1", "lowest": "_LOD0", "physics": "_PHYS"}
MEDIUM_TARGET = 4769
LOWER_TEXTURE_FALLBACKS = frozenset(("Hall1", "Wing3"))


def build_levels(args, source, authored_path):
    gates, planar, counts = baseline_sections(args, source, authored_path)
    planar["Wing3"] = []
    counts["Wing3"] = len(source.sections["Wing3"].positions)
    plan = plan_budget(args.library, source.sections, counts, target_triangles=MEDIUM_TARGET,
                       protected_nodes=(*GATES, "Wing2", "Hall1", "Wing3"), max_relative_error=0.005,
                       max_sampled_distance_m=0.05)
    replacements = {**gates, **plan.replacements}
    baked, stats = bake_named_sections(args, source, planar, replacements)
    medium = textured_sections(source, baked)
    levels = {"high": textured_sections(source), "medium": medium}
    plans = plan_testhall_levels(args.library, source, replacements,
                                 medium_sections={name: part.positions for name, part in medium.items()},
                                 low_target=args.low_target, lowest_target=args.lowest_target)
    bake_report = {"medium": stats}
    for level, proposal in plans.items():
        previous = next(reversed(levels.values()))
        updates = {name: proposal.replacements[name] for name, positions in proposal.sections.items()
                   if (name not in LOWER_TEXTURE_FALLBACKS and len(positions) < len(previous[name].positions) and
                       not np.array_equal(positions, previous[name].positions))}
        fallback = {name: part.arrays for name, part in previous.items()}
        distance = 0.12 if level == "low" else 0.24
        profile = SimpleNamespace(**{**vars(args), "projection_distance": distance})
        changed, bake_report[level] = bake_named_sections(profile, source, {}, updates, fallback=fallback,
                                                         gate_projection_distance=distance)
        levels[level] = {**previous, **{name: part for name, part in textured_sections(source, changed).items()
                                       if name in changed}}
    return levels, {name: plan.report for name, plan in plans.items()}, bake_report


def geometry_key(positions):
    return Counter(map(triangle_key, np.asarray(positions, dtype=np.float32)))


def model_bounds(positions):
    return np.stack((positions.min(axis=(0, 1)), positions.max(axis=(0, 1))))


def viewer_space(parts, high):
    return np.concatenate([decoded_upload_positions(positions, high[name]) for name, positions in parts.items()])


def write_visual(root, level, parts):
    suffix = SUFFIXES[level]
    dae = root / f"TestHall{suffix}.dae"
    write_lod_collada(dae, parts, suffix=suffix)
    groups, textures, names = {}, {}, {}
    for name, part in parts.items():
        groups[name + suffix] = indexed_primitives(*part.arrays[:4], part.corner_indices)
        textures.update(part.textures)
        names.update({label: f"{name}-slot-{label}" for label in part.textures})
    glb = dae.with_suffix(".glb")
    write_textured_glb(glb, groups, textures, material_names=names)
    gltf = dae.with_suffix(".gltf")
    binary = write_gltf_from_glb(glb, gltf)
    return {"dae": dae, "glb": glb, "gltf": gltf, "bin": binary}


def write_physics(root, parts):
    suffix = SUFFIXES["physics"]
    dae = root / f"TestHall{suffix}.dae"
    write_lod_collision_collada(dae, parts, suffix=suffix)
    groups = {}
    for name, positions in parts.items():
        vertices, indices = np.unique(positions.reshape(-1, 3), axis=0, return_inverse=True)
        groups[name + suffix] = [(vertices, None, None, indices, None)]
    glb = dae.with_suffix(".glb")
    write_textured_glb(glb, groups, None)
    gltf = dae.with_suffix(".gltf")
    binary = write_gltf_from_glb(glb, gltf)
    return {"dae": dae, "glb": glb, "gltf": gltf, "bin": binary}


def compare(first, second, frame, resolution):
    return compare_section(*first.render_arrays, *second.render_arrays,
                           frame_positions=frame, resolution=resolution)


def fidelity_metrics(metrics, mean, p95):
    return (metrics["mean_abs_error"] is not None and metrics["mean_abs_error"] <= mean and
            metrics["p95_abs_error"] <= p95)


def qualify_lower_appearance(levels, source, references, frame):
    """Keep lower geometry edits only when the composed textured scene passes."""
    report = {}
    for level, previous in (("low", "medium"), ("lowest", "low")):
        reference = references[level]
        base = levels[previous]
        edits = {name: part for name, part in levels[level].items()
                 if len(part.positions) < len(base[name].positions)}

        def render(parts):
            metrics = compare(source, combine_lod_parts(parts), frame, reference["resolution"]).metrics
            return {"mean_abs_error": metrics["mean_abs_error"], "p95_abs_error": metrics["p95_abs_error"],
                    "passed": fidelity_metrics(metrics, reference["mean_abs_error"], reference["p95_abs_error"])}

        complete = {**base, **edits}
        whole = render(complete)
        report[level] = {"whole_proposal": whole, "parts": {}}
        if whole["passed"]:
            levels[level] = complete
            continue

        kept = dict(base)
        for name in sorted(edits, key=lambda key: (len(edits[key].positions) - len(base[key].positions), key)):
            kept[name] = edits[name]
            result = render(kept)
            report[level]["parts"][name] = result
            if not result["passed"]:
                kept[name] = base[name]
        levels[level] = kept
    return report


def round_trip(args, level, parts, paths, resolution):
    combined = combine_lod_parts(parts)
    suffix = SUFFIXES[level]
    expected = {name + suffix: geometry_key(part.positions) for name, part in parts.items()}
    reports = {}
    for extension in ("dae", "glb", "gltf"):
        path = paths[extension]
        imported = load_textured_scene(args.library, path, path.parent if extension == "dae" else None)
        geometry = {name: geometry_key(part.positions) for name, part in imported.sections.items()}
        colors = compare(combined, imported, combined.positions, resolution).metrics
        reports[extension] = {"file": path.name, "named_geometry_matches": geometry == expected, "color": colors,
                              "passed": geometry == expected and fidelity_metrics(colors, 0.05, 1)}
    return reports


def run(args):
    root = args.output.resolve()
    work = root / "measurements"
    work.mkdir(parents=True, exist_ok=True)
    source_path = args.corpus / "TestHall_MESH.dae"
    authored_path = args.corpus / "TestHall_LOD2.dae"
    source = load_textured_scene(args.library, source_path, args.textures)
    authored = load_textured_scene(args.library, authored_path, args.textures,
                                   translation=(AUTHORED_TRANSLATE_X, 0, 0))
    print("Reproducing the qualified Medium and planning lower levels.", flush=True)
    trial = SimpleNamespace(**{**vars(args), "output": work})
    original, geometry_plans, baking = build_levels(trial, source, authored_path)
    measurements = {}
    frame = np.concatenate((source.positions, authored.positions))
    authored_geometry = benchmark_pair(args.benchmark, source_path, authored_path,
                                       work / "authored-vs-source.csv", translate_x=AUTHORED_TRANSLATE_X)
    authored_appearance = compare(source, authored, frame, args.view_size).metrics
    resolution = {"high": args.view_size, "medium": args.view_size, "low": 192, "lowest": 96}
    references = {"high": authored_appearance, "medium": authored_appearance,
                  **{level: compare(source, authored, frame, resolution[level]).metrics for level in ("low", "lowest")}}
    print("Checking lower-level textures and preparing shared material slots.", flush=True)
    appearance_selection = qualify_lower_appearance(original, source, references, frame)
    prepared, materials = share_lod_materials(original, max_size=args.texture_limit)
    targets = {"high": len(source.positions), "medium": MEDIUM_TARGET,
               "low": args.low_target, "lowest": args.lowest_target}
    high_positions = {name: part.positions for name, part in prepared["high"].items()}
    previous = source
    for level, parts in prepared.items():
        print(f"Writing and measuring {level}.", flush=True)
        result = combine_lod_parts(parts)
        raw = combine_lod_parts(original[level])
        paths = write_visual(root, level, parts)
        maps = compare(raw, result, frame, resolution[level]).metrics
        appearance = compare(source, result, frame, resolution[level])
        if level != "high":
            appearance.image.save(work / f"{level}-vs-high.png")
        geometry = benchmark_pair(args.benchmark, source_path, paths["dae"],
                                  work / f"{level}-vs-source.csv", resolution=384 if level in ("high", "medium") else resolution[level])
        round_trips = round_trip(args, level, parts, paths, min(128, resolution[level]))
        packed = viewer_space({name: part.positions for name, part in parts.items()}, high_positions)
        bounds_error = {name: float(np.abs(model_bounds(part.positions) -
                                           model_bounds(high_positions[name])).max())
                        for name, part in parts.items()}
        measurements[level] = {"triangles": len(result.positions),
                               "target_triangles": targets[level],
                               "triangles_over_target": max(0, len(result.positions) - targets[level]),
                               "parts": {name: len(part.positions) for name, part in parts.items()},
                               "degenerate_triangles": {name: int(visual_degenerate_mask(part.positions).sum())
                                                        for name, part in parts.items()},
                               "max_part_bound_changes_m": bounds_error, "geometry": geometry,
                               "appearance_against_source": appearance.metrics,
                               "appearance_against_previous_level": compare(previous, result, frame, resolution[level]).metrics,
                               "authored_reference_at_this_resolution": references[level],
                               "shared_slot_texel_error": maps, "shared_slots_preserved": fidelity_metrics(maps, 0.05, 1),
                               "walkability": testhall_traversal(result.positions, check_supported_walk=True),
                               "packed_walkability": testhall_traversal(packed, check_supported_walk=True),
                               "round_trips": round_trips}
        previous = result

    from lodestar_firestorm_collision import build_firestorm_collider

    print("Building the Firestorm triangle collider from the source mesh.", flush=True)
    source_physics = check_walkability(trial, source_path, source.positions)
    collider, collision_repairs = {}, {}
    for name, section in source.sections.items():
        collider[name], collision_repairs[name] = build_firestorm_collider(
            section.positions, fixed_positions=section.outside_positions)
    physics_paths = write_physics(root, collider)
    packed_physics = testhall_traversal(viewer_space(collider, high_positions), check_supported_walk=True)
    collision_reports = {}
    for extension in ("dae", "glb", "gltf"):
        path = physics_paths[extension]
        reloaded, _ = load_scene(args.library, path)
        walkability = testhall_traversal(reloaded, check_supported_walk=True)
        collision_reports[extension] = {"file": path.name, "triangles": len(reloaded),
                                        "geometry_matches_source": geometry_key(reloaded) == geometry_key(source.positions),
                                        "geometry_matches_collider": geometry_key(reloaded) == geometry_key(np.concatenate(list(collider.values()))),
                                        "walkability": walkability}
    medium = measurements["medium"]
    collision_degenerates = {
        name: {"uploader": int(physics_degenerate_mask(positions).sum()),
               "packed": int(packed_physics_degenerate_mask(positions).sum())}
        for name, positions in collider.items()}
    checks = {
        "medium_triangle_count": medium["triangles"] == MEDIUM_TARGET,
        "medium_geometry": all(medium["geometry"][key] <= authored_geometry[key] + 1e-9 for key in GEOMETRY_METRICS),
        "medium_appearance": fidelity_metrics(medium["appearance_against_source"],
                                             authored_appearance["mean_abs_error"], authored_appearance["p95_abs_error"]),
        "lower_levels_reduce": all(measurements[smaller]["triangles"] < measurements[larger]["triangles"]
                                   for larger, smaller in (("medium", "low"), ("low", "lowest"))),
        "lower_level_appearance": all(fidelity_metrics(measurements[level]["appearance_against_source"],
                                                       measurements[level]["authored_reference_at_this_resolution"]["mean_abs_error"],
                                                       measurements[level]["authored_reference_at_this_resolution"]["p95_abs_error"])
                                      for level in ("low", "lowest")),
        "model_bounds": all(error <= 0.0001 for item in measurements.values()
                            for error in item["max_part_bound_changes_m"].values()),
        "shared_slots": all(item["shared_slots_preserved"] for item in measurements.values()),
        "visual_files": all(report["passed"] for item in measurements.values() for report in item["round_trips"].values()),
        "visual_walks": all(item["walkability"]["passed"] and item["packed_walkability"]["passed"]
                            for item in measurements.values()),
        "zero_visual_degenerates": not any(count for item in measurements.values()
                                           for count in item["degenerate_triangles"].values()),
        "zero_physics_degenerates": not any(count for part in collision_degenerates.values() for count in part.values()),
        "physics_files": packed_physics["passed"] and all(
            item["geometry_matches_collider"] and item["walkability"]["passed"] for item in collision_reports.values()),
        "mesh_derived_stairs_fixture": source_physics["staircase_fixture"]["passed"],
    }
    report = {"target_firestorm_release": "7.2.5.81788", "viewer_preview_verified": False,
              "levels": measurements, "authored_medium": {"geometry": authored_geometry, "appearance": authored_appearance},
              "lower_lod_geometry_plans": geometry_plans, "baking": baking, "materials": materials,
              "lower_lod_texture_selection": appearance_selection,
              "lower_lod_medium_parts_retained_for_texture": sorted(LOWER_TEXTURE_FALLBACKS),
              "physics": collision_reports, "staircase": source_physics["staircase_fixture"],
              "physics_repairs": collision_repairs, "physics_after_packing": packed_physics,
              "physics_degenerate_triangles": collision_degenerates,
              "checks": checks, "passed_offline": all(checks.values())}
    report_path = root / "scorecard.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("TestHall " + ", ".join(f"{level} {item['triangles']:,}" for level, item in measurements.items()))
    print(report_path)
    if not report["passed_offline"]:
        raise ValueError("The package failed one or more offline checks. See scorecard.json.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("textures", type=Path)
    parser.add_argument("output", type=Path)
    for option in ("library", "benchmark", "lodestar"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--low-target", type=int, default=3400)
    parser.add_argument("--lowest-target", type=int, default=2400)
    parser.add_argument("--texture-limit", type=int, default=2048)
    parser.add_argument("--view-size", type=int, default=512)
    parser.set_defaults(texel_scale=1.0, max_tile=64, max_atlas=4096, planar_max_tile=256,
                        projection_distance=0.06, reuse_affine_uv=True)
    try:
        run(parser.parse_args())
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"TestHall Firestorm: {error.stderr or error}\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"TestHall Firestorm: {error}\n")


if __name__ == "__main__":
    main()
