"""Generate and qualify the textured TestHall target and source-derived collision."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from bake_section_atlas import build_target, build_tiles, pack_atlas, replace_patches, texture_bytes
from check_testhall_full import AUTHORED_TRANSLATE_X, PROFILE, benchmark_pair, export_geometry
from check_testhall_gates import GATES, GEOMETRY_METRICS, authored_gate, plan_gate, scene_arrays
from check_testhall_physics import load_scene, triangle_key
from lodestar_nonplanar_bake import bake_nonplanar
from lodestar_section_compare import compare_section
from lodestar_testhall_budget import plan_budget
from lodestar_testhall_textures import compose_textured_scene, load_textured_scene, write_scene_glb
from lodestar_uv_regions import plan_uv_regions
from lodestar_walkability import staircase_fixture, testhall_traversal, walk_route


def top_uv(section):
    return section.source_uvs * [1, -1] + [0, 1]


def baseline_sections(args, scene, authored_path):
    gates, planar, counts = {}, {}, {}
    profile = SimpleNamespace(library=args.library, benchmark=args.benchmark, gate_target=None,
                              relative_error=(0.002, 0.005, 0.01, 0.025), screen_components=True, lock_boundary=False)
    for name, section in scene.sections.items():
        if name in GATES:
            directory = args.output / "gate-baseline" / name
            directory.mkdir(parents=True, exist_ok=True)
            reference = authored_gate(args.library, authored_path, name)
            result, _ = plan_gate(profile, section, reference, directory)
            if result is None:
                raise ValueError(f"The qualified gate baseline did not reproduce for {name}")
            gates[name] = result
            counts[name] = len(result.positions)
        else:
            # The validated 5,455-triangle baseline retains Wing2 as imported.
            plans = [] if name == "Wing2" else plan_uv_regions(
                section, outside_positions=section.outside_positions, **PROFILE)[1]
            planar[name] = plans
            counts[name] = len(section.positions) - sum(len(layout.faces) - len(triangles)
                                                        for layout, _, _, triangles in plans)
    return gates, planar, counts


def combine_geometry(scene, planar, replacements):
    keep = np.ones(len(scene.positions), dtype=bool)
    sections = []
    for name, section in scene.sections.items():
        if name in replacements:
            arrays = scene_arrays(replacements[name].positions)
        elif planar.get(name):
            patches = [(layout.faces, points, uv, triangles, layout.normal)
                       for layout, points, uv, triangles in planar[name]]
            arrays = replace_patches(section, top_uv(section), patches, max(scene.textures) + 1)[:3]
        else:
            continue
        keep[section.face_indices] = False
        sections.append(arrays)
    return tuple(np.concatenate([attribute[keep], *[part[i] for part in sections]])
                 for i, attribute in enumerate(scene.arrays[:3]))


def collision_from_source(args, render_path, output):
    subprocess.run([str(args.lodestar.resolve()), "--export-physics", str(output), str(render_path)],
                   check=True, capture_output=True, text=True)
    source, _ = load_scene(args.library, render_path)
    collider, _ = load_scene(args.library, output)
    if Counter(map(triangle_key, collider)) != Counter(map(triangle_key, source)):
        raise ValueError(f"The derived collider differs from source geometry in {render_path.name}")
    return collider


def check_walkability(args, source_path, rendered):
    collider = collision_from_source(args, source_path, args.output / "source-collision.dae")
    triangles, start, route = staircase_fixture()
    stairs = args.output / "staircase-source.glb"
    export_geometry(stairs, *scene_arrays(triangles))
    stair_collider = collision_from_source(args, stairs, args.output / "staircase-collision.dae")
    stair_walk = walk_route(stair_collider, start, route, radius=0.22, height=1.69)
    return {"source_collider_triangles": len(collider), "source_collider_matches_mesh": True,
            "rendered_testhall": testhall_traversal(rendered, check_supported_walk=True),
            "testhall_source_collider": testhall_traversal(collider, check_supported_walk=True),
            "staircase_fixture": stair_walk}


def bake_named_sections(args, scene, planar, replacements, *, fallback=None, gate_projection_distance=0.04):
    changed, per_section = {}, {}
    for name, section in scene.sections.items():
        if name in replacements:
            target = replacements[name]
            gate = name in GATES
            try:
                uv, normals, materials, textures, statistics = bake_nonplanar(
                    section, target.positions, scene.textures,
                    max_projection_distance=gate_projection_distance if gate else args.projection_distance,
                    source_components=target.source_components, target_components=target.target_components,
                    texel_scale=0.5 if gate else args.texel_scale, max_tile=12 if gate else args.max_tile,
                    max_atlas=args.max_atlas, supersamples=2, retain_exact=not gate,
                    reuse_affine_uv=args.reuse_affine_uv and not gate)
            except ValueError as error:
                if fallback is None or name not in fallback:
                    raise
                changed[name] = fallback[name]
                per_section[name] = {"retained_previous_level": str(error)}
                continue
            arrays = target.positions, uv, normals, materials, textures
        elif planar.get(name):
            tiles = build_tiles(section, planar[name], scene.textures, args.texel_scale,
                                args.planar_max_tile or args.max_tile, 2)
            atlas, offsets = pack_atlas(tiles, args.max_atlas)
            arrays = build_target(section, top_uv(section), scene.textures, tiles, atlas, offsets)[:5]
            statistics = {"baked_regions": len(tiles), "atlas_size": list(atlas.size)}
        else:
            continue
        changed[name] = arrays
        per_section[name] = statistics
    return changed, per_section


def bake_sections(args, scene, planar, replacements):
    changed, per_section = bake_named_sections(args, scene, planar, replacements)
    return compose_textured_scene(scene, [(scene.sections[name].face_indices, arrays)
                                          for name, arrays in changed.items()]), per_section


def comparison(args, source, candidate, authored):
    frame = np.concatenate([source.positions, candidate.positions, authored.positions])
    result = compare_section(*source.render_arrays, *candidate.render_arrays, resolution=args.view_size, frame_positions=frame)
    baseline = compare_section(*source.render_arrays, *authored.render_arrays, resolution=args.view_size, frame_positions=frame)
    result.image.save(args.output / "comparison.png")
    baseline.image.save(args.output / "authored-comparison.png")
    passed = all(result.metrics[key] <= baseline.metrics[key] for key in ("mean_abs_error", "p95_abs_error"))
    return {"passed": passed, "candidate": result.metrics, "authored": baseline.metrics,
            "base_rgba8_bytes": {"source": texture_bytes(source.textures), "candidate": texture_bytes(candidate.textures),
                                  "authored": texture_bytes(authored.textures)}}


def check_render_formats(args, candidate):
    from lodestar_gltf_export import write_gltf_from_glb
    from lodestar_visual_collada import write_textured_collada

    glb, gltf, dae = (args.output / f"lod.{extension}" for extension in ("glb", "gltf", "dae"))
    binary = write_gltf_from_glb(glb, gltf)
    write_textured_collada(dae, candidate)
    expected = Counter(map(triangle_key, candidate.positions.astype("<f4")))
    reports = {}
    for extension, path in (("glb", glb), ("gltf", gltf), ("dae", dae)):
        imported = load_textured_scene(args.library, path, args.output if extension == "dae" else None)
        collider = collision_from_source(args, path, args.output / f"lod-{extension}-collision.dae")
        geometry_ok = Counter(map(triangle_key, collider)) == expected
        walkability = testhall_traversal(collider, check_supported_walk=True)
        compare = compare_section(*candidate.render_arrays, *imported.render_arrays,
                                  resolution=args.view_size, frame_positions=candidate.positions).metrics
        colors_ok = (compare["mean_abs_error"] is not None and compare["mean_abs_error"] <= 0.05 and
                     compare["p95_abs_error"] <= 1)
        reports[extension] = {"file": path.name, "triangles": len(imported.positions), "geometry_matches": geometry_ok,
                              "mean_abs_error": compare["mean_abs_error"], "p95_abs_error": compare["p95_abs_error"],
                              "walkability": walkability, "passed": geometry_ok and colors_ok and walkability["passed"]}
    if binary is not None:
        reports["gltf"]["buffer_file"] = binary.name
    return {"passed": all(item["passed"] for item in reports.values()), "exports": reports}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source_path = args.corpus / "TestHall_MESH.dae"
    authored_path = args.corpus / "TestHall_LOD2.dae"
    source = load_textured_scene(args.library, source_path, args.textures)
    authored = load_textured_scene(args.library, authored_path, args.textures, translation=(AUTHORED_TRANSLATE_X, 0, 0))
    gates, planar, counts = baseline_sections(args, source, authored_path)
    baseline_count = sum(counts.values())
    preserved = set(args.preserve_planar) | set(args.preserve_source)
    if preserved - planar.keys():
        raise ValueError(f"Expected non-gate source sections to preserve: {sorted(preserved - planar.keys())}")
    for name in args.preserve_source:
        planar[name] = []
        counts[name] = len(source.sections[name].positions)
    plan = plan_budget(args.library, source.sections, counts, target_triangles=args.target_triangles,
                       protected_nodes=(*GATES, "Wing2", *sorted(preserved)), max_relative_error=args.relative_error,
                       max_sampled_distance_m=args.component_distance)
    replacements = {**gates, **plan.replacements}
    geometry = combine_geometry(source, planar, replacements)
    if len(geometry[0]) != plan.candidate_triangles:
        raise ValueError("The assembled geometry disagrees with the budget plan")
    geometry_path = args.output / "lod-geometry.glb"
    export_geometry(geometry_path, *geometry)
    candidate = benchmark_pair(args.benchmark, source_path, geometry_path, args.output / "candidate-vs-source.csv")
    reference = benchmark_pair(args.benchmark, source_path, authored_path, args.output / "authored-vs-source.csv",
                               translate_x=AUTHORED_TRANSLATE_X)
    geom_ok = all(candidate[key] <= reference[key] + 1e-9 for key in GEOMETRY_METRICS)
    rendered, _ = load_scene(args.library, geometry_path)
    walkability = check_walkability(args, source_path, rendered)
    walk_ok = all(walkability[key]["passed"] for key in ("rendered_testhall", "testhall_source_collider", "staircase_fixture"))
    report = {"source_triangles": len(source.positions), "baseline_triangles": baseline_count,
              "candidate_triangles": candidate["triangles"], "authored_triangles": reference["triangles"],
              "target_triangles": args.target_triangles, "budget_passed": candidate["triangles"] == args.target_triangles,
              "geometry_passed": geom_ok, "geometry": {"candidate": candidate, "authored": reference},
              "walkability_passed": walk_ok, "walkability": walkability, "budget": plan.report,
              "preserve_planar": sorted(set(args.preserve_planar)), "preserve_source": sorted(set(args.preserve_source)),
              "texture_profile": {key: getattr(args, key) for key in ("texel_scale", "max_tile", "max_atlas",
                                                                      "planar_max_tile", "projection_distance",
                                                                      "reuse_affine_uv")},
              "passed": False}
    report_path = args.output / "scorecard.json"
    if geom_ok and walk_ok and report["budget_passed"]:
        try:
            baked, statistics = bake_sections(args, source, planar, replacements)
            for asset, name in ((source, "source"), (baked, "lod"), (authored, "authored")):
                write_scene_glb(args.output / f"{name}.glb", asset)
            textured, _ = load_scene(args.library, args.output / "lod.glb")
            if Counter(map(triangle_key, textured)) != Counter(map(triangle_key, rendered)):
                raise ValueError("Textured export changed the qualified geometry")
            report["bake"] = statistics
            report["appearance"] = comparison(args, source, baked, authored)
            if report["appearance"]["passed"]:
                report["formats"] = check_render_formats(args, baked)
                report["passed"] = report["formats"]["passed"]
        except ValueError as error:
            report["bake_error"] = str(error)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"TestHall: {len(source.positions):,} → {candidate['triangles']:,}, target {args.target_triangles:,}. {report_path}")
    if not report["passed"]:
        raise ValueError("The target failed one or more checks; see scorecard.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("textures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--lodestar", type=Path, required=True)
    parser.add_argument("--target-triangles", type=int, default=4769)
    parser.add_argument("--preserve-planar", action="append", default=["Hall1"], help="Keep a section's previous planar LOD")
    parser.add_argument("--preserve-source", action="append", default=["Wing3"], help="Keep a section's original triangles and textures")
    parser.add_argument("--relative-error", type=float, default=0.005)
    parser.add_argument("--component-distance", type=float, default=0.05, help="Sampled component distance budget in metres")
    parser.add_argument("--projection-distance", type=float, default=0.06)
    parser.add_argument("--texel-scale", type=float, default=1.0)
    parser.add_argument("--max-tile", type=int, default=64)
    parser.add_argument("--planar-max-tile", type=int, default=256, help="Planar chart texture cap; 0 uses --max-tile")
    parser.add_argument("--max-atlas", type=int, default=4096)
    parser.add_argument("--reuse-affine-uv", action=argparse.BooleanOptionalAction, default=True,
                        help="Retain source textures when sampled affine UVs agree")
    parser.add_argument("--view-size", type=int, default=512)
    try:
        run(parser.parse_args())
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"TestHall target: {error.stderr or error}\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"TestHall target: {error}\n")


if __name__ == "__main__":
    main()
