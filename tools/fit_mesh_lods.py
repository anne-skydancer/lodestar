"""Fit real diffuse mesh charts and calculate all visual LODs from surface integrals."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from types import SimpleNamespace
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import numpy as np

from bake_section_atlas import indexed_primitives
from lodestar_gltf_export import write_gltf_from_glb
from lodestar_lod_materials import combine_lod_parts, textured_sections
from lodestar_physics_select import prepare_physics
from lodestar_size_plan import size_proposals, apply_physical_containers
from lodestar_section_compare import compare_section
from lodestar_surface_fit import SurfaceCalculator, charts, fit_chart, generate_chart
from lodestar_surface_fit import correspondence_bound
from lodestar_chart_boundaries import coarsen_boundaries
from lodestar_testhall_textures import TexturedScene, load_textured_scene, _replacement_corners
from lodestar_textured_glb import write_textured_glb
from lodestar_visual_collada import write_lod_collada, write_lod_collision_collada
from lodestar_patch_source import _scene_mesh
from lodestar_firestorm_degenerate import physics_degenerate_mask, packed_physics_degenerate_mask
from lodestar_firestorm_degenerate import visual_degenerate_mask


LEVELS = ("high", "medium", "low", "lowest")
SUFFIXES = ("", "_LOD2", "_LOD1", "_LOD0")


def plan_levels(scene, library, pixels=(1024, 512, 192, 96), pixel_error=0.5, normal_angle=1.0):
    if (len(pixels) != 4 or any(not np.isfinite(p) or p <= 0 for p in pixels) or
            any(first <= second for first, second in zip(pixels, pixels[1:])) or
            not np.isfinite(pixel_error) or pixel_error <= 0 or
            not np.isfinite(normal_angle) or not 0 <= normal_angle < 90):
        raise ValueError("Expected four descending view sizes, positive pixel error and a normal limit below 90 degrees")
    extent = float(np.ptp(scene.positions, axis=(0, 1)).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("The source needs finite nonzero scene bounds")
    calculator = SurfaceCalculator(library)
    groups = charts(scene)
    node_by_face = {}
    node_pins = {}
    for name, section in scene.sections.items():
        node_by_face.update((int(face), name) for face in section.face_indices)
        points = scene.positions[section.face_indices].reshape(-1, 3)
        node_pins[name] = {tuple(points[index]) for extrema in (points.argmin(axis=0), points.argmax(axis=0))
                           for index in extrema}
    owners = defaultdict(set)
    for index, selected in enumerate(groups):
        for point in set(map(tuple, scene.positions[selected].reshape(-1, 3))):
            owners[point].add(index)

    fitted, reasons = {}, Counter()
    for index, selected in enumerate(groups):
        try:
            fitted[index] = fit_chart(scene, selected, calculator.fit, calculator.patch)
        except ValueError as error:
            reasons[str(error)] += 1
    previous = {index: chart.faces for index, chart in fitted.items()}
    previous_bounds = {index: {"surface_bound": 0.0, "normal_angle_bound_degrees": 0.0} for index in fitted}
    reports = {"charts": len(groups), "fitted_charts": len(fitted), "fallback_reasons": dict(reasons),
               "fit_reports": {str(i): chart.report for i, chart in fitted.items()}, "levels": {},
               "polynomial_basis": ["1", "u", "v", "u*u", "u*v", "v*v", "u*u*u", "u*u*v", "u*v*v", "v*v*v"]}
    levels = {}
    for level, resolution in zip(LEVELS, pixels):
        starting_indices = dict(previous)
        starting_bounds = {index: dict(bound) for index, bound in previous_bounds.items()}
        scale = extent / resolution
        seam_report = {"removed_seam_vertices": 0, "changed_charts": []}
        if level != "high":
            floors = {index: min(len(previous[index]), int((
                calculator.calculate_chart(chart, scale, pixel_error) if hasattr(calculator, "calculate_chart") else
                calculator.calculate(chart.controls, scale, pixel_error))["suggested_triangles"]))
                for index, chart in fitted.items()}
            seam_report = coarsen_boundaries(scene, groups, fitted, previous, floors,
                scale * pixel_error, normal_angle, set().union(*node_pins.values()))
            for index in seam_report["changed_charts"]:
                chart = fitted[index]
                previous_bounds[index] = {key: value for key, value in correspondence_bound(
                    chart.uv, chart.positions, chart.normals, chart.faces, previous[index],
                    source_normals=chart.corner_normals, target_normals=chart.normals_for(previous[index])).items()
                    if key in previous_bounds[index]}
        chunks = defaultdict(list)
        decisions = []
        part_fallbacks = {}
        for index, selected in enumerate(groups):
            name = node_by_face[int(selected[0])]
            chart = fitted.get(index)
            decision = {"chart": index, "node": name, "source_triangles": len(selected)}
            if chart is None:
                arrays = [array[selected] for array in scene.arrays[:4]]
                corners = scene.corner_indices[selected]
                ordering = selected
                decision.update(triangles=len(selected), fallback=True)
            else:
                before = previous[index]
                active = set(map(int, before.ravel()))
                locks = np.array([vertex in active and (len(owners[tuple(point)]) > 1 or tuple(point) in node_pins[name])
                                  for vertex, point in enumerate(chart.positions)], dtype=np.uint8)
                try:
                    if level == "high":
                        # Calculate the High budget while retaining its authoritative triangle stream.
                        budget = (calculator.calculate_chart(chart, scale, pixel_error) if hasattr(calculator, "calculate_chart")
                                  else calculator.calculate(chart.controls, scale, pixel_error))
                        indices = chart.faces
                        details = {"calculation": budget, "triangles": len(indices), "authoritative_high": True}
                    else:
                        indices, details = generate_chart(chart, calculator, scale, pixel_error,
                                                          normal_angle, locks, before)
                    decision.update(details)
                except (ValueError, np.linalg.LinAlgError) as error:
                    indices = before
                    decision.update(triangles=len(indices), fallback_reason=str(error))
                previous[index] = indices
                if decision.get("changed"):
                    previous_bounds[index] = {key: decision[key] for key in previous_bounds[index]}
                decision.update(previous_bounds[index])
                if index in seam_report["changed_charts"]:
                    decision.update(changed=True, seam_coarsened=True)
                material = int(scene.materials[selected[0]])
                arrays = [chart.positions[indices], chart.source_uv[indices], chart.normals_for(indices),
                          np.full(len(indices), material, dtype=np.int64)]
                # A geometric vertex may carry different sided shading normals.
                # Give each actual attribute tuple a distinct render identity.
                corners = _replacement_corners(*arrays[:3]) + (int(scene.corner_indices.max()) + 1) * (index + 1)
                ordering = selected if np.array_equal(indices, chart.faces) else np.full(len(indices), int(selected[0]))
            chunks[name].append((*arrays, corners, ordering))
            decisions.append(decision)
        parts = {}
        for name, pieces in chunks.items():
            arrays = [np.concatenate([piece[i] for piece in pieces]) for i in range(4)]
            corners = np.concatenate([piece[4] for piece in pieces])
            ordering = np.argsort(np.concatenate([piece[5] for piece in pieces]), kind="stable")
            arrays = [array[ordering] for array in arrays]
            corners = corners[ordering]
            bindings = {int(material): scene.textures[int(material)] for material in np.unique(arrays[3])}
            parts[name] = TexturedScene(*arrays, bindings, corners)
        if level == "high":
            # Preserve original face order, corner identities and attribute values in High.
            parts = textured_sections(scene)
        else:
            prior = levels[next(reversed(levels))]
            for name, part in list(parts.items()):
                if visual_degenerate_mask(part.positions).any():
                    parts[name] = prior[name]
                    part_fallbacks[name] = "candidate degenerates under uploader position packing"
                    for decision in decisions:
                        index = decision["chart"]
                        if decision["node"] == name and index in fitted:
                            previous[index] = starting_indices[index]
                            previous_bounds[index] = starting_bounds[index]
                            decision.update(starting_bounds[index], triangles=len(starting_indices[index]),
                                            changed=False, retained_for_packing=True)
        if part_fallbacks and seam_report["changed_charts"]:
            # A shared seam transaction must never be rolled back on one side.
            parts = dict(levels[next(reversed(levels))])
            previous = starting_indices
            previous_bounds = starting_bounds
            seam_report["rolled_back_for_packing"] = True
            for decision in decisions:
                index = decision["chart"]
                if index in fitted:
                    decision.update(starting_bounds[index], triangles=len(starting_indices[index]),
                                    changed=False, retained_for_packing=True)
        levels[level] = parts
        reports["levels"][level] = {
            "view_pixels": resolution, "world_units_per_pixel": scale, "maximum_pixel_error": pixel_error,
            "maximum_normal_angle_degrees": normal_angle, "triangles": sum(len(p.positions) for p in parts.values()),
            "render_vertices": sum(len(np.unique(part.corner_indices[part.materials == label]))
                                   for part in parts.values() for label in np.unique(part.materials)),
            "decisions": decisions, "part_fallbacks": part_fallbacks, "seam_coarsening": seam_report,
            "uploader_degenerate_triangles_by_part": {
                name: int(visual_degenerate_mask(part.positions).sum()) for name, part in parts.items()}}
    return levels, reports


def write_level(root, stem, parts, suffix, *, physical_frames=None):
    groups = {name + suffix: indexed_primitives(*part.arrays[:4], part.corner_indices)
              for name, part in parts.items()}
    textures = {material: binding for part in parts.values() for material, binding in part.textures.items()}
    glb = root / f"{stem}{suffix}.glb"
    write_textured_glb(glb, groups, textures)
    write_gltf_from_glb(glb, glb.with_suffix(".gltf"))
    write_lod_collada(glb.with_suffix(".dae"), parts, suffix=suffix, physical_frames=physical_frames)
    return glb


def verify_roundtrip(library, path, expected, suffix):
    imported = load_textured_scene(library, path, path.parent)
    wanted = {name + suffix: len(part.positions) for name, part in expected.items()}
    actual = {name: len(section.positions) for name, section in imported.sections.items()}
    if wanted != actual:
        raise ValueError(f"Export round trip changed named geometry in {path.name}")
    def images(bindings):
        return {(image.size, sha256(image.tobytes()).hexdigest(), bool(repeat))
                for image, repeat in bindings.values()}
    bindings = {label: binding for part in expected.values() for label, binding in part.textures.items()}
    if images(imported.textures) != images(bindings):
        raise ValueError(f"Export round trip changed diffuse images or sampler modes in {path.name}")
    return {"triangles": len(imported.positions), "named_parts": len(actual),
            "image_and_sampler_bindings_preserved": True}


def edited_masks(source, target):
    def keys(scene):
        corners = np.concatenate(scene.arrays[:3], axis=2).astype(np.float32)
        result = []
        for face, material in zip(corners, scene.materials):
            a, b, c = map(tuple, face)
            result.append((int(material), min((a, b, c), (b, c, a), (c, a, b))))
        return result
    before, after = keys(source), keys(target)
    original, generated = set(before), set(after)
    source_mask = np.array([face not in generated for face in before])
    target_mask = np.array([face not in original for face in after])
    return (source_mask[np.argsort(source.materials, kind="stable")],
            target_mask[np.argsort(target.materials, kind="stable")])


def write_physics_candidate(output, repaired, physics_report, *, force_diagnostic=False):
    """Publish the usual companion only if every triangle passes both checks."""
    ready = all(not (physics_degenerate_mask(part) | packed_physics_degenerate_mask(part)).any()
                for part in repaired.values())
    filename = "Lodestar_PHYS.dae" if ready and not force_diagnostic else "Lodestar_unresolved_PHYS.dae"
    if not ready and (output / "Lodestar_PHYS.dae").exists():
        raise ValueError("Unresolved physics cannot replace an existing upload companion. Use a fresh output folder.")
    physics_report.update(triangle_upload_ready=ready, file=filename,
                          status=("blocked_size_container_proposal" if force_diagnostic else
                                  "triangle_checks_passed" if ready else "blocked_unresolved_triangles"))
    write_lod_collision_collada(output / filename, repaired)
    return output / filename


def run(args):
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise ValueError("The source mesh does not exist")
    output.mkdir(parents=True, exist_ok=True)
    stem = "Lodestar"
    if source.parent == output:
        raise ValueError("Use a separate output directory for the generated LOD package")
    for suffix in (*SUFFIXES, "_PHYS", "_source_PHYS"):
        for extension in (".dae", ".glb", ".gltf"):
            target = output / f"{stem}{suffix}{extension}"
            if target.exists() and (target.samefile(source) or
                    (args.existing_physics and target.samefile(args.existing_physics))):
                raise ValueError("A generated output would overwrite an authoritative input")
    (output / "scorecard.json").write_text(json.dumps({"status": "building", "source": str(source)}), encoding="utf-8")
    source_hash = sha256(source.read_bytes()).hexdigest()
    started = perf_counter()
    scene = load_textured_scene(args.library, source, args.textures)
    levels, report = plan_levels(scene, args.library, tuple(args.pixels), args.pixel_error, args.normal_angle)
    report['size_planning'] = size_proposals(levels['high'], args.library, output, declared_bounds=scene.declared_bounds,
        import_scale=getattr(args, 'import_scale', 1.0))
    physical_frames, containers = apply_physical_containers(report['size_planning'], args.library, output)
    report.update(status="complete", source=source.name, source_sha256=source_hash,
                  visual_generator="lodestar_uv_cavity", external_simplifier_used=False,
                  authoritative_high_triangles=len(scene.positions),
                  floating_point_bounds="double arithmetic with scale-dependent allowances; interval certificates pending",
                  uv_mapping="High UV chart domains and source texture bindings retained; shared seam edits coupled",
                  normal_limit_degrees=args.normal_angle)
    report["source_diffuse_bindings"] = {str(label): {"size": list(image.size), "repeat": bool(repeat),
        "rgb_sha256": sha256(image.tobytes()).hexdigest()} for label, (image, repeat) in scene.textures.items()}
    high = combine_lod_parts(levels["high"])
    for level, suffix in zip(LEVELS, SUFFIXES):
        path = write_level(output, stem, levels[level], suffix, physical_frames=physical_frames)
        current = combine_lod_parts(levels[level])
        source_edited, target_edited = edited_masks(high, current)
        measured = compare_section(*high.render_arrays, *current.render_arrays, resolution=args.view_size,
                                   source_edited=source_edited, target_edited=target_edited,
                                   source_label="High diffuse", target_label=f"{level.capitalize()} diffuse")
        measured.image.save(output / f"{level}-comparison.png")
        report["levels"][level].update(glb_bytes=path.stat().st_size, comparison=measured.metrics)
        report["levels"][level]["roundtrips"] = {
            extension: verify_roundtrip(args.library, path.with_suffix(extension), levels[level], suffix)
            for extension in (".glb", ".gltf", ".dae")}
        print(f"{level}: {len(current.positions)} triangles")
    # Retain High for diagnosis; collision selection may use a generated lower
    # level, with visual attributes removed and an explicit composed bound.
    physics = {name: part.positions for name, part in levels["high"].items()}
    write_lod_collision_collada(output / "Lodestar_source_PHYS.dae", physics)
    collision_levels = {level: {name: SimpleNamespace(positions=containers[name]) if name in containers else part
        for name, part in parts.items()} for level, parts in levels.items()}
    repaired, physics_report = prepare_physics(collision_levels, report, args.library)
    if containers:
        # Both High and its physical container lie within these bounds. Their
        # diameter is a conservative bidirectional bound, not a form-fit claim.
        container_bound = max(float(np.linalg.norm(np.asarray(p['container_size']))) for p in physical_frames.values())
        physics_report['maximum_surface_change_before_packing'] += container_bound
        physics_report.update(physical_container_models=list(containers),
            physical_container_surface_bound=container_bound,
            physical_container_policy='legal physical bounds with unchanged authored render surface')
    physics_report['maximum_surface_change_after_import_scale_before_packing'] = (
        physics_report['maximum_surface_change_before_packing'] * report['size_planning']['import_scale'])
    physics_path = write_physics_candidate(output, repaired, physics_report,
        force_diagnostic=report['size_planning']['requires_frame_review'])
    if physical_frames:
        write_lod_collision_collada(physics_path, repaired, physical_frames=physical_frames)
    report["physics"] = physics_report
    if report['size_planning']['requires_frame_review']:
        report['status'] = 'container_proposed'
    elif not physics_report["triangle_upload_ready"]:
        report["status"] = "physics_blocked"
    unresolved = sum(len(part["unresolved_triangle_indices"]) for part in physics_report["parts"].values())
    print(f"physics: {sum(len(part) for part in repaired.values())} triangles, "
          f"{unresolved} unresolved triangle-uploader rejections")
    if args.existing_physics:
        with _scene_mesh(args.library, args.existing_physics) as (view, _, _):
            indices = np.ctypeslib.as_array(view.indices, shape=(view.index_count,)).reshape(-1, 3)
            vertices = np.ctypeslib.as_array(view.vertices, shape=(view.vertex_count,))
            old = np.asarray(vertices["position"][indices], dtype=np.float64)
        report["physics"]["existing_collider"] = {
            "file": args.existing_physics.name,
            "sha256": sha256(args.existing_physics.read_bytes()).hexdigest(),
            "triangles_after_native_import": len(old),
            "aggregate_uploader_rejections": int(np.count_nonzero(
                physics_degenerate_mask(old) | packed_physics_degenerate_mask(old))),
            "aggregate_check_scope": "diagnostic only; uploader normalization is per named model",
            "translation_for_comparison": args.existing_translation}
    if args.benchmark:
        for label, candidate, translation in (
            ("rebuilt", physics_path, (0, 0, 0)),
            ("existing", args.existing_physics, args.existing_translation)):
            if candidate is None:
                continue
            command = [str(args.benchmark.resolve()), "--authored-lod", str(candidate.resolve()),
                       "--authored-translate", *map(str, translation), str((output / "Lodestar.glb").resolve()),
                       "--repeats", "1", "--samples", "4096", "--resolution", "128"]
            measured = subprocess.run(command, text=True, capture_output=True, check=True)
            (output / f"{label}-physics-comparison.csv").write_text(measured.stdout, encoding="utf-8")
    report["elapsed_seconds"] = perf_counter() - started
    generated = {output / f"{stem}{suffix}{extension}" for suffix in SUFFIXES
                 for extension in (".dae", ".glb", ".gltf")}
    generated.update((physics_path, output / "Lodestar_source_PHYS.dae"))
    generated.add(output / 'size-proposals.json')
    generated.update((output / 'size-proposals').glob('*.dae'))
    for suffix in SUFFIXES:
        document = json.loads((output / f"{stem}{suffix}.gltf").read_text(encoding="utf-8"))
        generated.update(output / buffer["uri"] for buffer in document["buffers"])
        xml = ET.parse(output / f"{stem}{suffix}.dae")
        generated.update(output / unquote(item.text) for item in xml.findall(
            ".//{http://www.collada.org/2005/11/COLLADASchema}image/{http://www.collada.org/2005/11/COLLADASchema}init_from"))
    if sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError("The authoritative source changed while fitting")
    report["output_sha256"] = {str(path.relative_to(output)): sha256(path.read_bytes()).hexdigest()
                               for path in sorted(generated)}
    temporary = output / "scorecard.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "scorecard.json")
    return physics_report["triangle_upload_ready"] and not report['size_planning']['requires_frame_review']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--textures", type=Path, help="COLLADA diffuse texture directory")
    parser.add_argument("--pixels", nargs=4, type=int, default=(1024, 512, 192, 96))
    parser.add_argument("--pixel-error", type=float, default=0.5)
    parser.add_argument("--normal-angle", type=float, default=1.0)
    parser.add_argument("--view-size", type=int, default=384)
    parser.add_argument("--existing-physics", type=Path)
    parser.add_argument("--existing-translation", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--import-scale", type=float, default=1.0)
    args = parser.parse_args()
    if not 64 <= args.view_size <= 1024:
        parser.error("Use a comparison view size from 64 to 1024")
    if not np.isfinite(args.existing_translation).all():
        parser.error("The existing physics translation must be finite")
    try:
        ready = run(args)
        if not ready:
            score = json.loads((args.output / 'scorecard.json').read_text())
            if score['status'] == 'container_proposed':
                parser.exit(2, 'A minimum-size container is proposed. Review size-proposals.json; the authored mesh is unchanged.\n')
            parser.exit(2, "Visual LODs generated. Physics is blocked. See scorecard.json and the unresolved diagnostic companion.\n")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Lodestar surface fitting failed: {error}\n")


if __name__ == "__main__":
    main()
