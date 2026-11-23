"""Assemble and measure the full TestHall LOD in original scene space."""

import argparse
import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from bake_section_atlas import indexed_primitives, replace_patches
from lodestar_patch_source import _scene_mesh
from lodestar_section_source import by_id, load_section, reference
from lodestar_textured_glb import write_textured_glb
from lodestar_uv_regions import plan_uv_regions


PROFILE = {"max_boundary_deviation": 5e-5, "normal_tolerance": 5e-4, "plane_relative_tolerance": 1e-4}
AUTHORED_TRANSLATE_X = 13.59409


@dataclass
class FullAssembly:
    source: tuple
    candidate: tuple
    sections: list
    unsupported_nodes: list
    preserved_unsupported_triangles: int


def read_geometry(library, path):
    """Read positions, top-origin UVs, and normals for each corner of each triangle."""
    with _scene_mesh(library, path) as (view, _, _):
        indices = np.ctypeslib.as_array(view.indices, shape=(view.index_count,)).reshape(-1, 3)
        vertices = np.ctypeslib.as_array(view.vertices, shape=(view.vertex_count,))
        return vertices["position"][indices], vertices["uv"][indices] * [1, -1] + [0, 1], vertices["normal"][indices]


def node_names(path):
    root = ET.parse(path).getroot()
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    instance = root.find("./scene/instance_visual_scene")
    if instance is None:
        raise ValueError(f"{path.name} has no active visual scene")
    scene = by_id(root, "./library_visual_scenes/visual_scene", reference(instance.get("url", "")))
    names = {node.get("name") or node.get("id") for node in scene.iter("node")
             if node.find("instance_geometry") is not None}
    return sorted(name for name in names if name)


def _validate_geometry(arrays):
    positions, uv, normals = arrays
    count = len(positions)
    if (count == 0 or positions.shape != (count, 3, 3) or normals.shape != (count, 3, 3) or
            uv.shape != (count, 3, 2) or not all(np.isfinite(array).all() for array in arrays)):
        raise ValueError("Export requires finite per-corner positions, UVs, and normals")


def export_geometry(path, positions, uv, normals):
    """Write an untextured GLB with identical attributes at shared indices."""
    arrays = (positions, uv, normals)
    _validate_geometry(arrays)
    material = np.zeros(len(positions), dtype=np.int32)
    attributes = np.column_stack([array.reshape(-1, array.shape[-1]) for array in arrays])
    _, indices = np.unique(attributes, axis=0, return_inverse=True)
    write_textured_glb(path, indexed_primitives(*arrays, material, indices.reshape(-1, 3)), None)


def assemble_full(library, source, *, replace_section=None):
    """Apply the bounded planar pass or a validated replacement for each mapped node.

    The optional callback receives a section and returns positions, top-origin UVs,
    and normals. Returning None selects the bounded planar pass for that section.
    """
    original = read_geometry(library, source)
    covered = np.zeros(len(original[0]), dtype=bool)
    chunks, sections, unsupported = [], [], []
    for node in node_names(source):
        try:
            section = load_section(library, source, node)
        except ValueError as error:
            unsupported.append({"node": node, "reason": str(error)})
            continue
        if covered[section.face_indices].any():
            raise ValueError(f"{node} shares imported triangles with another selected node")
        covered[section.face_indices] = True
        entry = {"node": node, "source_triangles": len(section.positions)}
        replacement = replace_section(section) if replace_section else None
        if replacement is None:
            _, plans = plan_uv_regions(section, outside_positions=section.outside_positions, **PROFILE)
            patches = [(layout.faces, points, uv, triangles, layout.normal) for layout, points, uv, triangles in plans]
            replacement = replace_patches(section, section.source_uvs * [1, -1] + [0, 1], patches,
                                          max(section.materials) + 1)[:3]
            entry.update(method="bounded_planar", patches=len(plans),
                         max_boundary_shift_m=max((plan[0].boundary_error for plan in plans), default=0.0),
                         max_plane_residual_m=max((plan[0].plane_error for plan in plans), default=0.0))
        else:
            replacement = tuple(np.asarray(array) for array in replacement)
            _validate_geometry(replacement)
            entry["method"] = "replacement"
        entry["candidate_triangles"] = len(replacement[0])
        sections.append(entry)
        chunks.append(replacement)

    full = tuple(np.concatenate([attributes[~covered], *[part[i] for part in chunks]])
                 for i, attributes in enumerate(original))
    untouched = int((~covered).sum())
    if len(full[0]) != untouched + sum(section["candidate_triangles"] for section in sections):
        raise ValueError("The assembled triangle count disagrees with the selected replacements")
    return FullAssembly(original, full, sections, unsupported, untouched)


def benchmark_pair(benchmark, source, candidate, csv_path, *, translate_x=0.0, samples=16384, resolution=384):
    """Compare a candidate in the shared frame and retain the native benchmark CSV."""
    command = [str(Path(benchmark).resolve()), "--authored-lod", str(candidate), str(source),
               "--repeats", "3", "--samples", str(samples), "--resolution", str(resolution)]
    if translate_x:
        command.extend(("--authored-translate", str(translate_x), "0", "0"))
    run = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = [row for row in csv.DictReader(io.StringIO(run.stdout)) if row["representation"] == "authored_lod"]
    if len(rows) != 1 or rows[0]["status"] != "ok":
        raise ValueError(f"The native benchmark did not produce one measured comparison for {candidate}")
    row = rows[0]
    csv_path.write_text(run.stdout, encoding="utf-8")
    return {**{key: int(row[key]) for key in ("source_triangles", "triangles", "silhouette_views",
                                             "silhouette_resolution", "surface_samples_per_direction")},
            **{key: float(row[key]) for key in ("source_extent", "surface_rms_relative", "surface_max_relative",
                                                "silhouette_mean_iou_error", "silhouette_max_iou_error")}}


def run(corpus, output, library, *, benchmark=None, replace_section=None):
    source = corpus / "TestHall_MESH.dae"
    authored = corpus / "TestHall_LOD2.dae"
    assembly = assemble_full(library, source, replace_section=replace_section)
    source_count, lod_count = len(assembly.source[0]), len(assembly.candidate[0])
    authored_count = len(read_geometry(library, authored)[0])
    output.mkdir(exist_ok=True, parents=True)
    source_glb, lod_glb = output / "source.glb", output / "lod.glb"
    export_geometry(source_glb, *assembly.source)
    export_geometry(lod_glb, *assembly.candidate)
    if len(read_geometry(library, source_glb)[0]) != source_count or len(read_geometry(library, lod_glb)[0]) != lod_count:
        raise ValueError("The exported GLBs changed triangle count on reimport")
    report = dict(source_triangles=source_count, candidate_triangles=lod_count, authored_medium_triangles=authored_count,
                  triangle_reduction_fraction=1 - lod_count / source_count, above_authored_triangles=lod_count - authored_count,
                  preserved_unsupported_triangles=assembly.preserved_unsupported_triangles, profile=PROFILE,
                  sections=assembly.sections, unsupported_nodes=assembly.unsupported_nodes)
    if benchmark is not None:
        report["candidate_metrics"] = benchmark_pair(benchmark, source_glb, lod_glb, output / "candidate-vs-source.csv")
        report["authored_metrics"] = benchmark_pair(benchmark, source_glb, authored, output / "authored-vs-source.csv",
                                                   translate_x=AUTHORED_TRANSLATE_X)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Directory with original and authored TestHall COLLADA files")
    parser.add_argument("output", type=Path, help="Directory for the GLBs, JSON report, and optional benchmark CSVs")
    parser.add_argument("--library", type=Path, required=True, help="Path to the Lodestar C ABI library")
    parser.add_argument("--benchmark", type=Path, help="Path to lodestar_benchmark for candidate and authored measurements")
    try:
        args = parser.parse_args()
        report = run(args.corpus, args.output, args.library, benchmark=args.benchmark)
        print(json.dumps(report, indent=2, allow_nan=False))
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"TestHall full check: {error.stderr or error}\n")
    except (OSError, RuntimeError, ValueError, ET.ParseError) as error:
        parser.exit(1, f"TestHall full check: {error}\n")


if __name__ == "__main__":
    main()
