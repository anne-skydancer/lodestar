"""Check a Wing2Break geometry trial against TestHall's source and walkable openings."""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from check_testhall_physics import CAPSULE_HEIGHT, CAPSULE_RADIUS, floor_height, load_scene, route_endpoints, triangle_key
from lodestar_collision_probes import sweep_clearance
from lodestar_section_source import load_section


LIMITS = {
    "surface_max_relative": 0.001,
    "silhouette_max_iou_error": 0.005,
    "route_delta_m": 0.001,
    "floor_delta_m": 0.001,
}


def run(args):
    section = load_section(args.library, args.source, "Wing2Break")
    original, _ = load_scene(args.library, args.source)
    source_path, lod_path = args.trial / "source.glb", args.trial / "lod.glb"
    exported, _ = load_scene(args.library, source_path)
    lod, _ = load_scene(args.library, lod_path)
    if Counter(map(triangle_key, exported)) != Counter(map(triangle_key, section.positions)):
        raise ValueError("The trial source GLB must contain the original Wing2Break geometry")

    keep = np.ones(len(original), dtype=bool)
    keep[section.face_indices] = False
    candidate = np.concatenate((original[keep], lod))
    probes = {
        "east_west_opening": (*route_endpoints(-4.27), CAPSULE_RADIUS, CAPSULE_HEIGHT),
        "solid_wall": (*route_endpoints(-4.75), CAPSULE_RADIUS, CAPSULE_HEIGHT),
        "arch": ((11.72, 7.94, 0.20), (11.72, 7.94, -1.25), 0.28, 1.75),
    }
    routes = {}
    for name, (start, end, radius, height) in probes.items():
        before = sweep_clearance(original, start, end, radius, height)
        after = sweep_clearance(candidate, start, end, radius, height)
        routes[name] = {"source_clearance_m": before, "candidate_clearance_m": after,
                        "passed": (before > 0) == (after > 0) and abs(before - after) <= LIMITS["route_delta_m"]}
    floors = {
        label: floor_height(mesh, 10.5, -3.70, 7.65, 8.15)
        for label, mesh in (("source", original), ("candidate", candidate))
    }
    floor_ok = all(height is not None for height in floors.values()) and (
        abs(floors["source"] - floors["candidate"]) <= LIMITS["floor_delta_m"]
    )

    benchmark = subprocess.run([str(args.benchmark.resolve()), "--authored-lod", str(lod_path), str(source_path),
                                "--repeats", "3", "--samples", "16384", "--resolution", "384"],
                               check=True, capture_output=True, text=True)
    row, = [item for item in csv.DictReader(io.StringIO(benchmark.stdout)) if item["representation"] == "authored_lod"]
    metrics = {key: float(row[key]) for key in ("surface_rms_relative", "surface_max_relative",
                                               "silhouette_mean_iou_error", "silhouette_max_iou_error")}
    metrics.update(silhouette_views=int(row["silhouette_views"]),
                   surface_samples_per_direction=int(row["surface_samples_per_direction"]))
    geometry_ok = (metrics["surface_max_relative"] <= LIMITS["surface_max_relative"] and
                   metrics["silhouette_max_iou_error"] <= LIMITS["silhouette_max_iou_error"] and
                   all(route["passed"] for route in routes.values()) and floor_ok)
    inputs = json.loads((args.trial / "report.json").read_text(encoding="utf-8"))
    if inputs["target_triangles"] != len(lod):
        raise ValueError("Candidate GLB triangle count disagrees with the trial report")
    report = dict(source_triangles=len(exported), candidate_triangles=len(lod),
                  requested_triangles=inputs["requested_triangles"], triangle_target_met=inputs["triangle_target_met"],
                  geometry_ok=geometry_ok, limits=LIMITS, routes=routes, floor=floors, metrics=metrics)
    (args.trial / "geometry-comparison.csv").write_text(benchmark.stdout, encoding="utf-8")
    document = json.dumps(report, indent=2, allow_nan=False)
    (args.trial / "geometry-check.json").write_text(document + "\n", encoding="utf-8")
    print(document)
    if not geometry_ok:
        raise ValueError("Wing2Break geometry did not pass the trial tolerances")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original TestHall_MESH.dae")
    parser.add_argument("trial", type=Path, help="Directory with source.glb, lod.glb, and report.json")
    parser.add_argument("--library", type=Path, required=True, help="Compiled Lodestar C library")
    parser.add_argument("--benchmark", type=Path, required=True, help="Compiled Lodestar benchmark executable")
    try:
        run(parser.parse_args())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Wing2Break geometry check: {error}\n")


if __name__ == "__main__":
    main()
