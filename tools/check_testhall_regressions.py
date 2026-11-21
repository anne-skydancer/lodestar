"""Reproduce the TestHall geometry, collision, and optional texture baselines on CPU."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from check_testhall_full import node_names


TOOLS = Path(__file__).resolve().parent
WING_TRIALS = (
    (993, 993, ("--max-boundary-deviation", "1e-7")),
    (883, 882, ("--max-boundary-deviation", "5e-5", "--normal-tolerance", "5e-4",
                "--plane-relative-tolerance", "1e-4")),
)


def invoke(script, *arguments):
    command = [sys.executable, "-B", str(TOOLS / script), *map(str, arguments)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{script} exited with status {result.returncode}\n{result.stderr or result.stdout}")


def read_report(directory, name="report.json"):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wing_geometry(args, source, trial, target, expected, profile):
    build_options = ("--library", args.library, "--cross-uv", "--target-triangles", target, *profile)
    invoke("bake_section_atlas.py", source, args.corpus, trial, *build_options, "--geometry-only")
    invoke("check_wing2break_geometry.py", source, trial, "--library", args.library, "--benchmark", args.benchmark)
    report = read_report(trial, "geometry-check.json")
    require((report["source_triangles"], report["requested_triangles"], report["candidate_triangles"]) ==
            (1104, target, expected) and report["triangle_target_met"], f"Wing2Break {target} triangle count changed")
    require(report["geometry_ok"] and report["metrics"]["silhouette_views"] == 13 and
            report["metrics"]["silhouette_max_iou_error"] == 0.0, f"Wing2Break {target} silhouette changed")
    for name, route in report["routes"].items():
        require(route["passed"] and math.isclose(route["source_clearance_m"], route["candidate_clearance_m"], abs_tol=1e-6),
                f"Wing2Break {target} clearance changed at {name}")
    floor = report["floor"]
    require(math.isclose(floor["source"], floor["candidate"], abs_tol=1e-6), f"Wing2Break {target} floor height changed")
    summary = {"target": target, "source_triangles": 1104, "candidate_triangles": expected,
               "silhouette_views": 13, "geometry_report": str(trial / "geometry-check.json")}
    if args.textures is not None:
        baked = args.output / f"wing2break-{target}-texture"
        invoke("bake_section_atlas.py", source, args.textures, baked, *build_options, "--crop-repeats",
               "--texel-scale", "0.10", "--max-tile", "256", "--retire-material", "0")
        texture = read_report(baked)
        require(texture["target_triangles"] == expected and texture["comparison"]["silhouette_iou"] == 1.0,
                f"Wing2Break {target} texture export changed geometry")
        expected_color = {993: 7.19, 882: 7.82}[expected]
        color_error = texture["comparison"]["edited_mean_abs_error"]
        before_bytes = texture["crop_only_source"]["base_rgba8_bytes"]
        after_bytes = texture["target_base_rgba8_bytes"]
        require(math.isclose(color_error, expected_color, abs_tol=0.10) and after_bytes < before_bytes,
                f"Wing2Break {target} texture color or memory result changed")
        summary["texture"] = {"edited_mean_color_error": color_error, "cropped_source_rgba8_bytes": before_bytes,
                              "candidate_rgba8_bytes": after_bytes, "report": str(baked / "report.json")}
    return summary


def run(args):
    source = args.corpus / "TestHall_MESH.dae"
    authored = args.corpus / "TestHall_LOD2.dae"
    physics = args.corpus / "TestHall_PHYS.dae"
    for path in (source, authored, physics, args.lodestar, args.library, args.benchmark):
        require(path.is_file(), f"Required TestHall baseline input is missing at {path}")
    if args.textures is not None:
        require(args.textures.is_dir(), f"Texture directory is missing at {args.textures}")
    wings = [wing_geometry(args, source, args.output / f"wing2break-{target}", target, expected, profile)
             for target, expected, profile in WING_TRIALS]

    repair_dir = args.output / "physics"
    invoke("check_testhall_physics.py", args.corpus, repair_dir, "--library", args.library, "--lodestar", args.lodestar)
    repair = read_report(repair_dir)
    clearance = repair["routes"]["authored_blockage"]["signed_clearance_m"]
    require(repair["passed"] and math.isclose(clearance["authored"], -0.0287, abs_tol=0.0005) and
            math.isclose(clearance["repaired"], 0.0215, abs_tol=0.0005), "TestHall physics clearance baseline changed")
    authored_names = node_names(physics)
    repaired_names = node_names(repair_dir / "testhall-repaired-physics.dae")
    require(authored_names == repaired_names and repair["replaced_node"] in authored_names and len(authored_names) == 12 and
            repair["unchanged_triangles"] == 686, "TestHall physics repair did not preserve the other 11 parts")

    full_dir = args.output / "full"
    invoke("check_testhall_full.py", args.corpus, full_dir, "--library", args.library, "--benchmark", args.benchmark)
    full = read_report(full_dir)
    require((full["source_triangles"], full["candidate_triangles"], full["authored_medium_triangles"]) == (8602, 7336, 4769),
            "Full TestHall triangle baselines changed")
    require(full["candidate_metrics"]["silhouette_views"] == 13 and
            full["candidate_metrics"]["silhouette_max_iou_error"] == 0.0, "Full TestHall silhouette baseline changed")

    summary = {"passed": True, "wing2break": wings,
               "physics": {"authored_clearance_m": clearance["authored"], "repaired_clearance_m": clearance["repaired"],
                           "preserved_parts": 11, "report": str(repair_dir / "report.json")},
               "full": {"source_triangles": 8602, "candidate_triangles": 7336, "authored_triangles": 4769,
                        "report": str(full_dir / "report.json")}}
    document = json.dumps(summary, indent=2, allow_nan=False)
    (args.output / "scorecard.json").write_text(document + "\n", encoding="utf-8")
    print(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Directory with the original, authored medium, and physics TestHall files")
    parser.add_argument("output", type=Path, help="Directory for regression artifacts and scorecard")
    parser.add_argument("--library", type=Path, required=True, help="Path to the Lodestar C ABI library")
    parser.add_argument("--lodestar", type=Path, required=True, help="Path to inverseglod for CPU physics export")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to the native CPU benchmark")
    parser.add_argument("--textures", type=Path, help="Also reproduce textured GLBs using this extracted texture folder")
    try:
        run(parser.parse_args())
    except (OSError, RuntimeError, ValueError, ET.ParseError) as error:
        parser.exit(1, f"TestHall regression: {error}\n")


if __name__ == "__main__":
    main()
