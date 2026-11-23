"""Verify a repair of the authored false blockage in Second Life's TestHall fixture."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np

from lodestar_collada_physics import replace_physics_part
from lodestar_collision_probes import sweep_clearance
from lodestar_patch_source import _scene_mesh
from lodestar_section_source import load_section


SOURCE_NODE = "Wing2Break"
AUTHORED_NODE = "Wing2Break_PHYS"
AUTHORED_SURFACE_ID = 10
# The archived files are laid out side by side. Scene-space matching of shared
# Wing2Break vertices gives this translation from the visual to the physics file.
AUTHORED_TRANSLATION = (13.13098, 0.0, 0.0)
CAPSULE_RADIUS = 0.22
CAPSULE_HEIGHT = 1.69
FEET_Y = 7.94
ROUTES = {
    "authored_blockage": -4.27,
    "solid_wall": -4.75,
    "open_passage": -3.70,
}


def load_scene(library, path):
    with _scene_mesh(library, path) as (view, _, provenance):
        indices = np.ctypeslib.as_array(view.indices, shape=(view.index_count,)).reshape(-1, 3)
        vertices = np.ctypeslib.as_array(view.vertices, shape=(view.vertex_count,))
        triangles = np.asarray(vertices["position"][indices], dtype=np.float64)
        if provenance.triangle_surface_id_count != len(triangles):
            raise ValueError(f"Missing source surface IDs in {path.name}")
        surfaces = np.ctypeslib.as_array(provenance.triangle_surface_ids, shape=(len(triangles),)).copy()
        return triangles, surfaces


def triangle_key(triangle):
    corners = tuple(map(tuple, triangle))
    return min(corners[i:] + corners[:i] for i in range(3))


def generated_part(generated, section):
    remaining = Counter(map(triangle_key, section))
    part = []
    for triangle in generated:
        key = triangle_key(triangle)
        if remaining[key] > 0:
            part.append(triangle)
            remaining[key] -= 1
    if any(remaining.values()):
        raise ValueError("The CLI export omits triangles from the Wing2Break source")
    return np.array(part)


def route_endpoints(z):
    return (9.40, FEET_Y, z), (14.05, FEET_Y, z)


def floor_height(triangles, x, z, low_y, high_y):
    # Intersect a downward vertical ray with projected triangle interiors.
    a, b, c = np.moveaxis(triangles, 1, 0)
    u, v = b - a, c - a
    x_offset, z_offset = x - a[:, 0], z - a[:, 2]
    determinant = u[:, 0] * v[:, 2] - v[:, 0] * u[:, 2]
    valid = determinant != 0
    first = np.divide(x_offset * v[:, 2] - z_offset * v[:, 0], determinant,
                      out=np.zeros_like(determinant), where=valid)
    second = np.divide(z_offset * u[:, 0] - x_offset * u[:, 2], determinant,
                       out=np.zeros_like(determinant), where=valid)
    height = a[:, 1] + first * u[:, 1] + second * v[:, 1]
    hits = valid & (first >= 0) & (second >= 0) & (first + second <= 1) & (height >= low_y) & (height <= high_y)
    return float(height[hits].max()) if np.any(hits) else None


def run(args):
    visual_path = args.corpus / "TestHall_MESH.dae"
    authored_path = args.corpus / "TestHall_PHYS.dae"
    visual, _ = load_scene(args.library, visual_path)
    authored, authored_ids = load_scene(args.library, authored_path)
    section = load_section(args.library, visual_path, SOURCE_NODE).positions
    if len(visual) != 8602 or len(authored) != 958 or len(section) != 1104:
        raise ValueError("Unexpected TestHall triangle counts; use the original Second Life import-test archive")
    args.output.mkdir(parents=True, exist_ok=True)
    full_path = args.output / "testhall-full-physics.dae"
    repaired_path = args.output / "testhall-repaired-physics.dae"
    subprocess.run([str(args.lodestar.resolve()), "--export-physics", str(full_path), str(visual_path)], check=True)
    rebuilt, _ = load_scene(args.library, full_path)
    if Counter(map(triangle_key, rebuilt)) != Counter(map(triangle_key, visual)):
        raise ValueError("Lodestar's reimported export differs from the visual geometry")
    updated = generated_part(rebuilt, section)
    replace_physics_part(authored_path, repaired_path, AUTHORED_NODE, updated, AUTHORED_TRANSLATION)
    repaired, repaired_ids = load_scene(args.library, repaired_path)
    unchanged_old = authored[authored_ids != AUTHORED_SURFACE_ID]
    unchanged_new = repaired[repaired_ids != AUTHORED_SURFACE_ID]
    if (len(authored) - len(unchanged_old) != 272 or len(repaired) - len(unchanged_new) != len(section)
            or Counter(map(triangle_key, unchanged_old)) != Counter(map(triangle_key, unchanged_new))):
        raise ValueError("Reimport did not retain the other authored collider surfaces")

    meshes = {
        "visual": visual,
        "authored": authored - AUTHORED_TRANSLATION,
        "fully_regenerated": rebuilt,
        "repaired": repaired - AUTHORED_TRANSLATION,
    }
    results = {}
    for name, z in ROUTES.items():
        start, end = route_endpoints(z)
        results[name] = {
            "start_feet_m": start,
            "end_feet_m": end,
            "signed_clearance_m": {
                label: sweep_clearance(mesh, start, end, CAPSULE_RADIUS, CAPSULE_HEIGHT)
                for label, mesh in meshes.items()
            },
        }
    floor_x, floor_z = 10.5, -3.70
    floor_low, floor_high = 7.65, 8.15
    floors = {label: floor_height(mesh, floor_x, floor_z, floor_low, floor_high) for label, mesh in meshes.items()}
    fault, wall, open_passage = (results[name]["signed_clearance_m"] for name in ROUTES)
    passed = (
        fault["authored"] < -0.02
        and all(fault[label] > 0.01 for label in ("visual", "fully_regenerated", "repaired"))
        and all(value < 0 for value in wall.values())
        and all(value > 0 for value in open_passage.values())
        and all(value is not None and abs(value - floors["visual"]) < 0.001 for value in floors.values())
    )
    report = {
        "fixture": "Second Life Mesh Import Test / TestHall",
        "coordinate_space": "Y-up scene space in meters",
        "authored_physics": str(authored_path),
        "repaired_physics": str(repaired_path),
        "authored_to_visual_translation_m": [-value for value in AUTHORED_TRANSLATION],
        "replaced_node": AUTHORED_NODE,
        "old_node_triangles": len(authored) - len(unchanged_old),
        "new_node_triangles": len(updated),
        "unchanged_triangles": len(unchanged_new),
        "triangle_counts": {label: len(mesh) for label, mesh in meshes.items()},
        "capsule_radius_m": CAPSULE_RADIUS,
        "capsule_height_m": CAPSULE_HEIGHT,
        "routes": results,
        "floor_probe": {
            "start_m": (floor_x, floor_high, floor_z),
            "end_m": (floor_x, floor_low, floor_z),
            "first_hit_y_m": floors,
        },
        "passed": passed,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError(f"The repaired collider failed a TestHall probe. See {report_path}")
    print(f"Capsule clearance: {fault['authored']:+.4f} m authored, {fault['repaired']:+.4f} m after repair")
    print(f"Wall, passage, and floor controls passed. Wrote {repaired_path} and {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Directory with the original TestHall visual and physics files")
    parser.add_argument("output", type=Path, help="Directory for collision files and probe report")
    parser.add_argument("--library", type=Path, required=True, help="Path to the Lodestar C ABI library")
    parser.add_argument("--lodestar", type=Path, required=True, help="Path to the inverseglod executable")
    try:
        run(parser.parse_args())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"TestHall physics check: {error}\n")


if __name__ == "__main__":
    main()
