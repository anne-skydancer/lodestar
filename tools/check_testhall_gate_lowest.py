"""Compare stock and Lodestar gate geometry at the same actual triangle count."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from check_testhall_physics import load_scene
from lodestar_firestorm_degenerate import visual_degenerate_mask
from lodestar_gate_shape import compare_gate_shape
from lodestar_section_compare import BACKGROUND, _project_pair, _rasterize


def read_triangles(path, library):
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "positions" not in data:
                raise ValueError(f"{path.name} needs a positions array")
            return np.asarray(data["positions"], dtype=np.float64)
    if library is None:
        raise ValueError("Mesh inputs require --library. Position arrays can use .npy or .npz.")
    return load_scene(library, path)[0]


def bounds(positions):
    return np.stack((positions.min(axis=(0, 1)), positions.max(axis=(0, 1))))


def comparison_image(source, stock, candidate):
    """Render the same cull-aware gate views with a shared close-up frame."""
    size, spacing, header = 192, 12, 24
    angles = (("Front", (1, 0, 0)), ("Back", (-1, 0, 0)),
              ("Elevated", (1, 1, 1)), ("Edge", (0, 0, 1)))
    panels = ((source, f"Original · {len(source):,} triangles"),
              (stock, f"Firestorm Auto · {len(stock):,} triangles"),
              (candidate, f"Lodestar · {len(candidate):,} triangles"))
    sheet = Image.new("RGB", (3 * size + 4 * spacing, header + len(angles) * (size + spacing + header)), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    for column, (mesh, label) in enumerate(panels):
        x = spacing + column * (size + spacing)
        draw.text((x, spacing - 3), label, fill="white")
        for row, (angle, direction) in enumerate(angles):
            y = header + row * (size + spacing + header)
            if column == 0:
                draw.text((x, y), angle, fill="white")
            _, projected = _project_pair(source, mesh, direction, size, source)
            a, b = projected[:, 1, :2] - projected[:, 0, :2], projected[:, 2, :2] - projected[:, 0, :2]
            facing = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) < 0
            visible = _rasterize(projected[facing], size)[0] >= 0
            panel = np.zeros((size, size, 3), dtype=np.uint8)
            panel[:] = BACKGROUND
            panel[visible] = (215, 222, 232)
            sheet.paste(Image.fromarray(panel), (x, y + header))
    return sheet


def run(args):
    source = read_triangles(args.high, args.library)
    frame = read_triangles(args.scene_frame, args.library)
    inputs = {"firestorm": args.stock, "lodestar": args.candidate}
    positions = {name: read_triangles(path, args.library) for name, path in inputs.items()}
    counts = {name: len(mesh) for name, mesh in positions.items()}
    if counts["firestorm"] != counts["lodestar"]:
        raise ValueError(f"The comparison needs equal realized counts, got {counts}")
    args.output.mkdir(parents=True, exist_ok=True)
    comparison_image(source, positions["firestorm"], positions["lodestar"]).save(args.output / "comparison.png")
    reports = {}
    for name, mesh in positions.items():
        rendering = {"include_image": True, "cull_backfaces": True, "opposite_views": True}
        distance = compare_gate_shape(source, mesh, resolution=args.resolution, frame_positions=frame,
                                      **rendering)
        detail = compare_gate_shape(source, mesh, resolution=args.closeup, **rendering)
        distance.image.save(args.output / f"{name}-at-lowest.png")
        detail.image.save(args.output / f"{name}-closeup.png")
        reports[name] = {
            "file": str(inputs[name]),
            "triangles": counts[name],
            "max_bounds_change_m": float(np.abs(bounds(mesh) - bounds(source)).max()),
            "zero_area_before_or_after_packing": int(visual_degenerate_mask(mesh).sum()),
            "lowest_view": distance.metrics,
            "closeup": detail.metrics,
        }
    lowest = reports["lodestar"]["lowest_view"]
    aperture = lowest["front_apertures"]
    checks = {
        "matched_triangle_count": True,
        "lodestar_silhouette_within_one_pixel": lowest["silhouette_pixels_outside_one_pixel_envelope"] == 0,
        "lodestar_guarded_front_openings_preserved": (aperture["guarded_open_pixels"] > 0 and
                                                     aperture["newly_blocked_guarded_pixels"] == 0 and
                                                     aperture["sealed_opening_count"] == 0),
        "lodestar_nondegenerate": reports["lodestar"]["zero_area_before_or_after_packing"] == 0,
    }
    report = {"source_triangles": len(source), "compared_triangles": counts["lodestar"],
              "scene_frame": str(args.scene_frame), "stock_provenance": args.stock_provenance,
              "checks": checks, "results": reports}
    path = args.output / "scorecard.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("high", "stock", "candidate", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--scene-frame", type=Path, required=True)
    parser.add_argument("--stock-provenance", required=True, help="Stock version and how its geometry was obtained")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--resolution", type=int, default=96, help="Full TestHall viewport dimension at Lowest")
    parser.add_argument("--closeup", type=int, default=192, help="Gate-only diagnostic viewport dimension")
    try:
        run(parser.parse_args())
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
