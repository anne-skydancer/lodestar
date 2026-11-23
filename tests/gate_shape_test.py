import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_gate_shape import compare_gate_shape


def rectangle(left, right, bottom, top, depth=0.0):
    corners = np.array([(left, bottom, depth), (right, bottom, depth), (right, top, depth), (left, top, depth)])
    return corners[[[0, 1, 2], [0, 2, 3]]]


def open_arch():
    return np.concatenate((rectangle(-2, -1, 0, 3), rectangle(1, 2, 0, 3), rectangle(-1, 1, 2, 3)))


class GateShapeTest(unittest.TestCase):
    def test_open_bottom_is_protected_against_filling(self):
        source = open_arch()
        metrics = compare_gate_shape(source, rectangle(-2, 2, 0, 3), resolution=40, front_axis=2).metrics
        report = metrics["front_apertures"]
        self.assertEqual(len(metrics["views"]), 13)
        self.assertEqual(report["major_aperture_count"], 1)
        self.assertEqual(report["apertures"][0]["kind"], "open_between_sides")
        self.assertGreater(report["guarded_open_pixels"], 0)
        self.assertEqual(report["newly_blocked_guarded_pixels"], report["guarded_open_pixels"])

    def test_merging_holes_by_removing_a_bar_preserves_negative_space(self):
        rim = np.concatenate((open_arch(), rectangle(-1, 1, 0, 0.25)))
        source = np.concatenate((rim, rectangle(-0.08, 0.08, 0.25, 2)))
        report = compare_gate_shape(source, rim, resolution=64, front_axis=2).metrics["front_apertures"]
        self.assertEqual(report["major_aperture_count"], 2)
        self.assertEqual({aperture["kind"] for aperture in report["apertures"]}, {"enclosed"})
        self.assertEqual(report["guarded_retention"], 1.0)

    def test_closing_the_bottom_is_detected_when_the_middle_stays_clear(self):
        source = open_arch()
        sealed = np.concatenate((source, rectangle(-1, 1, -0.1, 0.2)))
        report = compare_gate_shape(source, sealed, resolution=40, front_axis=2).metrics["front_apertures"]
        self.assertEqual(report["sealed_opening_count"], 1)
        self.assertGreater(report["guarded_retention"], 0.75)

    def test_depth_detects_a_flattened_plane_with_a_matching_front_outline(self):
        source = np.concatenate((rectangle(-2, 0, 0, 3, 0.6), rectangle(0, 2, 0, 3)))
        candidate = rectangle(-2, 2, 0, 3, 0.6)
        metrics = compare_gate_shape(source, candidate, resolution=64, front_axis=2).metrics
        front = next(v for v in metrics["views"] if v["direction"] == [0, 0, 1])
        self.assertEqual(front["silhouette_iou"], 1)
        self.assertAlmostEqual(front["interior_visible_depth"]["p95_error_px"], 8.64)
        self.assertGreater(front["interior_visible_depth"]["over_one_pixel"], 0)

    def test_fixed_scene_frame_and_supersampling_keep_depth_in_screen_pixels(self):
        source, candidate = rectangle(-2, 2, 0, 3), rectangle(-2, 2, 0, 3, 0.5)
        frame = rectangle(-8, 8, -4, 8)
        for supersample in (1, 4):
            metrics = compare_gate_shape(source, candidate, resolution=64, frame_positions=frame,
                                         front_axis=2, supersample=supersample).metrics
            front = next(v for v in metrics["views"] if v["direction"] == [0, 0, 1])
            self.assertAlmostEqual(front["interior_visible_depth"]["p95_error_px"], 1.8)

    def test_opaque_culling_detects_reversed_winding_on_both_sides(self):
        source = rectangle(-2, 2, 0, 3)
        metrics = compare_gate_shape(source, source[:, [0, 2, 1]], resolution=48, front_axis=2,
                                     cull_backfaces=True, opposite_views=True).metrics
        views = {tuple(view["direction"]): view for view in metrics["views"]}
        self.assertEqual(len(views), 26)
        self.assertGreater(views[(0, 0, 1)]["reference_pixels"], 0)
        self.assertEqual(views[(0, 0, 1)]["candidate_pixels"], 0)
        self.assertEqual(views[(0, 0, -1)]["reference_pixels"], 0)
        self.assertGreater(views[(0, 0, -1)]["candidate_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
