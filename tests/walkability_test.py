from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_walkability import staircase_fixture, walk_route


def horizontal_quad(y, z0, z1):
    corners = np.array([[-0.5, y, z0], [-0.5, y, z1], [0.5, y, z1], [0.5, y, z0]])
    return corners[[[0, 1, 2], [0, 2, 3]]]


class WalkabilityTest(unittest.TestCase):
    def walk(self, triangles, start, waypoints, **limits):
        return walk_route(triangles, start, waypoints, radius=0.22, height=1.69, **limits)

    def test_walks_up_and_down_mesh_treads(self):
        triangles, start, waypoints = staircase_fixture()
        ascent = self.walk(triangles, start, waypoints)
        self.assertTrue(ascent["passed"], ascent)
        self.assertEqual(len(ascent["steps"]), 3)
        np.testing.assert_allclose([step["rise_m"] for step in ascent["steps"]], 0.17)
        self.assertAlmostEqual(ascent["feet_trace_m"][-1][1], 0.515)

        reversed_landings = [feet[::2] for feet in ascent["feet_trace_m"][-2::-1]]
        descent = self.walk(triangles, ascent["feet_trace_m"][-1], reversed_landings)
        self.assertTrue(descent["passed"], descent)
        np.testing.assert_allclose([step["rise_m"] for step in descent["steps"]], -0.17)

    def test_rejects_riser_above_the_step_height_limit(self):
        triangles, start, waypoints = staircase_fixture(rise=0.30)
        report = self.walk(triangles, start, waypoints)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_waypoint"], 0)
        self.assertEqual(report["failure"], "landing_has_no_support_within_step_limits")

    def test_rejects_lintel_while_lifting_to_the_next_step(self):
        triangles, start, waypoints = staircase_fixture()
        lintel = horizontal_quad(1.76, -0.15, 0.29)
        report = self.walk(np.concatenate((triangles, lintel)), start, waypoints)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "lift_is_obstructed")
        self.assertLess(report["steps"][0]["lift_clearance_m"], 0)

    def test_rejects_gap_between_two_supported_landings(self):
        triangles = np.concatenate((horizontal_quad(0, -0.3, 0.1), horizontal_quad(0, 0.29, 0.6)))
        report = self.walk(triangles, [0, 0, 0], [[0, 0.42]])
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "missing_support_between_waypoints")

    def test_tread_requires_support_not_just_open_volume(self):
        report = self.walk(horizontal_quad(0, -0.3, 0.1), [0, 0, 0], [[0, 0.42]])
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure"], "landing_has_no_support_within_step_limits")


if __name__ == "__main__":
    unittest.main()
