from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_collision_probes import sweep_clearance


def wall_panel(left, right, bottom, top):
    quad = np.array([[left, bottom, 0], [right, bottom, 0], [right, top, 0], [left, top, 0]])
    return quad[[[0, 1, 2], [0, 2, 3]]]


class CollisionProbeTest(unittest.TestCase):
    def test_capsule_passes_opening_and_hits_wall(self):
        mesh = np.concatenate((
            wall_panel(-2, -0.8, 0, 3),
            wall_panel(0.8, 2, 0, 3),
            wall_panel(-0.8, 0.8, 2.6, 3),
        ))
        self.assertAlmostEqual(sweep_clearance(mesh, [0, 0, -1], [0, 0, 1], 0.25, 1.8), 0.55)
        self.assertAlmostEqual(sweep_clearance(mesh, [1.2, 0, -1], [1.2, 0, 1], 0.25, 1.8), -0.25)

    def test_vertical_and_stationary_probes_find_floor(self):
        mesh = np.array([[[-3, 0, -2], [3, 0, -2], [0, 0, 4]]])
        self.assertAlmostEqual(sweep_clearance(mesh, [0, 1, 0], [0, 0.3, 0], 0.25, 1.8), 0.3)
        self.assertAlmostEqual(sweep_clearance(mesh, [0, 1, 0], [0, -0.5, 0], 0.25, 1.8), -0.25)
        self.assertAlmostEqual(sweep_clearance(mesh, [0, 0, 0], [0, 0, 0], 0.25, 1.8), 0)

    def test_oblique_edge_crosses_interior_of_swept_axis(self):
        mesh = np.array([[[0, 0.6, -1], [0.2, 0.6, 1], [0, 1.1, 1]]])
        self.assertAlmostEqual(sweep_clearance(mesh, [-2, 0, 0], [2, 0, 0], 0.2, 2.0), -0.2)

    def test_spherical_capsule_sweeps_as_segment(self):
        mesh = wall_panel(-1, 1, 0, 2)
        self.assertAlmostEqual(sweep_clearance(mesh, [0, 0, -1], [0, 0, 1], 0.25, 0.5), -0.25)

    def test_closest_points_can_be_inside_both_edges(self):
        mesh = np.array([[[0, -1, -1], [0, 1, -1], [0, 1, -1]]])
        self.assertAlmostEqual(sweep_clearance(mesh, [-1, 0, -2], [1, 0, -2], 0.25, 0.5), 0.75)


if __name__ == "__main__":
    unittest.main()
