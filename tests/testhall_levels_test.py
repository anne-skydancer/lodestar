from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_nonplanar import NonplanarResult
from lodestar_testhall_levels import _degenerate_triangles, _extrema_pins, _oriented_projection_distance, _silhouette


class TestHallLevelsTest(unittest.TestCase):
    def test_axis_bounds_are_pinned_without_locking_an_entire_plane(self):
        positions = np.array([[[0, 0, 0], [4, 0, 0], [2, 1, 0]], [[0, 3, 1], [0, 4, 0], [4, 4, 1]]])
        expected = np.stack((positions.min(axis=(0, 1)), positions.max(axis=(0, 1))))
        pins = np.array(list(_extrema_pins(positions)))
        np.testing.assert_array_equal(np.stack((pins.min(axis=0), pins.max(axis=0))), expected)
        self.assertNotIn((2, 1, 0), _extrema_pins(positions))

    def test_silhouette_guard_rejects_a_missing_opening_beyond_the_pixel_envelope(self):
        source = np.ones((1, 9, 9), dtype=bool)
        source[:, 2:7, 2:7] = False
        candidate = source.copy()
        candidate[:, 2:7, 2] = True
        self.assertEqual(_silhouette(source, candidate)["pixels_outside_one_pixel_envelope"], 0)
        candidate[:, 2:7, 3] = True
        self.assertEqual(_silhouette(source, candidate)["pixels_outside_one_pixel_envelope"], 3)

    def test_degeneracy_guard_keeps_real_faces_and_counts_collinear_faces(self):
        positions = np.array([[[0, 0, 0], [2, 2, 0], [1, 1, 0]], [[0, 0, 0], [2, 2, 0], [1, 2, 0]]])
        self.assertEqual(_degenerate_triangles(positions), 1)
        self.assertEqual(_degenerate_triangles(positions[1:]), 0)

    def test_packed_mesh_precision_rejects_a_subquantum_face(self):
        positions = np.array([[[0, 0, 0], [1e-6, 0, 0], [0, 1e-6, 0]],
                              [[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
        self.assertEqual(_degenerate_triangles(positions), 1)

    def test_texture_projection_requires_a_matching_surface_orientation(self):
        triangle = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        label = np.array([0])
        forward = NonplanarResult(triangle, label, label, [])
        reversed_face = NonplanarResult(triangle[:, ::-1], label, label, [])
        self.assertEqual(_oriented_projection_distance(triangle, forward, 0.1), 0.0)
        self.assertEqual(_oriented_projection_distance(triangle, reversed_face, 0.1), np.inf)


if __name__ == "__main__":
    unittest.main()
