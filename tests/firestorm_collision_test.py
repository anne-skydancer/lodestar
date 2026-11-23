import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_firestorm_collision import build_firestorm_collider
from lodestar_firestorm_degenerate import decoded_upload_positions, physics_degenerate_mask
from lodestar_walkability import staircase_fixture, walk_route


class FirestormCollisionTest(unittest.TestCase):
    def test_retriangulates_a_convex_plane_before_omitting_triangles(self):
        a, b, c, d = [0, 0, 0], [1, 0, 0], [0.5, 0.000001, 0], [0.5, -1, 0]
        source = np.array([(a, b, c), (b, a, d)])
        collider, report = build_firestorm_collider(source)
        np.testing.assert_array_equal(collider, [(c, d, b), (d, c, a)])
        self.assertEqual(report["diagonal_flips"], 1)
        self.assertEqual(report["omitted_after_local_repair"], 0)
        self.assertLessEqual(report["source_to_collider_sampled_error_m"], 1e-8)

    def test_omits_redundant_rejection_with_certified_coverage(self):
        source = np.array([
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0.2, 0.2, 0], [0.20005, 0.2, 0], [0.2, 0.20005, 0]],
        ])
        collider, report = build_firestorm_collider(source)
        np.testing.assert_array_equal(collider, source[:1])
        self.assertEqual(report["omitted_after_local_repair"], 1)
        self.assertLessEqual(report["source_to_collider_upper_bound_m"], 1e-8)
        self.assertEqual(report["collider_to_source_upper_bound_m"], 0)
        self.assertFalse(physics_degenerate_mask(collider).any())

    def test_refuses_to_remove_surface_beyond_tolerance(self):
        source = [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0.1], [1, 0, 0.1], [0, 1, 0.1]],
            [[0.2, 0.2, 0.04], [0.20005, 0.2, 0.04], [0.2, 0.20005, 0.04]],
        ]
        with self.assertRaisesRegex(ValueError, "surface error|surface change"):
            build_firestorm_collider(source)

    def test_keeps_step_support_through_upload_encoding(self):
        source, start, route = staircase_fixture()
        candidate, report = build_firestorm_collider(source)
        np.testing.assert_array_equal(candidate, source)
        self.assertEqual(report["omitted_after_local_repair"], 0)
        walked = walk_route(decoded_upload_positions(candidate, source), start, route, radius=0.22, height=1.69)
        self.assertTrue(walked["passed"])


if __name__ == "__main__":
    unittest.main()
