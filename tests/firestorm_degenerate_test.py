import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_firestorm_degenerate import (
    decoded_upload_positions,
    normalize_triangles,
    packed_physics_degenerate_mask,
    physics_degenerate_mask,
    quantized_triangles,
    visual_degenerate_mask,
)


class FirestormDegenerateTest(unittest.TestCase):
    def test_visual_packing_can_collapse_nonzero_geometry(self):
        triangles = [
            [[0, 0, 2], [1, 0, 2], [0, 1, 2]],
            [[0, 0, 2], [0.000001, 0, 2], [0, 0.000001, 2]],
            [[0, 0, 2], [0.25, 0.25, 2], [0.5, 0.5, 2]],
            [[0.5, 0.5, 2], [1, 0.5, 2], [0.5, 1, 2]],
        ]
        packed = quantized_triangles(triangles)
        np.testing.assert_array_equal(packed[1], np.zeros((3, 3)))
        np.testing.assert_array_equal(packed[3, 0], [32767, 32767, 0])
        np.testing.assert_array_equal(visual_degenerate_mask(triangles), [False, True, True, False])

    def test_physics_screen_keeps_triangles_above_both_area_thresholds(self):
        triangles = [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 0.001, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 0.003, 0]],
            [[0.2, 0.2, 0], [0.201, 0.2, 0], [0.2, 0.201, 0]],
            [[0.2, 0.2, 0], [0.205, 0.2, 0], [0.2, 0.205, 0]],
        ]
        expected = [False, True, False, True, False]
        np.testing.assert_array_equal(physics_degenerate_mask(triangles), expected)
        np.testing.assert_array_equal(packed_physics_degenerate_mask(triangles), expected)
        np.testing.assert_array_equal(visual_degenerate_mask(triangles), [False] * len(triangles))

    def test_import_normalization_uses_unit_span_for_nearly_flat_axis(self):
        triangles = [[[1, 4, 0], [3, 4, 0], [1, 4, 0.000002]]]
        model = normalize_triangles(triangles)
        np.testing.assert_array_equal(model[0, :, 0], [-0.5, 0.5, -0.5])
        np.testing.assert_array_equal(model[0, :, 1], [0, 0, 0])
        np.testing.assert_array_equal(model[0, :, 2], np.float32([-0.000001, -0.000001, 0.000001]))

    def test_checks_accept_empty_models_and_reject_nonfinite_positions(self):
        empty = np.empty((0, 3, 3))
        for check in (visual_degenerate_mask, physics_degenerate_mask, packed_physics_degenerate_mask):
            self.assertEqual(check(empty).shape, (0,))
            with self.assertRaisesRegex(ValueError, "finite triangle positions"):
                check(np.full((1, 3, 3), np.nan))

    def test_uploaded_tier_uses_the_high_models_center_and_span(self):
        high = [[[10, 3, -2], [14, 3, -2], [10, 5, -2]]]
        tier = [[[0, 0, 0], [2, 0, 0], [0, 1, 0]]]
        np.testing.assert_array_equal(decoded_upload_positions(tier, high), high)


if __name__ == "__main__":
    unittest.main()
