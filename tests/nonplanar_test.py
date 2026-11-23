import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_nonplanar import simplify_nonplanar


class NonplanarTest(unittest.TestCase):
    def test_sampled_distance_limit_rejects_topology_preserving_shape_error(self):
        points = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        faces = np.array([(0, 1, 2), (0, 2, 3)])

        def shrink(*_):
            return np.array([(0, 2, 3)], dtype=np.uint32), 0.0

        with patch("lodestar_nonplanar._bind", return_value=shrink):
            unchecked = simplify_nonplanar(None, points[faces], 1)
            checked = simplify_nonplanar(None, points[faces], 1, max_sampled_relative_error=0.01)
        self.assertEqual(len(unchecked.positions), 1)
        np.testing.assert_array_equal(checked.positions, points[faces])

    def test_rejects_filling_a_hole(self):
        points = np.array([(0, 0, 0), (3, 0, 0), (3, 3, 0), (0, 3, 0),
                           (1, 1, 0), (2, 1, 0), (2, 2, 0), (1, 2, 0)])
        faces = np.array([(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                          (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])

        def fill_hole(*_):
            return np.array([(0, 6, 7), (0, 7, 1)], dtype=np.uint32), 0.0

        with patch("lodestar_nonplanar._bind", return_value=fill_hole):
            result = simplify_nonplanar(None, points[faces], 2)
        np.testing.assert_array_equal(result.positions, points[faces])
        self.assertEqual(result.components[0]["boundary_loops"], 2)

    def test_marks_preexisting_nonmanifold_input_for_fallback(self):
        points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)])
        faces = np.array([(0, 1, 2), (1, 0, 3), (0, 1, 4)])

        def unexpected(*_):
            self.fail("Preexisting nonmanifold component reached native simplifier")

        with patch("lodestar_nonplanar._bind", return_value=unexpected):
            result = simplify_nonplanar(None, points[faces], 1)
        np.testing.assert_array_equal(result.positions, points[faces])
        self.assertFalse(result.components[0]["supported"])

    def test_rejects_removal_of_pinned_position(self):
        source = np.array([[(0, 0, 0), (1, 0, 0), (1, 1, 0)],
                           [(0, 0, 0), (1, 1, 0), (0, 1, 0)]])

        def skip_pin(_, faces, pins, *__):
            self.assertEqual(pins.tolist(), [1, 0, 0, 0])
            return np.array([(1, 2, 3)], dtype=faces.dtype), 0.0

        with patch("lodestar_nonplanar._bind", return_value=skip_pin):
            result = simplify_nonplanar(None, source, 1, outside_positions=[(0, 0, 0)])
        np.testing.assert_array_equal(result.positions, source)


if __name__ == "__main__":
    unittest.main()
