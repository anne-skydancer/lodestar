import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_section_source import face_fingerprint, matching_fingerprints


class SectionSourceTest(unittest.TestCase):
    def setUp(self):
        self.faces = np.array([[[1, 0, 0], [2, 0, 0], [1, 1, 0]],
                               [[4, 0, 0], [5, 0, 0], [4, 1, 0]]], dtype=np.float32)
        self.uv = np.array([[[0, 0], [1, 0], [0, 1]]] * 2, dtype=np.float32)

    def test_opt_in_position_ulp_matches_cyclic_order_only(self):
        expected = face_fingerprint(self.faces, self.uv)
        shifted = self.faces.copy()
        shifted[0, 0, 0] = np.nextafter(shifted[0, 0, 0], np.float32(np.inf))
        imported = face_fingerprint(shifted[::-1, [1, 2, 0]], self.uv[::-1, [1, 2, 0]])
        self.assertFalse(matching_fingerprints(expected, imported))
        self.assertTrue(matching_fingerprints(expected, imported, allow_position_ulp=True))
        reversed_winding = face_fingerprint(shifted[:, [0, 2, 1]], self.uv[:, [0, 2, 1]])
        self.assertFalse(matching_fingerprints(expected, reversed_winding, allow_position_ulp=True))

    def test_opt_in_position_ulp_refuses_ambiguous_assignments(self):
        shifted = np.repeat(self.faces[:1], 2, axis=0)
        expected_positions = shifted.copy()
        expected_positions[1, 0, 0] = np.nextafter(expected_positions[1, 0, 0], np.float32(np.inf))
        expected = face_fingerprint(expected_positions, self.uv)
        imported = face_fingerprint(shifted, self.uv)
        self.assertFalse(matching_fingerprints(expected, imported, allow_position_ulp=True))


if __name__ == "__main__":
    unittest.main()
