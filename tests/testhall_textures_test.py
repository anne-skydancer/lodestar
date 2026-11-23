import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from lodestar_section_source import face_fingerprint
from lodestar_testhall_textures import TexturedScene, _surface_match, compose_textured_scene


class TestHallTexturesTest(unittest.TestCase):
    def test_matching_retries_only_with_zero_area_faces_removed(self):
        triangle = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        zero_area = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=np.float32)
        uv = triangle[:, :, :2]
        native = face_fingerprint(triangle, uv)
        both = face_fingerprint(np.concatenate((triangle, zero_area)), np.concatenate((uv, uv)))
        groups = {1: [np.array([12])], 2: []}
        self.assertEqual(_surface_match(both, 2, groups, {12: native}, "Trim", "Wing2"), 12)

        nonzero = zero_area.copy()
        nonzero[0, 2, 1] = np.float32(1e-7)
        both = face_fingerprint(np.concatenate((triangle, nonzero)), np.concatenate((uv, uv)))
        with self.assertRaisesRegex(ValueError, "Cannot uniquely map"):
            _surface_match(both, 2, groups, {12: native}, "Trim", "Wing2")

    def test_composition_retains_native_indices_and_deduplicates_texture_bindings(self):
        positions = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                              [[1, 0, 0], [1, 1, 0], [0, 1, 0]]])
        uv = positions[:, :, :2].copy()
        normals = np.broadcast_to([0, 0, 1], positions.shape)
        corners = np.array([[7, 9, 11], [9, 12, 11]])
        original_image = Image.new("RGB", (2, 2), "blue")
        original = TexturedScene(positions, uv, normals, np.array([4, 4]), {4: (original_image, True)}, corners)
        changed = (positions[1:], uv[1:], normals[1:], np.array([1]), {1: (original_image.copy(), True)})
        result = compose_textured_scene(original, [(np.array([1]), changed)])
        np.testing.assert_array_equal(result.positions, positions)
        np.testing.assert_array_equal(result.corner_indices[0], corners[0])
        self.assertGreater(int(result.corner_indices[1].min()), int(corners.max()))
        self.assertEqual(len(result.textures), 1)
        self.assertEqual(result.materials[0], result.materials[1])
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            compose_textured_scene(original, [(np.array([1]), changed), (np.array([1]), changed)])


if __name__ == "__main__":
    unittest.main()
