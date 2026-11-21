import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bake_planar_patch import linear_image, sample
from lodestar_nonplanar_bake import bake_nonplanar


def section(triangles, materials):
    triangles = np.asarray(triangles, dtype=float)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return SimpleNamespace(positions=triangles, normals=np.broadcast_to(normals[:, None, :], triangles.shape),
                           source_uvs=triangles[:, :, :2].copy(), triangle_materials=np.array(materials))


class NonplanarBakeTest(unittest.TestCase):
    def test_preserves_existing_faces_and_corner_order(self):
        high = section([[[0, 0, 0], [1, 0, 0], [1, 1, 0]]], [3])
        original = Image.new("RGB", (4, 4), "green")
        order = [2, 0, 1]
        low = high.positions[:, order]
        uv, normals, materials, textures, report = bake_nonplanar(high, low, {3: (original, True)},
                                                                max_projection_distance=0.001)
        np.testing.assert_array_equal(uv, high.source_uvs[:, order] * [1, -1] + [0, 1])
        np.testing.assert_array_equal(normals, high.normals[:, order])
        self.assertEqual(materials.tolist(), [3])
        self.assertIs(textures[3][0], original)
        self.assertEqual(report["retained_faces"], 1)
        self.assertEqual(report["baked_faces"], 0)

    def test_nearby_opposite_side_cannot_bleed(self):
        high = section([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                        [[0, 0, .06], [0, 1, .06], [1, 0, .06]]], [3, 4])
        low = high.positions[[0]].copy()
        low[0, :, 2] = [.041, .045, .055]
        textures = {3: (Image.new("RGB", (8, 8), "red"), True),
                    4: (Image.new("RGB", (8, 8), "blue"), True)}
        uv, _, materials, textures, report = bake_nonplanar(high, low, textures, max_projection_distance=.06)
        atlas, repeat = textures[materials[0]]
        positions = np.vstack((uv[0], uv.mean(axis=1)))
        np.testing.assert_array_equal(sample(linear_image(atlas), positions, repeat=repeat),
                                      [[1., 0., 0.]] * 4)
        self.assertGreater(report["maximum_projection_distance"], .05)
        self.assertEqual(report["baked_faces"], 1)
        self.assertEqual(json.loads(json.dumps(report))["projection_samples"], report["projection_samples"])

    def test_projection_disallows_unmatched_samples(self):
        high = section([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], [2])
        low = high.positions.copy()
        low[0, 1, 0] = 1.05
        textures = {2: (Image.new("RGB", (8, 8), "red"), True)}
        with self.assertRaisesRegex(ValueError, "Projection distance"):
            bake_nonplanar(high, low, textures, max_projection_distance=.02)

    def test_component_constraint_isolates_nearby_shell(self):
        high = section([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                        [[0, 0, .05], [1, 0, .05], [0, 1, .05]]], [3, 4])
        low = high.positions[[0]].copy()
        low[0, :, 2] = .04
        textures = {3: (Image.new("RGB", (8, 8), "red"), True),
                    4: (Image.new("RGB", (8, 8), "blue"), True)}
        uv, _, materials, textures, report = bake_nonplanar(high, low, textures, max_projection_distance=.06,
                                                            source_components=[0, 1], target_components=[0])
        atlas, repeat = textures[materials[0]]
        np.testing.assert_array_equal(sample(linear_image(atlas), uv.mean(axis=1), repeat=repeat), [[1., 0., 0.]])
        self.assertAlmostEqual(report["maximum_projection_distance"], .04)

    def test_atlas_can_retire_original_when_all_faces_are_baked(self):
        high = section([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], [3])
        textures = {3: (Image.new("RGB", (8, 8), "red"), True)}
        _, _, labels, output, report = bake_nonplanar(high, high.positions, textures, max_projection_distance=.001,
                                                      retain_exact=False)
        self.assertEqual(labels.tolist(), [4])
        self.assertEqual(set(output), {4})
        self.assertEqual(report["baked_faces"], 1)


if __name__ == "__main__":
    unittest.main()
