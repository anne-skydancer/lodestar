import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from bake_planar_patch import linear_image, sample
from lodestar_lod_materials import combine_lod_parts, share_lod_materials
from lodestar_testhall_textures import TexturedScene


def texture(width, height, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def scene(uvs, materials, textures):
    count = len(materials)
    positions = np.broadcast_to([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], (count, 3, 3)).copy()
    positions[:, :, 0] += 2 * np.arange(count)[:, None]
    normals = np.broadcast_to([0., 0., 1.], positions.shape)
    return TexturedScene(positions, np.array(uvs), normals, np.array(materials), textures,
                         np.arange(count * 3).reshape(count, 3))


class LodMaterialsTest(unittest.TestCase):
    def assert_preserved(self, before, after):
        np.testing.assert_array_equal(before.positions, after.positions)
        np.testing.assert_array_equal(before.normals, after.normals)
        lattice = np.array([(a, b, 16 - a - b) for a in range(17) for b in range(17 - a)]) / 16
        for index, (original, packed) in enumerate(zip(before.materials, after.materials)):
            expected, repeat = before.textures[original]
            actual, new_repeat = after.textures[packed]
            before_uv = lattice @ before.uvs[index]
            after_uv = lattice @ after.uvs[index]
            np.testing.assert_allclose(sample(linear_image(expected), before_uv, repeat=repeat),
                                       sample(linear_image(actual), after_uv, repeat=new_repeat), atol=1e-12)

    def test_lower_bindings_share_exact_texels_and_use_real_high_faces(self):
        high = scene([[[-0.05, -0.04], [0.03, -0.04], [-0.05, 0.02]],
                      [[1.5, 2.0], [5.1, 2.0], [1.5, 2.2]]], [100, 100], {100: (texture(8, 8, 1), True)})
        low = scene([[[0, 0], [0.4, 0.05], [0.1, 0.7]]], [200], {200: (texture(16, 16, 2), False)})
        packed, report = share_lod_materials({"high": {"object": high}, "low": {"object": low}}, max_size=32)
        upper, lower = packed["high"]["object"], packed["low"]["object"]
        self.assertEqual(report["parts"]["object"]["high_atlas_anchor_face"], 0)
        self.assertEqual(upper.materials[0], lower.materials[0])
        self.assertNotEqual(upper.materials[1], lower.materials[0])
        self.assertIs(upper.textures[upper.materials[0]][0], lower.textures[lower.materials[0]][0])
        self.assert_preserved(high, upper)
        self.assert_preserved(low, lower)
        self.assertEqual(report["unique_images"], 2)
        self.assertEqual(report["base_rgba8_bytes"], sum(image.width * image.height * 4
                                                       for image, _ in upper.textures.values()))

    def test_atlas_crop_repackages_tall_image_and_retains_long_repeats(self):
        wide = [[2.0, 0.875], [12.1, 0.875], [2.0, 1.125]]
        compact = [[-0.2, 1.98], [-0.03, 1.98], [-0.2, 2.12]]
        high = scene([wide, compact], [10, 11], {10: (texture(16, 16, 3), True), 11: (texture(16, 16, 4), True)})
        charts = [[[0, -0.001], [0.8, 0], [0.2, 0.012]], [[0, 0.99], [0.6, 0.992], [0.2, 1.011]]]
        low = scene(charts, [20, 20], {20: (texture(8, 300, 5), False)})
        levels = {"high": {"first": high, "second": high}, "low": {"first": low, "second": low}}
        packed, report = share_lod_materials(levels, max_size=32, material_limit=2)
        first = packed["high"]["first"]
        self.assertEqual(report["parts"]["first"]["original_repeat_bindings"], 1)
        self.assertIsNone(report["parts"]["first"]["high_atlas_anchor_face"])
        self.assertTrue(first.textures[first.materials[0]][1])
        self.assertFalse(first.textures[first.materials[1]][1])
        for level, before in (("high", high), ("low", low)):
            self.assert_preserved(before, packed[level]["first"])
            self.assert_preserved(before, packed[level]["second"])
        combined = combine_lod_parts(packed["high"])
        self.assertEqual(len(combined.positions), 4)
        self.assertEqual(len(np.unique(combined.materials)), 4)
        self.assertEqual(len(np.unique(combined.corner_indices)), 12)
        self.assertEqual(report["unique_images"], 2)

    def test_different_part_layouts_and_impossible_density_fail(self):
        uv = [[[0., 0.], [1., 0.], [0., 1.]]]
        high = scene(uv, [7], {7: (texture(64, 64, 6), False)})
        with self.assertRaisesRegex(ValueError, "same named parts"):
            share_lod_materials({"high": {"first": high}, "low": {"second": high}}, max_size=32)
        with self.assertRaisesRegex(ValueError, "original density"):
            share_lod_materials({"high": {"first": high}}, max_size=32)

    def test_combining_local_atlases_keeps_distinct_textures(self):
        uv = [[[0., 0.], [1., 0.], [0., 1.]]]
        first = scene(uv, [7], {7: (texture(4, 4, 7), False)})
        second = scene(uv, [7], {7: (texture(4, 4, 8), False)})
        combined = combine_lod_parts({"first": first, "second": second})
        self.assertEqual(len(np.unique(combined.materials)), 2)
        self.assertIs(combined.textures[combined.materials[0]][0], first.textures[7][0])
        self.assertIs(combined.textures[combined.materials[1]][0], second.textures[7][0])


if __name__ == "__main__":
    unittest.main()
