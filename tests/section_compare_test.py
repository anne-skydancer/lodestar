from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_section_compare import compare_section


def quad(alternate=False, depth=0):
    positions = np.array([[-1, -1, depth], [1, -1, depth], [1, 1, depth], [-1, 1, depth]], dtype=float)
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=float)
    faces = np.array([[0, 1, 3], [1, 2, 3]] if alternate else [[0, 1, 2], [0, 2, 3]])
    return positions[faces], uv[faces], np.broadcast_to([0.0, 0.0, 1.0], (2, 3, 3)).copy()


class SectionCompareTest(unittest.TestCase):
    def test_matching_surface_survives_different_diagonals(self):
        diffuse = {0: (Image.new("RGB", (1, 1), (128, 200, 80)), False)}
        args = (*quad(), [0, 0], diffuse, *quad(alternate=True), [0, 0], diffuse)
        result = compare_section(*args, resolution=64)
        metrics = result.metrics
        self.assertEqual(len(metrics["views"]), 6)
        self.assertGreater(metrics["compared_pixels"], 0)
        self.assertEqual(metrics["rgb_samples"], 3 * metrics["compared_pixels"])
        self.assertEqual(metrics["silhouette_iou"], 1)
        self.assertEqual(metrics["mean_abs_error"], 0)
        self.assertEqual(metrics["p95_abs_error"], 0)
        self.assertEqual(metrics["max_abs_error"], 0)
        self.assertEqual(metrics["edited_pixels"], 0)
        self.assertIsNone(metrics["edited_mean_abs_error"])
        views = {view["name"]: view for view in metrics["views"]}
        self.assertEqual(views["Left"]["union_pixels"], 0)
        self.assertIsNone(views["Left"]["silhouette_iou"])
        self.assertEqual(result.image.mode, "RGB")
        repeated = compare_section(*args, resolution=64)
        self.assertEqual(repeated.metrics, metrics)
        self.assertEqual(repeated.image.tobytes(), result.image.tobytes())

    def test_depth_hides_the_rear_texture_from_the_front(self):
        far, near = quad(depth=-0.3), quad(depth=0.3)
        source = tuple(np.concatenate((a, b)) for a, b in zip(far, near))
        target = tuple(np.concatenate((a, b)) for a, b in zip(near, far))
        source_textures = {
            1: (Image.new("RGB", (1, 1), "red"), True),
            2: (Image.new("RGB", (1, 1), "green"), False),
        }
        target_textures = {**source_textures, 1: (Image.new("RGB", (1, 1), "blue"), True)}
        result = compare_section(*source, [1, 1, 2, 2], source_textures,
                                 *target, [2, 2, 1, 1], target_textures, resolution=64,
                                 source_edited=[True, True, False, False])
        views = {view["name"]: view for view in result.metrics["views"]}
        self.assertEqual(views["Front"]["mean_abs_error"], 0)
        self.assertEqual(views["Front"]["edited_pixels"], 0)
        self.assertIsNone(views["Front"]["edited_p95_abs_error"])
        self.assertGreater(views["Back"]["mean_abs_error"], 30)
        self.assertEqual(views["Back"]["edited_pixels"], views["Back"]["compared_pixels"])
        self.assertEqual(views["Back"]["edited_mean_abs_error"], views["Back"]["mean_abs_error"])
        self.assertEqual(result.metrics["silhouette_iou"], 1)

    def test_scene_displacement_changes_silhouette_in_the_shared_frame(self):
        source, uv, normals = quad()
        target = source + [0.6, 0, 0]
        diffuse = {0: (Image.new("RGB", (1, 1), "white"), True)}
        result = compare_section(source, uv, normals, [0, 0], diffuse,
                                 target, uv, normals, [0, 0], diffuse, resolution=64, source_edited=[True, True])
        self.assertLess(result.metrics["silhouette_iou"], 1)
        self.assertGreater(result.metrics["silhouette_iou"], 0)
        self.assertEqual(result.metrics["mean_abs_error"], 0)
        self.assertEqual(result.metrics["edited_pixels"], result.metrics["compared_pixels"])
        self.assertLess(result.metrics["edited_pixels"], result.metrics["source_pixels"])

    def test_edited_pixel_errors_are_measured_on_the_union_of_visible_face_masks(self):
        positions, uv, normals = quad()
        panels = np.concatenate((positions - [1.5, 0, 0], positions + [1.5, 0, 0]))
        corners = panels, np.tile(uv, (2, 1, 1)), np.tile(normals, (2, 1, 1)), [0, 0, 1, 1]
        white = Image.new("RGB", (1, 1), "white"), True
        blue = Image.new("RGB", (1, 1), "blue"), True
        args = (*corners, {0: white, 1: white}, *corners, {0: blue, 1: white})
        changed = compare_section(*args, resolution=64, target_edited=[True, True, False, False]).metrics
        self.assertGreater(changed["edited_pixels"], 0)
        self.assertLess(changed["edited_pixels"], changed["compared_pixels"])
        self.assertGreater(changed["edited_mean_abs_error"], changed["mean_abs_error"])
        self.assertEqual(changed["edited_max_abs_error"], changed["max_abs_error"])

        both = compare_section(*args, resolution=64, source_edited=[False, False, True, True],
                               target_edited=[True, True, False, False]).metrics
        self.assertEqual(both["edited_pixels"], both["compared_pixels"])
        self.assertEqual(both["edited_mean_abs_error"], both["mean_abs_error"])
        self.assertEqual(both["edited_p95_abs_error"], both["p95_abs_error"])

    def test_texture_orientation_and_interpolated_normal_matter(self):
        source, uv, normals = quad()
        pattern = np.empty((8, 8, 3), dtype=np.uint8)
        pattern[:4] = (255, 0, 0)
        pattern[4:] = (0, 0, 255)
        diffuse = {0: (Image.fromarray(pattern), False)}
        tilted = normals.copy()
        tilted[:, 0] = [0, 0, -1]
        result = compare_section(source, uv, normals, [0, 0], diffuse,
                                 source, uv, tilted, [0, 0], diffuse, resolution=64)
        preview = result.image
        # The first source tile displays the front view below the headings.
        red = preview.getpixel((40, 57))
        blue = preview.getpixel((40, 91))
        self.assertGreater(red[0], red[2] + 100)
        self.assertGreater(blue[2], blue[0] + 100)
        self.assertEqual(result.metrics["silhouette_iou"], 1)
        self.assertGreater(result.metrics["max_abs_error"], 30)

    def test_missing_material_texture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "material 2 needs a diffuse texture"):
            compare_section(*quad(), [0, 2], {0: (Image.new("RGB", (1, 1)), False)},
                            *quad(), [0, 0], {}, resolution=64)


if __name__ == "__main__":
    unittest.main()
