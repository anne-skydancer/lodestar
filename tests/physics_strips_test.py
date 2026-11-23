import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fit_mesh_lods import write_physics_candidate
from lodestar_firestorm_degenerate import physics_degenerate_mask, packed_physics_degenerate_mask
from lodestar_physics_strips import exact_strip_repair, exact_edge_refinement


def panel(width=.004, neighbor=False):
    a, b, c, d = (0., 0., 0.), (.25, 0., 0.), (.25, width, 0.), (0., width, 0.)
    faces = [(a, b, c), (a, c, d), ((0, 0, 1), (1, 0, 1), (0, 1, 1))]
    if neighbor:
        faces.extend(((b, a, (0, 0, .5)), ((0, width, .5), d, c)))
    return np.array(faces, dtype=np.float32).astype(np.float64)


def perimeter(triangles):
    edges = Counter(tuple(sorted((tuple(a), tuple(b)))) for tri in triangles
                    for a, b in zip(tri, np.roll(tri, -1, axis=0)))
    return {edge for edge, count in edges.items() if count == 1}


class PhysicsStripsTest(unittest.TestCase):
    def test_long_thin_panel_is_refined_without_omission_or_source_movement(self):
        source = panel()
        original = source.copy()
        result, report = exact_strip_repair(source)
        self.assertEqual(report["source_rejections"], 2)
        self.assertEqual(report["remaining_rejections"], 0)
        self.assertGreater(len(result), len(source))
        self.assertEqual(report["omitted_positive_area_triangles"], 0)
        self.assertEqual(report["ideal_surface_change"], 0)
        np.testing.assert_array_equal(source, original)
        self.assertTrue(set(map(tuple, source.reshape(-1, 3))) <= set(map(tuple, result.reshape(-1, 3))))
        plane = result[np.all(result[:, :, 2] == 0, axis=1)]
        cross = np.cross(plane[:, 1] - plane[:, 0], plane[:, 2] - plane[:, 0])
        self.assertTrue(np.all(cross[:, 2] > 0))
        self.assertEqual(cross[:, 2].sum() / 2, .25 * source[0, 2, 1])
        self.assertFalse(physics_degenerate_mask(result).any())
        self.assertFalse(packed_physics_degenerate_mask(result).any())

    def test_refinement_conforms_to_neighbors_across_real_creases(self):
        source = panel(neighbor=True)
        result, report = exact_strip_repair(source)
        self.assertEqual(report["remaining_rejections"], 0)
        self.assertEqual(report["incident_faces_refined"], 2)
        # The two long shared boundaries must not become unmatched T junctions.
        unmatched = perimeter(result)
        for a, b in unmatched:
            self.assertFalse(a[2] == b[2] == 0 and a[1] == b[1] and a[0] != b[0])
        for triangle in result:
            self.assertTrue(np.all(triangle[:, 2] == 0) or np.all(triangle[:, 2] == 1) or
                            np.all(triangle[:, 1] == 0) or np.all(triangle[:, 1] == source[0, 2, 1]))

    def test_small_real_warp_is_not_flattened(self):
        source = panel()
        source[1, 2, 2] = 1e-14
        result, report = exact_strip_repair(source)
        self.assertEqual(report["strip_pairs"], 0)
        np.testing.assert_array_equal(result, source)

    def test_explicit_bounded_mode_reports_a_real_warp_instead_of_calling_it_exact(self):
        source = panel()
        source[1, 2, 2] = np.float32(1e-5)
        result, report = exact_strip_repair(source, maximum_surface_displacement=.001)
        self.assertEqual(report['remaining_rejections'], 0)
        self.assertGreater(report['ideal_surface_change'], 0)
        self.assertLessEqual(report['ideal_surface_change'], .001)
        self.assertTrue(set(map(tuple, source.reshape(-1, 3))) <= set(map(tuple, result.reshape(-1, 3))))

    def test_unrepresentable_thin_panel_is_retained_and_blocks_companion(self):
        source = panel(width=.0005)
        result, report = exact_strip_repair(source)
        self.assertGreater(report["remaining_rejections"], 0)
        np.testing.assert_array_equal(result, source)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = {}
            path = write_physics_candidate(output, {"panel": result}, manifest)
            self.assertEqual(path.name, "Lodestar_unresolved_PHYS.dae")
            self.assertFalse((output / "Lodestar_PHYS.dae").exists())
            self.assertFalse(manifest["triangle_upload_ready"])
            self.assertEqual(manifest["status"], "blocked_unresolved_triangles")

    def test_stale_companion_cannot_masquerade_as_a_repaired_result(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            companion = output / "Lodestar_PHYS.dae"
            companion.write_bytes(b"previous companion")
            with self.assertRaisesRegex(ValueError, "fresh output folder"):
                write_physics_candidate(output, {"panel": panel(width=.0005)}, {})
            self.assertEqual(companion.read_bytes(), b"previous companion")

    def test_single_long_triangle_can_be_refined_without_requiring_a_planar_quad(self):
        source = np.array([[[10.4631795883, 7.9101643562, 7.8593053818],
                            [10.4631795883, 8.0040845871, 7.1924510002],
                            [10.4631891251, 7.9101643562, 7.9310584068]],
                           [[7., 7.5, 7], [15, 7.5, 7], [7, 9.9, 15]]], dtype=np.float32).astype(float)
        result, report = exact_edge_refinement(source)
        self.assertGreater(report['edge_bisections'], 0)
        self.assertEqual(report['remaining_rejections'], 0)
        self.assertEqual(report['ideal_surface_change'], 0)
        self.assertFalse((physics_degenerate_mask(result) | packed_physics_degenerate_mask(result)).any())


if __name__ == "__main__":
    unittest.main()
