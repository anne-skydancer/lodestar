import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_firestorm_collision import _exact_coplanar, _flip_edges
from lodestar_physics_rebuild import exact_planar_repair


class PhysicsRebuildTest(unittest.TestCase):
    def panel(self):
        points = np.array([(x, y, 0.) for y in np.linspace(0, 1, 5) for x in np.linspace(0, 1, 5)])
        faces = []
        for row in range(4):
            for column in range(4):
                a = row * 5 + column
                faces.extend(((a, a + 1, a + 6), (a, a + 6, a + 5)))
        return points[np.array(faces)]

    def test_exact_physics_retriangulation_removes_redundant_interior_faces(self):
        source = self.panel()
        result, count = exact_planar_repair(source)
        self.assertEqual(count, 1)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.all(result[:, :, 2] == 0))
        area = np.linalg.norm(np.cross(result[:, 1] - result[:, 0], result[:, 2] - result[:, 0]), axis=1).sum() / 2
        self.assertEqual(area, 1)

    def test_physics_retriangulation_retains_cross_part_contact_knots(self):
        contact = (.5, 0., 0.)
        result, _ = exact_planar_repair(self.panel(), (contact,))
        self.assertIn(contact, set(map(tuple, result.reshape(-1, 3))))

    def test_exact_physics_retriangulation_preserves_a_small_intentional_bulge(self):
        source = self.panel()
        middle = np.all(source[:, :, :2] == [.5, .5], axis=2)
        source[middle, 2] = 1e-14
        result, count = exact_planar_repair(source)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(result, source)

    def test_exact_coplanarity_rejects_a_small_real_warp(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        self.assertTrue(_exact_coplanar(*points))
        points[3, 2] = np.nextafter(0.0, 1.0)
        self.assertFalse(_exact_coplanar(*points))

    def test_exact_repair_changes_only_an_internal_planar_diagonal(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        def rejected(indices):
            return np.array([0 in triangle and 2 in triangle for triangle in indices])
        original = points.copy()
        self.assertEqual(_flip_edges(points, faces, rejected, exact=True), 1)
        self.assertFalse(rejected(faces).any())
        np.testing.assert_array_equal(points, original)
        triangles = points[faces]
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2
        self.assertEqual(area, 1)

    def test_exact_repair_retains_a_nonplanar_source_surface(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1e-7]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        original = faces.copy()
        def rejected(indices):
            return np.array([0 in triangle and 2 in triangle for triangle in indices])
        self.assertEqual(_flip_edges(points, faces, rejected, exact=True), 0)
        np.testing.assert_array_equal(faces, original)


if __name__ == "__main__":
    unittest.main()
