import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_uv_regions import plan_uv_regions


def section_from_faces(points, indices, materials):
    positions = np.asarray(points, dtype=np.float64)[indices]
    a, b, c = positions.transpose((1, 0, 2))
    normals = np.cross(b - a, c - a)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return SimpleNamespace(positions=positions, normals=np.repeat(normals[:, None], 3, axis=1),
                           triangle_materials=np.asarray(materials))


class UvRegionsTest(unittest.TestCase):
    def test_concave_region_crosses_diffuse_material_boundaries(self):
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0),
                  (1, 1, 0), (2, 1, 0), (0, 2, 0), (1, 2, 0)]
        indices = [(0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4), (3, 4, 7), (3, 7, 6)]
        section = section_from_faces(points, indices, [0, 0, 1, 1, 2, 2])
        candidates, plans = plan_uv_regions(section)
        self.assertEqual(candidates, [tuple(range(6))])
        layout, positions, _, triangles = plans[0]
        self.assertEqual(len(triangles), 4)
        self.assertIn((1, 1, 0), set(map(tuple, positions)))
        self.assertEqual(len(layout.texture_boundary), 8)
        edges = [(tuple(section.positions[face, first]), tuple(section.positions[face, last]))
                 for face, first, last in layout.texture_boundary]
        self.assertEqual([end for _, end in edges], [start for start, _ in edges[1:] + edges[:1]])

    def test_straight_shared_crease_keeps_outside_contacts(self):
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0), (0, 0, 1), (2, 0, 1)]
        indices = [(0, 1, 4), (1, 3, 4), (1, 2, 3), (2, 1, 6), (1, 5, 6), (1, 0, 5)]
        section = section_from_faces(points, indices, [0, 0, 0, 1, 1, 1])
        _, plans = plan_uv_regions(section)
        self.assertEqual(len(plans), 2)
        for _, positions, _, triangles in plans:
            self.assertEqual(len(triangles), 2)
            self.assertNotIn((1, 0, 0), set(map(tuple, positions)))
        _, pinned = plan_uv_regions(section, outside_positions=[(1, 0, 0)])
        self.assertEqual(pinned, [])
        _, targeted = plan_uv_regions(section, target_triangles=5)
        self.assertEqual(len(targeted), 2)
        _, unneeded = plan_uv_regions(section, target_triangles=6)
        self.assertEqual(unneeded, [])

    def test_bounded_mode_preserves_outside_contacts_and_sharp_corners(self):
        points = [(0, 0, 0), (1, 0.0001, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)]
        indices = [(0, 1, 4), (1, 3, 4), (1, 2, 3)]
        section = section_from_faces(points, indices, [0, 0, 0])
        for tolerance in (0.0, 0.00009):
            _, plans = plan_uv_regions(section, max_boundary_deviation=tolerance)
            self.assertEqual(plans, [])
        _, plans = plan_uv_regions(section, max_boundary_deviation=0.0002, target_triangles=2)
        layout, positions, _, triangles = plans[0]
        self.assertEqual(len(triangles), 2)
        self.assertNotIn(points[1], set(map(tuple, positions)))
        self.assertAlmostEqual(layout.boundary_error, 0.0001)
        _, pinned = plan_uv_regions(section, outside_positions=[points[1]], max_boundary_deviation=0.0002)
        self.assertEqual(pinned, [])

        small_corner = [(0, 0, 0), (0.01, -0.01, 0), (0.02, 0, 0), (0.02, 1, 0), (0, 1, 0)]
        section = section_from_faces(small_corner, indices, [0, 0, 0])
        _, plans = plan_uv_regions(section, max_boundary_deviation=0.02)
        self.assertEqual(plans, [])

    def test_bounded_mode_checks_the_complete_replacement_chord(self):
        points = [(0, 0, 0), (1, 0.003, 0), (2, 0.004, 0), (3, 0.003, 0), (4, 0, 0),
                  (4, 1, 0), (0, 1, 0)]
        indices = [(i, i + 1, 6) for i in range(5)]
        section = section_from_faces(points, indices, [0] * len(indices))
        _, plans = plan_uv_regions(section, max_boundary_deviation=0.0011)
        layout, positions, _, triangles = plans[0]
        self.assertGreater(len(triangles), 2)
        self.assertLess(len(triangles), len(indices))
        self.assertIn(points[2], set(map(tuple, positions)))
        self.assertLessEqual(layout.boundary_error, 0.0011)

    def test_wider_chart_fit_needs_normal_and_plane_tolerances(self):
        points = [(0, 0, 0), (1, 0.00001, 0.00003), (2, 0, 0), (2, 1, 0), (0, 1, 0)]
        indices = [(0, 1, 4), (1, 3, 4), (1, 2, 3)]
        section = section_from_faces(points, indices, [0, 0, 0])
        for loose in ({"normal_tolerance": 0.0001}, {"plane_relative_tolerance": 0.0001}):
            _, plans = plan_uv_regions(section, max_boundary_deviation=0.00005, **loose)
            self.assertEqual(plans, [])
        _, plans = plan_uv_regions(section, max_boundary_deviation=0.00005,
                                   normal_tolerance=0.0001, plane_relative_tolerance=0.0001)
        layout, _, _, triangles = plans[0]
        self.assertEqual(len(triangles), 2)
        self.assertGreater(layout.plane_error, 0.000002)
        self.assertGreater(layout.geometric_normal_error, 0.000001)
        self.assertLessEqual(layout.shading_normal_error, 0.0001)


if __name__ == "__main__":
    unittest.main()
