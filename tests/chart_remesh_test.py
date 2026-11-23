import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from surface_fit_test import grid_scene
from lodestar_surface_fit import fit_chart, generate_chart, correspondence_bound
from lodestar_nonplanar import _topology


class ChartRemeshTest(unittest.TestCase):
    def chart(self):
        scene = grid_scene(bend=0)
        return fit_chart(scene, np.arange(len(scene.positions)))

    def calculate(self, chart, locks):
        calculator = SimpleNamespace(calculate=lambda *args: {"converged": 1, "suggested_triangles": 2})
        return generate_chart(chart, calculator, .01, .5, 1., locks)

    def test_lodestar_reaches_the_planar_disk_bound_and_preserves_all_boundary_segments(self):
        chart = self.chart()
        result, report = self.calculate(chart, np.zeros(len(chart.positions), dtype=np.uint8))
        self.assertEqual(len(result), 14)  # 16 fixed boundary vertices, no interior vertices.
        self.assertEqual(report["remesher"], "lodestar_uv_cavity")
        self.assertEqual(_topology(chart.positions, result), _topology(chart.positions, chart.faces))
        bound = correspondence_bound(chart.uv, chart.positions, chart.normals, chart.faces, result)
        self.assertLess(bound["surface_bound"], 1e-8)
        self.assertEqual(bound["normal_angle_bound_degrees"], 0)
        for array in (chart.positions, chart.normals, chart.source_uv):
            self.assertTrue(np.isfinite(array).all())

    def test_protected_interior_contact_raises_the_minimum_without_moving_it(self):
        chart = self.chart()
        original = chart.positions.copy()
        center = int(np.flatnonzero(np.all(chart.uv == [.5, .5], axis=1))[0])
        locks = np.zeros(len(chart.positions), dtype=np.uint8)
        locks[center] = 1
        result, report = self.calculate(chart, locks)
        self.assertEqual(len(result), 16)
        self.assertIn(center, result)
        np.testing.assert_array_equal(chart.positions, original)


if __name__ == "__main__":
    unittest.main()
