import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from lodestar_surface_fit import charts, fit_chart, correspondence_bound
from lodestar_chart_boundaries import coarsen_boundaries
from surface_fit_test import grid_scene


class ChartBoundariesTest(unittest.TestCase):
    def scene(self, seam=False):
        scene = grid_scene(bend=0)
        if seam:
            # Two material charts share the middle horizontal geometric line.
            scene.materials[16:] = 1
        return scene

    def run_remesh(self, scene, pins=()):
        groups = charts(scene)
        fitted = {i: fit_chart(scene, group) for i, group in enumerate(groups)}
        current = {i: chart.faces for i, chart in fitted.items()}
        report = coarsen_boundaries(scene, groups, fitted, current,
            {i: 2 for i in fitted}, .001, 1., set(pins))
        return fitted, current, report

    def test_collinear_uv_boundary_knots_can_be_removed_without_losing_the_domain(self):
        fitted, current, report = self.run_remesh(self.scene())
        self.assertGreater(report['removed_seam_vertices'], 0)
        for index, chart in fitted.items():
            bound = correspondence_bound(chart.uv, chart.positions, chart.normals, chart.faces, current[index],
                source_normals=chart.corner_normals, target_normals=chart.normals_for(current[index]))
            self.assertLess(bound['surface_bound'], 1e-8)

    def test_material_seam_edits_are_coupled_and_contact_pins_remain(self):
        pin = (.5, .5, 0.)
        fitted, current, report = self.run_remesh(self.scene(seam=True), (pin,))
        seams = []
        for index, chart in fitted.items():
            points = chart.positions[np.unique(current[index])]
            seams.append(set(map(tuple, points[points[:, 1] == .5])))
        self.assertEqual(seams[0], seams[1])
        self.assertIn(pin, seams[0])


if __name__ == '__main__':
    unittest.main()
