import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_surface_fit import charts, correspondence_bound, fit_chart, patch_controls
from bake_planar_patch import boundary_loop
from lodestar_patch_layout import triangulate_boundary
from lodestar_testhall_textures import TexturedScene
from fit_mesh_lods import plan_levels


def grid_scene(size=5, bend=0.3):
    xy = np.array([(x, y) for y in np.linspace(0, 1, size) for x in np.linspace(0, 1, size)])
    positions = np.column_stack((xy, bend * xy[:, 0] ** 2)).astype(np.float32)
    normals = np.column_stack((-2 * bend * xy[:, 0], np.zeros(len(xy)), np.ones(len(xy))))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    triangles = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            triangles.extend(((a, a + 1, a + size + 1), (a, a + size + 1, a + size)))
    triangles = np.array(triangles)
    return SimpleNamespace(positions=positions[triangles], normals=normals[triangles], uvs=xy[triangles],
                           corner_indices=triangles, materials=np.zeros(len(triangles), dtype=int),
                           sections={"panel": SimpleNamespace(face_indices=np.arange(len(triangles)))})


class SurfaceFitTest(unittest.TestCase):
    def test_packing_rejection_restores_chart_state_for_following_levels(self):
        data = grid_scene(bend=0)
        scene = TexturedScene(data.positions, data.uvs, data.normals, data.materials,
                              {0: (Image.new("RGB", (1, 1), "white"), True)}, data.corner_indices, data.sections)
        chart = fit_chart(scene, np.arange(len(scene.positions)))
        boundary = np.array(boundary_loop(chart.faces))
        candidate = boundary[triangulate_boundary(chart.uv[boundary])].astype(np.uint32)
        coefficients = np.zeros((10, 3))
        coefficients[1, 0] = coefficients[2, 1] = 1
        calculator = SimpleNamespace(calculate=lambda *args: {"converged": 1, "suggested_triangles": len(candidate)},
                                     fit=lambda *args: (coefficients.copy(), 1), patch=patch_controls)
        def packed_mask(positions):
            return np.full(len(positions), len(positions) < len(scene.positions))
        with patch("fit_mesh_lods.SurfaceCalculator", return_value=calculator), \
                patch("fit_mesh_lods.visual_degenerate_mask", side_effect=packed_mask):
            levels, report = plan_levels(scene, "unused-library")
        for name, parts in levels.items():
            self.assertEqual(len(parts["panel"].positions), len(scene.positions))
            if name != "high":
                decision = report["levels"][name]["decisions"][0]
                self.assertTrue(decision["retained_for_packing"])
                self.assertFalse(decision["changed"])
                self.assertEqual(decision["triangles"], len(scene.positions))
                self.assertEqual(decision["surface_bound"], 0)

    def test_real_triangle_stream_fits_a_quadratic_without_changing_high(self):
        scene = grid_scene()
        original = scene.positions.copy()
        groups = charts(scene)
        self.assertEqual(len(groups), 1)
        fit = fit_chart(scene, groups[0])
        self.assertEqual(fit.report["degree"], 2)
        np.testing.assert_array_equal(scene.positions, original)
        for patch in fit.controls:
            # Cubic Bernstein corners interpolate the fitted reference graph.
            corners = patch[:3]
            np.testing.assert_allclose(corners[:, 2], 0.3 * corners[:, 0] ** 2, atol=1e-7)
        self.assertGreater(fit.report["model_fit_deviation_bound"], 0)

    def test_material_and_indexed_uv_seams_split_fitting_charts(self):
        scene = grid_scene()
        scene.materials[16:] = 1
        self.assertEqual(len(charts(scene)), 2)
        scene.materials[:] = 0
        scene.corner_indices[16:] += 100
        scene.uvs[16:] += [3, 0]
        self.assertEqual(len(charts(scene)), 2)

    def test_folded_uvs_are_retained_as_unsupported(self):
        scene = grid_scene()
        scene.uvs[0, [1, 2]] = scene.uvs[0, [2, 1]]
        with self.assertRaisesRegex(ValueError, "fold|collapsed|manifold disk"):
            fit_chart(scene, np.arange(len(scene.positions)))

    def test_shading_splits_do_not_split_uv_continuous_geometry(self):
        scene = grid_scene(bend=0)
        scene.corner_indices = np.arange(scene.positions.size // 3).reshape(-1, 3)
        scene.normals[::2] = [0.01, 0., 1.]
        self.assertEqual(len(charts(scene)), 1)
        chart = fit_chart(scene, charts(scene)[0])
        np.testing.assert_array_equal(chart.normals_for(chart.faces), scene.normals)
        # A reference shading discontinuity cannot be hidden by vertex welding.
        averaged = np.broadcast_to([0., 0., 1.], scene.normals.shape)
        bound = correspondence_bound(chart.uv, chart.positions, chart.normals, chart.faces, chart.faces,
            source_normals=chart.corner_normals, target_normals=averaged)
        self.assertGreater(bound['normal_angle_bound_degrees'], .5)

    def test_overlay_bounds_the_entire_surface_and_shading_fields(self):
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]])
        positions = np.column_stack((uv, [0, 0, 0, 0, .1]))
        normals = np.tile([0., 0., 1.], (5, 1))
        normals[4] = [.1, 0, .995]
        source = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        target = np.array([[0, 1, 2], [0, 2, 3]])
        bound = correspondence_bound(uv, positions, normals, source, target)
        self.assertAlmostEqual(bound["surface_bound"], .1, places=8)
        self.assertGreater(bound["normal_angle_bound_degrees"], 5)
        with self.assertRaisesRegex(ValueError, "complete source UV"):
            correspondence_bound(uv, positions, normals, source, target[:1])

    def test_retriangulation_preserves_an_affine_uv_parameterized_surface(self):
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        positions = np.column_stack((uv * [2, 3], uv[:, 0] * .5))
        normals = np.tile([-.5, 0., 1.], (4, 1))
        before = np.array([[0, 1, 2], [0, 2, 3]])
        after = np.array([[0, 1, 3], [1, 2, 3]])
        bound = correspondence_bound(uv, positions, normals, before, after)
        self.assertLess(bound["surface_bound"], 1e-8)
        self.assertLess(bound["normal_angle_bound_degrees"], 1e-8)

    def test_local_normal_bounds_allow_a_chart_wrapping_a_complete_cylinder(self):
        uv = np.array([(u, v) for v in (0., 1.) for u in np.linspace(0, 1, 13)])
        angle = uv[:, 0] * 2 * np.pi
        positions = np.column_stack((np.cos(angle), uv[:, 1], np.sin(angle)))
        normals = np.column_stack((np.cos(angle), np.zeros(len(uv)), np.sin(angle)))
        faces = np.array([triangle for x in range(12)
                          for triangle in ((x, x + 1, x + 14), (x, x + 14, x + 13))])
        bound = correspondence_bound(uv, positions, normals, faces, faces)
        self.assertLess(bound["surface_bound"], 1e-8)
        self.assertLess(bound["normal_angle_bound_degrees"], 1e-8)


if __name__ == "__main__":
    unittest.main()
