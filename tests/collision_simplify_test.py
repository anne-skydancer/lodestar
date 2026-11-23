import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from lodestar_collision_simplify import simplify_collision, simplify_collision_parts
from lodestar_nonplanar import _topology
from lodestar_firestorm_degenerate import physics_degenerate_mask, packed_physics_degenerate_mask


def panel(offset=0.):
    points = np.array([(x + offset, 0., z) for z in np.linspace(0, 1, 5) for x in np.linspace(0, 1, 5)])
    faces = np.array([face for z in range(4) for x in range(4) for a in [z * 5 + x]
                      for face in ((a, a + 1, a + 6), (a, a + 6, a + 5))])
    return points[faces]


class CollisionSimplifyTest(unittest.TestCase):
    def test_zero_displacement_retains_the_complete_source(self):
        source = panel()
        result, report = simplify_collision(source, 0)
        np.testing.assert_array_equal(source, result)
        self.assertEqual(report['accepted_collapses'], 0)

    def test_quotient_reduces_a_dense_floor_without_moving_its_support_height(self):
        source = panel()
        result, report = simplify_collision(source, .4)
        self.assertLess(len(result), len(source))
        self.assertTrue(np.all(result[:, :, 1] == 0))
        self.assertLessEqual(report['surface_correspondence_bound'], .4)
        self.assertFalse((physics_degenerate_mask(result) | packed_physics_degenerate_mask(result)).any())
        def topology(triangles):
            points, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
            return _topology(points, indices.reshape(-1, 3))[:2]
        self.assertEqual(topology(source), topology(result))

    def test_shared_model_seam_is_updated_on_both_sides(self):
        result, report = simplify_collision_parts({'left': panel(), 'right': panel(1)}, .4)
        seams = []
        for name in ('left', 'right'):
            points = result[name].reshape(-1, 3)
            seams.append(set(map(tuple, points[points[:, 0] == 1])))
            self.assertLess(len(result[name]), 32)
        self.assertEqual(seams[0], seams[1])
        self.assertLessEqual(report['surface_correspondence_bound'], .4)

    def test_a_single_component_cannot_disappear_to_an_isolated_edge(self):
        source = np.array([[[0., 0, 0], [1, 0, 0], [0, 0, 1]]])
        result, report = simplify_collision(source, 10)
        np.testing.assert_array_equal(result, source)

    def test_a_closed_tetrahedral_component_cannot_collapse_into_a_double_sided_triangle(self):
        outer = np.array([[[0., 0, 0], [10, 0, 0], [10, 0, 10]],
                          [[0., 0, 0], [10, 0, 10], [0, 0, 10]]])
        points = np.array([[2., 2, 2], [2.1, 2, 2], [2, 2.1, 2], [2, 2, 2.1]])
        tetra = points[np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])]
        source = np.concatenate((outer, tetra))
        result, report = simplify_collision(source, 1.)
        inner = result[np.all(result[:, :, 1] > 1, axis=1)]
        self.assertEqual(len(inner), 4)
        self.assertEqual(len({tuple(sorted(map(tuple, face))) for face in inner}), 4)
        self.assertEqual(report['closed_shells_checked'], 1)
        self.assertGreater(report['minimum_closed_shell_volume_ratio'], 0)


if __name__ == '__main__':
    unittest.main()
