import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from lodestar_physics_select import prepare_physics


class PhysicsSelectTest(unittest.TestCase):
    def inputs(self):
        triangle = np.array([[[0., 0, 0], [1, 0, 0], [0, 0, 1]]])
        levels = {level: {'part': SimpleNamespace(positions=triangle * scale)}
                  for level, scale in (('high', 4), ('medium', 3), ('low', 2), ('lowest', 1))}
        report = {'levels': {level: {'world_units_per_pixel': .01, 'maximum_pixel_error': .5,
                   'decisions': [{'surface_bound': .002}]} for level in levels}}
        return levels, report

    def result(self, source, library, ready=True):
        return source, {'parts': {'part': {'unresolved_triangle_indices': [] if ready else [0]}},
                        'all_parts_have_supported_representation': ready,
                        'maximum_surface_change_before_packing': 0.0}

    def test_valid_lowest_representation_is_used_instead_of_resetting_to_high(self):
        levels, report = self.inputs()
        with patch('lodestar_physics_select.rebuild_physics', side_effect=self.result):
            parts, result = prepare_physics(levels, report, 'unused')
        np.testing.assert_array_equal(parts['part'], levels['lowest']['part'].positions)
        self.assertEqual(result['source_level'], 'lowest')
        self.assertEqual(result['maximum_surface_change_before_packing'], .002)
        self.assertFalse(result['exact_surface_preservation'])

    def test_blocked_candidates_are_never_relabelled_valid(self):
        levels, report = self.inputs()
        with patch('lodestar_physics_select.rebuild_physics', side_effect=lambda s, l: self.result(s, l, False)):
            _, result = prepare_physics(levels, report, 'unused')
        self.assertFalse(result['all_parts_have_supported_representation'])
        self.assertEqual(len(result['selection_attempts']), 20)

    def test_low_hulls_take_precedence_over_a_lowest_triangle_fallback(self):
        levels, report = self.inputs()
        def rebuild(source, library):
            parts, status = self.result(source, library)
            status['parts']['part']['convex_conversion'] = {'supported': float(source['part'].max()) == 2.}
            return parts, status
        with patch('lodestar_physics_select.rebuild_physics', side_effect=rebuild):
            _, result = prepare_physics(levels, report, 'unused')
        self.assertEqual(result['source_level'], 'low')
        self.assertEqual(result['selected_representation'], 'native_hulls')


if __name__ == '__main__':
    unittest.main()
