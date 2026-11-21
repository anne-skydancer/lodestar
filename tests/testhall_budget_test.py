from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lodestar_nonplanar import NonplanarResult
from lodestar_testhall_budget import _choose_restorations, plan_budget


class TestHallBudgetTest(unittest.TestCase):
    def test_exact_budget_prefers_greater_quality_recovery(self):
        choices = [("single", 4, 2.0), ("pair-a", 2, 4.0), ("pair-b", 2, 1.0)]
        self.assertEqual(_choose_restorations(choices, 4), {"pair-a", "pair-b"})
        self.assertEqual(_choose_restorations(choices, 3), {"pair-a"})

    def test_budget_retains_real_faces_and_component_labels(self):
        triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        source = np.stack([triangle + [0., 0., index] for index in range(6)])
        section = SimpleNamespace(positions=source, outside_positions={(0., 0., 0.)})
        components = [dict(source_triangles=3, candidate_triangles=1, estimated_relative_error=0.001,
                           sampled_relative_error=deviation) for deviation in (0.001, 0.003)]
        simplified = NonplanarResult(source[[0, 3]], np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1]), components)
        with patch("lodestar_testhall_budget.simplify_nonplanar", return_value=simplified) as simplify:
            result = plan_budget(None, {"hall": section, "gate": section}, {"hall": 6, "gate": 2},
                                 protected_nodes=("gate",), target_triangles=6)
        self.assertEqual(simplify.call_count, 1)
        self.assertTrue(result.report["exact_target"])
        self.assertEqual(set(result.replacements), {"hall"})
        hall = result.replacements["hall"]
        np.testing.assert_array_equal(hall.positions, source[[0, 3, 4, 5]])
        np.testing.assert_array_equal(hall.target_components, [0, 1, 1, 1])
        self.assertTrue(hall.components[1]["retained_for_budget"])


if __name__ == "__main__":
    unittest.main()
