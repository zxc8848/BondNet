import unittest

import numpy as np

from bondnet.utils.metrics import BondNetMetrics


class MetricsDiagnosticsTests(unittest.TestCase):
    def test_candidate_prevalence_and_confusion_matrix(self):
        metrics = BondNetMetrics()
        metrics.update_connectivity(
            np.array([1, 0, 1, 0, 0]),
            np.array([1, 0, 0, 0, 1]),
        )
        metrics.update_bond_types(
            np.array([0, 1, 1, 3]),
            np.array([0, 0, 1, 3]),
        )

        result = metrics.compute()
        self.assertEqual(result["conn_n_candidates_directed"], 5)
        self.assertEqual(result["conn_n_positive_directed"], 2)
        self.assertEqual(result["conn_n_negative_directed"], 3)
        self.assertAlmostEqual(result["conn_positive_prevalence"], 0.4)
        self.assertAlmostEqual(result["conn_negative_to_positive_ratio"], 1.5)
        self.assertEqual(
            result["bond_type_support"],
            {"single": 2, "double": 1, "triple": 0, "aromatic": 1},
        )
        self.assertEqual(
            result["bond_type_confusion_matrix"],
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
        )


if __name__ == "__main__":
    unittest.main()
