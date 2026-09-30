import unittest

import numpy as np

from scripts.evaluate_connectivity_thresholds import _score_threshold


class ConnectivityThresholdTests(unittest.TestCase):
    def test_edge_and_molecule_metrics(self):
        probabilities = np.array([0.9, 0.2, 0.8, 0.7, 0.1])
        labels = np.array([True, False, False, True, False])
        molecule_ids = np.array([0, 0, 0, 1, 1])
        result = _score_threshold(
            probabilities,
            labels,
            molecule_ids,
            n_molecules=2,
            threshold=0.5,
        )
        self.assertEqual(result["tp_directed"], 2)
        self.assertEqual(result["fp_directed"], 1)
        self.assertEqual(result["fn_directed"], 0)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["molecule_connectivity_exact"], 0.5)


if __name__ == "__main__":
    unittest.main()
