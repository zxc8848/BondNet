import unittest

import torch

from bondnet.data.noise_augment import apply_dynamic_candidate_mask


class DynamicCandidateMaskTests(unittest.TestCase):
    def test_heavy_and_hydrogen_cutoffs_are_applied_to_current_distances(self):
        batch = {
            "elems": torch.tensor([6, 6, 1]),
            "edge_index": torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2]]),
            "edge_dist": torch.tensor([2.4, 2.6, 1.2, 1.6, 1.4]),
            "bond_mask": torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool),
        }
        result = apply_dynamic_candidate_mask(batch, heavy_cutoff=2.5, hydrogen_cutoff=1.5)
        self.assertEqual(
            result["train_edge_mask"].tolist(),
            [True, False, True, False, True],
        )
        self.assertEqual(
            result["train_bond_mask"].tolist(),
            [True, False, False, False, False],
        )
        self.assertEqual(
            result["h_candidate_edge_mask"].tolist(),
            [False, False, True, False, True],
        )

    def test_bond_type_targets_follow_active_true_hh_edges(self):
        batch = {
            "elems": torch.tensor([6, 6, 8, 1]),
            "edge_index": torch.tensor(
                [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0]]
            ),
            "edge_dist": torch.tensor([1.4, 1.4, 2.6, 2.6, 1.0, 1.0]),
            "bond_mask": torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),
            # Labels follow the six true positions in bond_mask.
            "bond_type_full": torch.tensor([1, 1, 0, 0, 0, 0]),
        }
        result = apply_dynamic_candidate_mask(batch, heavy_cutoff=2.5, hydrogen_cutoff=1.5)
        self.assertEqual(
            result["train_bond_mask"].tolist(),
            [True, True, False, False, False, False],
        )
        self.assertEqual(result["bond_type_train_active"].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
