"""Paired inference audit of the fixed test cache's clean 3.0-A envelope.

For each existing joint checkpoint at sigma=0.20, evaluate the original cache
path and an otherwise identical path that adds non-H--H directed candidates
which are within 2.5 A after noise but absent from the clean cache. The same
molecule-keyed noise, coordinates, model, and scorer are used in both arms.
No checkpoint, threshold, or sample is selected from the audit results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate as ev
from bondnet.data.dataset import CachedBondNetDataset
from bondnet.data.noise_augment import GaussianNoiseAugment
from scripts import export_per_molecule_stats as stats_mod


CACHE = ROOT / "data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt"
AUDIT = ROOT / "results/revision_v3_undirected_audit/joint"
OUTPUT = ROOT / "results/revision_v3_candidate_envelope"
SIGMA = 0.2
SEED = 20260921
CUTOFF = 2.5
CHECKPOINTS = {
    seed: ROOT / f"checkpoints/revision_v2_direction_fixed_lr1e4_seed{seed}/one_stage_joint/best_e2e.pt"
    for seed in (42, 43, 44)
}


def add_missing_active_edges(batch: dict, cutoff: float) -> tuple[dict, int]:
    """Append both directions of newly active non-H--H pairs, labelled non-bond.

    The original cache remains intact, including its inactive envelope edges.
    The reference-bond audit found no true bonds outside that envelope; assert
    that none of the new pairs overlaps a known true-bond record.
    """
    coord = batch["coord"]
    elems = batch["elems"]
    old_edges = batch["edge_index"]
    counts = [int(n) for n in batch["num_atoms_per_mol"].tolist()]
    new_edges = []
    new_mol = []
    offset = 0
    for mol_i, n_atoms in enumerate(counts):
        local = coord[offset:offset + n_atoms]
        local_elems = elems[offset:offset + n_atoms]
        diff = local.unsqueeze(0) - local.unsqueeze(1)
        distances = diff.norm(dim=-1)
        eligible = (distances <= cutoff) & (distances > 0)
        eligible &= ~((local_elems[:, None] == 1) & (local_elems[None, :] == 1))
        old_here = batch["edge_mol_idx"] == mol_i
        old_local = old_edges[old_here] - offset
        eligible[old_local[:, 0], old_local[:, 1]] = False
        missing = eligible.nonzero(as_tuple=False)
        if missing.numel():
            new_edges.append(missing + offset)
            new_mol.append(torch.full((missing.shape[0],), mol_i, dtype=batch["edge_mol_idx"].dtype))
        offset += n_atoms
    if not new_edges:
        return batch, 0

    added = torch.cat(new_edges, dim=0)
    src, dst = added[:, 0], added[:, 1]
    known_bonds = {tuple(pair) for pair in batch["edge_index_bond"].tolist()}
    if any(tuple(pair) in known_bonds for pair in added.tolist()):
        raise AssertionError("An added candidate is a known reference bond")
    delta = coord[dst] - coord[src]
    n_new = added.shape[0]
    out = dict(batch)
    out["edge_index"] = torch.cat((old_edges, added), dim=0)
    out["edge_diff"] = torch.cat((batch["edge_diff"], delta), dim=0)
    out["edge_dist"] = torch.cat((batch["edge_dist"], delta.norm(dim=-1)), dim=0)
    out["edge_mol_idx"] = torch.cat((batch["edge_mol_idx"], torch.cat(new_mol)), dim=0)
    for key in ("bond_exists", "bond_mask", "train_edge_mask", "train_bond_mask"):
        old = batch[key]
        out[key] = torch.cat((old, torch.zeros(n_new, dtype=old.dtype)), dim=0)
    h_edge = (elems[src] == 1) ^ (elems[dst] == 1)
    out["h_candidate_edge_mask"] = torch.cat((batch["h_candidate_edge_mask"], h_edge), dim=0)
    return out, n_new


def summarize(stats: dict) -> dict:
    _, f1 = stats_mod.f1_from_stats(stats["u_pipe_tp"], stats["u_pipe_fp"], stats["u_pipe_fn"])
    return {
        "n_molecules": int(len(stats["mol_idx"])),
        "pipeline_macro_f1": f1,
        "hh_graph_exact_rate": float(stats["u_hh_graph_exact"].mean()),
        "true_pair_exact_rate": float(stats["u_true_pair_exact"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_mols", type=int, default=None, help="Smoke test only; omit for the final audit")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = ev._get_device(args.device)
    dataset = CachedBondNetDataset(str(CACHE))
    if args.max_mols is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_mols, len(dataset))))
    config = argparse.Namespace(eval_noise_seed=SEED, cutoff=CUTOFF, h_cutoff=CUTOFF,
                                conn_threshold=0.5, batch_size=128)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_predict = stats_mod.predict
    report = {"sigma": SIGMA, "noise_seed": SEED, "cutoff_angstrom": CUTOFF,
              "cache": str(CACHE.relative_to(ROOT)), "max_mols": args.max_mols,
              "seeds": {}}

    for seed, checkpoint in CHECKPOINTS.items():
        model = ev.load_model(str(checkpoint), device)
        model.eval()
        added_records = 0
        affected_batches = 0

        def expanded_predict(model_, stage2, batch, device_, config_, sigma):
            nonlocal added_records, affected_batches
            aug = GaussianNoiseAugment(sigma=SIGMA, sigma_min=SIGMA,
                                       sigma_max=SIGMA, cutoff=CUTOFF)
            noise_seed = SEED + int(round(SIGMA * 10000.0))
            noisy = aug.augment_batch(batch, per_molecule=True,
                                      base_seed=noise_seed, epoch=0)
            expanded, n_added = add_missing_active_edges(noisy, CUTOFF)
            added_records += n_added
            affected_batches += int(n_added > 0)
            return original_predict(model_, stage2, expanded, device_, config_, 0.0)

        stats_mod.predict = original_predict
        baseline = stats_mod.collect_sigma(model, None, dataset, device, config, SIGMA)
        stats_mod.predict = expanded_predict
        try:
            expanded = stats_mod.collect_sigma(model, None, dataset, device, config, SIGMA)
        finally:
            stats_mod.predict = original_predict
        if not np.array_equal(baseline["mol_idx"], expanded["mol_idx"]):
            raise AssertionError("Paired molecule order changed")
        base_summary = summarize(baseline)
        expanded_summary = summarize(expanded)
        if args.max_mols is None:
            audit_rows = json.loads((AUDIT / f"seed{seed}/aggregate.json").read_text(encoding="utf-8"))
            reference = next(row for row in audit_rows if float(row["sigma"]) == SIGMA)
            report["seeds"].setdefault(str(seed), {})["main_table_baseline_delta"] = {
                "pipeline_macro_f1": base_summary["pipeline_macro_f1"] - reference["undirected_pipeline_macro_f1"],
                "hh_graph_exact_rate": base_summary["hh_graph_exact_rate"] - reference["undirected_hh_graph_exact_rate"],
            }
            # GPU scatter reductions are not bitwise deterministic between
            # separate inference runs; retain the exact replay delta and use a
            # gate that catches protocol drift rather than floating-point jitter.
            for ours, theirs, tolerance in (
                (base_summary["pipeline_macro_f1"], reference["undirected_pipeline_macro_f1"], 1e-4),
                (base_summary["hh_graph_exact_rate"], reference["undirected_hh_graph_exact_rate"], 5e-4),
            ):
                if abs(ours - theirs) > tolerance:
                    raise AssertionError(f"Original path failed to reproduce main table within {tolerance}: {ours} != {theirs}")
        report["seeds"][str(seed)] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "main_table_baseline_delta": report["seeds"].get(str(seed), {}).get("main_table_baseline_delta"),
            "baseline": base_summary,
            "expanded": expanded_summary,
            "delta_f1": expanded_summary["pipeline_macro_f1"] - base_summary["pipeline_macro_f1"],
            "delta_hh_graph_exact": expanded_summary["hh_graph_exact_rate"] - base_summary["hh_graph_exact_rate"],
            "added_directed_candidate_records": added_records,
            "affected_batches": affected_batches,
            "molecules_changed_exact_status": int(np.count_nonzero(
                baseline["u_hh_graph_exact"] != expanded["u_hh_graph_exact"])),
        }
        print(f"seed={seed} baseline={base_summary} expanded={expanded_summary} "
              f"added={added_records}", flush=True)
    path = OUTPUT / ("smoke.json" if args.max_mols is not None else "summary.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
