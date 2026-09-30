# BondNet

**Molecular bond perception from 3D coordinates with explicit hydrogen context.**

BondNet recovers a molecular graph — connectivity plus heavy–heavy bond types
(single, double, triple, aromatic) — from atom identities and 3D Cartesian
coordinates. The primary model is a **joint** PaiNN network: one shared backbone
on a distance-derived candidate graph feeds a connectivity head and a four-class
heavy–heavy bond-type head, optimized in a single run. Explicit hydrogens take
part in candidate construction and message passing but are excluded from
bond-type supervision and scoring. A heavy-only joint model, an oracle
hydrogen-count control, and a hard two-stage recipe are provided as comparators.

- Paper: *BondNet: Molecular Bond Perception from 3D Coordinates with Explicit
  Hydrogen Context* (Zhang & Zeng, revised manuscript).
- Checkpoints, split labels, external-cohort files and all result files:
  **Zenodo — [doi:10.5281/zenodo.23054626](https://doi.org/10.5281/zenodo.23054626)**.
- Baselines: RDKit `DetermineBonds` (RDKit's implementation of xyz2mol, default
  and post-hoc `useHueckel=True` configurations) and OpenBabel; an exploratory
  comparison with YuelBond.

This repository contains source code, scripts and tests only. Datasets,
feature caches and checkpoints are on Zenodo.

---

## Repository layout

```
bondnet/        Python package (data/featurizer, noise augmentation, PaiNN model, losses, metrics)
train.py        Train the joint model (or a connectivity-only Stage 1 with --connectivity_only)
train_stage2.py Train the Stage-2 bond-type network of the hard two-stage comparator
evaluate.py     General evaluation entry point
scripts/        Data preparation, evaluation, audits and paper-reproduction scripts
tests/          Regression tests (fixed split, edge direction, keyed noise, scoring, ...)
```

## Installation

Python ≥ 3.9; a CUDA GPU is recommended for training.

```bash
git clone https://github.com/zxc8848/BondNet.git
cd BondNet
pip install -r requirements.txt
pip install -e .
# OpenBabel baseline (optional): conda install -c conda-forge openbabel
python -m pytest -q
```

**Open Babel data directory.** `PerceiveBondOrders` reads its bond-typing rules
from the Open Babel data directory (`bondtyp.txt`). If `BABEL_DATADIR` is not
set and the library cannot find the directory, Open Babel silently falls back to
built-in tables and its bond orders become much worse (clean GEOM HH-graph exact
match 64% instead of 93%). Check your installation with
`python scripts/diagnose_openbabel_datadir.py`, and set `BABEL_DATADIR` (for a
conda installation typically `<conda prefix>/share/openbabel`) before running
any Open Babel baseline.

The revision experiments used Python 3.10, PyTorch 2.5.1, RDKit 2026.03.6 and
Open Babel 3.1.1 (conda-forge; reports 3.1.0) on Windows with an RTX 4090. The
`.ps1` runners are Windows PowerShell wrappers around the Python commands; the
Python scripts themselves are platform-independent.

---

## Data and checkpoints (Zenodo)

Download the Zenodo record and unpack the archives in the repository root:

| Archive | Contents |
|---|---|
| `checkpoints_v4_weights.zip` | Selected checkpoints (`best_e2e.pt`, model weights without optimizer state) and training logs for the joint explicit-H, joint heavy-only, oracle hydrogen-count and hard two-stage models, seeds 42–44 |
| `geom_fixed_split_and_test_inputs.zip` | Fixed 80/10/10 molecule-group split labels for all 269,739 GEOM-DRUGS random1/re10 molecules; the 27,240-molecule fixed-test SDFs at σ = 0, 0.10, 0.20 Å with molecule-keyed noise |
| `geom_random1_re10_source_sdf.zip` | The GEOM-DRUGS random1/re10 source SDF used to build all GEOM caches |
| `external_pubchem3d_cohort_v4.zip` | The frozen 10,000-molecule PubChem3D cohort: manifest, CID list, cohort and perturbed SDFs, feature caches, deviation records 001 and 002 |
| `results_v4.zip` | Every result file behind the tables and figures of the revised manuscript |
| `BondNet_code_v4.zip` | Snapshot of this repository at the release commit |

Unpacked paths follow the repository layout: `checkpoints/…`, `data/…`,
`results/…`. `checkpoints/SOURCE_PATHS.txt` maps the released checkpoint
names to the paths used by the original run scripts.

The fixed split is a deterministic FNV-1a hash of `geom_mol_idx` (seed 42;
`bondnet.data.dataset._assign_split_label`). Build feature caches from the
source SDF with

```bash
python scripts/precompute_features.py --data_path data/geom_drugs_all_random1_rel10.sdf \
    --output data/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed.pt \
    --cutoff 3.0 --h_cutoff 3.0 --explicit_h           # heavy-only: omit --explicit_h
python scripts/extract_fixed_split_cache.py --input <cache.pt> --output_dir data/fixed_split_caches
```

---

## Reproducing the revised manuscript

### Training (three seeds each)

Joint explicit-H model (`scripts/run_revision_v2_onestage_direction_fixed.ps1`):

```bash
python train.py --explicit_h --hidden_size 360 --epochs 50 --lr 0.0001 \
  --cache_path data/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed.pt \
  --split_key geom_mol_idx --batch_size 128 --noise_min 0 --noise_max 0.15 --noise_clean_prob 0 \
  --cutoff 2.5 --h_cutoff 2.5 --e2e_val_every 5 --e2e_val_max_mols 8192 --save_every 5 \
  --selection_noise_levels 0 0.1 --selection_noise_seed 20260921 \
  --seed 42 --output_dir checkpoints/joint_explicit_h/seed42
```

Joint heavy-only model: the same command with `--no_explicit_h` and the
heavy-only cache (`scripts/run_revision_v3_onestage_heavy.ps1`).
Oracle hydrogen-count control: `scripts/run_revision_v4_heavy_hcount.ps1`
(builds heavy-only caches whose atom tokens encode the reference number of
attached hydrogens with `scripts/build_heavy_hcount_cache.py`, then trains and
evaluates with the heavy-only recipe; this input is an oracle and is not
available to a coordinate-only pipeline).
Hard two-stage comparator: `scripts/run_revision_v2.ps1` (Stage 1:
`train.py --connectivity_only --hidden_size 256`, lr 3e-4, 25 epochs;
Stage 2: `train_stage2.py --train_topology teacher`, lr 1e-4, 25 epochs).

### Evaluation on the GEOM fixed test

| Result | Script |
|---|---|
| Main learned-model scores, undirected-pair audit, bootstrap | `scripts/export_per_molecule_stats.py`, `scripts/run_revision_v3_undirected_audit.ps1`, `scripts/summarize_revision_v3_undirected.py`, `scripts/bootstrap_revision_v3_undirected_hydrogen.py` |
| RDKit / OpenBabel on the same keyed-noise SDFs | `scripts/p0d_unified_robustness.py run --methods rdkit openbabel`; Open Babel with its data directory: `scripts/run_revision_v4_openbabel_datadir.ps1`, `scripts/summarize_revision_v4_openbabel_fixed.py` |
| RDKit configuration audit (incl. `useHueckel=True`), rule-label audit | `scripts/audit_rdkit_configurations.py`, `scripts/audit_rule_label_disagreements.py` |
| Full-molecule identity of all methods; RDKit label-mismatch cross-tabulation | `scripts/evaluate_full_molecule_identity.py`, `scripts/crosstab_rule_label_identity.py` |
| Chemical validity, structured distortions | `scripts/run_revision_v3_joint_secondary.ps1` |
| Test-time H corruption | `scripts/run_revision_v3_joint_h_corruption.ps1` |
| Shared-heavy-atom-noise H sensitivity | `scripts/evaluate_paired_hydrogen_noise.py` |
| Oracle hydrogen-count control | `scripts/run_revision_v4_heavy_hcount.ps1`, `scripts/summarize_revision_v4_heavy_hcount.py` |
| Element-pair strata and confusion matrices | `scripts/run_revision_v3_joint_pair_strata.ps1` |
| Candidate-envelope sensitivity, graphs rebuilt from perturbed coordinates, epoch-50 check | `scripts/audit_revision_v3_candidate_envelope.py`, `scripts/build_rebuilt_noisy_caches.py`, `scripts/run_revision_v4_rebuilt_graph.ps1`, `scripts/summarize_revision_v4_rebuilt_graph.py`, `scripts/audit_epoch50_validation.py` |
| Runtime | `scripts/run_revision_v4_runtime_obfix.ps1` |
| YuelBond same-cohort comparison (478 molecules) | `scripts/p0c_yuelbond_compare.py`, `scripts/run_revision_v3_joint_shared_cohort.ps1` |
| Figures 1–2 | `scripts/plot_revision_v3_metrics.py` |

### External PubChem3D cohort

The cohort was selected, materialized and frozen before any model was
evaluated on it; `data/external_v3/manifest.json` records every hash, the
evaluation plan and the reporting rules.

```bash
python scripts/build_external_cohort_v3.py verify        # re-checks all frozen hashes
powershell scripts/run_revision_v3_external.ps1          # one-shot evaluation (skips completed outputs)
python scripts/summarize_revision_v3_external.py
python scripts/summarize_revision_v3_external_subset.py  # Deviation 001 sensitivity analysis
```

To rebuild the cohort from the public PubChem3D download
(`select` → `materialize` → `freeze`), see the docstring of
`scripts/build_external_cohort_v3.py`. Two post-freeze deviations are recorded:
`data/external_v3/DEVIATION_001.md` (105 molecules of an earlier cohort not
excluded) and `data/external_v3/DEVIATION_002.md` (Open Babel re-run with its
data directory configured).

### Notes

- Edge vectors follow the convention x_j − x_i for every directed record; the
  featurizer and cache reader recompute them from coordinates
  (`tests/test_edge_direction_consistency.py`). Checkpoints trained before this
  fix are not part of the release.
- Headline scores count each unordered heavy–heavy pair once (an exported bond
  exists if either directed record predicts it); per-class counts are summed
  over the whole cohort before F1 is computed.
- The bond-type loss is the focal-loss variant documented in
  `bondnet/loss/focal_loss.py` (class weight inside the focusing term).
- Scripts named `p0*`, `run_ablations.py`, `run_experiment_plan.py` and older
  `revision_*` runners reproduce analyses from earlier manuscript versions and
  are kept for traceability.

---

## Citation

```bibtex
@article{bondnet2026,
  title  = {BondNet: Molecular Bond Perception from 3D Coordinates with Explicit Hydrogen Context},
  author = {Zhang, Xiaochen and Zeng, Hui},
  year   = {2026},
  note   = {Revised manuscript}
}

@misc{bondnet_zenodo_v4,
  title     = {BondNet revision release: checkpoints, split labels, external cohort and results},
  author    = {Zhang, Xiaochen and Zeng, Hui},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23054626}
}
```

## License

See `LICENSE`.
