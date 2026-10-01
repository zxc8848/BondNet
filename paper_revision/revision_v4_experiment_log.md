# Revision experiments and provenance (2026-09-29)

This log distinguishes the frozen primary graph-scoring evaluation from later
diagnostics. None of the experiments below changes a trained checkpoint, the
external cohort manifest, or a prespecified primary prediction file.

## Endpoint-50 validation audit

- Script: `scripts/audit_epoch50_validation.py`
- Data: first 8,192 fixed GEOM validation molecules; sigma 0 and 0.10 Å;
  original molecule-keyed noise and checkpoint-selection score.
- Output: `results/revision_v3_epoch50_validation/summary.json`
- Epoch 50 minus selected epoch 46: +0.0000128, -0.0000009, -0.0000097
  for joint explicit-H seeds 42–44. The primary checkpoint choice is unchanged.

## Shared-heavy-atom-noise hydrogen sensitivity

- Script: `scripts/evaluate_paired_hydrogen_noise.py`; regression tests:
  `tests/test_paired_hydrogen_noise.py`.
- The original explicit-H noise draw is made at the full atom count. Its
  heavy-atom displacements are applied to the corresponding heavy-only atoms;
  stereo-retained H atoms are mapped by unique clean coordinates.
- Outputs: `results/revision_v4_paired_hydrogen/geom/` and
  `results/revision_v4_paired_hydrogen/external/`, including per-seed,
  per-molecule NPZ files and `summary.json`.
- Three-seed mean HH exactness, explicit-H versus paired heavy-only:
  GEOM 99.62%/91.66% at sigma 0.10 and 69.56%/51.27% at sigma 0.20;
  PubChem3D 97.94%/82.99% and 64.60%/40.40%, respectively.
- This is post-hoc inference, not part of the frozen external primary plan.

## Rule-label and RDKit-configuration audits

- `scripts/audit_rule_label_disagreements.py` reads saved clean GEOM RDKit
  predictions and verifies their SDF hash. Output:
  `results/revision_v4_rule_label_audit/geom_rdkit_clean.json`.
  Among 27,240 molecules: 25,866 exact HH graphs, 16 tool failures, 19
  connectivity errors, and 1,339 same-connectivity label errors. This
  partition does not prove chemical equivalence of the label mismatches.
- `scripts/audit_rdkit_configurations.py` uses the same atom-pair label map
  and SDFs as the primary baseline. The default-configuration rerun exactly
  reproduces the GEOM/external high-noise HH-exact values 3.76%/3.64%.
- Validation-only settings, 26,940 molecules at sigma 0.20 Å: default 3.91%,
  cap 2,000 3.95%, `useVdw=True` 1.14%, `embedChiral=False` 0.15%,
  `allowChargedFragments=False` 3.80%, `useHueckel=True` 53.51% HH exact.
- Post-hoc `useHueckel=True` on GEOM test: 94.93%, 94.47%, 53.63% HH exact
  at sigma 0/0.10/0.20 Å; on external PubChem3D: 85.33%, 85.24%, 46.73%.
  Joint explicit-H means are 99.83%, 99.62%, 69.56% and
  98.43%, 97.94%, 64.60%, respectively. The Hückel runs are not frozen
  primary endpoints.
- Same-scorer pipeline macro-F1 reruns are in
  `results/revision_v4_rdkit_configs_f1/`. The Hückel GEOM values are
  0.98827/0.98510/0.72204 and external values are
  0.95650/0.95598/0.73002 at sigma 0/0.10/0.20 Å. The default high-noise
  reruns reproduce the primary GEOM/external F1 values 0.23669/0.23053.
- JSON outputs are under `results/revision_v4_rdkit_configs/`. The original
  GEOM/external high-noise default runs report explicit iteration-limit
  exceptions for 495/20,469 and 150/7,732 failed molecules, respectively.

## Verification and release blockers

- `D:\minconda\python.exe -m pytest -q`: 31 passed.
- The built-in LaTeX compiler is unavailable on this Windows host. Local
  MiKTeX draft-mode passes (with BibTeX) complete with no undefined references
  or overfull boxes. Draft mode does not update `paper_revision/main.pdf`.
- Do not submit until the corrected GitHub release and permanent archive DOI
  have been published and verified, the manuscript and both response letters
  have their final identifiers, and an up-to-date PDF is generated.
