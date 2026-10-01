# Revision changelog

## 2026-09-20

- Preserved the submitted manuscript under `paper/` and created the independent
  working revision under `paper_revision/`.
- Rewrote the abstract around the scientific question, controlled evidence, and
  limitations rather than a list of validation-set metrics.
- Reframed one-stage versus hard two-stage prediction as a controlled hypothesis.
- Replaced ambiguous heavy/hydrogen notation with A/H notation and explicitly
  defined the candidate, bonded, and labelled heavy-heavy edge sets.
- Moved the Stage-1 threshold rule into the Stage-1 subsection.
- Linked each contribution to the research questions.
- Added success-conditional metrics to the historical robustness table and
  explained why edge-level F1 and molecule-level exact match can rank methods
  differently.
- Removed the causal interpretation of the historical diversity-versus-scale
  comparison.
- Expanded threats to validity to disclose the historical split override and
  cached candidate-topology issue.
- Rewrote the conclusion to distinguish bond perception from complete chemically
  valid molecular reconstruction.
- Created a point-by-point response draft and a tracked experiment plan.

## Code and evaluation changes

- Fixed monolithic and sharded cache splitting so stable train/validation/test
  labels take precedence over the legacy grouped re-split.
- Added `--legacy_resplit` only for historical reproducibility.
- Added tests proving that nominal test samples do not enter training.
- Added active candidate-mask reconstruction from current coordinates inside a
  larger cached edge envelope, including local-geometry recomputation.
- Added candidate prevalence, class support, and full bond-type confusion matrices
  to evaluation outputs.
- Added a one-pass connectivity threshold sweep with edge PR/F1 and molecule-level
  exact connectivity.
- Added conditional-on-success F1 and exact-match metrics to the unified scorer.
- Added a matched architecture protocol manifest. The final primary protocol
  uses 6,443,825 one-stage parameters versus 6,491,657 total two-stage parameters
  (0.74% relative difference) and an equal aggregate 50-epoch budget.
- Verified five unit tests covering fixed splitting, candidate masks, diagnostic
  metrics, and threshold scoring.

## Verification update (2026-09-25)

- Re-ran the fixed-split and dynamic-candidate-mask regression suites after
  installing pytest: all 4 tests passed in 1.45 s.
- The revision experiment programme and fixed-test evaluations are complete; no
  historical checkpoint is used for the held-out accuracy claim.

## Independent-audit corrections (2026-09-26)

- Replaced separate learned/rule-based headline tables with a single fixed-test,
  same-metric comparison. The abstract, discussion, and conclusion now state
  that RDKit has higher molecule-level exactness at clean and moderate noise,
  while BondNet's advantage is limited to edge-level pipeline F1 and the
  strong-noise regime where rule-based graph construction often fails.
- Withdrew the claim of a demonstrated two-stage architectural advantage. The
  revised text distinguishes molecule-bootstrap uncertainty from uncertainty
  over three training seeds and reports the seed-44 reversal at sigma=0.10.
- Corrected the explicit-H ablation to compare teacher-forced explicit-H and
  heavy-only models with the same 25+25 epoch budget. Removed the unequal
  fine-tuned-versus-non-fine-tuned comparison and the obsolete 99.1% to 19.7%
  interpretation.
- Added persistent molecule identifiers to every training/evaluation collate and
  molecule-keyed Gaussian noise. Corresponding heavy atoms now receive bitwise
  identical perturbations in explicit-H and heavy-only representations,
  independent of batch order and size.
- Added deterministic DataLoader generators, complete file logging, fail-fast
  validation shape checks, and errors for missing validation labels. Removed
  arbitrary forward-pass exception swallowing.
- Corrected candidate prevalence from 27.53% for the 3.0-A cache envelope to
  40.36% for the actual clean 2.5-A active graph, and disclosed that the finite
  envelope misses an active pair in 18.9% of molecules at sigma=0.20 under
  molecule-keyed test perturbations.
- Audited all 269,739 GEOM random1/re10 structures and disclosed that none has a
  nonzero formal charge, so charged-molecule performance is untested.
- Added regression tests for test-as-validation refusal, paired explicit-H/
  heavy-only noise, batch-order independence, and equality between materialized
  rule-baseline coordinates and model-evaluation perturbations.
- Started a fully logged three-seed rerun under the unified implementation.

## Matched-rerun results (2026-09-27)

- Completed and evaluated the three-seed rerun on all 27,240 fixed-test
  molecules, including a corrected seed-44 one-stage continuation. Validation
  selected its epoch-11 checkpoint from a 50-epoch run; the last epoch is not
  used as a substitute for the selected best checkpoint.
- Replaced the manuscript's principal table with same-metric pipeline F1,
  exact-type, and strict heavy--heavy graph exact rates for two-stage,
  one-stage, RDKit, and OpenBabel.
- Replaced the explicit-H/heavy-only comparison with like-for-like teacher-forced
  training and removed the historical mixed-protocol difference as evidence.
- Audited the 3.0-A candidate cache against molecule-keyed test noise: at
  sigma=0.20, 5,139 molecules contain 6,075 active pairs outside the cache,
  all reference non-bonds.
- Exported per-molecule sufficient statistics for the final checkpoints and
  computed 2,000 paired molecule-bootstrap replicates. Updated the reviewer
  response and manuscript to distinguish this conditional uncertainty from
  variation across three training seeds.
- Corrected hydrogen provenance: the GEOM parser uses `Chem.AddHs` on source
  SMILES to establish H count/identities before matching source XYZ positions.
  The H coordinates are source coordinates, but the count is graph-informed.
- Flagged public release of the revised checkpoints, code, and results as a
  pre-resubmission task because the cited public records could not be verified.

## Second-review corrections (2026-09-27)

- Replaced old-generation secondary tables with v2 inference. Chemical-validity
  and structured-distortion results now cover seeds 42/43/44 with sample SD;
  pair-support and confusion matrices remain explicitly labelled seed-42
  diagnostics. The high-noise sanitizable rate varies from 37.31% to 48.95%
  across training seeds.
- Reran the Stage-1 threshold sweep on v2, selecting 0.79 on validation before
  reporting its trade-off on test. The default 0.50 remains the headline system.
- Added separate v2 figures for pipeline macro-F1 and HH-graph exact match.
- Replaced one-shot test wording with disclosure that earlier revision rounds
  inspected the same molecule-disjoint fixed test cohort.
- Qualified the architecture comparison: data, augmentation, budget, and
  capacity were near-matched, but optimizer tuning was not symmetric and the
  one-stage training trajectory deteriorated.
- Removed the title's unqualified "Noise-Robust" claim and old validation-cohort
  robustness figure; rewrote the introduction and opening discussion around v2.
- Corrected the clean active heavy--heavy candidate count to 1,520,211
  unordered pairs, distinct from the 3.0-A cache envelope count.
- Removed the last train/test-to-validation fallback when no `split_key` is
  supplied and extended the regression tests.
- Reran the v2 seed-42 runtime benchmark on the molecule-keyed SDF cohort and
  replaced the old-generation runtime table and reviewer response.
- Added a 2,000-replicate paired molecule bootstrap for explicit-H versus
  heavy-only F1, reported separately from the three-training-seed interval.
- Added the RDKit iteration-cap failure-breakdown caveat and documented why
  `--allow_stage2_cache_mismatch` is used with a validated test-only cache.
- Rechecked the full test suite: 17 passed. The built-in LaTeX compiler on this
  host still cannot locate its standard directories, so PDF regeneration remains
  a separate pre-submission step.

## Follow-up factual and protocol corrections (2026-09-27)

- Corrected the v1 `revision_matched` versus v2 explanation: both used fixed
  splits and dynamic candidate masking. Stage-2 learning rate, validation
  checkpoint selection, molecule-keyed noise, and the separate v1
  predicted-topology fine-tuning arm are the relevant differences.
- Marked the architecture comparison as incomplete because joint one-stage
  optimization was not retuned symmetrically.
- Reran the Stage-0 and train-only distance lookup controls on the v2 CPU
  molecule-keyed noise coordinates, correcting their manuscript and reply values.
- Reran intact and six H-specific corruption conditions with all three v2
  explicit-H checkpoints; replaced the old v1 H-corruption paragraph with a
  three-seed table of pipeline F1 and strict HH-graph exactness.
- Moved the two historical clean/robustness tables and their discussion to a
  labelled appendix so the v2 fixed-test main comparison leads Results.
- Narrowed the canonical-SMILES explanation to include both symmetry-equivalent
  atom permutations and aromatic/Kekule representation normalization.
- Reduced primary-table spacing and shortened its conditional-F1 heading;
  visual layout still needs compilation on a working LaTeX host.
- Completed the v2 three-seed predicted-topology Stage-2 continuation and
  same-cohort evaluation. High-noise pipeline F1 improved on average, but
  strict HH-graph exact match did not; the added ten epochs lack a matched
  teacher-forced continuation control. Replaced v1-only M9 claims with this
  qualified v2 result and added the paired table to the manuscript.

## Text corrections (2026-09-28, audit items 2-6)

- Reviewer response M9 now states "partially addressed": the v2 predicted-topology
  continuation did not improve molecule-level HH-graph exactness, and its effect
  is not attributed to topology exposure.
- Manuscript and response now state that end-to-end checkpoint scoring in the
  continuation occurred only at epochs 1 and 6 (epochs 7-10 trained but never
  scored), so the selected checkpoints add six epochs; table caption updated.
- The distance-lookup baseline is now described as fitted only on clean training
  coordinates; its noisy-coordinate collapse is no longer used to rule out a
  noise-aware lookup. Response M12 relabelled "partially addressed".
- Availability statement and editor response rewritten in submission form. The
  new Zenodo version DOI is a visible red placeholder `[REVISION-ZENODO-DOI]`
  that must be replaced after release (also in references.bib, bondnetzenodo).
- Removed "pending release verification" wording from Methods and the response
  preamble; fixed an appendix cross-reference ("reported below").
- Predicted-topology table column spacing reduced; the manuscript now compiles
  with no overfull boxes or undefined references (38 pages).
- Pre-edit copies saved in `paper_revision/_backup_20260928/`.

## Text corrections (2026-09-29, noise pairing and external-cohort methods)

- Methods (noise model): removed the claim that heavy-only inputs receive the same heavy-atom displacements as explicit-H inputs. The keyed sampler draws a tensor sized to each representation and is not prefix-stable, so the two arms get independent draws at matched sigma and seeding. Contributions list and the explicit-H vs heavy-only table caption updated accordingly; response letter R1 hydrogen item notes the correction.
- Methods (external confirmation cohort): replaced the placeholder with the pre-specified protocol implemented in `scripts/build_external_cohort_v3.py` (source, seed 20260929, N = 10,000, rule list, identity matching, keyed noise, manifest/verification). States that historical pre-correction models used subsets of the same PubChem3D download, while no corrected checkpoint touched PubChem3D. Exclusion counts and freeze time/hash remain red placeholders.
- Datasets table: PubChem3D conformers described as computed (not deposited); external row N = 10,000.
- Backup: `paper_revision/_backup_20260929b/`.

## Deviation 001 and external-cohort wording (2026-09-29)

- The historical 50,000-molecule PubChem3D transfer cohort (source records 0-49,999) was not excluded at selection; 105 frozen molecules belong to it. Recorded in `data/external_v3/DEVIATION_001.md` before any external summary; the primary analysis is unchanged and a 9,895-molecule sensitivity analysis is added (`scripts/summarize_revision_v3_external_subset.py`, recomputation only).
- Methods: states the omission and the sensitivity analysis; replaces "archived before evaluation" with a description of the actual evidence (local timestamps plus runner start record, no independent time-stamping, full reproducibility of the cohort from source + script + seed).
- Methods (noise/heavy-only): heavy-only inputs come from RDKit RemoveHs, which keeps stereo-defining H (19/27,240 GEOM test, 75/10,000 external molecules).
- Backup: `paper_revision/_backup_20260929b/main_before_deviation001.tex`.

## External cohort results filled in (2026-09-29)

- Table `tab:external` filled from `results/revision_v3_external/summary.json` and the rule-baseline score JSONs; caption states the Deviation 001 sensitivity bound (max change 0.0011 F1, 0.10 points exact match; from `sensitivity_excl_historical50k/summary.json`).
- New Results text (Section 3.3), abstract sentence, Discussion paragraph, Threats sentence, Methods split paragraph and Conclusion sentence replace the external-cohort placeholders. Added `\label{sec:methods_noise}`.
- Side-effect fixed: `run_revision_v3_external.ps1` called `p0d_unified_robustness.py table` without `--output`, which overwrote `results/robustness_26940_unified/summary.csv` with external rows. The historical file was regenerated from its unchanged score JSONs, and the external table was moved to `results/revision_v3_external/rule_baselines/summary.csv`.
- Backup: `paper_revision/_backup_20260929b/main_before_external_results.tex`.

## Figures and joint pair-strata tooling (2026-09-29)

- New Figs. 1-2 (`revision_v3_pipeline_f1.pdf`, `revision_v3_hh_graph_exact.pdf`): pipeline macro-F1 and HH-graph exact match versus sigma for the five systems, GEOM fixed test and PubChem3D external side by side; seed-SD error bars, discrete levels with dashed guides (R2 minor 5). Generated by `scripts/plot_revision_v3_metrics.py` from the result files; plotted values in `revision_v3_figure_values.csv`. Inserted after Table `tab:external`, referenced from Sections 3.1 and 3.3.
- `scripts/analyze_pair_strata.py`: `--stage2_ckpt` now optional (joint model), `--train_distribution_csv` reuses the fixed-split support counts, and `--pair_policy undirected_or` (new default) scores pairs as in the main table. `directed_src_lt_dst` reproduces the earlier v2 files. Original kept at `tmp/claude_ext_test/analyze_pair_strata.orig.py`.
- New `scripts/run_revision_v3_joint_pair_strata.ps1` and `scripts/summarize_revision_v3_joint_pair_strata.py` (three-seed summary; refuses results whose macro-F1 does not reproduce the undirected audit).
- Backup: `paper_revision/_backup_20260929b/main_before_figures.tex`.

## Joint-model pair strata filled in (2026-09-29)

- Section 3.6 "Joint-model element-pair strata and confusion matrices": text plus Tables `tab:joint_pair_support` and `tab:joint_pair_confusion` from `results/revision_v3_joint_pair_strata/three_seed_summary.json` (three seeds, undirected OR; macro-F1 gate against the undirected audit passed). No red placeholders remain in the manuscript.
- Appendix B staged confusion caption now states that it used the earlier i<j directed-record convention.
- Backup: `paper_revision/_backup_20260929b/main_before_strata.tex`.

## Response letters and availability statement (2026-09-29)

- `response_to_reviewers.md`: removed the working-draft banner; added the external cohort to the opening and to M2, M3, M5, M6, M7, M11 and M13; M3 now describes the frozen cohort, Deviation 001 and the missing third-party timestamp; R2-3 rewritten with the joint-model strata (Tables 11-12); section locations updated to the restructured manuscript (Appendix A/B/C); Minor comments mention Figs. 1-2 and carry a `[RELEASE: ...]` placeholder.
- `response_to_editor.md`: removed the banner; Point 1 rewritten as a release statement with `[RELEASE: ...]` placeholders; Point 2 adds the external-cohort result and its two limitations; the heavy-only comparison notes independent noise draws.
- `main.tex` Availability statement rewritten with red `[RELEASE: ...]` placeholders and the archive contents.
- Backups: `paper_revision/_backup_20260929b/response_to_*_before_v3final.md`.

## Final text alignment with the external cohort (2026-09-29)

- Contributions list: new item for the one-time frozen PubChem3D confirmation.
- Datasets paragraph: PubChem3D now described as the source of the frozen external cohort; its historical role (transfer source, Diverse-80k pool) is stated separately.
- Summary of findings and Practical recommendations: external-cohort sentence added.
- Removed the "Use of AI-assisted tools" subsection from the manuscript at the authors' request.
- Appendices moved after the reference list; appendix tables and figures are numbered per appendix (A1, B1, C1, ...).
- Backup: `paper_revision/_backup_20260929b/main_before_final_text.tex`.

## Read-through fixes on the 16:39 (CST) version (2026-09-29)

Applied to `main.tex` SHA-256 7d251156... (backup `_backup_20260929b/main_1639_before_fix15.tex`); result d0ce7adc...
- "one-stage" -> "joint" in the Noise model and Results protocol sentences.
- Removed the obsolete sentence about historical robustness-diagnostic augmentation; "historical diagnostics" -> "earlier exploratory work" for the PubChem3D source.
- Defined Diverse-80k where it is used in the external-cohort exclusions.
- Error-analysis intro now lists the joint element-pair analysis.
- "such an external cohort" -> "a generator-output cohort"; "roughly 28" -> "roughly 27" HH bonds; grammar fix in Discussion.
- Appendix staged table caption "V2" removed; duplicate "frozen" removed from the external-table caption; CDG defined in the YuelBond caption.
- External results now cite the shared-noise heavy-only values (Table paired_h_noise).
- Abstract notes that a post-hoc RDKit Hückel setting narrows but does not close the strong-noise gap.

- Abstract statement heading restored to "Scientific contribution." (journal requirement).

## Response to second internal review (2026-09-29)

Backup: `_backup_20260929b/main_04531ee9_before_review2.tex`.
- Abstract: heavy-only model described as "same epoch recipe"; RDKit Hückel numbers (53.63% GEOM, 46.73% external at sigma 0.20) added; dependence on complete explicit H stated (25% H removal -> 39.97%).
- Introduction: scope paragraph (neutral molecules with reliable explicit H, computed conformers; generation not evaluated; graph-exact metrics apply to reconstruction).
- Methods: cosine-cutoff argument for the clean-coordinate envelope; edge-feature, head and type-loss definitions; logit symmetrization formula; checkpoint-selection score S; training-configuration table (Table train_config) and GPU-hours; metric aggregation (cohort-level TP/FP/FN).
- "Confirmation" wording -> "frozen-protocol external evaluation"; "equal budget/recipe" -> "same epoch recipe".
- Figures 1-2: post-hoc RDKit Hückel series added (hollow diamonds, dotted).
- Red PENDING placeholders for the three new analyses: rebuilt-graph evaluation, full-molecule identity for all methods, oracle H-count control.
- New scripts: build_rebuilt_noisy_caches.py, run_revision_v4_rebuilt_graph.ps1, summarize_revision_v4_rebuilt_graph.py, evaluate_full_molecule_identity.py, build_heavy_hcount_cache.py, run_revision_v4_heavy_hcount.ps1, summarize_revision_v4_heavy_hcount.py.

## Rebuilt-graph results and identity-script fix (2026-09-30)

Backup: `_backup_20260929b/main_c76c7159_before_rebuilt.tex`.
- Candidate-graph paragraph, Threats, Practical recommendations: PENDING replaced with the rebuilt-graph result (all nine checkpoints, both cohorts; HH exact changes at most 0.14 points for explicit-H and staged models). New appendix Table `tab:rebuilt_graph` from `results/revision_v4_rebuilt_graph/summary.json`.
- Training-configuration table: Open Babel 3.1.0.
- `scripts/evaluate_full_molecule_identity.py`: the Windows run stalled after GEOM RDKit sigma 0 (multiprocessing.Pool waits forever if a native RDKit/Hückel call hangs or crashes a worker). Rule tools now run one molecule per task with a per-molecule wall-clock limit (default 60 s, counted as tool failure and listed in the summary), pool restart, and resumable partial files; completed CSVs are reused.

## Identity runner and H-count cache fixes (2026-09-30)
- `scripts/evaluate_full_molecule_identity.py`: replaced the `multiprocessing.Pool` runner. On Windows, 7 of 8 Hückel pool tasks submitted at each pool (re)start never returned, so ~90 molecules per σ were recorded as spurious 60 s timeouts. Now: dedicated worker processes, each warmed up (imports + one tool call) and started one at a time before the per-molecule clock starts; only the stuck worker is replaced; every provisional timeout is retried once in a fresh worker and counts as a failure only if the retry also fails. Completed CSVs with un-retried timeouts are re-scored automatically (only those molecules). Cloud check on GEOM σ=0: all 91 provisional Hückel timeouts succeed on retry.
- `scripts/build_heavy_hcount_cache.py`: heavy atoms are now aligned by order among non-H atoms with an element-sequence check (retained isotope/stereo H may appear anywhere, e.g. the leading D of geom_mol_idx 36337); n_H counts all attached H including retained ones.

## Full-molecule identity results (2026-09-30, main.tex 6a2a87a3 -> ec3333d4)
- Replaced the identity PENDING placeholder in Sec. 3.6.1 (label `sec:chem_validity`) with the protocol description, new Table `tab:identity` (joint BondNet seeds 42-44, RDKit, RDKit Hückel post-hoc, OpenBabel; both cohorts; σ = 0/0.10/0.20; sanitizable / SMILES match / InChI match) and an interpretation paragraph. Source: `results/revision_v4_identity/summary.json` (after the timeout-retry pass: 0 final timeouts).
- Added one sentence after the RDKit label-mismatch audit (Sec. 3.1) pointing to the identity check (99.86% vs 94.52% clean GEOM SMILES match).
- Backup: `_backup_20260929b/main_6a2a87a3_before_identity.tex`. Compiles cleanly, 48 pages.

## Abstract rewrite (2026-09-30, main.tex ec3333d4 -> a0a6cc81)
- Abstract restructured as application need -> method -> key results (few headline numbers) -> scope; Scientific contribution restated as the method's value plus the benchmark. 326 words (J Cheminform limit 350). All numbers unchanged from the tables (HH exact 99.83/69.56 vs RDKit 3.76; identity 99.86 vs 94.52; external 98.43/85.32, 64.60/3.64; heavy-only 50.70).
- Backup: `_backup_20260929b/main_ec3333d4_before_abstract.tex`.

## Audit round 3 (2026-09-30, main.tex a0a6cc81 -> 1d78fe2e)
1. Identity "Sanitizable": `evaluate_full_molecule_identity.py` now records stages separately (sanitizable = first SanitizeMol; normalized = Kekulize/re-sanitize/RemoveHs/SMILES; inchi_ok) plus fail_stage/fail_error, and writes `results/revision_v4_identity_v2/` (old CSVs lacking stage fields are not reused). Cloud check with RDKit/OpenBabel: every failure occurred at the first SanitizeMol (AtomValenceException for OpenBabel); no molecule passed sanitization and then failed normalization. Final values need the rerun on the evaluation machine.
2. New `scripts/crosstab_rule_label_identity.py`: molecule-level cross-tab of the 1,339 clean-GEOM RDKit label-only mismatches with identity. Provisional cloud result (RDKit 2026.03.6 reproduced 1,320 audited predictions): 0 SMILES match, 421 InChI-only, 899 differ under both. Text now states "mostly, but not entirely"; red provisional placeholder until the evaluation-machine run.
3. Focal loss: Methods now names the implemented weighted-CE focal variant and contrasts it with the alpha-balanced form (added \cite{focal}, Lin et al. 2017); `bondnet/loss/focal_loss.py` docstring corrected (code unchanged).
4. Rebuilt-graph wording: per-checkpoint bounds (explicit-H <= 0.06 pp, staged <= 0.30 pp, F1 <= 0.0021), means (explicit-H <= 0.05, staged <= 0.14), heavy-only compared with the shared-noise cached evaluation (+0.01/-0.12 GEOM, 0.00/-0.07 external); absolute "does not depend" wording removed.
5. Abstract: "after common aromaticity normalization"; GEOM noted as used during development; noise training described without causal claim; missing-H wording corrected; external Hückel 46.7% vs 64.6% at 0.20 A. 347 words.
6. Identity analysis labelled post-hoc (text and caption); sigma=0.20 sentence now reports SMILES values per method/cohort.
Backups: `_backup_20260929b/main_a0a6cc81_before_audit3.tex`, `references_before_audit3.bib`, `focal_loss_before_docstring.py`.

## Identity runner: worker-crash handling (2026-09-30)
- On Windows a Hückel worker that dies in native code closes its pipe, and `poll()` raised BrokenPipeError, aborting the run. The runner now treats a closed pipe or dead process as a provisional `worker_crash` (same path as a timeout): the worker is replaced, the molecule is retried once in a fresh worker, and only a repeated failure counts (fail_stage `worker_crash`). Tested in the cloud by killing two workers mid-run: both molecules succeeded on retry. Resumes from the existing partial files.

## RDKit label-mismatch cross-tab, final (2026-09-30, main.tex 1d78fe2e -> 36e7ee7a)
- Evaluation machine (RDKit 2026.03.6; all 1,339 audited predictions reproduced, 0 mismatches vs saved): of the 1,339 clean-GEOM RDKit label-only mismatches, 0 match the normalized reference SMILES, 431 match only the non-stereo InChI (268/738 single-double-only, 163/601 aromatic-label), 908 differ under both. 19 connectivity errors all differ; 16 tool failures. Red provisional placeholder replaced. Source: `results/revision_v4_rule_label_audit/geom_rdkit_clean_identity_crosstab.json`.
- Note: the cloud run with the same RDKit version on Linux reproduced only 1,320 predictions (platform-dependent DetermineBonds assignments for 20 molecules); the evaluation-machine values are used.
- OpenBabel Python binding on the evaluation machine: 3.1.0. A cloud check with openbabel-wheel 3.1.1.23 gave clean-GEOM macro-F1 0.985 / HH exact 92.9% versus 0.892 / 64.2% in the primary results: OpenBabel baseline is strongly build-dependent (pending decision on a sensitivity run).

## Open Babel discrepancy traced to missing data files (2026-09-30)
- Cloud rerun of the primary Open Babel call on both cohorts (openbabel-wheel, data files found) gives clean HH exact 92.86% GEOM / 89.04% external versus 64.19% / 65.39% in the primary tables. Hiding only `bondtyp.txt` reproduces the primary value (63.9% on 2,000 GEOM molecules). Not a version effect: several wheel builds (3.1.1.14, 3.1.1.23) give the same result, and all report OBReleaseVersion 3.1.0. Results: `results/revision_v4_openbabel_datadir_check/`. Diagnostic for the evaluation machine: `scripts/diagnose_openbabel_datadir.py`. Manuscript not yet changed.

## Open Babel data-directory cause confirmed on the evaluation machine (2026-09-30)
- `diagnose_openbabel_datadir.py`: BABEL_DATADIR unset in the Python environment used for all runs -> HH exact 63.90% (first 2,000 clean GEOM); with BABEL_DATADIR = D:\minconda\share\openbabel (conda openbabel 3.1.1, OBReleaseVersion 3.1.0) -> 92.75%, identical to the cloud wheel. All primary Open Babel numbers were produced without bondtyp.txt.
- New `scripts/run_revision_v4_openbabel_datadir.ps1` reruns the Open Babel baselines (GEOM, frozen external, 478-molecule shared subset) with the data directory set, into `results/revision_v4_openbabel_fixed/` (archived predictions untouched).
- `evaluate_full_molecule_identity.py` now sets/validates BABEL_DATADIR before any Open Babel scoring and records it in the summary; the earlier Open Babel identity results were moved to `results/revision_v4_identity_v2/_superseded_openbabel_no_datadir/`.
- Pending: rerun, then update Tables fixed_learned / external / yuel_shared / identity, Figures 1-2, the Open Babel text, and record external-cohort Deviation 002. Runtime table Open Babel row was also measured without data files.

## Open Babel corrected throughout (2026-09-30, main.tex 36e7ee7a -> 84832193)
- `run_revision_v4_openbabel_datadir.ps1` completed on the evaluation machine (BABEL_DATADIR = D:\minconda\share\openbabel). Corrected values replace the primary Open Babel numbers in Tables fixed_learned, external, yuel_shared and identity, Figures 1-2 (`plot_revision_v3_metrics.py` now reads `results/revision_v4_openbabel_fixed/summary.json`) and all Open Babel text. GEOM clean/0.10/0.20 HH exact 92.86/46.12/0.72% (F1 0.9853/0.8580/0.4507); external 89.02/41.21/0.95% (F1 0.98302/0.85375/0.43921); shared-478 clean/0.10 90.59/43.51%.
- New Baselines paragraph explains the data-directory issue; external Methods paragraph and `data/external_v3/DEVIATION_002.md` record Deviation 002 with original vs corrected external values; Discussion and Threats updated. Deviation-001 subset recomputed from corrected predictions (`summarize_revision_v4_openbabel_fixed.py`): max |Δ| F1 0.0011, HH 0.078 pp (existing caption statement still holds).
- Identity table: Open Babel rows corrected (GEOM all σ, external σ=0/0.10); RDKit Hückel GEOM σ=0 99.93/94.49/96.51 and σ=0.20 sanitizable 77.94 from the crash-free v2 run. External Open Babel σ=0.20 identity was stale (computed by the earlier run after the folder move) and was moved to `_superseded_openbabel_no_datadir/external/openbabel_stale_sigma020`; red PEND until rerun.
- Efficiency table caption notes the Open Babel timing predates the correction.
- Config table: Open Babel 3.1.1 (conda-forge; reports 3.1.0) with BABEL_DATADIR set.
- Backups: `_backup_20260929b/main_36e7ee7a_before_obfix.tex`, `figs_before_obfix/`, `plot_revision_v3_metrics_before_obfix.py`.

## Identity: external Open Babel sigma=0.20 filled (2026-09-30, main.tex 84832193 -> cc004c83)
- Rerun with BABEL_DATADIR = D:\minconda\share\openbabel: sanitizable 93.03%, SMILES 0.17%, InChI 2.68% (697 AtomValenceException). Table identity and the sigma=0.20 sentence updated; identity table now complete. Only remaining red placeholder: oracle H-count control.

## Language polish merged (2026-09-30, main.tex -> 0db9b87c)
- Base: the user's polished main.tex (f01f61ea, saved 09:21; copy in `_backup_20260929b/main_f01f61ea_user_polish_copy.tex`). An alternative Claude polish is kept as `main_polished_claude.tex` for reference only.
- Targeted fixes on the user's version: restored the J Cheminform "Scientific contribution" statement (abstract 304 words); added the application scenarios to the first abstract sentence; replaced the leftover "threshold sweep is labelled diagnostic" sentence; "unexpected hydrogen input" -> "incomplete or displaced hydrogen input"; removed "main-table staged checkpoints are unchanged"; removed "corrected split" wording in the YuelBond section and caption. Numbers unchanged. Compiles cleanly, 45 pages.

## Oracle H-count control filled (2026-09-30, user's main.tex 4045a49a -> f517209b)
- Base: user's revised main.tex 4045a49a (saved 10:44 CST). Results from `results/revision_v4_heavy_hcount/summary.json`: heavy-only + reference H-count tokens reach HH exact 99.88/99.79/82.03% (GEOM) and 98.72/98.33/76.52% (PubChem3D) at sigma 0/0.10/0.20, versus explicit H 99.83/99.62/69.56 and 98.43/97.94/64.60; seed SD at 0.20 is 0.13/0.15 vs 2.03/1.93.
- Added: Methods paragraph describing the oracle control (post-hoc; token map; heavy-only noise draws); Results paragraph + new Table `tab:hcount` replacing the last red placeholder; one abstract sentence (abstract now 254 words); Discussion (source of the hydrogen benefit is mainly valence information) and Practical implications (supplying counts directly; not evaluated with predicted counts); Conclusions clause. No red placeholders remain. Compiles cleanly, 45 pages.

## Runtime re-timed and joint identity re-run (2026-09-30, main.tex f517209b -> a98e15e2)
- `scripts/run_revision_v4_runtime_obfix.ps1` (BABEL_DATADIR set; environment now recorded by `benchmark_revision_runtime.py`) -> `results/revision_v4_runtime/`. Table efficiency now from one session: joint 1.837 (0.534) / 2.159 ms, 463 mol/s, 758 MiB; staged 1.711 (0.392) / 2.030 ms, 493 mol/s, 550 MiB; RDKit 0.544 / 0.866 ms, 1,154 mol/s; OpenBabel 0.361 / 0.683 ms, 1,465 mol/s (Open Babel timing essentially unchanged by the data directory). Text updated (forward 14.55 s, 1,873 mol/s; featurization 35.30 s; transfer 0.18 s; SDF 8.77 s; size strata 1.52/1.81/2.19 ms; staged now faster end to end); "OpenBabel timing predates" caption note removed; practical-implications throughput 463 mol/s.
- `evaluate_full_molecule_identity.py --methods joint` (stage-wise): BondNet values unchanged except first-stage sanitizability at sigma=0.20, GEOM 72.35 (1.99) and external 69.97 (2.05), because at most two molecules per seed pass SanitizeMol but fail the later kekulization/re-sanitization. Identity table and text updated; this completes audit point 1 (Sanitizable definition).

## Response letters updated for review round 2 (2026-09-30)
- Both letters updated against main.tex 413ecb43 (user's latest). Added: OpenBabel data-directory correction (all OpenBabel numbers corrected; disclosed as a post-freeze departure, both runs archived); full-molecule identity comparison of four methods and the 1,339-mismatch cross-tab (0 SMILES, 431 InChI-only); RDKit Hückel identity values; rebuilt-graph check; oracle H-count control (82.03% / 76.52% at sigma 0.20); stage-wise Sanitizable definition; re-timed runtime (463 vs 493 mol/s; RDKit 0.866, OpenBabel 0.683 ms); metric aggregation; focal-loss variant, training-config table, GPU-hours; scope statement; section references updated to current headings ("confirmation" wording removed; "not a universal decrease" replaced because all methods are lower on PubChem3D after the OpenBabel fix).
- Every number in the letters is traceable to main.tex or the result files. Backups: `_backup_20260929b/response_to_*_before_round2.md`.

## Release v4 packaging (2026-09-30)
- New `scripts/build_release_v4.py` assembles `release_v4/` (release_v3 left untouched): github/ code snapshot (155 files) with updated README (Open Babel data-directory note, new v4 scripts); zenodo/ checkpoints_v4.zip (+3 oracle H-count models), external_pubchem3d_cohort_v4.zip (+DEVIATION_002.md), results_v4.zip (v3 result dirs + rebuilt graph, identity_v2, openbabel_fixed, datadir check, heavy_hcount, runtime), unchanged GEOM zips, BondNet_code_v4.zip, README_zenodo.md, SHA256SUMS.txt; `release_v4/上传说明.md`. Zips are built by the user on Windows (`D:\minconda\python.exe scripts\build_release_v4.py`); the device VM cannot run long background jobs.
