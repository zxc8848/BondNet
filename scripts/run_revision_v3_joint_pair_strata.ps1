param(
    [string]$Python = 'D:\minconda\python.exe',
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$DryRun
)

# Joint explicit-H element-pair support strata, molecule-size strata and
# reference-bond confusion matrices on the GEOM fixed test (inference only).
# Pairs are scored with the undirected OR rule of the main table; the
# summarizer refuses results whose macro-F1 does not reproduce
# results/revision_v3_undirected_audit/joint/seed*/aggregate.json.

$ErrorActionPreference = 'Stop'
$TestCache = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt'
# Training-split support counts from the earlier run on the same fixed split;
# avoids re-reading the 6.6 GB training cache.
$TrainDist = 'results/revision_v2_pair_strata/seed42/train_pair_bond_distribution.csv'
foreach ($Path in @($Python, $TestCache, $TrainDist)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required input not found: $Path" }
}

foreach ($Seed in $Seeds) {
    $Checkpoint = "checkpoints/revision_v2_direction_fixed_lr1e4_seed$Seed/one_stage_joint/best_e2e.pt"
    if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "Missing checkpoint $Checkpoint" }
    $Output = "results/revision_v3_joint_pair_strata/seed$Seed"
    if (Test-Path -LiteralPath "$Output/summary.json") { Write-Host "[SKIP] seed=$Seed"; continue }
    $Args = @(
        'scripts/analyze_pair_strata.py',
        '--train_distribution_csv', $TrainDist,
        '--test_cache', $TestCache,
        '--checkpoint', $Checkpoint,
        '--output_dir', $Output,
        '--pair_policy', 'undirected_or',
        '--noise_levels', '0', '0.1', '0.2',
        '--eval_noise_seed', '20260921',
        '--batch_size', '128', '--cutoff', '2.5', '--h_cutoff', '2.5',
        '--conn_threshold', '0.5', '--device', 'cuda'
    )
    Write-Host "[RUN] joint pair strata seed=$Seed"
    if ($DryRun) { Write-Host "$Python $($Args -join ' ')"; continue }
    & $Python @Args
    if ($LASTEXITCODE -ne 0) { throw "pair strata failed for seed $Seed" }
}
if (-not $DryRun) {
    & $Python scripts/summarize_revision_v3_joint_pair_strata.py
    if ($LASTEXITCODE -ne 0) { throw 'summary failed (macro-F1 gate or missing files)' }
}
