param(
    [string]$Python = 'D:\minconda\python.exe',
    [int[]]$Seeds = @(42, 43, 44),
    [ValidateSet('structured', 'validity')]
    [string[]]$Analyses = @('structured', 'validity'),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Cache = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt'
$ReferenceSdf = 'data/robustness_27240_keyed/sigma_000.sdf'
foreach ($Path in @($Python, $Cache, $ReferenceSdf)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required input not found: $Path" }
}

foreach ($Seed in $Seeds) {
    $Checkpoint = "checkpoints/revision_v2_direction_fixed_lr1e4_seed$Seed/one_stage_joint/best_e2e.pt"
    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        throw "Missing corrected joint checkpoint for seed $Seed"
    }
    foreach ($Analysis in $Analyses) {
        $Output = if ($Analysis -eq 'structured') { "results/revision_v3_joint_structured_directed/seed$Seed" } else { "results/revision_v3_joint_$Analysis/seed$Seed" }
        $Expected = if ($Analysis -eq 'structured') { "$Output/results.json" } else { "$Output/summary.json" }
        if (Test-Path -LiteralPath $Expected) {
            Write-Host "[SKIP] $Analysis seed=$Seed"
            continue
        }
        if ($Analysis -eq 'structured') {
            $Args = @(
                'scripts/evaluate_structured_distortions.py',
                '--cache', $Cache, '--checkpoint', $Checkpoint,
                '--output_dir', $Output, '--device', 'cuda',
                '--batch_size', '128', '--cutoff', '2.5', '--h_cutoff', '2.5',
                '--conn_threshold', '0.5'
            )
        } else {
            $Args = @(
                'scripts/evaluate_chemical_validity.py',
                '--checkpoint', $Checkpoint, '--cache', $Cache,
                '--reference_sdf', $ReferenceSdf,
                '--output_dir', $Output, '--device', 'cuda',
                '--noise_levels', '0', '0.1', '0.2',
                '--eval_noise_seed', '20260921',
                '--batch_size', '128', '--cutoff', '2.5', '--h_cutoff', '2.5',
                '--conn_threshold', '0.5'
            )
        }
        Write-Host "[RUN] $Analysis seed=$Seed"
        if ($DryRun) {
            Write-Host "$Python $($Args -join ' ')"
            continue
        }
        & $Python @Args
        if ($LASTEXITCODE -ne 0) {
            throw "$Analysis failed for seed $Seed with exit code $LASTEXITCODE"
        }
        if (-not (Test-Path -LiteralPath $Expected)) {
            throw "$Analysis exited successfully but did not produce $Expected"
        }
    }
}
if (-not $DryRun) { Write-Host '[DONE] corrected joint secondary analyses complete.' }
