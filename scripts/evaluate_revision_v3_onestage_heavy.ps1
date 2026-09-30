param(
    [string]$Python = 'D:\minconda\python.exe',
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$TestCache = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_heavy_fixed_test.pt'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $TestCache)) { throw "Test cache not found: $TestCache" }

foreach ($Seed in $Seeds) {
    $CheckpointDir = "checkpoints/revision_v3_one_stage_heavy_seed$Seed"
    $Checkpoint = "$CheckpointDir/best_e2e.pt"
    $Complete = (Test-Path -LiteralPath "$CheckpointDir/epoch_0050.pt") -and
        (Test-Path -LiteralPath $Checkpoint) -and
        (Select-String -LiteralPath "$CheckpointDir/train.log" -Pattern 'Done\. Best val loss:' -Quiet)
    if (-not $Complete) { throw "Training incomplete for heavy-only seed $Seed" }

    $Output = "results/revision_v3_one_stage_heavy_test/seed$Seed"
    if (Test-Path -LiteralPath "$Output/results.json") {
        Write-Host "[SKIP] test result already present for seed $Seed"
        continue
    }
    $EvalArgs = @(
        'evaluate.py', '--checkpoint', $Checkpoint,
        '--cache_path', $TestCache, '--no_explicit_h',
        '--noise_levels', '0', '0.1', '0.2',
        '--eval_noise_seed', '20260921',
        '--batch_size', '128', '--num_workers', '0', '--device', 'cuda',
        '--eval_split', 'test', '--split_key', 'geom_mol_idx',
        '--cutoff', '2.5', '--h_cutoff', '2.5',
        '--conn_threshold', '0.5', '--hh_only',
        '--output_dir', $Output
    )
    Write-Host "[EVAL] heavy-only one-stage seed $Seed"
    if ($DryRun) {
        Write-Host "$Python $($EvalArgs -join ' ')"
        continue
    }
    & $Python @EvalArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Heavy-only one-stage evaluation failed for seed $Seed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath "$Output/results.json")) {
        throw "Evaluation exited successfully but results.json is absent for seed $Seed"
    }
}
Write-Host '[DONE] v3 heavy-only one-stage three-seed fixed-test evaluation complete.'
