param([string]$Python = 'D:\minconda\python.exe', [switch]$DryRun)

$ErrorActionPreference = 'Stop'
$Sdf = 'data/robustness_27240_keyed/sigma_000.sdf'
$Stage1 = 'checkpoints/revision_v2_seed42/explicit_stage1/best_e2e.pt'
$Stage2 = 'checkpoints/revision_v2_seed42/explicit_stage2/best_e2e.pt'
$Joint = 'checkpoints/revision_v2_direction_fixed_lr1e4_seed42/one_stage_joint/best_e2e.pt'
foreach ($Path in @($Python, $Sdf, $Stage1, $Stage2, $Joint)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing runtime input: $Path" }
}
foreach ($Model in @('joint', 'staged')) {
    $Output = "results/revision_v3_runtime/seed42_$Model"
    if (Test-Path -LiteralPath "$Output/runtime.json") {
        Write-Host "[SKIP] runtime $Model"
        continue
    }
    $Args = @(
        'scripts/benchmark_revision_runtime.py',
        '--sdf', '0', $Sdf,
        '--output_dir', $Output,
        '--batch_size', '128', '--device', 'cuda',
        '--cutoff', '2.5', '--h_cutoff', '2.5', '--conn_threshold', '0.5'
    )
    if ($Model -eq 'joint') { $Args += @('--checkpoint', $Joint) }
    else { $Args += @('--stage1', $Stage1, '--stage2', $Stage2) }
    Write-Host "[RUN] runtime $Model"
    if ($DryRun) {
        Write-Host "$Python $($Args -join ' ')"
        continue
    }
    & $Python @Args
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath "$Output/runtime.json")) {
        throw "Runtime benchmark failed for $Model"
    }
}
if (-not $DryRun) { Write-Host '[DONE] representative seed-42 clean runtime benchmarking complete.' }
