param(
    [string]$Python = 'D:\minconda\python.exe',
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
foreach ($SharedSdf in @('data/shared_bench_clean.sdf', 'data/shared_bench_noisy.sdf')) {
    if (-not (Test-Path -LiteralPath $SharedSdf)) { throw "Missing SDF: $SharedSdf" }
}
foreach ($Seed in $Seeds) {
    $Checkpoint = "checkpoints/revision_v2_direction_fixed_lr1e4_seed$Seed/one_stage_joint/best_e2e.pt"
    if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "Missing checkpoint: $Checkpoint" }
    $Args = @(
        'scripts/p0c_yuelbond_compare.py', 'joint',
        '--checkpoint', $Checkpoint, '--seed', [string]$Seed,
        '--batch-size', '64', '--device', 'cuda'
    )
    Write-Host "[RUN] shared 5k joint seed=$Seed"
    if ($DryRun) { Write-Host "$Python $($Args -join ' ')"; continue }
    & $Python @Args
    if ($LASTEXITCODE -ne 0) { throw "Shared cohort failed for seed=$Seed" }
    foreach ($Condition in @('clean', 'noisy')) {
        $Result = "results/p0c/joint_v3/seed$($Seed)_$($Condition)_score.json"
        if (-not (Test-Path -LiteralPath $Result)) { throw "Missing result: $Result" }
        $Row = Get-Content -LiteralPath $Result -Raw | ConvertFrom-Json
        if ([int]$Row.n_molecules -ne 478 -or [string]$Row.fixed_split_subset -ne 'test' -or [double]$Row.success_rate -ne 1.0) {
            throw "Unexpected cohort or success: $Result"
        }
        Write-Host "[RESULT] seed=$Seed $Condition F1=$($Row.f1_macro_pipeline) HH=$($Row.full_graph_exact_match)"
    }
}
