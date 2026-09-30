param(
    [string]$Python = 'D:\minconda\python.exe',
    [int[]]$Seeds = @(42, 43, 44),
    [string[]]$ConditionNames = @(),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Cache = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Cache)) { throw "Test cache not found: $Cache" }

$Conditions = @(
    @{ Name = 'intact'; Drop = '0'; Noise = '0' },
    @{ Name = 'drop025'; Drop = '0.25'; Noise = '0' },
    @{ Name = 'drop050'; Drop = '0.5'; Noise = '0' },
    @{ Name = 'drop100'; Drop = '1'; Noise = '0' },
    @{ Name = 'hnoise010'; Drop = '0'; Noise = '0.1' },
    @{ Name = 'hnoise020'; Drop = '0'; Noise = '0.2' },
    @{ Name = 'hnoise030'; Drop = '0'; Noise = '0.3' }
)

foreach ($Seed in $Seeds) {
    $Checkpoint = "checkpoints/revision_v2_direction_fixed_lr1e4_seed$Seed/one_stage_joint/best_e2e.pt"
    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        throw "Missing corrected joint checkpoint for seed $Seed"
    }
    foreach ($Condition in $Conditions) {
        if ($ConditionNames.Count -gt 0 -and $Condition.Name -notin $ConditionNames) {
            continue
        }
        $Output = "results/revision_v3_joint_h_corruption/seed$Seed/$($Condition.Name)"
        if (Test-Path -LiteralPath "$Output/results.json") {
            Write-Host "[SKIP] seed=$Seed condition=$($Condition.Name)"
            continue
        }
        $EvalArgs = @(
            'evaluate.py', '--checkpoint', $Checkpoint,
            '--cache_path', $Cache, '--explicit_h',
            '--noise_levels', '0', '--eval_noise_seed', '20260921',
            '--hydrogen_drop_fraction', $Condition.Drop,
            '--hydrogen_noise_sigma', $Condition.Noise,
            '--hydrogen_corruption_seed', '20260925',
            '--batch_size', '128', '--num_workers', '0', '--device', 'cuda',
            '--eval_split', 'test', '--split_key', 'geom_mol_idx',
            '--cutoff', '2.5', '--h_cutoff', '2.5',
            '--conn_threshold', '0.5', '--hh_only',
            '--output_dir', $Output
        )
        Write-Host "[EVAL] seed=$Seed condition=$($Condition.Name)"
        if ($DryRun) {
            Write-Host "$Python $($EvalArgs -join ' ')"
            continue
        }
        & $Python @EvalArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Hydrogen corruption failed for seed=$Seed condition=$($Condition.Name)"
        }
        if (-not (Test-Path -LiteralPath "$Output/results.json")) {
            throw "Missing results.json for seed=$Seed condition=$($Condition.Name)"
        }
    }
}
if (-not $DryRun) {
    Write-Host '[DONE] corrected joint hydrogen-corruption evaluation complete.'
}
