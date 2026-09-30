param(
    [string]$Python = 'D:\minconda\python.exe',
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Cache = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Cache)) { throw "Cache not found: $Cache" }
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
    if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "Checkpoint not found: $Checkpoint" }
    foreach ($Condition in $Conditions) {
        $Output = "results/revision_v3_secondary_pair_audit/h_corruption/seed$Seed/$($Condition.Name)"
        if (Test-Path -LiteralPath "$Output/aggregate.json") {
            Write-Host "[SKIP] H seed=$Seed $($Condition.Name)"
            continue
        }
        $Args = @(
            'scripts/export_per_molecule_stats.py',
            '--checkpoint', $Checkpoint, '--cache', $Cache,
            '--output_dir', $Output, '--noise_levels', '0',
            '--hydrogen_drop_fraction', $Condition.Drop,
            '--hydrogen_noise_sigma', $Condition.Noise,
            '--hydrogen_corruption_seed', '20260925',
            '--batch_size', '128', '--cutoff', '2.5', '--h_cutoff', '2.5',
            '--conn_threshold', '0.5', '--device', 'cuda'
        )
        Write-Host "[RUN] H seed=$Seed $($Condition.Name)"
        if ($DryRun) { Write-Host "$Python $($Args -join ' ')"; continue }
        & $Python @Args
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath "$Output/aggregate.json")) {
            throw "H pair audit failed: seed=$Seed $($Condition.Name)"
        }
    }
    $StructuredOutput = "results/revision_v3_secondary_pair_audit/structured/seed$Seed"
    if (Test-Path -LiteralPath "$StructuredOutput/results.json") {
        Write-Host "[SKIP] structured seed=$Seed"
    } else {
        $Args = @(
            'scripts/evaluate_structured_distortions.py',
            '--checkpoint', $Checkpoint, '--cache', $Cache,
            '--output_dir', $StructuredOutput, '--seed', '20260921',
            '--batch_size', '128', '--cutoff', '2.5', '--h_cutoff', '2.5',
            '--conn_threshold', '0.5', '--device', 'cuda'
        )
        Write-Host "[RUN] structured seed=$Seed"
        if ($DryRun) { Write-Host "$Python $($Args -join ' ')"; continue }
        & $Python @Args
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath "$StructuredOutput/results.json")) {
            throw "Structured pair audit failed: seed=$Seed"
        }
    }
}
