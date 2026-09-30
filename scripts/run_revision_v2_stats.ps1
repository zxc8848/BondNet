param([string]$Python = 'D:\minconda\python.exe')

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$ExplicitTest = 'data\fixed_split_caches\geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt'
$HeavyTest = 'data\fixed_split_caches\geom_drugs_all_random1_re10_c30_heavy_fixed_test.pt'

foreach ($Seed in 42, 43, 44) {
    foreach ($Mode in 'one_stage', 'two_stage', 'heavy_two_stage') {
        $Root = "checkpoints\revision_v2_seed$Seed"
        $Cache = $ExplicitTest
        $Stage2 = $null
        if ($Mode -eq 'one_stage') {
            if ($Seed -eq 44) {
                $Checkpoint = 'checkpoints\revision_v2_seed44_rerun\one_stage_joint\best_e2e.pt'
            } else {
                $Checkpoint = "$Root\one_stage_joint\best_e2e.pt"
            }
        } elseif ($Mode -eq 'two_stage') {
            $Checkpoint = "$Root\explicit_stage1\best_e2e.pt"
            $Stage2 = "$Root\explicit_stage2\best_e2e.pt"
        } else {
            $Checkpoint = "$Root\heavy_stage1\best_e2e.pt"
            $Stage2 = "$Root\heavy_stage2\best_e2e.pt"
            $Cache = $HeavyTest
        }
        $Output = "results\revision_v2_bootstrap\stats\seed$Seed\$Mode"
        $Expected = @('sigma_000.npz', 'sigma_010.npz', 'sigma_020.npz', 'aggregate.json')
        $Complete = $true
        foreach ($Name in $Expected) {
            if (-not (Test-Path -LiteralPath (Join-Path $Output $Name))) {
                $Complete = $false
            }
        }
        if ($Complete) {
            Write-Output "SKIP $Seed $Mode"
            continue
        }
        $Arguments = @(
            'scripts\export_per_molecule_stats.py',
            '--checkpoint', $Checkpoint,
            '--cache', $Cache,
            '--output_dir', $Output,
            '--noise_levels', '0', '0.1', '0.2',
            '--batch_size', '128', '--device', 'cuda'
        )
        if ($Stage2) {
            $Arguments += @('--stage2_ckpt', $Stage2)
        }
        Write-Output "RUN $Seed $Mode"
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Stats export failed: seed=$Seed mode=$Mode exit=$LASTEXITCODE"
        }
    }
}

Write-Output 'All per-molecule revision-v2 statistics exported.'
