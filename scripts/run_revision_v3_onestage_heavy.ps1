param(
    [string]$Python = 'D:\minconda\python.exe',
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Cache = 'data/geom_drugs_all_random1_re10_c30_heavy_fixed.pt'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Cache)) { throw "Cache not found: $Cache" }

foreach ($Seed in $Seeds) {
    $Output = "checkpoints/revision_v3_one_stage_heavy_seed$Seed"
    $Resume = @()
    if (Test-Path -LiteralPath $Output) {
        $Completed = (Test-Path -LiteralPath "$Output/epoch_0050.pt") -and
            (Test-Path -LiteralPath "$Output/train.log") -and
            (Select-String -LiteralPath "$Output/train.log" -Pattern 'Done\. Best val loss:' -Quiet)
        if ($Completed) {
            Write-Host "[SKIP] completed heavy-only seed $Seed"
            continue
        }
        $Latest = Get-ChildItem -LiteralPath $Output -Filter 'epoch_*.pt' -File |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($null -eq $Latest) {
            throw "Partial run without periodic checkpoint at $Output; inspect before restarting."
        }
        $Resume = @('--resume', $Latest.FullName)
        Write-Host "[RESUME] heavy-only seed $Seed from $($Latest.FullName)"
    } else {
        Write-Host "[RUN] heavy-only one-stage seed $Seed"
    }

    $TrainArgs = @(
        'train.py', '--no_explicit_h', '--hidden_size', '360',
        '--epochs', '50', '--lr', '0.0001', '--output_dir', $Output,
        '--cache_path', $Cache, '--split_key', 'geom_mol_idx',
        '--batch_size', '128', '--noise_min', '0', '--noise_max', '0.15',
        '--noise_clean_prob', '0', '--cutoff', '2.5', '--h_cutoff', '2.5',
        '--num_workers', '0', '--device', 'cuda',
        '--e2e_val_every', '5', '--e2e_val_max_mols', '8192',
        '--save_every', '5', '--selection_noise_levels', '0', '0.1',
        '--selection_noise_seed', '20260921', '--seed', "$Seed"
    ) + $Resume
    if ($DryRun) {
        Write-Host "$Python $($TrainArgs -join ' ')"
        continue
    }
    & $Python @TrainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Heavy-only one-stage training failed for seed $Seed with exit code $LASTEXITCODE"
    }
}
Write-Host '[DONE] v3 heavy-only one-stage three-seed training complete.'
