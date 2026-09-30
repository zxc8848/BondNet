param(
    [string]$Python = 'D:\minconda\python.exe',
    [int[]]$Seeds = @(42, 43, 44)
)

$ErrorActionPreference = 'Stop'

foreach ($Seed in $Seeds) {
    $Output = "checkpoints/revision_v2_direction_fixed_lr1e4_seed$Seed/one_stage_joint"
    $Resume = @()
    if (Test-Path -LiteralPath $Output) {
        $Completed = (Test-Path -LiteralPath "$Output/epoch_0050.pt") -and
            (Test-Path -LiteralPath "$Output/train.log") -and
            (Select-String -LiteralPath "$Output/train.log" -Pattern 'Done\. Best val loss:' -Quiet)
        if ($Completed) {
            Write-Host "[SKIP] completed seed $Seed"
            continue
        }
        $Latest = Get-ChildItem -LiteralPath $Output -Filter 'epoch_*.pt' -File |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($null -eq $Latest) {
            throw "Partial run without a periodic checkpoint at $Output; inspect before restarting."
        }
        $Resume = @('--resume', $Latest.FullName)
        Write-Host "[RESUME] seed $Seed from $($Latest.FullName)"
    } else {
        Write-Host "[RUN] direction-fixed one-stage seed $Seed"
    }

    & $Python train.py `
        --explicit_h --hidden_size 360 --epochs 50 --lr 0.0001 `
        --output_dir $Output `
        --cache_path data/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed.pt `
        --split_key geom_mol_idx --batch_size 128 `
        --noise_min 0 --noise_max 0.15 --noise_clean_prob 0 `
        --cutoff 2.5 --h_cutoff 2.5 --num_workers 0 --device cuda `
        --e2e_val_every 5 --e2e_val_max_mols 8192 --save_every 5 `
        --selection_noise_levels 0 0.1 --selection_noise_seed 20260921 `
        --seed $Seed @Resume
    if ($LASTEXITCODE -ne 0) {
        throw "Direction-fixed one-stage training failed for seed $Seed with exit code $LASTEXITCODE"
    }
}
Write-Host '[DONE] direction-fixed one-stage three-seed training complete.'
