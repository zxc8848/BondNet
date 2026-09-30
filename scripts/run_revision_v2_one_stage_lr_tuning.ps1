param(
    [string]$Python = 'D:\minconda\python.exe',
    [ValidateSet('1e-4', '3e-5')]
    [string]$Rate = '1e-4',
    [ValidateSet(42, 43, 44)]
    [int]$Seed = 42
)

$ErrorActionPreference = 'Stop'
$RateValue = if ($Rate -eq '1e-4') { '0.0001' } else { '0.00003' }
$RateTag = if ($Rate -eq '1e-4') { 'lr1e4' } else { 'lr3e5' }
$Output = "checkpoints/revision_v2_one_stage_${RateTag}_seed$Seed"

if (Test-Path -LiteralPath $Output) {
    $Complete = (Test-Path -LiteralPath "$Output/epoch_0050.pt") -and
        (Test-Path -LiteralPath "$Output/train.log") -and
        (Select-String -LiteralPath "$Output/train.log" -Pattern 'Done\. Best val loss:' -Quiet)
    if ($Complete) {
        Write-Host "[SKIP] completed $Output"
        exit 0
    }
    throw "Partial run exists at $Output; inspect it before an explicit resume."
}

Write-Host "[RUN] one-stage seed=$Seed lr=$RateValue output=$Output"
& $Python train.py `
    --explicit_h --hidden_size 360 --epochs 50 --lr $RateValue `
    --output_dir $Output `
    --cache_path data/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed.pt `
    --split_key geom_mol_idx --batch_size 128 `
    --noise_min 0 --noise_max 0.15 --noise_clean_prob 0 `
    --cutoff 2.5 --h_cutoff 2.5 --num_workers 0 --device cuda `
    --e2e_val_every 5 --e2e_val_max_mols 8192 --save_every 10 `
    --selection_noise_levels 0 0.1 --selection_noise_seed 20260921 `
    --seed $Seed
if ($LASTEXITCODE -ne 0) {
    throw "One-stage learning-rate run failed with exit code $LASTEXITCODE"
}
Write-Host "[DONE] $Output"
