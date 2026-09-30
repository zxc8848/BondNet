param(
    [string]$Python = 'python',
    [int[]]$Seeds = @(42, 43, 44)
)

$ErrorActionPreference = 'Stop'
$Cache = 'data/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed.pt'

foreach ($Seed in $Seeds) {
    $Stage1 = "checkpoints/revision_v2_seed$Seed/explicit_stage1/best_e2e.pt"
    $Stage2 = "checkpoints/revision_v2_seed$Seed/explicit_stage2/best_e2e.pt"
    $Output = "checkpoints/revision_v2_predicted_topology_seed$Seed"
    if (Test-Path -LiteralPath $Output) {
        $FinalEpoch = Test-Path -LiteralPath "$Output/epoch_0010.pt"
        $DoneLine = if (Test-Path -LiteralPath "$Output/train.log") {
            Select-String -LiteralPath "$Output/train.log" -Pattern 'Done\. Best val loss:' -Quiet
        } else { $false }
        if ($FinalEpoch -and $DoneLine) {
            Write-Host "[SKIP] completed v2 predicted-topology seed $Seed"
            continue
        }
        throw "Partial v2 predicted-topology run exists for seed $Seed at $Output; inspect or resume explicitly."
    }
    Write-Host "[RUN] v2 predicted-topology seed $Seed"
    & $Python train_stage2.py `
        --stage1_ckpt $Stage1 --init_checkpoint $Stage2 `
        --cache_path $Cache --output_dir $Output `
        --train_topology predicted --conn_threshold 0.5 `
        --epochs 10 --batch_size 128 --lr 0.0001 `
        --noise_min 0 --noise_max 0.15 --noise_clean_prob 0 `
        --split_key geom_mol_idx --seed $Seed --device cuda `
        --num_workers 0 --val_every 1 --e2e_val_every 5 `
        --e2e_val_max_mols 8192 --selection_noise_levels 0 0.1 `
        --selection_noise_seed 20260921
    if ($LASTEXITCODE -ne 0) { throw "V2 predicted-topology fine-tuning failed for seed $Seed" }
}
Write-Host '[DONE] v2 predicted-topology fine-tuning complete.'
