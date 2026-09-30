param(
    [string]$Python = 'python',
    [int[]]$Seeds = @(42, 43, 44)
)

$ErrorActionPreference = 'Stop'
$TestCache = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt'

foreach ($Seed in $Seeds) {
    $Stage1 = "checkpoints/revision_v2_seed$Seed/explicit_stage1/best_e2e.pt"
    $Stage2 = "checkpoints/revision_v2_predicted_topology_seed$Seed/best_e2e.pt"
    $Output = "results/revision_v2_predicted_topology/seed$Seed"
    if (Test-Path -LiteralPath "$Output/results.json") {
        Write-Host "[SKIP] completed v2 predicted-topology evaluation seed $Seed"
        continue
    }
    Write-Host "[EVAL] v2 predicted-topology seed $Seed"
    & $Python evaluate.py `
        --checkpoint $Stage1 --stage2_ckpt $Stage2 `
        --allow_stage2_cache_mismatch `
        --cache_path $TestCache --explicit_h `
        --cutoff 2.5 --h_cutoff 2.5 `
        --eval_split test --noise_levels 0 0.1 0.2 `
        --output_dir $Output --batch_size 128 --num_workers 0 `
        --device cuda --split_key geom_mol_idx --conn_threshold 0.5 `
        --eval_noise_seed 20260921
    if ($LASTEXITCODE -ne 0) { throw "V2 predicted-topology evaluation failed for seed $Seed" }
}
Write-Host '[DONE] v2 predicted-topology evaluation complete.'
