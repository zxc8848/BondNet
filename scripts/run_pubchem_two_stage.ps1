param(
    [string]$Python = "D:\minconda\python.exe",
    [string]$DataPath = "data\pubchem3d_1M.sdf",
    [string]$CachePath = "data\pubchem3d_1M_c25_explicit_h_sharded",
    [int]$MaxSamples = 200000,
    [int]$Epochs = 100,
    [double]$NoiseMax = 0.02,
    [switch]$SkipStage1,
    [switch]$SkipStage2
)

$ErrorActionPreference = "Stop"

$Stage1Output = "checkpoints\pubchem3d_1M"
$Stage2Output = "checkpoints\pubchem3d_1M_stage2_c25"
$Stage1Ckpt = Join-Path $Stage1Output "best_e2e.pt"

Write-Host "== BondNet PubChem two-stage run =="
Write-Host "Python:      $Python"
Write-Host "Cache:       $CachePath"
Write-Host "MaxSamples:  $MaxSamples"
Write-Host "NoiseMax:    $NoiseMax"
Write-Host ""

if (-not $SkipStage1) {
    Write-Host "== Stage 1: PaiNN connectivity training =="
    & $Python train.py `
        --data_path $DataPath `
        --output_dir $Stage1Output `
        --cutoff 2.5 `
        --explicit_h `
        --val_split 0.01 `
        --pos_weight 10 `
        --num_interactions 4 `
        --hidden_size 256 `
        --edge_embedding_size 32 `
        --epochs $Epochs `
        --lr 3e-4 `
        --grad_clip 0.5 `
        --weight_decay 1e-4 `
        --dropout 0.1 `
        --cache_path $CachePath `
        --batch_size 256 `
        --num_workers 0 `
        --cache_max_samples $MaxSamples `
        --noise_min 0.0 `
        --noise_max $NoiseMax `
        --shard_cache_size 10 `
        --noise_warmup_epochs 3
}

if (-not $SkipStage2) {
    if (-not (Test-Path $Stage1Ckpt)) {
        throw "Stage 1 checkpoint not found: $Stage1Ckpt"
    }

    Write-Host "== Stage 2: topology bond-type training =="
    & $Python train_stage2.py `
        --stage1_ckpt $Stage1Ckpt `
        --cache_path $CachePath `
        --output_dir $Stage2Output `
        --val_split 0.05 `
        --num_layers 4 `
        --edge_embedding_size 32 `
        --batch_size 256 `
        --epochs $Epochs `
        --lr 5e-4 `
        --grad_clip 1.0 `
        --noise_min 0.0 `
        --noise_max $NoiseMax `
        --cache_max_samples $MaxSamples
}

Write-Host "== Done =="
