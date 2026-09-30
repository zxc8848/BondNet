param(
    [string]$Python = "D:\minconda\python.exe",
    [string]$DataPath = "data\pubchem3d_1M.sdf",
    [int]$MaxSamples = 500000,
    [int]$ShardSize = 20000,
    [int]$Stage1Epochs = 10,
    [int]$Stage2Epochs = 20,
    [double]$NoiseMax = 0.15,
    [ValidateSet("both", "explicit", "heavy")]
    [string]$Mode = "both",
    [switch]$SkipStage1,
    [switch]$SkipStage2,
    [switch]$RunSanityEval
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([string]$Name, [scriptblock]$Command)
    Write-Host ""
    Write-Host "== $Name =="
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Run-OneMode {
    param(
        [string]$Tag,
        [bool]$ExplicitH
    )

    $CachePath = if ($ExplicitH) {
        "data\pubchem3d_1M_c25_explicit_h_sharded"
    } else {
        "data\pubchem3d_1M_c25_heavy_sharded"
    }
    $HFlag = if ($ExplicitH) { "--explicit_h" } else { "--no_explicit_h" }
    $Root = "checkpoints\pubchem_h_ablation"
    $Stage1Out = Join-Path $Root "${Tag}_stage1_conn_n$($NoiseMax)"
    $Stage2Out = Join-Path $Root "${Tag}_stage2_n$($NoiseMax)"
    $Stage1Ckpt = Join-Path $Stage1Out "best_e2e.pt"

    Write-Host ""
    Write-Host "############################################"
    Write-Host "# Mode: $Tag"
    Write-Host "# Cache: $CachePath"
    Write-Host "# MaxSamples: $MaxSamples"
    Write-Host "############################################"

    if (-not $SkipStage1) {
        Invoke-Checked "Stage1 $Tag connectivity" {
            & $Python train.py `
                --data_path $DataPath `
                --cache_path $CachePath `
                --cache_shard_size $ShardSize `
                --cache_max_samples $MaxSamples `
                --output_dir $Stage1Out `
                --cutoff 2.5 `
                --h_cutoff 2.0 `
                $HFlag `
                --connectivity_only `
                --val_split 0.01 `
                --pos_weight 10 `
                --num_interactions 4 `
                --hidden_size 256 `
                --edge_embedding_size 32 `
                --vector_norm_limit 3.0 `
                --dropout 0.1 `
                --epochs $Stage1Epochs `
                --lr 1e-4 `
                --lr_warmup_epochs 0 `
                --min_lr_factor 0.1 `
                --grad_clip 0.25 `
                --weight_decay 1e-4 `
                --noise_min 0.0 `
                --noise_max $NoiseMax `
                --noise_warmup_epochs 3 `
                --batch_size 256 `
                --num_workers 1 `
                --prefetch_factor 2 `
                --shard_cache_size 1 `
                --e2e_val_every 5 `
                --e2e_val_max_mols 1024 `
                --profile_batches 20
        }
    }

    if (-not $SkipStage2) {
        if (-not (Test-Path $Stage1Ckpt)) {
            throw "Stage1 checkpoint not found: $Stage1Ckpt"
        }
        Invoke-Checked "Stage2 $Tag bond typing" {
            & $Python train_stage2.py `
                --stage1_ckpt $Stage1Ckpt `
                --cache_path $CachePath `
                --cache_max_samples $MaxSamples `
                --output_dir $Stage2Out `
                --val_split 0.05 `
                --num_layers 4 `
                --edge_embedding_size 32 `
                --batch_size 256 `
                --epochs $Stage2Epochs `
                --lr 1e-4 `
                --weight_decay 1e-5 `
                --grad_clip 0.5 `
                --noise_min 0.0 `
                --noise_max $NoiseMax `
                --noise_clean_prob 0.3 `
                --num_workers 1 `
                --prefetch_factor 2 `
                --shard_cache_size 1 `
                --e2e_val_every 5 `
                --e2e_val_max_mols 1024 `
                --profile_batches 20
        }
    }

    if ($RunSanityEval) {
        $Stage2Ckpt = Join-Path $Stage2Out "best_e2e.pt"
        if (-not (Test-Path $Stage2Ckpt)) {
            $Stage2Ckpt = Join-Path $Stage2Out "best.pt"
        }
        if (-not (Test-Path $Stage2Ckpt)) {
            throw "Stage2 checkpoint not found under $Stage2Out"
        }
        Invoke-Checked "Eval $Tag PubChem sanity" {
            & $Python evaluate.py `
                --checkpoint $Stage1Ckpt `
                --stage2_ckpt $Stage2Ckpt `
                --cache_path $CachePath `
                $HFlag `
                --max_mols 50000 `
                --noise_levels 0.0 0.05 0.1 0.15 0.2 `
                --batch_size 256 `
                --num_workers 0 `
                --output_dir "results\pubchem_h_ablation_${Tag}"
        }
    }
}

Write-Host "== PubChem H ablation retraining =="
Write-Host "Mode:          $Mode"
Write-Host "MaxSamples:    $MaxSamples"
Write-Host "NoiseMax:      $NoiseMax"
Write-Host "Stage1Epochs:  $Stage1Epochs"
Write-Host "Stage2Epochs:  $Stage2Epochs"

if ($Mode -eq "both" -or $Mode -eq "explicit") {
    Run-OneMode -Tag "explicit_h" -ExplicitH $true
}
if ($Mode -eq "both" -or $Mode -eq "heavy") {
    Run-OneMode -Tag "heavy_only" -ExplicitH $false
}

Write-Host ""
Write-Host "== Done =="
