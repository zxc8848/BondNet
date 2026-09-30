param(
    [string]$Python = "D:\minconda\python.exe",
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$TrainOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ExplicitCache = "data\geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed.pt"
$HeavyCache = "data\geom_drugs_all_random1_re10_c30_heavy_fixed.pt"
$ExplicitTest = "data\fixed_split_caches\geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt"
$HeavyTest = "data\fixed_split_caches\geom_drugs_all_random1_re10_c30_heavy_fixed_test.pt"
$EvalSeed = 20260921

function Invoke-Step {
    param([string]$Marker, [string[]]$Arguments)
    if (Test-Path -LiteralPath $Marker) {
        Write-Host "SKIP completed: $Marker"
        return
    }
    Write-Host "RUN: $Python $($Arguments -join ' ')"
    if ($DryRun) { return }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
    New-Item -ItemType File -Path $Marker -Force | Out-Null
}

function Get-ResumeArgs {
    param([string]$Directory)
    if (-not (Test-Path -LiteralPath $Directory)) { return @() }
    $Latest = Get-ChildItem -LiteralPath $Directory -Filter "epoch_*.pt" -File |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if ($null -eq $Latest) { return @() }
    Write-Host "RESUME: $($Latest.FullName)"
    return @("--resume", $Latest.FullName)
}

foreach ($Seed in $Seeds) {
    $Root = "checkpoints\revision_v2_seed$Seed"
    $ResultRoot = "results\revision_v2_seed$Seed"
    New-Item -ItemType Directory -Path $Root, $ResultRoot -Force | Out-Null

    $Common = @(
        "--cache_path", $ExplicitCache,
        "--split_key", "geom_mol_idx",
        "--batch_size", "128",
        "--noise_min", "0.0", "--noise_max", "0.15",
        "--noise_clean_prob", "0.0",
        "--cutoff", "2.5", "--h_cutoff", "2.5",
        "--num_workers", "0", "--device", "cuda",
        "--e2e_val_every", "5", "--e2e_val_max_mols", "8192",
        "--save_every", "10", "--seed", "$Seed"
    )

    $OneDir = "$Root\one_stage_joint"
    New-Item -ItemType Directory -Path $OneDir -Force | Out-Null
    $OneArgs = @(
        "train.py", "--explicit_h", "--hidden_size", "360",
        "--epochs", "50", "--lr", "0.0003", "--output_dir", $OneDir
    ) + $Common + (Get-ResumeArgs $OneDir)
    Invoke-Step "$OneDir\.complete" $OneArgs

    $Stage1Dir = "$Root\explicit_stage1"
    New-Item -ItemType Directory -Path $Stage1Dir -Force | Out-Null
    $Stage1Args = @(
        "train.py", "--explicit_h", "--connectivity_only", "--hidden_size", "256",
        "--epochs", "25", "--lr", "0.0003", "--output_dir", $Stage1Dir
    ) + $Common + (Get-ResumeArgs $Stage1Dir)
    Invoke-Step "$Stage1Dir\.complete" $Stage1Args

    $Stage2Dir = "$Root\explicit_stage2"
    New-Item -ItemType Directory -Path $Stage2Dir -Force | Out-Null
    $Stage2Args = @(
        "train_stage2.py", "--stage1_ckpt", "$Stage1Dir\best_e2e.pt",
        "--cache_path", $ExplicitCache, "--output_dir", $Stage2Dir,
        "--train_topology", "teacher", "--epochs", "25", "--batch_size", "128",
        "--lr", "0.0001", "--noise_min", "0.0", "--noise_max", "0.15",
        "--noise_clean_prob", "0.0", "--split_key", "geom_mol_idx",
        "--seed", "$Seed", "--num_workers", "0", "--device", "cuda",
        "--e2e_val_every", "5", "--e2e_val_max_mols", "8192", "--save_every", "10"
    ) + (Get-ResumeArgs $Stage2Dir)
    Invoke-Step "$Stage2Dir\.complete" $Stage2Args

    $HeavyStage1Dir = "$Root\heavy_stage1"
    New-Item -ItemType Directory -Path $HeavyStage1Dir -Force | Out-Null
    $HeavyCommon = $Common.Clone()
    $HeavyCommon[1] = $HeavyCache
    $HeavyStage1Args = @(
        "train.py", "--no_explicit_h", "--connectivity_only", "--hidden_size", "256",
        "--epochs", "25", "--lr", "0.0003", "--output_dir", $HeavyStage1Dir
    ) + $HeavyCommon + (Get-ResumeArgs $HeavyStage1Dir)
    Invoke-Step "$HeavyStage1Dir\.complete" $HeavyStage1Args

    $HeavyStage2Dir = "$Root\heavy_stage2"
    New-Item -ItemType Directory -Path $HeavyStage2Dir -Force | Out-Null
    $HeavyStage2Args = @(
        "train_stage2.py", "--stage1_ckpt", "$HeavyStage1Dir\best_e2e.pt",
        "--cache_path", $HeavyCache, "--output_dir", $HeavyStage2Dir,
        "--train_topology", "teacher", "--epochs", "25", "--batch_size", "128",
        "--lr", "0.0001", "--noise_min", "0.0", "--noise_max", "0.15",
        "--noise_clean_prob", "0.0", "--split_key", "geom_mol_idx",
        "--seed", "$Seed", "--num_workers", "0", "--device", "cuda",
        "--e2e_val_every", "5", "--e2e_val_max_mols", "8192", "--save_every", "10"
    ) + (Get-ResumeArgs $HeavyStage2Dir)
    Invoke-Step "$HeavyStage2Dir\.complete" $HeavyStage2Args

    if (-not $TrainOnly) {
        $EvalCommon = @(
            "--noise_levels", "0.0", "0.1", "0.2",
            "--eval_noise_seed", "$EvalSeed", "--batch_size", "128",
            "--device", "cuda", "--hh_only", "--cutoff", "2.5", "--h_cutoff", "2.5"
        )
        $OneResult = "$ResultRoot\one_stage"
        New-Item -ItemType Directory -Path $OneResult -Force | Out-Null
        Invoke-Step "$OneResult\.complete" (@(
            "evaluate.py", "--checkpoint", "$OneDir\best_e2e.pt",
            "--cache_path", $ExplicitTest, "--output_dir", $OneResult
        ) + $EvalCommon)

        $ExplicitResult = "$ResultRoot\explicit_two_stage"
        New-Item -ItemType Directory -Path $ExplicitResult -Force | Out-Null
        Invoke-Step "$ExplicitResult\.complete" (@(
            "evaluate.py", "--checkpoint", "$Stage1Dir\best_e2e.pt",
            "--stage2_ckpt", "$Stage2Dir\best_e2e.pt", "--allow_stage2_cache_mismatch",
            "--cache_path", $ExplicitTest, "--output_dir", $ExplicitResult
        ) + $EvalCommon)

        $HeavyResult = "$ResultRoot\heavy_two_stage"
        New-Item -ItemType Directory -Path $HeavyResult -Force | Out-Null
        Invoke-Step "$HeavyResult\.complete" (@(
            "evaluate.py", "--checkpoint", "$HeavyStage1Dir\best_e2e.pt",
            "--stage2_ckpt", "$HeavyStage2Dir\best_e2e.pt", "--allow_stage2_cache_mismatch",
            "--cache_path", $HeavyTest, "--no_explicit_h", "--output_dir", $HeavyResult
        ) + $EvalCommon)
    }
}
