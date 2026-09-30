param(
    [string]$Python = "D:\minconda\python.exe",
    [ValidateSet("painn", "clof")]
    [string]$Backbone = "painn",
    [switch]$SkipCacheBuild,
    [switch]$SkipTwoStageDiagnostic,
    [switch]$Tune,
    [switch]$NoAutoClsWeights,
    [double]$CleanProb = -1.0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$SourceSdf = "data\geom_drugs_all_random1_rel10.sdf"
$CacheC25H30 = "data\geom_drugs_all_random1_re10_c25_h30_explicit_h_fixed.pt"
$CacheC30H30 = "data\geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed.pt"
$CacheC30Heavy = "data\geom_drugs_all_random1_re10_c30_heavy_fixed.pt"

if (-not $SkipCacheBuild) {
    & $Python scripts\precompute_features.py `
        --data_path $SourceSdf `
        --output $CacheC25H30 `
        --cutoff 2.5 `
        --h_cutoff 3.0 `
        --explicit_h

    & $Python scripts\precompute_features.py `
        --data_path $SourceSdf `
        --output $CacheC30H30 `
        --cutoff 3.0 `
        --h_cutoff 3.0 `
        --explicit_h

    & $Python scripts\precompute_features.py `
        --data_path $SourceSdf `
        --output $CacheC30Heavy `
        --cutoff 3.0 `
        --h_cutoff 3.0
}

$Epochs = "50"
$NoiseTrainMaxes = @("0.0", "0.05", "0.10", "0.15", "0.20")
$NoiseLevels = @("0.0", "0.01", "0.03", "0.05", "0.10", "0.15", "0.20")
$Cutoffs = @("2.5", "3.0")
$E2EValMaxMols = "8192"
$E2EValEvery = "5"
$MainTrainNoiseMax = "0.15"
$TrainNoiseCleanProb = "0.0"
$OutputRoot = "checkpoints/one_stage_main_study_$Backbone"
$ResultsRoot = "results/one_stage_main_study_$Backbone"
$MainCutoff = "3.0"

if ($CleanProb -ge 0.0) {
    $TrainNoiseCleanProb = $CleanProb.ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)
}

if ($Tune) {
    if ($CleanProb -lt 0.0) {
        $TrainNoiseCleanProb = "0.3"
    }
    $Epochs = "10"
    $NoiseTrainMaxes = @("0.0", "0.05")
    $NoiseLevels = @("0.0", "0.05", "0.10")
    $Cutoffs = @("2.5")
    $E2EValMaxMols = "1024"
    $E2EValEvery = "5"
    $MainTrainNoiseMax = "0.05"
    $CleanTag = $TrainNoiseCleanProb.Replace(".", "p")
    $OutputRoot = "checkpoints/one_stage_tune_${Backbone}_cp$CleanTag"
    $ResultsRoot = "results/one_stage_tune_${Backbone}_cp$CleanTag"
    $MainCutoff = "2.5"
}

$runnerArgs = @(
    "scripts\run_one_stage_main_study.py",
    "--python", $Python,
    "--cache_path", $CacheC25H30,
    "--cache_map", "2.5=$CacheC25H30", "3.0=$CacheC30H30",
    "--heavy_cache_path", $CacheC30Heavy,
    "--main_cutoff", $MainCutoff,
    "--h_cutoff", "3.0",
    "--cutoffs"
)
$runnerArgs += $Cutoffs
$runnerArgs += @(
    "--noise_train_maxes"
)
$runnerArgs += $NoiseTrainMaxes
$runnerArgs += @(
    "--main_train_noise_max", $MainTrainNoiseMax,
    "--train_noise_clean_prob", $TrainNoiseCleanProb,
    "--noise_levels"
)
$runnerArgs += $NoiseLevels
$runnerArgs += @(
    "--one_stage_epochs", $Epochs,
    "--stage1_epochs", "20",
    "--stage2_epochs", "50",
    "--batch_size", "256",
    "--hidden_size", "256",
    "--backbone", $Backbone,
    "--num_interactions", "4",
    "--edge_embedding_size", "32",
    "--e2e_val_max_mols", $E2EValMaxMols,
    "--e2e_val_every", $E2EValEvery,
    "--output_root", $OutputRoot,
    "--results_root", $ResultsRoot,
    "--skip-existing",
    "--stop-on-error"
)

if (-not $NoAutoClsWeights) {
    $runnerArgs += "--auto_cls_weights"
}
if ($Tune) {
    $runnerArgs += "--no_run_h_ablation"
    $runnerArgs += "--no_run_cutoff_ablation"
}
if (-not $SkipTwoStageDiagnostic -and -not $Tune) {
    $runnerArgs += "--run_two_stage_diagnostic"
}
if ($DryRun) {
    $runnerArgs += "--dry-run"
}

& $Python @runnerArgs
