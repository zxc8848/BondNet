param(
    [int]$TrainingProcessId,
    [string]$Python = 'D:\minconda\python.exe',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

$Checkpoint = 'checkpoints\revision_v2_seed44_rerun\one_stage_joint\best_e2e.pt'
$TrainLog = 'checkpoints\revision_v2_seed44_rerun\one_stage_joint\train.log'
$TrainFinal = 'checkpoints\revision_v2_seed44_rerun\one_stage_joint\epoch_0050.pt'
$EvalDir = 'results\revision_v2_seed44_rerun\one_stage'
$EvalDone = Join-Path $EvalDir '.complete'
$Curve = Join-Path $EvalDir 'robustness_curve.csv'

if ($DryRun) {
    Write-Output "Training PID: $TrainingProcessId"
    Write-Output "Checkpoint: $Checkpoint"
    Write-Output "Evaluation: $EvalDir"
    exit 0
}

if ($TrainingProcessId -le 0) {
    throw 'TrainingProcessId must be a positive process ID.'
}

Write-Output "Waiting for training process $TrainingProcessId"
while (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

if (-not (Test-Path -LiteralPath $TrainFinal)) {
    throw "Training process exited without final epoch checkpoint: $TrainFinal"
}
if (-not (Select-String -LiteralPath $TrainLog -Pattern 'Done. Best val loss:' -SimpleMatch -Quiet)) {
    throw "Training process exited without completion log: $TrainLog"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Missing selected checkpoint: $Checkpoint"
}

New-Item -ItemType Directory -Path $EvalDir -Force | Out-Null
if (-not (Test-Path -LiteralPath $EvalDone)) {
    Write-Output "Evaluating $Checkpoint"
    & $Python evaluate.py `
        --checkpoint $Checkpoint `
        --cache_path data\fixed_split_caches\geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt `
        --output_dir $EvalDir `
        --noise_levels 0.0 0.1 0.2 `
        --eval_noise_seed 20260921 `
        --batch_size 128 `
        --device cuda `
        --hh_only `
        --cutoff 2.5 `
        --h_cutoff 2.5
    if ($LASTEXITCODE -ne 0) {
        throw "Seed-44 one-stage evaluation failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $Curve)) {
        throw "Evaluation exited without robustness curve: $Curve"
    }
    New-Item -ItemType File -Path $EvalDone -Force | Out-Null
}

Write-Output 'Summarizing all revision-v2 evaluations'
& $Python scripts\summarize_revision_v2.py `
    --seed44-one-stage $Curve `
    --output-dir results\revision_v2_final_analysis
if ($LASTEXITCODE -ne 0) {
    throw "Final analysis failed with exit code $LASTEXITCODE"
}
Write-Output 'Seed-44 rerun, evaluation, and final numerical summary completed.'
