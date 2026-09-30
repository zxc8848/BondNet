param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingProcessId,
    [string]$Python = 'D:\minconda\python.exe'
)

$ErrorActionPreference = 'Stop'
Write-Host "[WAIT] direction-fixed training process $TrainingProcessId"
Wait-Process -Id $TrainingProcessId

foreach ($Seed in @(42, 43, 44)) {
    $Output = "checkpoints/revision_v2_direction_fixed_lr1e4_seed$Seed/one_stage_joint"
    $Complete = (Test-Path -LiteralPath "$Output/epoch_0050.pt") -and
        (Test-Path -LiteralPath "$Output/train.log") -and
        (Select-String -LiteralPath "$Output/train.log" -Pattern 'Done\. Best val loss:' -Quiet)
    if (-not $Complete) {
        throw "Training stopped before all three seeds completed; refusing test evaluation (seed $Seed)."
    }
}

Write-Host '[EVAL] All three seeds complete; starting fixed-test evaluation.'
& 'E:\zxcproject\bondnet\scripts\evaluate_revision_v2_direction_fixed.ps1' -Python $Python
if ($LASTEXITCODE -ne 0) {
    throw "Direction-fixed test evaluation failed with exit code $LASTEXITCODE"
}
Write-Host '[DONE] direction-fixed training and fixed-test evaluation.'
