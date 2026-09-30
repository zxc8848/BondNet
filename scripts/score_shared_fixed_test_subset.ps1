param([string]$Python = 'D:\minconda\python.exe')

$ErrorActionPreference = 'Stop'
foreach ($Condition in @('clean', 'noisy')) {
    $Sdf = "data/shared_bench_$Condition.sdf"
    if (-not (Test-Path -LiteralPath $Sdf)) { throw "Missing SDF: $Sdf" }
    foreach ($Method in @('yuelbond', 'rdkit', 'openbabel')) {
        $Pred = "data/$($Method)_shared_bench_$($Condition)_preds.json"
        $Output = "results/p0c/joint_v3/$($Method)_$($Condition)_fixedtest478_score.json"
        if (-not (Test-Path -LiteralPath $Pred)) { throw "Missing predictions: $Pred" }
        $Args = @(
            'scripts/p0c_yuelbond_compare.py', 'score',
            '--sdf', $Sdf, '--pred', $Pred,
            '--name', $Method, '--subset-split', 'test',
            '--output', $Output
        )
        Write-Host "[SCORE] $Method $Condition, fixed-test overlap only"
        & $Python @Args
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Output)) {
            throw "Scoring failed: $Method $Condition"
        }
        $Row = Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
        if ([int]$Row.n_molecules -ne 478 -or [string]$Row.fixed_split_subset -ne 'test') {
            throw "Unexpected overlap subset: $Output"
        }
    }
}
