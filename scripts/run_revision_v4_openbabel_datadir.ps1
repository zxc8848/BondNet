param(
    [string]$Python = 'D:\minconda\python.exe',
    [string]$DataDir = 'D:\minconda\share\openbabel',
    [switch]$DryRun
)

# Re-run every Open Babel baseline with its data files available.
# The primary runs had BABEL_DATADIR unset, so PerceiveBondOrders could not read
# bondtyp.txt and fell back to built-in tables (clean GEOM HH exact 64% instead
# of ~93%; see results/revision_v4_openbabel_datadir_check). Same code, SDFs and
# scorer as the primary runs; outputs go to new directories, nothing is overwritten.

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath (Join-Path $DataDir 'bondtyp.txt'))) { throw "bondtyp.txt not found in $DataDir" }
$env:BABEL_DATADIR = $DataDir
$Out = 'results/revision_v4_openbabel_fixed'

function Invoke-Py([string[]]$A) {
    if ($DryRun) { Write-Host "$Python $($A -join ' ')"; return }
    & $Python @A
    if ($LASTEXITCODE -ne 0) { throw "failed: $($A -join ' ')" }
}

# Record the environment actually seen by Python.
Invoke-Py @('-c', "import os,json; from openbabel import openbabel as ob; print(json.dumps({'OBReleaseVersion': ob.OBReleaseVersion(), 'BABEL_DATADIR': os.environ.get('BABEL_DATADIR')}))")

# 1. GEOM fixed test and frozen PubChem3D cohort (primary graph-scoring protocol).
foreach ($C in @(@('geom', 'data/robustness_27240_keyed'), @('external', 'data/external_v3'))) {
    $R = "$Out/$($C[0])"
    Invoke-Py @('scripts/p0d_unified_robustness.py', 'run', '--data-dir', $C[1], '--results-dir', $R,
        '--sigmas', '0', '0.1', '0.2', '--methods', 'openbabel')
    Invoke-Py @('scripts/p0d_unified_robustness.py', 'table', '--results-dir', $R, '--output', "$R/summary.csv")
}

# 2. 478-molecule shared-SDF comparison (YuelBond section). Predictions are written
#    next to a copy of each SDF so the archived prediction files stay untouched.
$Shared = "$Out/shared"
New-Item -ItemType Directory -Force -Path $Shared | Out-Null
foreach ($Condition in @('clean', 'noisy')) {
    $Src = "data/shared_bench_$Condition.sdf"
    $Sdf = "$Shared/shared_bench_$Condition.sdf"
    if (-not (Test-Path -LiteralPath $Sdf)) { Copy-Item -LiteralPath $Src -Destination $Sdf }
    Invoke-Py @('scripts/p0c_yuelbond_compare.py', 'rule', '--sdf', $Sdf, '--methods', 'openbabel')
    Invoke-Py @('scripts/p0c_yuelbond_compare.py', 'score', '--sdf', $Sdf,
        '--pred', "$Shared/openbabel_shared_bench_$($Condition)_preds.json",
        '--name', 'openbabel', '--subset-split', 'test',
        '--output', "$Shared/openbabel_$($Condition)_fixedtest478_score.json")
}
Write-Host "[DONE] Open Babel baselines with data files -> $Out"
