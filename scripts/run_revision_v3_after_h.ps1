param([string]$Python = 'D:\minconda\python.exe')

$ErrorActionPreference = 'Stop'
$HRoot = 'results/revision_v3_joint_h_corruption'
$HLog = "$HRoot/run.stderr.log"
while ($true) {
    $Count = @(Get-ChildItem -LiteralPath $HRoot -Recurse -Filter results.json -ErrorAction SilentlyContinue).Count
    if ($Count -eq 21) { break }
    if (-not (Get-Process -Id 31316 -ErrorAction SilentlyContinue)) {
        throw "H-corruption process stopped with only $Count/21 result files; inspect $HLog"
    }
    Start-Sleep -Seconds 20
}
Write-Host '[GATE] all 21 H-corruption results are present.'

& powershell.exe -NoProfile -File 'scripts/run_revision_v3_joint_secondary.ps1' -Python $Python -Seeds 42 -Analyses structured
if ($LASTEXITCODE -ne 0) { throw 'Structured-distortion seed 42 failed.' }

$Structured = Get-Content 'results/revision_v3_joint_structured_directed/seed42/results.json' -Raw | ConvertFrom-Json
$Clean = $Structured | Where-Object { $_.distortion -eq 'clean' }
$Main = Get-Content 'results/revision_v2_direction_fixed_test/seed42/one_stage/results.json' -Raw | ConvertFrom-Json
if (@($Clean).Count -ne 1) { throw 'Expected exactly one clean structured-distortion result.' }
$F1Delta = [math]::Abs([double]$Clean.pipeline_macro_f1 - [double]$Main.BondNet[0].f1_macro_pipeline)
$ExactDelta = [math]::Abs([double]$Clean.hh_graph_exact_rate - [double]$Main.BondNet[0].full_graph_exact_match)
Write-Host "[GATE] clean-control deltas: F1=$F1Delta HH-exact=$ExactDelta"
if ($F1Delta -gt 0.000001 -or $ExactDelta -gt 0.000001) {
    throw 'Structured clean control does not reproduce the primary scorer. Stop and inspect the protocol.'
}

foreach ($Seed in @(43, 44)) {
    & powershell.exe -NoProfile -File 'scripts/run_revision_v3_joint_secondary.ps1' -Python $Python -Seeds $Seed -Analyses structured
    if ($LASTEXITCODE -ne 0) { throw "Structured-distortion seed $Seed failed." }
}
foreach ($Seed in @(42, 43, 44)) {
    & powershell.exe -NoProfile -File 'scripts/run_revision_v3_joint_secondary.ps1' -Python $Python -Seeds $Seed -Analyses validity
    if ($LASTEXITCODE -ne 0) { throw "Joint chemical-validity seed $Seed failed." }
}
Write-Host '[DONE] corrected joint secondary evaluation finished.'
