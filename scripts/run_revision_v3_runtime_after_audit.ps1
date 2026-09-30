param([string]$Python = 'D:\minconda\python.exe')

$ErrorActionPreference = 'Stop'
while (Get-Process -Id 5576 -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
}
$Count = @(Get-ChildItem -LiteralPath 'results/revision_v3_undirected_audit' -Recurse -Filter aggregate.json -ErrorAction SilentlyContinue).Count
if ($Count -ne 9) { throw "Scoring audit is incomplete ($Count/9); runtime benchmark will not start." }
& powershell.exe -NoProfile -File 'scripts/run_revision_v3_runtime.ps1' -Python $Python
if ($LASTEXITCODE -ne 0) { throw 'Clean runtime benchmark failed.' }
Write-Host '[DONE] runtime benchmark complete after the scoring audit.'
