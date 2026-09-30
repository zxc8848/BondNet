param([string]$Python = 'D:\minconda\python.exe')

$ErrorActionPreference = 'Stop'
while ($true) {
    $Count = @(Get-ChildItem -LiteralPath 'results/revision_v3_joint_validity' -Recurse -Filter summary.json -ErrorAction SilentlyContinue).Count
    if ($Count -eq 3) { break }
    if (-not (Get-Process -Id 4948 -ErrorAction SilentlyContinue)) {
        throw "Joint validity evaluation stopped with only $Count/3 summaries."
    }
    Start-Sleep -Seconds 20
}
& $Python 'scripts/summarize_revision_v3_joint_secondary.py'
if ($LASTEXITCODE -ne 0) { throw 'Secondary summaries failed validation.' }
& powershell.exe -NoProfile -File 'scripts/run_revision_v3_undirected_audit.ps1' -Python $Python
if ($LASTEXITCODE -ne 0) { throw 'Undirected scoring audit failed.' }
& powershell.exe -NoProfile -File 'scripts/run_revision_v3_runtime.ps1' -Python $Python
if ($LASTEXITCODE -ne 0) { throw 'Joint/staged runtime benchmark failed.' }
Write-Host '[DONE] joint secondary analyses, scoring audit, and runtime benchmark complete.'
