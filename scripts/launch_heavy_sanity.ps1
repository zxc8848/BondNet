$ErrorActionPreference = "Stop"

$LogDir = "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "pubchem_h_ablation_heavy_sanity.transcript.log"

Start-Transcript -Path $LogPath -Force | Out-Null
try {
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_pubchem_h_ablation.ps1 `
        -Mode heavy `
        -MaxSamples 200000 `
        -Stage1Epochs 3 `
        -Stage2Epochs 5 `
        -NoiseMax 0.15 `
        -RunSanityEval
}
finally {
    Stop-Transcript | Out-Null
}
