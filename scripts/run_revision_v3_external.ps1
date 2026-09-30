param(
    [string]$Python = 'D:\minconda\python.exe',
    [switch]$DryRun
)

# One-shot evaluation on the frozen v3 external cohort (data/external_v3).
# Every model/baseline is run once with the settings recorded in
# data/external_v3/manifest.json. Re-running only fills in missing outputs;
# completed outputs are never overwritten.

$ErrorActionPreference = 'Stop'
$Data = 'data/external_v3'
$Results = 'results/revision_v3_external'
$Manifest = "$Data/manifest.json"
$ExplicitCache = "$Data/cache_explicit_c30_h30.pt"
$HeavyCache = "$Data/cache_heavy_c30.pt"
$Log = "$Results/run_log.txt"

foreach ($Path in @($Python, $Manifest, "$Data/manifest.sha256", $ExplicitCache, $HeavyCache)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required input not found: $Path (run build_external_cohort_v3.py select/materialize/freeze first)" }
}

& $Python scripts/build_external_cohort_v3.py verify
if ($LASTEXITCODE -ne 0) { throw 'Frozen cohort, protocol code or checkpoints changed; refusing to evaluate.' }

New-Item -ItemType Directory -Force -Path $Results | Out-Null
$ManifestHash = ((Get-Content "$Data/manifest.sha256" -Raw).Trim() -split '\s+')[0]
if (-not (Test-Path -LiteralPath "$Results/START.json")) {
    if (-not $DryRun) {
        @{ manifest_sha256 = $ManifestHash; started_utc = (Get-Date).ToUniversalTime().ToString('o') } |
            ConvertTo-Json | Set-Content -Encoding utf8 "$Results/START.json"
    }
} else {
    $Start = Get-Content "$Results/START.json" -Raw | ConvertFrom-Json
    if ($Start.manifest_sha256 -ne $ManifestHash) { throw 'START.json refers to a different manifest.' }
}

function Write-Log([string]$Message) {
    $Line = "$((Get-Date).ToUniversalTime().ToString('o')) $Message"
    Write-Host $Line
    if (-not $DryRun) { Add-Content -Path $Log -Value $Line -Encoding utf8 }
}

function Invoke-Once([string]$Name, [string]$Expected, [string[]]$Arguments) {
    if (Test-Path -LiteralPath $Expected) { Write-Log "[SKIP] $Name (exists: $Expected)"; return }
    Write-Log "[RUN] $Name"
    if ($DryRun) { Write-Host "$Python $($Arguments -join ' ')"; return }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Expected)) {
        throw "$Name failed (exit $LASTEXITCODE)"
    }
    Write-Log "[DONE] $Name"
}

$Plan = (Get-Content $Manifest -Raw | ConvertFrom-Json).evaluation_plan.models
foreach ($Seed in @(42, 43, 44)) {
    foreach ($Model in @('joint', 'heavy', 'staged')) {
        $Entry = $Plan."${Model}_seed$Seed"
        if (-not $Entry) { throw "Manifest has no entry ${Model}_seed$Seed" }
        $Output = "$Results/$Model/seed$Seed"
        $Arguments = @(
            'scripts/export_per_molecule_stats.py',
            '--checkpoint', $Entry.model_file.path, '--cache', $Entry.cache,
            '--output_dir', $Output,
            '--noise_levels', '0', '0.1', '0.2',
            '--eval_noise_seed', '20260921',
            '--batch_size', '128', '--cutoff', '2.5', '--h_cutoff', '2.5',
            '--conn_threshold', '0.5', '--device', 'cuda'
        )
        if ($Entry.stage2_file) { $Arguments += @('--stage2_ckpt', $Entry.stage2_file.path) }
        Invoke-Once "$Model seed=$Seed" "$Output/aggregate.json" $Arguments
    }
    $ValidityOut = "$Results/joint_validity/seed$Seed"
    Invoke-Once "joint validity seed=$Seed" "$ValidityOut/summary.json" @(
        'scripts/evaluate_chemical_validity.py',
        '--checkpoint', $Plan."joint_seed$Seed".model_file.path, '--cache', $ExplicitCache,
        '--reference_sdf', "$Data/sigma_000.sdf",
        '--output_dir', $ValidityOut, '--device', 'cuda',
        '--noise_levels', '0', '0.1', '0.2',
        '--eval_noise_seed', '20260921',
        '--batch_size', '128', '--cutoff', '2.5', '--h_cutoff', '2.5',
        '--conn_threshold', '0.5'
    )
}

$RuleDone = @('000', '010', '020') | ForEach-Object {
    (Test-Path "$Results/rule_baselines/scores/rdkit_sigma_$_.json") -and
    (Test-Path "$Results/rule_baselines/scores/openbabel_sigma_$_.json")
}
if ($RuleDone -contains $false) {
    Write-Log '[RUN] RDKit and OpenBabel'
    if ($DryRun) {
        Write-Host "$Python scripts/p0d_unified_robustness.py run --data-dir $Data --results-dir $Results/rule_baselines --sigmas 0 0.1 0.2 --methods rdkit openbabel"
    } else {
        & $Python scripts/p0d_unified_robustness.py run --data-dir $Data --results-dir "$Results/rule_baselines" --sigmas 0 0.1 0.2 --methods rdkit openbabel
        if ($LASTEXITCODE -ne 0) { throw 'Rule baselines failed' }
        & $Python scripts/p0d_unified_robustness.py table --results-dir "$Results/rule_baselines"
        if ($LASTEXITCODE -ne 0) { throw 'Rule baseline table failed' }
        Write-Log '[DONE] RDKit and OpenBabel'
    }
} else {
    Write-Log '[SKIP] rule baselines (all scores exist)'
}

if (-not $DryRun) {
    & $Python scripts/summarize_revision_v3_external.py
    if ($LASTEXITCODE -ne 0) { throw 'Summary failed' }
    Write-Log '[DONE] external evaluation complete; see results/revision_v3_external/summary.md'
}
