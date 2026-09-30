param(
    [string]$Python = 'D:\minconda\python.exe',
    [ValidateSet('joint', 'heavy', 'staged')]
    [string[]]$Models = @('joint', 'heavy', 'staged'),
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$ExplicitCache = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt'
$HeavyCache = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_heavy_fixed_test.pt'
foreach ($Model in $Models) {
    foreach ($Seed in $Seeds) {
        $Cache = if ($Model -eq 'heavy') { $HeavyCache } else { $ExplicitCache }
        $Checkpoint = switch ($Model) {
            'joint' { "checkpoints/revision_v2_direction_fixed_lr1e4_seed$Seed/one_stage_joint/best_e2e.pt" }
            'heavy' { "checkpoints/revision_v3_one_stage_heavy_seed$Seed/best_e2e.pt" }
            'staged' { "checkpoints/revision_v2_seed$Seed/explicit_stage1/best_e2e.pt" }
        }
        $Stage2 = if ($Model -eq 'staged') { "checkpoints/revision_v2_seed$Seed/explicit_stage2/best_e2e.pt" } else { $null }
        $PrimaryResult = switch ($Model) {
            'joint' { "results/revision_v2_direction_fixed_test/seed$Seed/one_stage/results.json" }
            'heavy' { "results/revision_v3_one_stage_heavy_test/seed$Seed/results.json" }
            'staged' { "results/revision_v2_direction_fixed_test/seed$Seed/explicit_two_stage/results.json" }
        }
        $Output = "results/revision_v3_undirected_audit/$Model/seed$Seed"
        $Expected = "$Output/aggregate.json"
        foreach ($Path in @($Python, $Cache, $Checkpoint, $PrimaryResult)) {
            if (-not (Test-Path -LiteralPath $Path)) { throw "Required input not found: $Path" }
        }
        if ($Stage2 -and -not (Test-Path -LiteralPath $Stage2)) { throw "Missing Stage 2 checkpoint: $Stage2" }
        if (Test-Path -LiteralPath $Expected) {
            Write-Host "[SKIP] $Model seed=$Seed"
        } else {
            $Args = @(
                'scripts/export_per_molecule_stats.py',
                '--checkpoint', $Checkpoint, '--cache', $Cache,
                '--output_dir', $Output,
                '--noise_levels', '0', '0.1', '0.2',
                '--eval_noise_seed', '20260921',
                '--batch_size', '128', '--cutoff', '2.5', '--h_cutoff', '2.5',
                '--conn_threshold', '0.5', '--device', 'cuda'
            )
            if ($Stage2) { $Args += @('--stage2_ckpt', $Stage2) }
            Write-Host "[RUN] $Model seed=$Seed"
            if ($DryRun) {
                Write-Host "$Python $($Args -join ' ')"
                continue
            }
            & $Python @Args
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Expected)) {
                throw "Undirected audit failed for $Model seed=$Seed"
            }
        }
        if ($DryRun) { continue }
        $Rows = Get-Content $Expected -Raw | ConvertFrom-Json
        $Main = (Get-Content $PrimaryResult -Raw | ConvertFrom-Json).BondNet
        foreach ($Row in $Rows) {
            $Reference = $Main | Where-Object { [double]$_.sigma -eq [double]$Row.sigma }
            if (@($Reference).Count -ne 1) { throw "Missing primary result for $Model seed=$Seed sigma=$($Row.sigma)" }
            $F1Delta = [math]::Abs([double]$Row.pipeline_macro_f1 - [double]$Reference.f1_macro_pipeline)
            $ExactDelta = [math]::Abs([double]$Row.hh_graph_exact_rate - [double]$Reference.full_graph_exact_match)
            Write-Host "[GATE] $Model seed=$Seed sigma=$($Row.sigma) directed deltas: F1=$F1Delta exact=$ExactDelta"
            # The independent per-molecule accumulator can differ by a few
            # 1e-6 in high-noise F1 while exact-molecule counts agree. Preserve
            # the discrepancy in the gate log and reject larger deviations.
            if ($F1Delta -gt 0.00001 -or $ExactDelta -gt 0.000000000001) {
                throw "Audit scorer does not reproduce primary result for $Model seed=$Seed sigma=$($Row.sigma)"
            }
        }
    }
}
if (-not $DryRun) { Write-Host '[DONE] all directed and undirected pair scores audited.' }
