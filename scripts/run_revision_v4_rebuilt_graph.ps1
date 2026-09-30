param(
    [string]$Python = 'D:\minconda\python.exe',
    [string[]]$Cohorts = @('geom', 'external'),
    [switch]$DryRun
)

# Inference-only check of the cached-graph protocol: every input graph
# (3.0 A envelope, message-passing edges, 2.5 A candidates) is rebuilt from the
# noisy coordinates, and the nine selected checkpoints are re-scored with the
# primary scorer. Nothing is trained or selected.

$ErrorActionPreference = 'Stop'
& $Python scripts/build_rebuilt_noisy_caches.py --cohorts @Cohorts
if ($LASTEXITCODE -ne 0) { throw 'Building rebuilt noisy caches failed' }

$Models = [ordered]@{
    joint  = @{ ckpt = 'checkpoints/revision_v2_direction_fixed_lr1e4_seed{0}/one_stage_joint/best_e2e.pt'; rep = 'explicit' }
    heavy  = @{ ckpt = 'checkpoints/revision_v3_one_stage_heavy_seed{0}/best_e2e.pt'; rep = 'heavy' }
    staged = @{ ckpt = 'checkpoints/revision_v2_seed{0}/explicit_stage1/best_e2e.pt'; rep = 'explicit';
                stage2 = 'checkpoints/revision_v2_seed{0}/explicit_stage2/best_e2e.pt' }
}
foreach ($Cohort in $Cohorts) {
    foreach ($Tag in @('000', '010', '020')) {
        foreach ($Name in $Models.Keys) {
            foreach ($Seed in @(42, 43, 44)) {
                $M = $Models[$Name]
                $Cache = "data/rebuilt_noisy_v4/$Cohort/$($M.rep)_sigma_$Tag.pt"
                $Out = "results/revision_v4_rebuilt_graph/$Cohort/$Name/seed$Seed/sigma_$Tag"
                if (Test-Path -LiteralPath "$Out/aggregate.json") { Write-Host "[SKIP] $Out"; continue }
                $Args = @(
                    'scripts/export_per_molecule_stats.py',
                    '--checkpoint', ($M.ckpt -f $Seed), '--cache', $Cache,
                    '--output_dir', $Out, '--noise_levels', '0',
                    '--batch_size', '128', '--cutoff', '2.5', '--h_cutoff', '2.5',
                    '--conn_threshold', '0.5', '--device', 'cuda'
                )
                if ($M.stage2) { $Args += @('--stage2_ckpt', ($M.stage2 -f $Seed)) }
                Write-Host "[RUN] $Cohort $Name seed=$Seed sigma=$Tag"
                if ($DryRun) { Write-Host "$Python $($Args -join ' ')"; continue }
                & $Python @Args
                if ($LASTEXITCODE -ne 0) { throw "failed: $Out" }
            }
        }
    }
}
if (-not $DryRun) {
    & $Python scripts/summarize_revision_v4_rebuilt_graph.py
    if ($LASTEXITCODE -ne 0) { throw 'summary failed' }
}
