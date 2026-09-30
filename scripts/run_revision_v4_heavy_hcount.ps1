param(
    [string]$Python = 'D:\minconda\python.exe',
    [int[]]$Seeds = @(42, 43, 44),
    [ValidateSet('build', 'train', 'eval', 'all')]
    [string]$Stage = 'all',
    [switch]$DryRun
)

# ORACLE diagnostic: heavy-only joint model whose atom tokens encode the number
# of attached H atoms taken from the reference graph. Same recipe as the
# heavy-only joint model (scripts/run_revision_v3_onestage_heavy.ps1); only the
# atom token differs. Not a deployable input.

$ErrorActionPreference = 'Stop'
$Dir = 'data/revision_v4_hcount'
$TrainCache = "$Dir/geom_drugs_all_random1_re10_c30_heavy_hcount_oracle_fixed.pt"
$TestCache = "$Dir/geom_fixed_test_c30_heavy_hcount_oracle.pt"
$ExtCache = "$Dir/external_v3_c30_heavy_hcount_oracle.pt"

function Invoke-Py([string[]]$A) {
    if ($DryRun) { Write-Host "$Python $($A -join ' ')"; return }
    & $Python @A
    if ($LASTEXITCODE -ne 0) { throw "failed: $($A -join ' ')" }
}

if ($Stage -in @('build', 'all')) {
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    $Jobs = @(
        @($TrainCache, 'data/geom_drugs_all_random1_re10_c30_heavy_fixed.pt', 'data/geom_drugs_all_random1_rel10.sdf'),
        @($TestCache, 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_heavy_fixed_test.pt', 'data/geom_drugs_all_random1_rel10.sdf'),
        @($ExtCache, 'data/external_v3/cache_heavy_c30.pt', 'data/external_v3/cohort.sdf')
    )
    foreach ($J in $Jobs) {
        if (Test-Path -LiteralPath $J[0]) { Write-Host "[SKIP] $($J[0])"; continue }
        Invoke-Py @('scripts/build_heavy_hcount_cache.py', '--cache', $J[1], '--sdf', $J[2], '--output', $J[0])
    }
}

if ($Stage -in @('train', 'all')) {
    foreach ($Seed in $Seeds) {
        $Output = "checkpoints/revision_v4_heavy_hcount_oracle_seed$Seed"
        $Resume = @()
        if (Test-Path -LiteralPath $Output) {
            $Done = (Test-Path -LiteralPath "$Output/epoch_0050.pt") -and
                (Select-String -LiteralPath "$Output/train.log" -Pattern 'Done\. Best val loss:' -Quiet)
            if ($Done) { Write-Host "[SKIP] seed $Seed trained"; continue }
            $Latest = Get-ChildItem -LiteralPath $Output -Filter 'epoch_*.pt' -File |
                Sort-Object Name -Descending | Select-Object -First 1
            if ($null -eq $Latest) { throw "Partial run without periodic checkpoint at $Output" }
            $Resume = @('--resume', $Latest.FullName)
        }
        Invoke-Py (@(
            'train.py', '--no_explicit_h', '--hidden_size', '360',
            '--epochs', '50', '--lr', '0.0001', '--output_dir', $Output,
            '--cache_path', $TrainCache, '--split_key', 'geom_mol_idx',
            '--batch_size', '128', '--noise_min', '0', '--noise_max', '0.15',
            '--noise_clean_prob', '0', '--cutoff', '2.5', '--h_cutoff', '2.5',
            '--num_workers', '0', '--device', 'cuda',
            '--e2e_val_every', '5', '--e2e_val_max_mols', '8192',
            '--save_every', '5', '--selection_noise_levels', '0', '0.1',
            '--selection_noise_seed', '20260921', '--seed', "$Seed"
        ) + $Resume)
    }
}

if ($Stage -in @('eval', 'all')) {
    foreach ($Seed in $Seeds) {
        $Ckpt = "checkpoints/revision_v4_heavy_hcount_oracle_seed$Seed/best_e2e.pt"
        foreach ($C in @(@('geom', $TestCache), @('external', $ExtCache))) {
            $Out = "results/revision_v4_heavy_hcount/$($C[0])/seed$Seed"
            if (Test-Path -LiteralPath "$Out/aggregate.json") { Write-Host "[SKIP] $Out"; continue }
            Invoke-Py @('scripts/export_per_molecule_stats.py', '--checkpoint', $Ckpt, '--cache', $C[1],
                '--output_dir', $Out, '--noise_levels', '0', '0.1', '0.2',
                '--eval_noise_seed', '20260921', '--batch_size', '128',
                '--cutoff', '2.5', '--h_cutoff', '2.5', '--conn_threshold', '0.5', '--device', 'cuda')
        }
    }
    if (-not $DryRun) { Invoke-Py @('scripts/summarize_revision_v4_heavy_hcount.py') }
}
