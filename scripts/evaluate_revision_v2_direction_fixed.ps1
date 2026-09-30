param(
    [string]$Python = 'D:\minconda\python.exe',
    [int[]]$Seeds = @(42, 43, 44),
    [ValidateSet('one_stage', 'explicit_two_stage', 'heavy_two_stage')]
    [string[]]$Arms = @('one_stage', 'explicit_two_stage', 'heavy_two_stage')
)

$ErrorActionPreference = 'Stop'
$ExplicitTest = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt'
$HeavyTest = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_heavy_fixed_test.pt'

foreach ($Seed in $Seeds) {
    $OneDir = "checkpoints/revision_v2_direction_fixed_lr1e4_seed$Seed/one_stage_joint"
    foreach ($Arm in $Arms) {
        if ($Arm -eq 'one_stage' -and
            -not ((Test-Path -LiteralPath "$OneDir/epoch_0050.pt") -and
                  (Test-Path -LiteralPath "$OneDir/train.log") -and
                  (Select-String -LiteralPath "$OneDir/train.log" -Pattern 'Done\. Best val loss:' -Quiet))) {
            throw "Direction-fixed one-stage training is not complete for seed $Seed"
        }
        $Output = "results/revision_v2_direction_fixed_test/seed$Seed/$Arm"
        if (Test-Path -LiteralPath "$Output/results.json") {
            Write-Host "[SKIP] completed seed $Seed arm $Arm"
            continue
        }
        $Common = @(
            '--noise_levels', '0', '0.1', '0.2',
            '--eval_noise_seed', '20260921',
            '--batch_size', '128', '--num_workers', '0', '--device', 'cuda',
            '--eval_split', 'test', '--split_key', 'geom_mol_idx',
            '--cutoff', '2.5', '--h_cutoff', '2.5',
            '--conn_threshold', '0.5', '--hh_only',
            '--output_dir', $Output
        )
        if ($Arm -eq 'one_stage') {
            $Args = @(
                '--checkpoint', "$OneDir/best_e2e.pt",
                '--cache_path', $ExplicitTest, '--explicit_h'
            ) + $Common
        } elseif ($Arm -eq 'explicit_two_stage') {
            $Root = "checkpoints/revision_v2_seed$Seed"
            $Args = @(
                '--checkpoint', "$Root/explicit_stage1/best_e2e.pt",
                '--stage2_ckpt', "$Root/explicit_stage2/best_e2e.pt",
                '--allow_stage2_cache_mismatch',
                '--cache_path', $ExplicitTest, '--explicit_h'
            ) + $Common
        } else {
            $Root = "checkpoints/revision_v2_seed$Seed"
            $Args = @(
                '--checkpoint', "$Root/heavy_stage1/best_e2e.pt",
                '--stage2_ckpt', "$Root/heavy_stage2/best_e2e.pt",
                '--allow_stage2_cache_mismatch',
                '--cache_path', $HeavyTest, '--no_explicit_h'
            ) + $Common
        }
        Write-Host "[EVAL] seed $Seed arm $Arm"
        & $Python evaluate.py @Args
        if ($LASTEXITCODE -ne 0) {
            throw "Direction-fixed evaluation failed for seed $Seed arm $Arm"
        }
    }
}
Write-Host '[DONE] direction-fixed learned-system evaluation complete.'
