param(
    [string]$Python = 'python',
    [int[]]$Seeds = @(43, 44)
)

$ErrorActionPreference = 'Stop'
$TestCache = 'data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt'
$ReferenceSdf = 'data/robustness_27240_keyed/sigma_000.sdf'

foreach ($Seed in $Seeds) {
    $Stage1 = "checkpoints/revision_v2_seed$Seed/explicit_stage1/best_e2e.pt"
    $Stage2 = "checkpoints/revision_v2_seed$Seed/explicit_stage2/best_e2e.pt"
    $ValidityOutput = "results/revision_v2_chemical_validity/seed$Seed"
    $DistortionOutput = "results/revision_v2_structured_distortions/seed$Seed"
    Write-Host "V2 secondary analysis seed ${Seed}: chemical validity"
    & $Python scripts/evaluate_chemical_validity.py `
        --stage1 $Stage1 --stage2 $Stage2 --cache $TestCache `
        --reference_sdf $ReferenceSdf --output_dir $ValidityOutput `
        --device cuda --batch_size 128
    if ($LASTEXITCODE -ne 0) { throw "Chemical validity failed for seed $Seed" }
    Write-Host "V2 secondary analysis seed ${Seed}: structured distortions"
    & $Python scripts/evaluate_structured_distortions.py `
        --cache $TestCache --checkpoint $Stage1 --stage2_ckpt $Stage2 `
        --output_dir $DistortionOutput --device cuda --batch_size 128
    if ($LASTEXITCODE -ne 0) { throw "Structured distortion failed for seed $Seed" }
}
Write-Host 'V2 secondary analyses completed.'
