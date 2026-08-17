[CmdletBinding()]
param(
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$brainRoot = Join-Path $repoRoot 'brain'
$script = Join-Path $PSScriptRoot 'build_baseline_architecture_manifest.py'
$output = Join-Path $brainRoot 'configs\baseline-architecture-manifest.json'
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python 3.11 was not found at the configured machine path: $python"
}
if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
    throw "The architecture-manifest builder is missing: $script"
}

$previousManifestSha256 = $null
if (Test-Path -LiteralPath $output -PathType Leaf) {
    $previous = Get-Content -LiteralPath $output -Raw -Encoding utf8 | ConvertFrom-Json
    $previousManifestSha256 = [string]$previous.manifest_sha256
}

Write-Host "ACTION=update-baseline-architecture-manifest"
Write-Host "OUTPUT=$output"
Write-Host "ISOLATED_PYTHON=-I"
if ($previousManifestSha256) {
    Write-Host "PREVIOUS_MANIFEST_SHA256=$previousManifestSha256"
}
if ($PlanOnly) {
    Write-Host 'PLAN_ONLY=true'
    return
}

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HIP_VISIBLE_DEVICES = '-1'
$env:ROCR_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'
$env:RAYON_NUM_THREADS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

& $python -I $script --output $output
if ($LASTEXITCODE -ne 0) {
    throw "Architecture-manifest builder failed with exit code $LASTEXITCODE."
}

$updated = Get-Content -LiteralPath $output -Raw -Encoding utf8 | ConvertFrom-Json
$manifestSha256 = [string]$updated.manifest_sha256
$sourceBundleSha256 = [string]$updated.implementation.source_bundle_sha256
Write-Host "MANIFEST_SHA256=$manifestSha256"
Write-Host "SOURCE_BUNDLE_SHA256=$sourceBundleSha256"
if ($previousManifestSha256 -and $previousManifestSha256 -ne $manifestSha256) {
    Write-Host 'MANIFEST_IDENTITY=new_comparison'
} elseif ($previousManifestSha256) {
    Write-Host 'MANIFEST_IDENTITY=unchanged'
}
foreach ($name in @($updated.implementation.source_files.PSObject.Properties.Name | Sort-Object)) {
    Write-Host ("SOURCE_FILE_SHA256 {0}={1}" -f $name, $updated.implementation.source_files.$name)
}
