[CmdletBinding()]
param(
    [string]$ConfiguredPython = 'C:\Users\Demon\AppData\Local\Programs\Python\Python311\python.exe'
)

$ErrorActionPreference = 'Stop'
$pythonPath = $ConfiguredPython
$brainRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python 3.11 was not found at the configured machine path: $pythonPath"
}

$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'
$env:RAYON_NUM_THREADS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HIP_VISIBLE_DEVICES = '-1'
$env:ROCR_VISIBLE_DEVICES = '-1'
$env:PYTHONPATH = Join-Path $brainRoot 'src'

$currentProcess = [System.Diagnostics.Process]::GetCurrentProcess()
try {
    $currentProcess.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
} catch {
    Write-Warning "Could not lower test process priority: $($_.Exception.Message)"
}

$testsRoot = Join-Path $brainRoot 'tests'
$testFiles = @(Get-ChildItem -LiteralPath $testsRoot -Filter 'test_*.py' -File | Sort-Object Name)
if ($testFiles.Count -eq 0) {
    throw "No play-safe tests were found under $testsRoot"
}

# Each module gets a fresh interpreter. This keeps the Phase-0 import-isolation
# assertion meaningful even when a later opt-in PyTorch test module imports
# torch during unittest discovery.
foreach ($testFile in $testFiles) {
    & $pythonPath -m unittest discover -s $testsRoot -t $brainRoot -p $testFile.Name -v
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
