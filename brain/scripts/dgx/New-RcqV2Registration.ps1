[CmdletBinding()]
param(
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$git = Get-DgxApplication -Name 'git'
$gitTopLevel = @(& $git -C $repoRoot rev-parse --show-toplevel)
if ($LASTEXITCODE -ne 0 -or $gitTopLevel.Count -ne 1 -or
    -not [IO.Path]::GetFullPath(([string]$gitTopLevel[0]).Trim()).Equals(
        $repoRoot,
        [StringComparison]::OrdinalIgnoreCase
    )) {
    throw 'Registration must run from the canonical Pseudo-Brain Git root.'
}
$marker = Join-Path $repoRoot '.pseudo-brain-workspace-v2'
if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
    throw 'The canonical Pseudo-Brain workspace marker is missing.'
}
$markerItem = Get-Item -LiteralPath $marker -Force
if (($markerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
    (Get-Content -LiteralPath $marker -Raw) -ne "pseudo-brain-workspace-v2`n") {
    throw 'The canonical Pseudo-Brain workspace marker is linked or has non-exact bytes.'
}

$sourceRoot = Join-Path $repoRoot 'brain\src'
$packageRoot = Join-Path $sourceRoot 'irene_brain'
$sourceEntries = @(Get-ChildItem -LiteralPath $sourceRoot -Force)
if ($sourceEntries.Count -ne 1 -or
    $sourceEntries[0] -isnot [IO.DirectoryInfo] -or
    -not $sourceEntries[0].Name.Equals('irene_brain', [StringComparison]::Ordinal) -or
    ($sourceEntries[0].Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw 'brain/src must contain exactly the regular irene_brain directory before registration.'
}
foreach ($entry in Get-ChildItem -LiteralPath $packageRoot -Force -Recurse) {
    if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        ($entry.PSIsContainer -and $entry.Name -eq '__pycache__') -or
        (-not $entry.PSIsContainer -and
            ($entry -isnot [IO.FileInfo] -or -not $entry.Extension.Equals('.py', [StringComparison]::Ordinal)))) {
        throw "The registration source tree contains a linked, cached, or non-.py entry: '$($entry.FullName)'"
    }
}

$config = Join-Path $repoRoot 'brain\configs\training\dgx-rcq-v2-reference.toml'
$registration = Join-Path $repoRoot 'registrations\rcq-v2-reference-v1.json'
if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw 'The fixed RCQ-v2 reference configuration is missing.'
}
if (Test-Path -LiteralPath $registration) {
    throw 'The fixed registration already exists; replacement is forbidden.'
}

$python = Get-DgxApplication -Name 'python'
# Windows PowerShell strips double quotes from an unquoted -c one-liner, which
# turns the module name into a NameError. A single-quoted here-string keeps the
# Python quotes intact and still runs under python -I.
$bootstrap = @'
import runpy, sys
source, root, config = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, source)
sys.argv = [
    "irene_brain.evaluation.rcq_v2_registration",
    "--training-release-root",
    root,
    "--config",
    config,
]
runpy.run_module("irene_brain.evaluation.rcq_v2_registration", run_name="__main__")
'@
if ($PlanOnly) {
    Write-Host 'PLAN action=create-target-blind-rcq-v2-registration output=registrations/rcq-v2-reference-v1.json isolated_python=-I'
    return
}

& $python -I -c $bootstrap $sourceRoot $repoRoot $config
if ($LASTEXITCODE -ne 0) {
    throw "Target-blind RCQ-v2 registration failed with exit code $LASTEXITCODE."
}
if (-not (Test-Path -LiteralPath $registration -PathType Leaf)) {
    throw 'The target-blind builder did not publish the fixed registration.'
}
$published = Get-Item -LiteralPath $registration -Force
if (($published.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw 'The published registration is a reparse point.'
}
$registrationSha256 = (Get-FileHash -LiteralPath $registration -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "REGISTRATION_SHA256=$registrationSha256"
Write-Host 'EXTERNAL_PIN_REQUIRED=true'
