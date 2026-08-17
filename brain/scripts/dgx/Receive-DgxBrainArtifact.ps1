[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

    [Parameter(Mandatory)]
    [string]$RemoteWorkDir,

    [Parameter(Mandatory)]
    [string]$RunId,

    [Parameter(Mandatory)]
    [ValidateSet('Logs', 'Checkpoints', 'ResumeAttempts')]
    [string]$Kind,

    [Parameter(Mandatory)]
    [string]$LocalDestination,

    [switch]$HistoricalIreneWorkspace,

    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

$workspaceAccess = if ($HistoricalIreneWorkspace) {
    'HistoricalIreneRead'
} else {
    'PseudoBrainWrite'
}
$remoteAction = if ($HistoricalIreneWorkspace) { 'historical_artifact' } else { 'artifact' }
Assert-DgxRemoteWorkDir -RemoteWorkDir $RemoteWorkDir -Access $workspaceAccess
Assert-DgxSlug -Value $RunId -Label 'RunId'
$kindValue = switch ($Kind) {
    'Logs' { 'logs' }
    'Checkpoints' { 'checkpoints' }
    'ResumeAttempts' { 'resume-attempts' }
}
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$destinationInput = if ([IO.Path]::IsPathRooted($LocalDestination)) {
    $LocalDestination
} else {
    Join-Path $repoRoot $LocalDestination
}
$destination = [IO.Path]::GetFullPath($destinationInput)
$repoPrefix = $repoRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
if (-not $destination.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "LocalDestination must remain inside the Pseudo-Brain workspace: $repoRoot"
}
$destinationRoot = [IO.Path]::GetPathRoot($destination)
if ($destination.TrimEnd('\', '/') -eq $destinationRoot.TrimEnd('\', '/')) {
    throw 'LocalDestination cannot be a drive or filesystem root.'
}
if (Test-Path -LiteralPath $destination) {
    throw "LocalDestination already exists; refusing to merge or overwrite: $destination"
}
$destinationParent = Split-Path -Parent $destination
if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) {
    throw "LocalDestination parent does not exist: $destinationParent"
}
$cursor = (Resolve-Path -LiteralPath $destinationParent).Path
while ($true) {
    $item = Get-Item -LiteralPath $cursor -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "LocalDestination parent chain cannot contain a reparse point: $cursor"
    }
    if ($cursor.Equals($repoRoot, [StringComparison]::OrdinalIgnoreCase)) {
        break
    }
    if (-not $cursor.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'LocalDestination parent escaped the Pseudo-Brain workspace.'
    }
    $cursor = Split-Path -Parent $cursor
}

$target = Resolve-DgxSshTarget -SshTarget $SshTarget

if ($PlanOnly) {
    Write-Host "PLAN remote=$SshTarget run=$RunId artifact=$kindValue access=$workspaceAccess destination=$destination"
    return
}

$artifactProbe = @(Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @($remoteAction, $RemoteWorkDir, $RunId, $kindValue))
$artifactPathLines = @($artifactProbe | Where-Object { $_ -match '^artifact_path=(/[A-Za-z0-9._/-]+)$' })
if ($artifactPathLines.Count -ne 1) {
    throw 'Remote artifact probe did not return one canonical, absolute artifact path.'
}
$canonicalRemotePath = ([string]$artifactPathLines[0]).Substring('artifact_path='.Length)

New-Item -ItemType Directory -Path $destination -ErrorAction Stop | Out-Null
$scp = Get-DgxApplication -Name 'scp'
$scpArguments = Get-DgxScpArguments
switch ($kindValue) {
    'logs' {
        $remoteSource = '{0}:{1}' -f $target.Target, $canonicalRemotePath
        & $scp @scpArguments $remoteSource $destination
    }
    'checkpoints' {
        $remoteSource = '{0}:{1}' -f $target.Target, $canonicalRemotePath
        & $scp @scpArguments -r $remoteSource $destination
    }
    'resume-attempts' {
        $remoteSource = '{0}:{1}' -f $target.Target, $canonicalRemotePath
        & $scp @scpArguments -r $remoteSource $destination
    }
}
if ($LASTEXITCODE -ne 0) {
    throw "Artifact retrieval failed with exit code $LASTEXITCODE. Partial output, if any, remains at $destination."
}
Write-Host "ARTIFACT_DESTINATION=$destination"
