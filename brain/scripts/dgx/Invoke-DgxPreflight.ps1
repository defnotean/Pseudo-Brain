[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

    [Parameter(Mandatory)]
    [string]$RemoteWorkDir,

    [Parameter(Mandatory)]
    [string]$ContainerImage,

    [Parameter(Mandatory)]
    [ValidateRange(1, 4096)]
    [int]$MinFreeDiskGiB,

    [Parameter(Mandatory)]
    [ValidateRange(4, 4096)]
    [int]$MinAvailableMemoryGiB,

    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

Assert-DgxRemoteWorkDir -RemoteWorkDir $RemoteWorkDir
Assert-DgxImageReference -ContainerImage $ContainerImage
$target = Resolve-DgxSshTarget -SshTarget $SshTarget

Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @(
        'preflight',
        $RemoteWorkDir,
        $ContainerImage,
        $MinFreeDiskGiB.ToString(),
        $MinAvailableMemoryGiB.ToString()
    ) `
    -PlanOnly:$PlanOnly
