[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

    [Parameter(Mandatory)]
    [string]$RemoteWorkDir,

    [Parameter(Mandatory)]
    [string]$ReleaseId,

    [Parameter(Mandatory)]
    [string]$ContainerImage,

    [Parameter(Mandatory)]
    [string]$ConfigRelativePath,

    [Parameter(Mandatory)]
    [ValidateRange(1, 4096)]
    [int]$MinFreeDiskGiB,

    [Parameter(Mandatory)]
    [ValidateRange(4, 4096)]
    [int]$MinAvailableMemoryGiB,

    [Parameter(Mandatory)]
    [ValidateRange(1, 20)]
    [int]$ContainerCpuCount,

    [Parameter(Mandatory)]
    [ValidateRange(4, 120)]
    [int]$ContainerMemoryGiB,

    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

Assert-DgxRemoteWorkDir -RemoteWorkDir $RemoteWorkDir
Assert-DgxSlug -Value $ReleaseId -Label 'ReleaseId'
Assert-DgxImageReference -ContainerImage $ContainerImage
Assert-DgxRelativePath -Path $ConfigRelativePath
if ($ConfigRelativePath -eq 'brain/configs/training/dgx-rcq-v2-reference.toml') {
    throw 'the RCQ-v2 reference configuration requires Invoke-DgxRcqV2Smoke.ps1'
}
$target = Resolve-DgxSshTarget -SshTarget $SshTarget

Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @(
        'smoke',
        $RemoteWorkDir,
        $ReleaseId,
        $ContainerImage,
        $ConfigRelativePath,
        $MinFreeDiskGiB.ToString(),
        $MinAvailableMemoryGiB.ToString(),
        $ContainerCpuCount.ToString(),
        $ContainerMemoryGiB.ToString()
    ) `
    -PlanOnly:$PlanOnly
