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
    [string]$RunId,

    [Parameter(Mandatory)]
    [ValidateSet('Foreground', 'Tmux')]
    [string]$LaunchMode,

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

    [switch]$AcknowledgeDetached,

    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

if ($LaunchMode -eq 'Tmux' -and -not $AcknowledgeDetached) {
    throw 'Tmux launch requires -AcknowledgeDetached. No persistent job is started implicitly.'
}
if ($LaunchMode -eq 'Foreground' -and $AcknowledgeDetached) {
    throw '-AcknowledgeDetached is only valid with -LaunchMode Tmux.'
}

Assert-DgxRemoteWorkDir -RemoteWorkDir $RemoteWorkDir
Assert-DgxSlug -Value $ReleaseId -Label 'ReleaseId'
Assert-DgxSlug -Value $RunId -Label 'RunId'
Assert-DgxImageReference -ContainerImage $ContainerImage
Assert-DgxRelativePath -Path $ConfigRelativePath
if ($ConfigRelativePath -eq 'brain/configs/training/dgx-rcq-v2-reference.toml' -or $RunId -eq 'dgx-rcq-v2-reference-seed-1702') {
    throw 'the RCQ-v2 reference config/run requires Start-DgxRcqV2Reference.ps1'
}
if ($ConfigRelativePath -eq 'brain/configs/training/dgx-rcq-v2-staging-canary.toml') {
    throw 'the schema-3 staging canary must use Invoke-DgxRcqStagingCanary.ps1'
}
$target = Resolve-DgxSshTarget -SshTarget $SshTarget

Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @(
        'train',
        $RemoteWorkDir,
        $ReleaseId,
        $ContainerImage,
        $ConfigRelativePath,
        $RunId,
        $LaunchMode.ToLowerInvariant(),
        $MinFreeDiskGiB.ToString(),
        $MinAvailableMemoryGiB.ToString(),
        $ContainerCpuCount.ToString(),
        $ContainerMemoryGiB.ToString()
    ) `
    -PlanOnly:$PlanOnly
