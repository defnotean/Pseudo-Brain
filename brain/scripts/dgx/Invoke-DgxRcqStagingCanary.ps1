[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

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

$target = Resolve-DgxSshTarget -SshTarget $SshTarget

# The remote action derives release, approved image, and a collision-resistant
# fixed run identity from the create-only pretraining pin. It executes both
# phases in the foreground; there is deliberately no detached launch option.
Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @(
        'rcq_staging_canary',
        $MinFreeDiskGiB.ToString(),
        $MinAvailableMemoryGiB.ToString(),
        $ContainerCpuCount.ToString(),
        $ContainerMemoryGiB.ToString()
    ) `
    -PlanOnly:$PlanOnly
