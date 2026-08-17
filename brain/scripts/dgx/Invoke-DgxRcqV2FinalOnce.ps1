[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

    [switch]$AcknowledgePermanentTestRetirement,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

if (-not $AcknowledgePermanentTestRetirement) {
    throw 'Final-once permanently retires the registered TEST ranges. Pass -AcknowledgePermanentTestRetirement after external authorization review.'
}
$target = Resolve-DgxSshTarget -SshTarget $SshTarget
Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @('rcq_v2_final_once_v1') `
    -PlanOnly:$PlanOnly
