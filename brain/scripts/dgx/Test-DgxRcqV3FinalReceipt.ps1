[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

$target = Resolve-DgxSshTarget -SshTarget $SshTarget
Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @('rcq_v3_verify_receipt_v1') `
    -PlanOnly:$PlanOnly
