[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

    [switch]$AcknowledgeDetached,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

if (-not $AcknowledgeDetached) {
    throw 'Reference training is fixed to Tmux and requires -AcknowledgeDetached. No persistent job is started implicitly.'
}
$target = Resolve-DgxSshTarget -SshTarget $SshTarget
Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @('rcq_v3_reference_train_v1') `
    -PlanOnly:$PlanOnly
