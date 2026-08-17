[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

    [Parameter(Mandatory)]
    [string]$RemoteWorkDir,

    [Parameter(Mandatory)]
    [string]$RunId,

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
$remoteAction = if ($HistoricalIreneWorkspace) { 'historical_status' } else { 'status' }
Assert-DgxRemoteWorkDir -RemoteWorkDir $RemoteWorkDir -Access $workspaceAccess
Assert-DgxSlug -Value $RunId -Label 'RunId'
$target = Resolve-DgxSshTarget -SshTarget $SshTarget

Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @($remoteAction, $RemoteWorkDir, $RunId) `
    -PlanOnly:$PlanOnly
