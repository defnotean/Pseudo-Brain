[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

    [Parameter(Mandatory)]
    [string]$PretrainingPinSha256,

    [Parameter(Mandatory)]
    [string]$LatestPointerSha256,

    [Parameter(Mandatory)]
    [string]$EntryCheckpointSha256,

    [Parameter(Mandatory)]
    [string]$CheckpointSha256,

    [Parameter(Mandatory)]
    [string]$ReadinessSha256,

    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

Assert-DgxSha256 -Value $PretrainingPinSha256 -Label 'PretrainingPinSha256'
Assert-DgxSha256 -Value $LatestPointerSha256 -Label 'LatestPointerSha256'
Assert-DgxSha256 -Value $EntryCheckpointSha256 -Label 'EntryCheckpointSha256'
Assert-DgxSha256 -Value $CheckpointSha256 -Label 'CheckpointSha256'
Assert-DgxSha256 -Value $ReadinessSha256 -Label 'ReadinessSha256'
$target = Resolve-DgxSshTarget -SshTarget $SshTarget

# These are the five externally reviewed file pins. The remote action derives
# every path and identity from the immutable pretraining record before it
# publishes the separate create-only authorization.
Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @(
        'rcq_v2_authorize_final_v1',
        $PretrainingPinSha256,
        $LatestPointerSha256,
        $EntryCheckpointSha256,
        $CheckpointSha256,
        $ReadinessSha256
    ) `
    -PlanOnly:$PlanOnly
