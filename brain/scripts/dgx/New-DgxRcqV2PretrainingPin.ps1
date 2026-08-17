[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

    [Parameter(Mandatory)]
    [string]$ReleaseId,

    [Parameter(Mandatory)]
    [string]$ReleaseArchiveSha256,

    [Parameter(Mandatory)]
    [string]$RegistrationSha256,

    [Parameter(Mandatory)]
    [string]$ContainerImage,

    [Parameter(Mandatory)]
    [string]$ContainerImageId,

    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

Assert-DgxSlug -Value $ReleaseId -Label 'ReleaseId'
Assert-DgxSha256 -Value $ReleaseArchiveSha256 -Label 'ReleaseArchiveSha256'
Assert-DgxSha256 -Value $RegistrationSha256 -Label 'RegistrationSha256'
Assert-DgxImageReference -ContainerImage $ContainerImage
Assert-DgxImageId -ContainerImageId $ContainerImageId
$target = Resolve-DgxSshTarget -SshTarget $SshTarget

# This is the only action that accepts production identity values. It publishes
# their canonical create-only trust root before smoke, canary, or training.
Invoke-DgxRemoteScript `
    -TargetInfo $target `
    -ScriptPath (Get-DgxRemoteHelperPath) `
    -RemoteArguments @(
        'rcq_v2_pin_pretraining_v1',
        $ReleaseId,
        $ReleaseArchiveSha256,
        $RegistrationSha256,
        $ContainerImage,
        $ContainerImageId
    ) `
    -PlanOnly:$PlanOnly
