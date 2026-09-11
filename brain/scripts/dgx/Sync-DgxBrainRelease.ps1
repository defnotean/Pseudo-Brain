[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SshTarget,

    [Parameter(Mandatory)]
    [string]$RemoteWorkDir,

    [Parameter(Mandatory)]
    [ValidateRange(1, 4096)]
    [int]$MinFreeDiskGiB,

    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Dgx.Common.ps1')

Assert-DgxRemoteWorkDir -RemoteWorkDir $RemoteWorkDir
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$git = Get-DgxApplication -Name 'git'
$localMarker = Join-Path $repoRoot '.pseudo-brain-workspace-v2'
if (-not (Test-Path -LiteralPath $localMarker -PathType Leaf)) {
    throw 'The local Pseudo-Brain v2 workspace marker is missing.'
}
$markerItem = Get-Item -LiteralPath $localMarker -Force
if (($markerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
    (Get-Content -LiteralPath $localMarker -Raw).Trim() -ne 'pseudo-brain-workspace-v2') {
    throw 'The local Pseudo-Brain v2 workspace marker is invalid or linked.'
}
$gitTopLevel = @(& $git -C $repoRoot rev-parse --show-toplevel)
if ($LASTEXITCODE -ne 0 -or $gitTopLevel.Count -ne 1) {
    throw 'Could not prove the local Git repository root.'
}
$canonicalGitTopLevel = [IO.Path]::GetFullPath(([string]$gitTopLevel[0]).Trim())
if (-not $canonicalGitTopLevel.Equals($repoRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The DGX sync scripts are not running from the canonical Pseudo-Brain Git root.'
}

$registrationRelativePath = 'registrations/rcq-v2-reference-v2.json'
$registrationV3RelativePath = 'registrations/rcq-v3-reference-v1.json'
$allowedRegistrationPaths = @($registrationRelativePath, $registrationV3RelativePath)
$sourceFiles = @(& $git -C $repoRoot ls-files --cached --others --exclude-standard -- brain $registrationRelativePath $registrationV3RelativePath)
if ($LASTEXITCODE -ne 0) {
    throw 'git ls-files failed while building the source-only release manifest.'
}

$denyPattern = '(?i)(?:^|/)(?:\.env(?:$|\.)|\.git(?:/|$)|__pycache__(?:/|$)|\.pytest_cache(?:/|$)|\.venv(?:/|$)|venv(?:/|$)|cache(?:/|$))|(?i)^brain/(?:runs?|logs?|checkpoints?|artifacts?|datasets?)(?:/|$)|(?i)\.(?:pt|pth|ckpt|bin|safetensors|onnx|np[yz]|mp4|mkv|avi|zip|tar|tgz|gz|py[co]|pyd|so|dll|dylib)$'
$sourceFiles = @($sourceFiles |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
      Where-Object { $_ -notmatch $denyPattern } |
      # The bounded training release has a Python-only import tree; the local
      # console's static UI is not a training input or executable dependency.
      Where-Object { $_ -notmatch '^brain/src/irene_brain/console/web(?:/|$)' } |
    Sort-Object -Unique)

if ($sourceFiles.Count -eq 0 -or
    $sourceFiles -notcontains 'brain/pyproject.toml' -or
    $sourceFiles -notcontains 'brain/src/irene_brain/__init__.py' -or
    $sourceFiles -notcontains $registrationRelativePath) {
    throw "The source manifest is incomplete; build the fixed create-only '$registrationRelativePath' before sync."
}
foreach ($relativePath in $sourceFiles) {
    if ($relativePath -match "[`r`n]" -or -not $relativePath.StartsWith('brain/')) {
        if ($allowedRegistrationPaths -notcontains $relativePath) {
            throw "Unsafe source path was rejected: '$relativePath'"
        }
    }
    if ($relativePath.StartsWith('brain/src/') -and
        (-not $relativePath.StartsWith('brain/src/irene_brain/') -or
            -not $relativePath.EndsWith('.py'))) {
        throw "brain/src may contain only regular .py files below irene_brain: '$relativePath'"
    }
    $item = Get-Item -LiteralPath (Join-Path $repoRoot $relativePath) -Force
    if ($item -isnot [IO.FileInfo] -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Symbolic links and reparse points are not synced: '$relativePath'"
    }
}

$target = Resolve-DgxSshTarget -SshTarget $SshTarget

if ($PlanOnly) {
    Write-Host "PLAN remote=$SshTarget source_files=$($sourceFiles.Count) workspace=$RemoteWorkDir"
    return
}

$tar = Get-DgxApplication -Name 'tar'
$scp = Get-DgxApplication -Name 'scp'
$nonce = [Guid]::NewGuid().ToString('N')
$tempRoot = [IO.Path]::GetTempPath()
$manifestPath = Join-Path $tempRoot "pseudo-brain-$nonce.manifest"
$archivePath = Join-Path $tempRoot "pseudo-brain-$nonce.tgz"

try {
    [IO.File]::WriteAllLines(
        $manifestPath,
        [string[]]$sourceFiles,
        [Text.UTF8Encoding]::new($false)
    )

    Push-Location $repoRoot
    try {
        & $tar -czf $archivePath -T $manifestPath
        if ($LASTEXITCODE -ne 0) {
            throw 'tar failed while creating the source-only release archive.'
        }
    } finally {
        Pop-Location
    }

    $archiveSha = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $shortSha = $archiveSha.Substring(0, 12)
    $releaseStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ').ToLowerInvariant()
    $releaseId = "r$releaseStamp-$shortSha"
    $archiveName = "pseudo-brain-$shortSha.tgz"

    Invoke-DgxRemoteScript `
        -TargetInfo $target `
        -ScriptPath (Get-DgxRemoteHelperPath) `
        -RemoteArguments @('prepare_sync', $RemoteWorkDir, $MinFreeDiskGiB.ToString())

    $scpArguments = Get-DgxScpArguments
    $remoteArchive = '{0}:{1}/incoming/{2}' -f $target.Target, $RemoteWorkDir, $archiveName
    & $scp @scpArguments $archivePath $remoteArchive
    if ($LASTEXITCODE -ne 0) {
        throw "Source archive upload failed with exit code $LASTEXITCODE."
    }

    Invoke-DgxRemoteScript `
        -TargetInfo $target `
        -ScriptPath (Get-DgxRemoteHelperPath) `
        -RemoteArguments @('install_sync', $RemoteWorkDir, $archiveName, $releaseId, $archiveSha)

    Write-Host "RELEASE_ID=$releaseId"
    Write-Host "ARCHIVE_SHA256=$archiveSha"
} finally {
    foreach ($temporaryFile in @($manifestPath, $archivePath)) {
        if (Test-Path -LiteralPath $temporaryFile -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryFile -Force
        }
    }
}
