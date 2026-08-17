Set-StrictMode -Version Latest

function Get-DgxApplication {
    param(
        [Parameter(Mandatory)]
        [string]$Name
    )

    $normalized = $Name.ToLowerInvariant()
    $isWindowsRuntime = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
    if ($isWindowsRuntime) {
        $systemRoot = [Environment]::GetFolderPath([Environment+SpecialFolder]::Windows)
        $programFiles = [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFiles)
        $localApplicationData = [Environment]::GetFolderPath(
            [Environment+SpecialFolder]::LocalApplicationData
        )
        $candidate = switch ($normalized) {
            'ssh' { Join-Path $systemRoot 'System32\OpenSSH\ssh.exe' }
            'scp' { Join-Path $systemRoot 'System32\OpenSSH\scp.exe' }
            'tar' { Join-Path $systemRoot 'System32\tar.exe' }
            'git' { Join-Path $programFiles 'Git\cmd\git.exe' }
            'python' {
                Join-Path $localApplicationData 'Programs\Python\Python311\python.exe'
            }
            default { throw "Application '$Name' has no canonical DGX tooling path." }
        }
    } else {
        if ($normalized -notin @('ssh', 'scp', 'tar', 'git', 'python')) {
            throw "Application '$Name' has no canonical DGX tooling path."
        }
        $candidate = "/usr/bin/$normalized"
        if ($normalized -eq 'python') {
            $candidate = '/usr/bin/python3'
        }
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Required canonical application '$candidate' is unavailable."
    }
    $item = Get-Item -LiteralPath $candidate -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Required canonical application '$candidate' is a reparse point."
    }
    return $item.FullName
}

function Assert-DgxRemoteWorkDir {
    param(
        [Parameter(Mandatory)]
        [string]$RemoteWorkDir,

        [ValidateSet('PseudoBrainWrite', 'HistoricalIreneRead')]
        [string]$Access = 'PseudoBrainWrite'
    )

    if ($RemoteWorkDir -notmatch '^(?:~/|/)[A-Za-z0-9._/-]+$') {
        throw 'RemoteWorkDir must be an explicit POSIX path containing only letters, numbers, dot, underscore, dash, and slash.'
    }
    if ($RemoteWorkDir -match '(?:^|/)\.\.(?:/|$)' -or $RemoteWorkDir -match '//') {
        throw 'RemoteWorkDir cannot contain parent traversal or an empty path segment.'
    }

    $pathSegments = @($RemoteWorkDir.TrimEnd('/') -split '/')
    $leaf = $pathSegments[-1]
    $parentLeaf = if ($pathSegments.Count -ge 2) { $pathSegments[-2] } else { '' }
    if ($parentLeaf -ne 'projects') {
        throw "RemoteWorkDir must be a direct child of the remote 'projects' directory."
    }
    if ($Access -eq 'PseudoBrainWrite') {
        if ($leaf -notmatch '^pseudo-brain(?:-[A-Za-z0-9._-]+)?$') {
            throw "New DGX work must use the Pseudo-Brain v2 workspace and end in 'pseudo-brain' (or 'pseudo-brain-<suffix>'). The historical Irene workspace is read-only."
        }
        if (('/' + $RemoteWorkDir.Trim('~', '/') + '/') -match '/irene-brain/') {
            throw 'A Pseudo-Brain write workspace cannot be nested under the historical Irene workspace.'
        }
    } elseif ($leaf -ne 'irene-brain') {
        throw "Historical retrieval requires the exact legacy 'irene-brain' workspace leaf."
    }
}

function Assert-DgxRelativePath {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [string]$RequiredPrefix = 'brain/'
    )

    if ($Path -notmatch '^[A-Za-z0-9._/-]+$' -or $Path.StartsWith('/')) {
        throw "Relative path '$Path' contains unsupported characters or is absolute."
    }
    if ($Path -match '(?:^|/)\.\.(?:/|$)' -or $Path -match '//' -or -not $Path.StartsWith($RequiredPrefix)) {
        throw "Relative path '$Path' must remain under '$RequiredPrefix' without traversal."
    }
}

function Assert-DgxImageReference {
    param(
        [Parameter(Mandatory)]
        [string]$ContainerImage
    )

    $imagePattern = '^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}|@sha256:[a-f0-9]{64})$'
    if ($ContainerImage -notmatch $imagePattern) {
        throw 'ContainerImage must be an explicit, shell-safe Docker image tag or sha256 digest reference.'
    }
}

function Assert-DgxImageId {
    param(
        [Parameter(Mandatory)]
        [string]$ContainerImageId
    )

    if ($ContainerImageId -notmatch '^sha256:[a-f0-9]{64}$') {
        throw 'ContainerImageId must be an immutable sha256: digest with 64 lowercase hexadecimal characters.'
    }
}

function Assert-DgxSlug {
    param(
        [Parameter(Mandatory)]
        [string]$Value,

        [Parameter(Mandatory)]
        [string]$Label
    )

    if ($Value -notmatch '^[a-z0-9][a-z0-9._-]{0,62}$') {
        throw "$Label must be 1-63 lowercase shell-safe characters."
    }
}

function Assert-DgxSha256 {
    param(
        [Parameter(Mandatory)]
        [string]$Value,

        [string]$Label = 'SHA-256'
    )

    if ($Value -notmatch '^[a-f0-9]{64}$') {
        throw "$Label must be exactly 64 lowercase hexadecimal characters."
    }
}

function Resolve-DgxSshTarget {
    param(
        [Parameter(Mandatory)]
        [string]$SshTarget
    )

    if ($SshTarget -notmatch '^(?:[A-Za-z_][A-Za-z0-9_.-]{0,31}@)?[A-Za-z0-9][A-Za-z0-9_.-]*$') {
        throw 'SshTarget must be an explicit shell-safe OpenSSH alias, hostname, or user@hostname.'
    }

    $forbiddenNames = @(
        'localhost',
        'localhost.localdomain',
        '127.0.0.1',
        '0.0.0.0',
        '::1',
        '::'
    )
    $machineNames = @(@(
            [Environment]::MachineName,
            [System.Net.Dns]::GetHostName()
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

    if ($forbiddenNames -contains $SshTarget.ToLowerInvariant()) {
        throw "Refusing local SSH target '$SshTarget'. DGX execution must be remote."
    }

    $ssh = Get-DgxApplication -Name 'ssh'
    $sshConfig = & $ssh -G $SshTarget 2>$null
    if ($LASTEXITCODE -ne 0 -or $null -eq $sshConfig) {
        throw "OpenSSH could not resolve configuration for target '$SshTarget'."
    }

    $resolvedHost = $null
    $resolvedUser = $null
    $resolvedPort = $null
    foreach ($line in $sshConfig) {
        if ($line -match '^hostname\s+(.+)$' -and $null -eq $resolvedHost) {
            $resolvedHost = $Matches[1].Trim()
        } elseif ($line -match '^user\s+(.+)$' -and $null -eq $resolvedUser) {
            $resolvedUser = $Matches[1].Trim()
        } elseif ($line -match '^port\s+([0-9]+)$' -and $null -eq $resolvedPort) {
            $resolvedPort = [int]$Matches[1]
        }
    }

    if ([string]::IsNullOrWhiteSpace($resolvedHost) -or
        $resolvedHost -notmatch '^[A-Za-z0-9][A-Za-z0-9_.:-]*$') {
        throw "SSH target '$SshTarget' did not resolve to a safe hostname."
    }
    if ($forbiddenNames -contains $resolvedHost.ToLowerInvariant() -or
        $machineNames.Where({ $_.Equals($resolvedHost, [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0) {
        throw "SSH alias '$SshTarget' resolves to this workstation ('$resolvedHost'); refusing local GPU execution."
    }

    try {
        $targetAddresses = @([System.Net.Dns]::GetHostAddresses($resolvedHost))
    } catch {
        $targetAddresses = @()
    }

    # Windows OpenSSH can resolve mDNS ``.local`` hosts even when the .NET DNS
    # API cannot. In that case, retain the fail-remote property by requiring a
    # strict, noninteractive SSH connection to identify an ARM64 Spark before
    # any launcher action is allowed. A failed/unknown probe remains fatal.
    if ($targetAddresses.Count -eq 0) {
        $probeArguments = @(
            '-T',
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=10',
            '-o', 'LogLevel=ERROR'
        )
        $probeOutput = @(
            & $ssh @probeArguments $SshTarget `
                '/usr/bin/env -i PATH=/usr/sbin:/usr/bin /usr/bin/uname -m' 2>$null
        )
        $remoteArchitecture = if ($probeOutput.Count -eq 1) {
            ([string]$probeOutput[0]).Trim().ToLowerInvariant()
        } else {
            ''
        }
        if ($LASTEXITCODE -ne 0 -or $remoteArchitecture -notin @('aarch64', 'arm64')) {
            throw "SSH target '$SshTarget' could not be proven to be a remote ARM64 DGX Spark."
        }
    }

    $localAddresses = @()
    foreach ($name in $machineNames) {
        try {
            $localAddresses += [System.Net.Dns]::GetHostAddresses($name)
        } catch {
            # A secondary local hostname may not be registered. Other names are still checked.
        }
    }
    $localAddressStrings = @($localAddresses | ForEach-Object { $_.ToString() } | Select-Object -Unique)
    foreach ($address in $targetAddresses) {
        if ([System.Net.IPAddress]::IsLoopback($address) -or $localAddressStrings -contains $address.ToString()) {
            throw "SSH target '$SshTarget' resolves to local address $address; refusing local GPU execution."
        }
    }

    [pscustomobject]@{
        Target = $SshTarget
        HostName = $resolvedHost
        User = $resolvedUser
        Port = $resolvedPort
        SshPath = $ssh
    }
}

function Get-DgxSshArguments {
    @(
        '-T',
        '-o', 'BatchMode=yes',
        '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=10',
        '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=2',
        '-o', 'LogLevel=ERROR'
    )
}

function Get-DgxScpArguments {
    @(
        '-B',
        '-q',
        '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=10',
        '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=2',
        '-o', 'LogLevel=ERROR'
    )
}

function Invoke-DgxRemoteScript {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$TargetInfo,

        [Parameter(Mandatory)]
        [string]$ScriptPath,

        [Parameter(Mandatory)]
        [string[]]$RemoteArguments,

        [switch]$PlanOnly
    )

    if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
        throw "Remote helper is missing: $ScriptPath"
    }
    foreach ($argument in $RemoteArguments) {
        if ([string]::IsNullOrWhiteSpace($argument) -or $argument -notmatch '^[A-Za-z0-9_./:@=+~,-]+$') {
            throw "Unsafe or empty remote argument was rejected: '$argument'"
        }
    }

    if ($PlanOnly) {
        Write-Host "PLAN remote=$($TargetInfo.Target) helper=$(Split-Path -Leaf $ScriptPath) action=$($RemoteArguments[0])"
        return
    }

    $sshArguments = Get-DgxSshArguments
    $scriptContent = (
        (Get-Content -LiteralPath $ScriptPath -Raw).Replace("`r`n", "`n").TrimEnd("`r", "`n") +
        "`n# PowerShell native-pipe newline sentinel"
    )
    $scriptContent |
        & $TargetInfo.SshPath @sshArguments $TargetInfo.Target `
            '/usr/bin/env -i PATH=/usr/sbin:/usr/bin /bin/bash -p --noprofile --norc -s --' `
            @RemoteArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Remote DGX action '$($RemoteArguments[0])' failed with exit code $LASTEXITCODE."
    }
}

function Get-DgxRemoteHelperPath {
    Join-Path $PSScriptRoot '_remote_dispatch.sh'
}
