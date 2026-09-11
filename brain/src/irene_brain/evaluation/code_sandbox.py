"""Linux-only, fail-closed Bubblewrap execution for bounded Python diagnostics.

Protects the host using namespaces, a minimal read-only runtime, seccomp and
resource limits. The ordinary same-interpreter benchmark harness is not an
adversarially tamper-proof grader. No execution is allowed without these layers.
"""
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import tempfile


SYSCALLS = '''read write readv writev close fstat newfstatat stat lstat statx lseek
pread64 openat access faccessat faccessat2 readlink readlinkat getdents64 fcntl ioctl
mmap mprotect munmap brk mremap madvise rt_sigaction rt_sigprocmask rt_sigreturn
sigaltstack arch_prctl set_tid_address set_robust_list rseq prlimit64 getrandom
clock_gettime clock_nanosleep nanosleep futex getpid gettid getuid geteuid getgid
getegid getcwd uname sched_getaffinity sysinfo wait4 exit exit_group execve'''.split()


def export_filter(stream):
    library = ctypes.util.find_library('seccomp')
    if library is None:
        raise RuntimeError('libseccomp is required')
    lib = ctypes.CDLL(library)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x80000000)  # SCMP_ACT_KILL_PROCESS
    if not context:
        raise RuntimeError('Cannot initialize seccomp')
    try:
        for name in SYSCALLS:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number < 0 or lib.seccomp_rule_add(context, 0x7fff0000, number, 0) != 0:
                raise RuntimeError('Cannot allow required syscall: ' + name)
        if lib.seccomp_export_bpf(context, stream.fileno()) != 0:
            raise RuntimeError('Cannot export seccomp filter')
        stream.seek(0)
    finally:
        lib.seccomp_release(context)


def command(filter_fd, program):
    if os.name != 'posix' or shutil.which('bwrap') is None:
        raise RuntimeError('Authorized Linux runtime with Bubblewrap is required')
    paths = ('/usr/bin/python3.12', '/usr/lib/python3.12', '/usr/lib/x86_64-linux-gnu', '/lib64')
    if not all(Path(path).exists() for path in paths):
        raise RuntimeError('Expected Python 3.12 system runtime is unavailable')
    args = ['bwrap', '--unshare-all', '--unshare-user', '--cap-drop', 'ALL',
            '--new-session', '--die-with-parent', '--clearenv', '--setenv', 'PATH', '/usr/bin']
    for path in paths:
        args += ['--ro-bind', path, path]
    # No host /proc, workspace, home, credentials, accelerator or socket mounts.
    args += ['--symlink', 'usr/lib', '/lib', '--dev', '/dev',
             '--size', '16777216', '--tmpfs', '/tmp', '--chdir', '/tmp',
             '--remount-ro', '/', '--remount-ro', '/dev',
             '--seccomp', str(filter_fd), '/usr/bin/python3.12', '-I', '-S', '-B', '-c', program]
    return args


def limits():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def run_python(program, payload='', timeout=6):
    """Run only on an authorized remote Linux host; never fall back to plain exec."""
    if len(program.encode()) + len(payload.encode()) > 2 * 1024 * 1024:
        raise ValueError('Sandbox input exceeds 2 MiB')
    if os.name != 'posix':
        raise RuntimeError('Remote Linux execution is required')
    with tempfile.TemporaryFile() as filt, tempfile.TemporaryFile() as output:
        export_filter(filt)
        process = subprocess.Popen(command(filt.fileno(), program), stdin=subprocess.PIPE,
            stdout=output, stderr=output, pass_fds=(filt.fileno(),), close_fds=True,
            preexec_fn=limits, start_new_session=True, env={'PATH': '/usr/bin:/bin'})
        timed_out = False
        try:
            process.communicate(payload.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
        output.seek(0)
        return {'returncode': process.returncode, 'timed_out': timed_out,
                'output': output.read(65536).decode(errors='replace')}


def check_program(code, test, entry_point):
    """Run the official trusted check function against generated code in isolation."""
    marker = 'PB_CHECK_PASSED_' + secrets.token_hex(16)
    bootstrap = '''import json, sys
payload = json.load(sys.stdin)
namespace = {}
exec(compile(payload['code'], '<candidate>', 'exec'), namespace)
exec(compile(payload['test'], '<tests>', 'exec'), namespace)
namespace['check'](namespace[payload['entry_point']])
print(payload['marker'], flush=True)
'''
    result = run_python(bootstrap, json.dumps({'code': code, 'test': test,
                         'entry_point': entry_point, 'marker': marker}))
    result['passed'] = result['returncode'] == 0 and not result['timed_out'] and marker in result['output'].splitlines()
    result['output'] = result['output'].replace(marker, '[PASS_MARKER]')
    return result
