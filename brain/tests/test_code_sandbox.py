"""Remote Linux integration tests; every executed snippet stays inside Bubblewrap."""
from irene_brain.evaluation.code_sandbox import run_python, check_program


def test_canonical_and_wrong_function_and_premature_exit():
    test = 'def check(candidate):\n    assert candidate(3) == 6\n    assert candidate(-2) == -4'
    assert check_program('def twice(x): return 2*x', test, 'twice')['passed']
    assert not check_program('def twice(x): return 3*x', test, 'twice')['passed']
    assert not check_program('raise SystemExit(0)', test, 'twice')['passed']


def test_filesystem_environment_and_devices_are_isolated():
    result = run_python('''import os
assert not os.path.exists('/content')
assert not os.path.exists('/root')
assert not os.path.exists('/proc')
assert not os.path.exists('/dev/nvidia0')
assert set(os.environ) <= {'PATH', 'PWD', 'LC_CTYPE'}
for path in ('/usr/lib/python3.12/pb_sandbox_write_probe', '/pb_write_probe', '/dev/pb_write_probe'):
    try:
        open(path, 'x')
    except OSError:
        pass
    else:
        raise AssertionError('Runtime is writable: ' + path)
print('ISOLATED')
''')
    assert result['returncode'] == 0, result
    assert 'ISOLATED' in result['output']


def test_network_process_creation_and_new_namespaces_are_killed():
    for snippet in ('import socket; socket.socket()', 'import os; os.fork()',
                    'import ctypes; ctypes.CDLL(None).unshare(0x10000000)'):
        result = run_python(snippet)
        assert result['returncode'] != 0, result


def test_cpu_memory_and_output_are_bounded():
    for snippet in ('while True: pass', 'x = bytearray(1024**3)',
                    'import os\nwhile True: os.write(1, b"x"*8192)'):
        result = run_python(snippet)
        assert result['returncode'] != 0, result
        assert len(result['output'].encode()) <= 65536
