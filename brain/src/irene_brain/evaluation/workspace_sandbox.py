"""Bounded read-only workspace snapshots for remote Python test feedback.

Supports zero-argument test_* functions and unittest.TestCase. Pytest plugins,
fixtures and dependencies are not supplied by this minimal runtime. The harness
shares an interpreter with candidate code and is not a tamper-proof grader.
"""
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import signal
import subprocess
import tempfile

from irene_brain.evaluation import code_sandbox


def checked_path(value):
    if not isinstance(value,str) or not value or '\\' in value or ':' in value:
        raise ValueError('Expected a relative POSIX file path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('Path must stay inside the workspace')
    if any(ord(char)<32 for char in value) or len(value)>240:
        raise ValueError('Invalid workspace path')
    return value


def checked_snapshot(files):
    if not isinstance(files,dict) or len(files)>64:
        raise ValueError('Workspace supports at most 64 files')
    total = 0
    names = set()
    for name, text in files.items():
        checked_path(name)
        if not isinstance(text,str) or len(text.encode())>65536:
            raise ValueError('Workspace file exceeds 64 KiB')
        names.add(name)
        total += len(text.encode())
    if total>1024*1024:
        raise ValueError('Workspace exceeds 1 MiB')
    for name in names:
        if any(str(parent) in names for parent in PurePosixPath(name).parents):
            raise ValueError('A workspace file conflicts with a directory')
    return dict(files)


BOOTSTRAP = '''import inspect, json, os, runpy, sys, unittest
payload = json.load(sys.stdin)
sys.path.insert(0, '/workspace')
suite = unittest.TestSuite()
for path in payload['tests']:
    namespace = runpy.run_path('/workspace/' + path, run_name='__workspace_test__')
    for name, obj in sorted(namespace.items()):
        if inspect.isfunction(obj) and name.startswith('test_') and obj.__module__ == '__workspace_test__':
            if inspect.iscoroutinefunction(obj) or inspect.isgeneratorfunction(obj) or inspect.isasyncgenfunction(obj):
                raise RuntimeError('Async and generator test functions are unsupported')
            suite.addTest(unittest.FunctionTestCase(obj))
        elif inspect.isclass(obj) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(obj))
count = suite.countTestCases()
if count == 0:
    raise RuntimeError('No supported tests discovered')
result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)
if result.skipped or result.expectedFailures:
    raise RuntimeError('Skipped or expected-failure tests cannot establish a pass')
print(payload['marker'] + str(count), flush=True)
'''


def run_workspace_tests(files, target=''):
    """Execute only inside the existing remote Linux Bubblewrap boundary."""
    if os.name!='posix':
        raise RuntimeError('Authorized remote Linux execution is required')
    files = checked_snapshot(files)
    if target:
        checked_path(target)
        if target in files:
            tests = [target] if target.endswith('.py') else []
        else:
            tests = sorted(name for name in files if name.startswith(target+'/')
                           and PurePosixPath(name).name.startswith('test_') and name.endswith('.py'))
    else:
        tests = sorted(name for name in files if PurePosixPath(name).name.startswith('test_') and name.endswith('.py'))
    if not tests:
        return {'passed':False,'returncode':1,'timed_out':False,'output':'No supported tests found','tests':[]}
    marker = 'PB_WORKSPACE_TESTS_'+secrets.token_hex(16)+':'
    with tempfile.TemporaryDirectory(prefix='pb-workspace-') as folder, tempfile.TemporaryFile() as filt, tempfile.TemporaryFile() as output:
        snapshot = Path(folder)
        for name, text in files.items():
            path = snapshot/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text,encoding='utf-8')
        code_sandbox.export_filter(filt)
        args = code_sandbox.command(filt.fileno(),BOOTSTRAP)
        args[1:1] = ['--ro-bind',str(snapshot),'/workspace']
        args[args.index('--chdir')+1] = '/workspace'
        process = subprocess.Popen(args,stdin=subprocess.PIPE,stdout=output,stderr=output,
            pass_fds=(filt.fileno(),),close_fds=True,preexec_fn=code_sandbox.limits,
            start_new_session=True,env={'PATH':'/usr/bin:/bin'})
        timed_out = False
        try:
            process.communicate(json.dumps({'tests':tests,'marker':marker}).encode(),timeout=6)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid,signal.SIGKILL)
            process.communicate(timeout=5)
        output.seek(0)
        text = output.read(65536).decode(errors='replace')
        receipts = [line.removeprefix(marker) for line in text.splitlines() if line.startswith(marker)]
        passed = process.returncode==0 and not timed_out and len(receipts)==1 and receipts[0].isdigit() and int(receipts[0])>0
        if not text:
            text = f'Sandbox exited with code {process.returncode}; no passing test receipt (timeout={timed_out})'
        return {'passed':passed,'returncode':process.returncode,'timed_out':timed_out,
                'output':text.replace(marker,'[TEST_COUNT]'),'tests':tests,
                'grading_limit':'Same-interpreter harness is not adversarially tamper-proof'}
