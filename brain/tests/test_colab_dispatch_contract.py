import importlib.util
import ast
from pathlib import Path
import subprocess
import sys

import pytest

source = Path(__file__).resolve().parents[2] / "scripts/colab_dispatch.py"
spec = importlib.util.spec_from_file_location("pb_colab_dispatch", source)
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


@pytest.mark.parametrize("code,status", [
    ("print('done')", 0), ("raise SystemExit(2)", 2),
    ("raise RuntimeError('failed gate')", 1), ("raise SystemExit('not a success')", 1),
])
def test_remote_python_wrapper_reports_real_failure(code, status, tmp_path):
    script = tmp_path / "wrapped.py"
    script.write_text(dispatch._result_wrapper(code, "test.py", "FIXED_RESULT"))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.stdout.splitlines()[-1] == f"FIXED_RESULT {status}"


def test_shell_transport_quotes_command_without_interpolation(monkeypatch):
    captured = []
    def inspect(path, session, timeout):
        captured.append(path.read_text())
        compile(captured[-1], "transport", "exec")
        assert session == "test-session" and timeout == 60
        return 9
    monkeypatch.setattr(dispatch, "cmd_exec_file", inspect)
    command = "printf '%s' 'a quote \" and $literal'"
    assert dispatch.cmd_exec(command, "test-session") == 9
    assert repr(command) in captured[0]


@pytest.mark.parametrize("remote_status,expected", [(2, 2), (None, 1)])
def test_zero_cli_exit_does_not_hide_failed_or_missing_remote_result(monkeypatch, tmp_path, remote_status, expected):
    script = tmp_path / "job.py"
    script.write_text("raise SystemExit(2)")
    monkeypatch.setattr(dispatch, "is_windows", lambda: False)
    monkeypatch.setattr(dispatch, "get_colab_command_prefix", lambda: ["fake-colab"])
    class Process:
        def __init__(self, command, **kwargs):
            wrapper_path = Path(command[command.index("--file") + 1])
            tree = ast.parse(wrapper_path.read_text())
            marker = ast.literal_eval(tree.body[-1].value.args[0])
            self.stdout = iter([] if remote_status is None else [f"{marker} {remote_status}\n"])
        def wait(self):
            return 0
    monkeypatch.setattr(dispatch.subprocess, "Popen", Process)
    assert dispatch.cmd_exec_file(script, "test-session") == expected
