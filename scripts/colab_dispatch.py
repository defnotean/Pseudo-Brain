#!/usr/bin/env python3
"""Automated Google Colab CLI Remote Dispatch Bridge.

Bridges local workstations (Windows / WSL / Linux / macOS) to Google Colab
using the official Google Colab CLI (`google-colab-cli`), enabling remote
training dispatch directly to cloud GPUs (A100, L4, T4) with automated
Google Drive persistence and live local watching.

Requirements:
  - Google Colab account (Google AI Ultra / Pro recommended for background & A100 compute).
  - On Windows: WSL2 (automatically detected and utilized).
  - `google-colab-cli` (installed via `python scripts/colab_dispatch.py install`).

Commands:
  # 1. First-time setup:
  python scripts/colab_dispatch.py install
  python scripts/colab_dispatch.py login

  # 2. Check active remote sessions:
  python scripts/colab_dispatch.py status

  # 3. Launch training remotely on Google Colab (e.g. A100 GPU):
  python scripts/colab_dispatch.py run --experiment online_adaptation --models thoughtlet --gpu A100 --steps 1500

  # 4. Stop remote session to preserve compute units:
  python scripts/colab_dispatch.py stop --session <name>
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def is_windows() -> bool:
    return sys.platform == "win32"


def get_colab_command_prefix() -> List[str]:
    """Resolves the command prefix needed to invoke `colab` CLI.

    On Windows, `google-colab-cli` requires Unix PTY and is executed via WSL.
    On Linux/macOS, it is executed natively.
    """
    if is_windows():
        return ["wsl", "/root/.local/bin/colab"]
    colab_path = shutil.which("colab") or os.path.expanduser("~/.local/bin/colab")
    return [str(colab_path)]


def run_colab_cli(args: List[str], interactive: bool = False) -> int:
    """Executes a colab-cli sub-command with proper terminal passthrough."""
    cmd = get_colab_command_prefix() + args
    if interactive:
        # Direct tty passthrough for interactive prompts (OAuth login, REPL)
        return subprocess.call(cmd)
    else:
        res = subprocess.run(cmd)
        return res.returncode


def cmd_install() -> int:
    """Installs or updates google-colab-cli via uv."""
    print("=" * 65)
    print("INSTALLING / VERIFYING GOOGLE COLAB CLI")
    print("=" * 65)

    if is_windows():
        print("Detected Windows environment. Ensuring uv and google-colab-cli in WSL...")
        # Ensure uv exists in WSL
        wsl_uv_check = subprocess.run(["wsl", "which", "/root/.local/bin/uv"], capture_output=True)
        if wsl_uv_check.returncode != 0:
            print("Installing uv inside WSL...")
            subprocess.run(["wsl", "bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])

        print("Installing google-colab-cli inside WSL...")
        ret = subprocess.call(["wsl", "bash", "-c", "/root/.local/bin/uv tool install --upgrade google-colab-cli"])
    else:
        print("Ensuring google-colab-cli locally...")
        ret = subprocess.call(["uv", "tool", "install", "--upgrade", "google-colab-cli"])

    if ret == 0:
        print("\n[SUCCESS] Google Colab CLI installed successfully.")
        print("Next step: Run `python scripts/colab_dispatch.py login` to authenticate.")
    else:
        print("\n[ERROR] Failed to install google-colab-cli.")
    return ret


def cmd_login() -> int:
    """Launches interactive Google OAuth authorization."""
    print("=" * 65)
    print("AUTHENTICATING GOOGLE COLAB CLI")
    print("=" * 65)
    print("Opening Google Colab authorization flow...")
    print("Follow the URL presented in your browser, approve access, and paste the authorization code below.\n")
    return run_colab_cli(["sessions"], interactive=True)


def cmd_status() -> int:
    """Lists currently active Colab sessions."""
    print("=" * 65)
    print("ACTIVE GOOGLE COLAB SESSIONS")
    print("=" * 65)
    return run_colab_cli(["sessions"])


def cmd_stop(session_name: Optional[str] = None) -> int:
    """Terminates an active session."""
    args = ["stop"]
    if session_name:
        args.extend(["-s", session_name])
    return run_colab_cli(args)


def cmd_exec_file(path: Path, session_name: Optional[str] = None, timeout: int = 60) -> int:
    """Run a Python file and propagate its result even when CLI exec returns zero."""
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("Remote execution requires a regular local file")
    if timeout < 1:
        raise ValueError("timeout must be positive")
    marker = "PB_REMOTE_RESULT_" + uuid.uuid4().hex
    wrapper = _result_wrapper(path.read_text(encoding="utf-8"), path.name, marker)
    with tempfile.TemporaryDirectory(prefix="pb-colab-result-") as directory:
        wrapped = Path(directory) / "wrapped.py"
        wrapped.write_text(wrapper, encoding="utf-8")
        local_path = str(wrapped)
        if is_windows():
            local_path = subprocess.run(["wsl", "wslpath", "-a", local_path.replace("\\", "/")],
                                        capture_output=True, text=True, check=True).stdout.strip()
        args = ["exec", "--file", local_path, "--timeout", str(timeout)]
        if session_name:
            args.extend(["-s", session_name])
        proc = subprocess.Popen(get_colab_command_prefix() + args, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        remote_status = None
        for line in proc.stdout:
            match = re.fullmatch(re.escape(marker) + r" (-?\d+)\s*", line)
            if match:
                remote_status = int(match[1])
            else:
                print(re.sub(r"([?&]colab-runtime-proxy-token=)[^&\s]+",
                             r"\1[redacted]", line), end="", flush=True)
        cli_status = proc.wait()
    if cli_status:
        return cli_status
    if remote_status is None:
        print("[ERROR] Remote result marker is missing; refusing to report success.")
        return 1
    return remote_status


def _result_wrapper(source: str, filename: str, marker: str) -> str:
    return ("import traceback\n"
            "_pb_status = 0\n"
            "try:\n"
            "    _ns = dict(get_ipython().user_ns) if ('get_ipython' in dir(__builtins__) or 'get_ipython' in globals()) else {}\n"
            f"    _ns.update({{'__name__': '__main__', '__file__': {filename!r}}})\n"
            f"    exec(compile({source!r}, {filename!r}, 'exec'), _ns)\n"
            "    if 'get_ipython' in dir(__builtins__) or 'get_ipython' in globals():\n"
            "        get_ipython().user_ns.update({k: v for k, v in _ns.items() if not k.startswith('__')})\n"
            "except SystemExit as exc:\n"
            "    _pb_status = 0 if exc.code is None else exc.code if isinstance(exc.code, int) else 1\n"
            "except BaseException:\n"
            "    traceback.print_exc()\n"
            "    _pb_status = 1\n"
            f"print({marker!r}, _pb_status, flush=True)\n")


def cmd_exec(command_str: str, session_name: Optional[str] = None, timeout: int = 60) -> int:
    """Execute shell code through a Python file; Colab exec has no command argument.

    The wrapper propagates remote failures and refuses missing completion markers.
    """
    source = ("import subprocess\n"
              f"result = subprocess.run(['bash', '-lc', {command_str!r}], "
              "capture_output=True, text=True)\n"
              "print(result.stdout, end='', flush=True)\n"
              "print(result.stderr, end='', flush=True)\n"
              "print('REMOTE_PROCESS_EXIT', result.returncode, flush=True)\n"
              "raise SystemExit(result.returncode)\n")
    with tempfile.TemporaryDirectory(prefix="pb-colab-exec-") as directory:
        script = Path(directory) / "execute.py"
        script.write_text(source, encoding="utf-8")
        return cmd_exec_file(script, session_name, timeout)


def cmd_run(
    experiment: str,
    models: List[str],
    seeds: List[int],
    steps: int,
    gpu: str,
    session_name: Optional[str],
    keep: bool,
    webhook: Optional[str],
    wandb: bool,
) -> int:
    """Dispatches a full training run to a remote Colab VM."""
    print("=" * 70)
    print("🧠 DISPATCHING EXPERIMENT TO GOOGLE COLAB")
    print(f"  Experiment:   {experiment}")
    print(f"  Models:       {models}")
    print(f"  Seeds:        {seeds}")
    print(f"  Steps:        {steps}")
    print(f"  Accelerator:  {gpu} GPU")
    print(f"  Auto-Release: {'No (--keep)' if keep else 'Yes (shuts down on completion)'}")
    print("=" * 70)

    sess = session_name or f"pb-{experiment[:8]}"

    # 1. Create or ensure remote session
    print(f"\n[1/4] Provisioning Colab runtime ({gpu} GPU, session='{sess}')...")
    create_args = ["new", "--session", sess]
    if gpu.upper() != "CPU":
        create_args.extend(["--gpu", gpu])
    ret = run_colab_cli(create_args)
    if ret != 0:
        print("[ERROR] Session creation failed; refusing to execute in an existing or unknown runtime.")
        return ret

    # 2. Setup environment and clone repo
    print("\n[2/4] Setting up remote environment and synchronizing repository...")
    setup_cmd = (
        "python3 -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")' && "
        "mountpoint -q /content/drive && "
        "mkdir -p /content/drive/MyDrive/PseudoBrain/runs && "
        "if [ ! -d /content/Pseudo-Brain ]; then "
        "  git clone https://github.com/defnotean/Pseudo-Brain.git /content/Pseudo-Brain; "
        "else "
        "  cd /content/Pseudo-Brain && git pull; "
        "fi"
    )
    ret = cmd_exec(setup_cmd, sess, timeout=300)
    if ret != 0:
        print("[ERROR] Remote setup failed. This legacy recipe requires an actual Google Drive mount.")
        if not keep:
            run_colab_cli(["stop", "-s", sess])
        return ret

    # 3. Build training command
    models_str = " ".join(models)
    seeds_str = " ".join(str(s) for s in seeds)
    webhook_flag = f"--webhook {webhook}" if webhook else ""
    wandb_flag = "--wandb" if wandb else ""

    run_cmd = (
        f"cd /content/Pseudo-Brain && "
        f"PYTHONPATH=. python3 -m brain.experiments.run "
        f"--experiment {experiment} "
        f"--models {models_str} "
        f"--seeds {seeds_str} "
        f"--steps {steps} "
        f"--device auto "
        f"--output_dir /content/drive/MyDrive/PseudoBrain/runs "
        f"{webhook_flag} {wandb_flag}"
    )

    print(f"\n[3/4] Launching training on remote {gpu} runtime...")
    print(f"Command: {run_cmd}\n")
    train_ret = cmd_exec(run_cmd, sess, timeout=3600)

    # 4. Cleanup if keep is False
    if not keep:
        print("\n[4/4] Shutting down remote Colab runtime to preserve compute units...")
        run_colab_cli(["stop", "-s", sess])
    else:
        print(f"\n[4/4] Runtime kept active. Manage with `python scripts/colab_dispatch.py stop -s {sess}`")

    return train_ret


def main():
    parser = argparse.ArgumentParser(description="Pseudo-Brain Google Colab CLI Remote Dispatch Bridge")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # install
    subparsers.add_parser("install", help="Install or upgrade google-colab-cli in WSL/locally")

    # login
    subparsers.add_parser("login", help="Authenticate with Google Colab OAuth")

    # status
    subparsers.add_parser("status", help="List active Colab sessions")

    # stop
    stop_p = subparsers.add_parser("stop", help="Stop an active Colab session")
    stop_p.add_argument("--session", "-s", type=str, default=None, help="Session name to stop")

    # exec
    exec_p = subparsers.add_parser("exec", help="Execute command in active session")
    exec_p.add_argument("command", type=str, help="Shell command to execute")
    exec_p.add_argument("--session", "-s", type=str, default=None, help="Target session name")
    exec_p.add_argument("--timeout", type=int, default=60)
    file_p = subparsers.add_parser("exec-file", help="Execute a local Python file in Colab")
    file_p.add_argument("file", type=Path)
    file_p.add_argument("--session", "-s", required=True)
    file_p.add_argument("--timeout", type=int, default=60)

    # run
    run_p = subparsers.add_parser("run", help="Dispatch full experiment to Google Colab")
    run_p.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["online_adaptation", "memory_benchmark", "self_correction"],
        help="Experiment suite",
    )
    run_p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["thoughtlet"],
        help="List of models to benchmark",
    )
    run_p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Seeds to run",
    )
    run_p.add_argument("--steps", type=int, default=1500, help="Training steps per model")
    run_p.add_argument(
        "--gpu",
        type=str,
        default="A100",
        choices=["A100", "L4", "T4", "H100", "CPU"],
        help="Accelerator type (defaults to A100 for Google AI Ultra)",
    )
    run_p.add_argument("--session", "-s", type=str, default=None, help="Session name")
    run_p.add_argument("--keep", action="store_true", help="Keep VM running after experiment completes")
    run_p.add_argument("--webhook", type=str, default=None, help="Discord/Slack webhook URL for live alerts")
    run_p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    if args.subcommand == "install":
        sys.exit(cmd_install())
    elif args.subcommand == "login":
        sys.exit(cmd_login())
    elif args.subcommand == "status":
        sys.exit(cmd_status())
    elif args.subcommand == "stop":
        sys.exit(cmd_stop(args.session))
    elif args.subcommand == "exec":
        sys.exit(cmd_exec(args.command, args.session, args.timeout))
    elif args.subcommand == "exec-file":
        sys.exit(cmd_exec_file(args.file, args.session, args.timeout))
    elif args.subcommand == "run":
        sys.exit(
            cmd_run(
                experiment=args.experiment,
                models=args.models,
                seeds=args.seeds,
                steps=args.steps,
                gpu=args.gpu,
                session_name=args.session,
                keep=args.keep,
                webhook=args.webhook,
                wandb=args.wandb,
            )
        )


if __name__ == "__main__":
    main()
