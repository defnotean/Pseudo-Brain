"""Neural Software Environment & Primitives for Pseudo-Brain.

Exposes a clean, domain-agnostic execution environment containing ONLY atomic
operating primitives (the "hands and eyes" of the agent) with ZERO domain
branching, zero preset templates, and zero keyword classifiers:

1. ACTION: WRITE_FILE <path>\n<content>
2. ACTION: READ_FILE <path>
3. ACTION: EDIT_FILE <path>\n<<<TARGET\n...\n===\n...\n>>>
4. ACTION: RUN_TESTS [optional args]
5. ACTION: RETRIEVE_MEMORY <query>
6. ACTION: FINISH <summary>

Security & Safety:
- Path Traversal Containment: All paths are strictly bounded inside `workspace_dir`.
- External Task Validator: If configured, `ACTION: FINISH` verifies task success
  against hidden criteria before declaring success.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from irene_brain.memory.parametric_memory import ParametricStaticKnowledgeCore, StaticRecallResult


@dataclass
class EnvironmentObservation:
    """Feedback from the environment returned after an action execution."""
    action_type: str
    success: bool
    observation_text: str
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    elapsed_ms: float = 0.0
    files_created_or_modified: List[str] = field(default_factory=list)
    verified_completion: bool = False


class NeuralSoftwareEnvironment:
    """Domain-agnostic software execution environment.

    Provides ONLY execution primitives on the local machine with zero domain logic:
    - File system read / write / patch (strictly sandboxed within workspace_dir)
    - Sandboxed AST syntax validation
    - Subprocess unit test runner
    - Memory knowledge retrieval
    - External task completion validation
    """

    def __init__(
        self,
        workspace_dir: Path,
        memory_core: Optional[ParametricStaticKnowledgeCore] = None,
        python_exe: Optional[str] = None,
        task_validator: Optional[Callable[[NeuralSoftwareEnvironment], Tuple[bool, str]]] = None,
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.memory_core = memory_core or ParametricStaticKnowledgeCore()
        self.python_exe = python_exe or sys.executable
        self.task_validator = task_validator
        self.action_history: List[str] = []
        self.observation_history: List[EnvironmentObservation] = []

    def _resolve_safe_path(self, rel_path: str) -> Optional[Path]:
        """Safely resolve path ensuring it stays strictly inside workspace_dir.

        Blocks directory traversal (e.g. '../../escape.py') and unauthorized filesystem writes.
        """
        try:
            clean = rel_path.strip().lstrip("/\\")
            target = (self.workspace_dir / clean).resolve()
            if not target.is_relative_to(self.workspace_dir):
                return None
            return target
        except Exception:
            return None

    def execute_action(self, action_str: str) -> EnvironmentObservation:
        """Parse and execute a raw action string emitted by Pseudo-Brain."""
        t0 = time.perf_counter()
        raw = action_str.strip()
        self.action_history.append(raw)

        # 1. ACTION: WRITE_FILE <rel_path>\n<content>
        if raw.startswith("ACTION: WRITE_FILE"):
            first_line, _, body = raw.partition("\n")
            rel_path = first_line.replace("ACTION: WRITE_FILE", "").strip()
            obs = self._handle_write_file(rel_path, body)

        # 2. ACTION: READ_FILE <rel_path>
        elif raw.startswith("ACTION: READ_FILE"):
            rel_path = raw.replace("ACTION: READ_FILE", "").strip().split("\n")[0]
            obs = self._handle_read_file(rel_path)

        # 3. ACTION: EDIT_FILE <rel_path>\n<<<TARGET\n...\n===\n...\n>>>
        elif raw.startswith("ACTION: EDIT_FILE"):
            first_line, _, body = raw.partition("\n")
            rel_path = first_line.replace("ACTION: EDIT_FILE", "").strip()
            obs = self._handle_edit_file(rel_path, body)

        # 4. ACTION: RUN_TESTS [test_path]
        elif raw.startswith("ACTION: RUN_TESTS"):
            arg = raw.replace("ACTION: RUN_TESTS", "").strip()
            obs = self._handle_run_tests(arg)

        # 5. ACTION: RETRIEVE_MEMORY <query>
        elif raw.startswith("ACTION: RETRIEVE_MEMORY"):
            query = raw.replace("ACTION: RETRIEVE_MEMORY", "").strip()
            obs = self._handle_retrieve_memory(query)

        # 6. ACTION: FINISH <summary>
        elif raw.startswith("ACTION: FINISH"):
            summary = raw.replace("ACTION: FINISH", "").strip()
            if self.task_validator is not None:
                passed, details = self.task_validator(self)
                if passed:
                    obs = EnvironmentObservation(
                        action_type="FINISH",
                        success=True,
                        observation_text=f"[OBSERVATION: Task verified and passed. Summary: {summary}. {details}]",
                        verified_completion=True,
                    )
                else:
                    obs = EnvironmentObservation(
                        action_type="FINISH",
                        success=False,
                        observation_text=f"[OBSERVATION: Task incomplete: {details}. Continue working.]",
                        stderr=details,
                        return_code=1,
                        verified_completion=False,
                    )
            else:
                obs = EnvironmentObservation(
                    action_type="FINISH",
                    success=False,
                    observation_text=f"[OBSERVATION: Task completion requires an external task validator. None configured. Summary: {summary}]",
                    stderr="No external validator configured",
                    return_code=1,
                    verified_completion=False,
                )

        # Unknown / Unparsable Action
        else:
            obs = EnvironmentObservation(
                action_type="UNKNOWN",
                success=False,
                observation_text=f"[OBSERVATION: Unrecognized action format: {raw[:80]}. Valid actions are WRITE_FILE, READ_FILE, EDIT_FILE, RUN_TESTS, RETRIEVE_MEMORY, FINISH]",
                stderr="Syntax/Format Error",
                return_code=-1,
            )

        obs.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.observation_history.append(obs)
        return obs

    def _handle_write_file(self, rel_path: str, content: str) -> EnvironmentObservation:
        """Write a file to disk and validate AST syntax if it is a Python file."""
        if not rel_path:
            return EnvironmentObservation(
                action_type="WRITE_FILE",
                success=False,
                observation_text="[OBSERVATION: WRITE_FILE failed: No file path specified]",
                stderr="Missing path",
                return_code=1,
            )

        target_file = self._resolve_safe_path(rel_path)
        if target_file is None:
            return EnvironmentObservation(
                action_type="WRITE_FILE",
                success=False,
                observation_text=f"[OBSERVATION: WRITE_FILE access denied: Path '{rel_path}' escapes workspace directory]",
                stderr="Access Denied: Path Traversal",
                return_code=1,
            )

        target_file.parent.mkdir(parents=True, exist_ok=True)

        # Pre-validate AST syntax for Python files
        if rel_path.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as e:
                return EnvironmentObservation(
                    action_type="WRITE_FILE",
                    success=False,
                    observation_text=f"[OBSERVATION: WRITE_FILE failed: SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}]",
                    stderr=f"SyntaxError: {e.msg}",
                    return_code=1,
                )

        try:
            target_file.write_text(content, encoding="utf-8")
        except OSError as e:
            return EnvironmentObservation(
                action_type="WRITE_FILE",
                success=False,
                observation_text=f"[OBSERVATION: WRITE_FILE failed with OS error: {e}]",
                stderr=str(e),
                return_code=1,
            )

        return EnvironmentObservation(
            action_type="WRITE_FILE",
            success=True,
            observation_text=f"[OBSERVATION: Successfully wrote {len(content)} bytes to {rel_path}]",
            files_created_or_modified=[rel_path],
        )

    def _handle_read_file(self, rel_path: str) -> EnvironmentObservation:
        """Read a file from disk."""
        target_file = self._resolve_safe_path(rel_path)
        if target_file is None:
            return EnvironmentObservation(
                action_type="READ_FILE",
                success=False,
                observation_text=f"[OBSERVATION: READ_FILE access denied: Path '{rel_path}' escapes workspace directory]",
                stderr="Access Denied: Path Traversal",
                return_code=1,
            )

        if not target_file.exists():
            return EnvironmentObservation(
                action_type="READ_FILE",
                success=False,
                observation_text=f"[OBSERVATION: File not found: {rel_path}]",
                stderr="FileNotFoundError",
                return_code=1,
            )

        content = target_file.read_text(encoding="utf-8", errors="replace")
        return EnvironmentObservation(
            action_type="READ_FILE",
            success=True,
            observation_text=f"[OBSERVATION: File {rel_path} content:\n{content}]",
            stdout=content,
        )

    def _handle_edit_file(self, rel_path: str, diff_body: str) -> EnvironmentObservation:
        """Edit a target file by replacing target block with replacement block."""
        target_file = self._resolve_safe_path(rel_path)
        if target_file is None:
            return EnvironmentObservation(
                action_type="EDIT_FILE",
                success=False,
                observation_text=f"[OBSERVATION: EDIT_FILE access denied: Path '{rel_path}' escapes workspace directory]",
                stderr="Access Denied: Path Traversal",
                return_code=1,
            )

        if not target_file.exists():
            return EnvironmentObservation(
                action_type="EDIT_FILE",
                success=False,
                observation_text=f"[OBSERVATION: EDIT_FILE failed: {rel_path} does not exist]",
                stderr="FileNotFoundError",
                return_code=1,
            )

        content = target_file.read_text(encoding="utf-8")

        # Parse <<<TARGET\n...\n===\n...\n>>> format
        m = re.search(r"<<<TARGET\s*\n(.*?)\n===\s*\n(.*?)>>>", diff_body, re.DOTALL)
        if m:
            target_str = m.group(1)
            replacement_str = m.group(2)
        else:
            parts = diff_body.split("===")
            if len(parts) == 2:
                target_str = parts[0].strip("\n")
                replacement_str = parts[1].strip("\n")
            else:
                return EnvironmentObservation(
                    action_type="EDIT_FILE",
                    success=False,
                    observation_text="[OBSERVATION: EDIT_FILE failed: Invalid diff syntax. Expected format: <<<TARGET\n<old>\n===\n<new>\n>>>]",
                    stderr="Diff format error",
                    return_code=1,
                )

        if target_str not in content:
            return EnvironmentObservation(
                action_type="EDIT_FILE",
                success=False,
                observation_text=f"[OBSERVATION: EDIT_FILE failed: Target snippet not found in {rel_path}]",
                stderr="Snippet mismatch",
                return_code=1,
            )

        new_content = content.replace(target_str, replacement_str, 1)

        # Validate syntax
        if rel_path.endswith(".py"):
            try:
                ast.parse(new_content)
            except SyntaxError as e:
                return EnvironmentObservation(
                    action_type="EDIT_FILE",
                    success=False,
                    observation_text=f"[OBSERVATION: EDIT_FILE failed: SyntaxError in replacement at line {e.lineno}: {e.msg}]",
                    stderr=f"SyntaxError: {e.msg}",
                    return_code=1,
                )

        target_file.write_text(new_content, encoding="utf-8")
        return EnvironmentObservation(
            action_type="EDIT_FILE",
            success=True,
            observation_text=f"[OBSERVATION: Successfully edited {rel_path}]",
            files_created_or_modified=[rel_path],
        )

    def _handle_run_tests(self, target_arg: str = "") -> EnvironmentObservation:
        """Execute pytest/unittest definitions with discovery, or standalone test scripts."""
        test_files: List[Path] = []
        if target_arg:
            specific_test = self._resolve_safe_path(target_arg)
            if specific_test is None:
                return EnvironmentObservation(
                    action_type="RUN_TESTS",
                    success=False,
                    observation_text=f"[OBSERVATION: RUN_TESTS access denied: Path '{target_arg}' escapes workspace directory]",
                    stderr="Access Denied: Path Traversal",
                    return_code=1,
                )
            if specific_test.is_dir():
                test_files.extend(sorted(specific_test.rglob("test_*.py")))
            elif specific_test.is_file():
                test_files.append(specific_test)
            else:
                return EnvironmentObservation(
                    action_type="RUN_TESTS", success=False,
                    observation_text=f"[OBSERVATION: RUN_TESTS failed: Test path '{target_arg}' does not exist]",
                    stderr="Test path does not exist", return_code=1,
                )

        if not test_files and not target_arg:
            # Auto-discover test files in workspace
            for p in sorted(self.workspace_dir.rglob("test_*.py")):
                test_files.append(p)

        if not test_files:
            return EnvironmentObservation(
                action_type="RUN_TESTS",
                success=False,
                observation_text="[OBSERVATION: RUN_TESTS: No test files found (looking for tests/test_*.py)]",
                stderr="No tests found",
                return_code=1,
            )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.workspace_dir) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

        test_outputs = []
        overall_success = True
        exit_code = 0

        for t_path in test_files:
            rel = t_path.relative_to(self.workspace_dir)
            cmd = [self.python_exe, str(t_path.resolve())]
            try:
                tree = ast.parse(t_path.read_text(encoding="utf-8"))
                has_test_definitions = any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
                    or isinstance(node, ast.ClassDef) and (
                        node.name.startswith("Test") or any(
                            isinstance(base, ast.Attribute) and base.attr == "TestCase"
                            or isinstance(base, ast.Name) and base.id == "TestCase"
                            for base in node.bases))
                    for node in ast.walk(tree)
                )
                if has_test_definitions:
                    cmd = [self.python_exe, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(t_path.resolve())]
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.workspace_dir),
                    capture_output=True,
                    text=True,
                    timeout=15,
                    env=env,
                )
                out = proc.stdout.strip()
                err = proc.stderr.strip()
                out = re.sub(r" in \d+\.\d+s", " in 0.00s", out)
                err = re.sub(r" in \d+\.\d+s", " in 0.00s", err)
                if proc.returncode != 0:
                    overall_success = False
                    exit_code = proc.returncode
                    test_outputs.append(f"[{rel} FAILED (exit {proc.returncode})]:\n{err or out}")
                else:
                    test_outputs.append(f"[{rel} PASSED]:\n{out}")
            except subprocess.TimeoutExpired:
                overall_success = False
                exit_code = 124
                test_outputs.append(f"[{rel} TIMED OUT (>15s)]")
            except Exception as e:
                overall_success = False
                exit_code = 1
                test_outputs.append(f"[{rel} EXCEPTION: {e}]")

        summary_text = "\n\n".join(test_outputs)
        status_label = "PASSED" if overall_success else "FAILED"
        return EnvironmentObservation(
            action_type="RUN_TESTS",
            success=overall_success,
            observation_text=f"[OBSERVATION: RUN_TESTS {status_label}:\n{summary_text}]",
            stdout=summary_text if overall_success else "",
            stderr=summary_text if not overall_success else "",
            return_code=exit_code,
        )

    def _handle_retrieve_memory(self, query: str) -> EnvironmentObservation:
        """Retrieve factual / algorithmic knowledge from the parametric memory core."""
        res: Optional[StaticRecallResult] = self.memory_core.query_static_knowledge(query)
        if res is not None and res.confidence >= self.memory_core.confidence_threshold:
            obs_str = f"[OBSERVATION: Retrieved Memory for '{res.topic}' ({res.category}): {res.summary}"
            if res.code_example:
                obs_str += f"\nCode:\n{res.code_example}"
            obs_str += "]"
            return EnvironmentObservation(
                action_type="RETRIEVE_MEMORY",
                success=True,
                observation_text=obs_str,
                stdout=res.summary,
            )

        return EnvironmentObservation(
            action_type="RETRIEVE_MEMORY",
            success=False,
            observation_text=f"[OBSERVATION: No high-confidence memory found for '{query}']",
            stderr="Memory miss",
        )
