"""Controlled Tools and Registry for Pseudo-Brain Autonomous Agent."""
from __future__ import annotations

import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import torch


@dataclass
class ToolResult:
    """Outcome of a tool invocation."""
    success: bool
    output: str
    error: Optional[str] = None
    reward: float = 0.0


class Tool(ABC):
    """Abstract base class for agent tools."""
    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass


class FileReadTool(Tool):
    name = "read_file"
    description = "Read contents of a text file from disk."

    def execute(self, path: str, max_lines: int = 100) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}", reward=-0.2)
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            content = "\n".join(lines[:max_lines])
            return ToolResult(success=True, output=content, reward=0.1)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e), reward=-0.5)


class FileWriteTool(Tool):
    name = "write_file"
    description = "Write text content to a target file on disk."

    def execute(self, path: str, content: str) -> ToolResult:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Successfully wrote {len(content)} characters to {path}", reward=0.2)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e), reward=-0.5)


class FileGrepTool(Tool):
    name = "grep_files"
    description = "Fast regex or substring search across files in a directory tree."

    def execute(
        self,
        pattern: str,
        path: str = ".",
        extension: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        is_regex: bool = False,
        case_sensitive: bool = True,
        max_matches: int = 50,
    ) -> ToolResult:
        try:
            if not pattern:
                return ToolResult(success=False, output="", error="Search pattern cannot be empty.", reward=-0.2)

            root = Path(path)
            if not root.exists():
                return ToolResult(success=False, output="", error=f"Path not found: {path}", reward=-0.2)

            allowed_exts: Optional[Set[str]] = None
            if extensions:
                allowed_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
            elif extension:
                allowed_exts = {extension.lower() if extension.startswith(".") else f".{extension.lower()}"}

            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(pattern if is_regex else re.escape(pattern), flags)
            except re.error as err:
                return ToolResult(success=False, output="", error=f"Regex error: {err}", reward=-0.3)

            skip_dirs = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".idea", ".vscode", "node_modules"}
            matches: List[str] = []
            files_to_search: List[Path] = []

            if root.is_file():
                files_to_search = [root]
            else:
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
                    for fname in filenames:
                        fp = Path(dirpath) / fname
                        if allowed_exts is not None and fp.suffix.lower() not in allowed_exts:
                            continue
                        files_to_search.append(fp)

            for fp in files_to_search:
                if len(matches) >= max_matches:
                    break
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                for line_idx, line in enumerate(text.splitlines(), start=1):
                    if compiled.search(line):
                        try:
                            rel_path = fp.relative_to(root).as_posix()
                        except ValueError:
                            rel_path = fp.as_posix()
                        matches.append(f"{rel_path}:{line_idx}: {line.strip()}")
                        if len(matches) >= max_matches:
                            break

            if matches:
                return ToolResult(success=True, output="\n".join(matches), reward=0.2)
            return ToolResult(success=True, output=f"No matches found for pattern: '{pattern}'", reward=0.1)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e), reward=-0.5)


class DirectoryListTool(Tool):
    name = "list_directory"
    description = "Recursive directory tree or file listing with size and extension filtering."

    def execute(
        self,
        path: str = ".",
        recursive: bool = True,
        max_depth: int = 4,
        extension: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        max_entries: int = 100,
        include_size: bool = True,
    ) -> ToolResult:
        try:
            root = Path(path)
            if not root.exists():
                return ToolResult(success=False, output="", error=f"Path not found: {path}", reward=-0.2)
            if not root.is_dir():
                return ToolResult(success=False, output="", error=f"Path is not a directory: {path}", reward=-0.2)

            allowed_exts: Optional[Set[str]] = None
            if extensions:
                allowed_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
            elif extension:
                allowed_exts = {extension.lower() if extension.startswith(".") else f".{extension.lower()}"}

            skip_dirs = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".idea", ".vscode"}
            entries: List[str] = []

            for dirpath, dirnames, filenames in os.walk(root):
                rel_dir = Path(dirpath).relative_to(root)
                depth = len(rel_dir.parts)

                dirnames[:] = [d for d in dirnames if d not in skip_dirs]

                if recursive and depth >= max_depth:
                    dirnames.clear()
                    continue
                if not recursive and depth > 0:
                    dirnames.clear()
                    continue

                for d in sorted(dirnames):
                    d_rel = (rel_dir / d).as_posix()
                    entries.append(f"[DIR]  {d_rel}")
                    if len(entries) >= max_entries:
                        break

                if len(entries) >= max_entries:
                    break

                for f in sorted(filenames):
                    fp = Path(dirpath) / f
                    if allowed_exts is not None and fp.suffix.lower() not in allowed_exts:
                        continue
                    f_rel = (rel_dir / f).as_posix()
                    if include_size:
                        try:
                            sz = fp.stat().st_size
                            entries.append(f"[FILE] {f_rel} ({sz} bytes)")
                        except Exception:
                            entries.append(f"[FILE] {f_rel}")
                    else:
                        entries.append(f"[FILE] {f_rel}")
                    if len(entries) >= max_entries:
                        break

                if len(entries) >= max_entries:
                    break

            if not entries:
                return ToolResult(success=True, output="Directory is empty.", reward=0.1)
            return ToolResult(success=True, output="\n".join(entries[:max_entries]), reward=0.1)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e), reward=-0.5)


class FilePatchTool(Tool):
    name = "patch_file"
    description = "Safe selective line or substring replacement without rewriting entire files."

    def execute(
        self,
        path: str,
        target_text: Optional[str] = None,
        replacement_text: Optional[str] = None,
        line_number: Optional[int] = None,
        allow_multiple: bool = False,
    ) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}", reward=-0.2)
            if not p.is_file():
                return ToolResult(success=False, output="", error=f"Path is not a regular file: {path}", reward=-0.2)

            if replacement_text is None:
                return ToolResult(success=False, output="", error="replacement_text must be provided.", reward=-0.2)

            content = p.read_text(encoding="utf-8", errors="replace")

            if line_number is not None:
                lines = content.splitlines(keepends=True)
                if line_number < 1 or line_number > len(lines):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Line number {line_number} out of bounds (1..{len(lines)}).",
                        reward=-0.3,
                    )
                orig_line = lines[line_number - 1]
                if target_text is not None and target_text not in orig_line:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Target text '{target_text}' not found on line {line_number}: '{orig_line.strip()}'.",
                        reward=-0.3,
                    )
                has_newline = orig_line.endswith("\n") or orig_line.endswith("\r\n")
                new_line = replacement_text
                if has_newline and not (new_line.endswith("\n") or new_line.endswith("\r\n")):
                    new_line += "\n"
                lines[line_number - 1] = new_line
                new_content = "".join(lines)
            elif target_text is not None:
                if target_text not in content:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Target text not found in {path}.",
                        reward=-0.3,
                    )
                count = content.count(target_text)
                if count > 1 and not allow_multiple:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Target text matched {count} times in {path}; specify line_number or set allow_multiple=True.",
                        reward=-0.3,
                    )
                new_content = content.replace(target_text, replacement_text) if allow_multiple else content.replace(target_text, replacement_text, 1)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error="Either target_text or line_number must be specified for patching.",
                    reward=-0.2,
                )

            p.write_text(new_content, encoding="utf-8")
            return ToolResult(success=True, output=f"Successfully patched {path}", reward=0.2)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e), reward=-0.5)


class CommandTool(Tool):
    name = "run_command"
    description = "Execute a shell command with timeout."

    def execute(self, command: str, cwd: Optional[str] = None, timeout: float = 10.0) -> ToolResult:
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = res.stdout + ("\n" + res.stderr if res.stderr else "")
            success = (res.returncode == 0)
            reward = 0.2 if success else -0.1
            return ToolResult(success=success, output=out[:2000], reward=reward)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Command timed out", reward=-0.5)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e), reward=-0.5)


class GitStatusTool(Tool):
    name = "git_status"
    description = "Check git status, working tree changes, and active branch."

    def execute(self, cwd: Optional[str] = None, short: bool = True) -> ToolResult:
        try:
            cmd = ["git", "status", "--short"] if short else ["git", "status"]
            res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10.0)
            if res.returncode != 0:
                err = (res.stderr or res.stdout).strip()
                return ToolResult(success=False, output="", error=f"Git status failed: {err}", reward=-0.2)

            branch_res = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            branch = branch_res.stdout.strip() if branch_res.returncode == 0 else "unknown"
            status_body = res.stdout.strip() if res.stdout.strip() else "working tree clean"
            out = f"Branch: {branch}\n{status_body}"
            return ToolResult(success=True, output=out, reward=0.1)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Git status timed out.", reward=-0.5)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e), reward=-0.5)


class GitCommitTool(Tool):
    name = "git_commit"
    description = "Stage files and commit changes to git repository."

    def execute(
        self,
        message: str,
        cwd: Optional[str] = None,
        add_all: bool = True,
        files: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> ToolResult:
        try:
            if add_all:
                add_res = subprocess.run(["git", "add", "-A"], cwd=cwd, capture_output=True, text=True, timeout=timeout)
                if add_res.returncode != 0:
                    return ToolResult(success=False, output="", error=f"git add failed: {add_res.stderr.strip()}", reward=-0.2)
            elif files:
                add_res = subprocess.run(["git", "add"] + files, cwd=cwd, capture_output=True, text=True, timeout=timeout)
                if add_res.returncode != 0:
                    return ToolResult(success=False, output="", error=f"git add failed: {add_res.stderr.strip()}", reward=-0.2)

            res = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if res.returncode == 0:
                return ToolResult(success=True, output=res.stdout.strip(), reward=0.2)
            else:
                err = (res.stderr or res.stdout).strip()
                return ToolResult(success=False, output=res.stdout.strip(), error=f"git commit failed: {err}", reward=-0.2)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Git commit timed out.", reward=-0.5)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e), reward=-0.5)


class GitBranchTool(Tool):
    name = "git_branch"
    description = "Create, switch, or inspect git branches."

    def execute(
        self,
        branch_name: Optional[str] = None,
        create: bool = False,
        cwd: Optional[str] = None,
        timeout: float = 10.0,
    ) -> ToolResult:
        try:
            if not branch_name:
                res = subprocess.run(["git", "branch"], cwd=cwd, capture_output=True, text=True, timeout=timeout)
                if res.returncode == 0:
                    return ToolResult(success=True, output=res.stdout.strip(), reward=0.1)
                return ToolResult(success=False, output="", error=res.stderr.strip(), reward=-0.2)

            cmd = ["git", "checkout", "-b", branch_name] if create else ["git", "checkout", branch_name]
            res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            out = (res.stdout + "\n" + res.stderr).strip()
            if res.returncode == 0:
                return ToolResult(success=True, output=out, reward=0.2)
            else:
                return ToolResult(success=False, output=out, error=res.stderr.strip() or out, reward=-0.2)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Git branch operation timed out.", reward=-0.5)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e), reward=-0.5)


class TestVerifyTool(Tool):
    name = "verify_goal"
    description = "Check if the current autonomous goal completion condition is satisfied."

    def __init__(self, verification_fn: Optional[Any] = None):
        self.verification_fn = verification_fn

    def execute(self, **kwargs) -> ToolResult:
        if self.verification_fn is None:
            return ToolResult(success=True, output="No verification function configured.", reward=1.0)
        try:
            passed = bool(self.verification_fn())
            if passed:
                return ToolResult(success=True, output="GOAL VERIFIED: Task complete!", reward=1.0)
            else:
                return ToolResult(success=False, output="VERIFICATION FAILED: Task not satisfied.", reward=-0.5)
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Verification exception: {e}", reward=-1.0)


class ToolRegistry:
    """Maintains indexed access to tools for policy selection."""

    def __init__(self, tools: Optional[List[Tool]] = None):
        self.tools: List[Tool] = list(tools) if tools else []
        self._name_to_idx: Dict[str, int] = {t.name: i for i, t in enumerate(self.tools)}

    def register(self, tool: Tool) -> int:
        if tool.name in self._name_to_idx:
            idx = self._name_to_idx[tool.name]
            self.tools[idx] = tool
            return idx
        idx = len(self.tools)
        self.tools.append(tool)
        self._name_to_idx[tool.name] = idx
        return idx

    def get_by_index(self, idx: int) -> Tool:
        return self.tools[idx % len(self.tools)]

    def get_by_name(self, name: str) -> Optional[Tool]:
        idx = self._name_to_idx.get(name)
        return self.tools[idx] if idx is not None else None

    def has_tool(self, name: str) -> bool:
        return name in self._name_to_idx

    def get_tool_names(self) -> List[str]:
        return [t.name for t in self.tools]

    def describe_tools(self) -> Dict[str, str]:
        return {t.name: t.description for t in self.tools}

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_idx

    def __iter__(self):
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    @classmethod
    def create_default(cls, verification_fn: Optional[Any] = None) -> "ToolRegistry":
        """Level 15 default registry (basic 4 tools)."""
        return cls([
            FileReadTool(),
            FileWriteTool(),
            CommandTool(),
            TestVerifyTool(verification_fn),
        ])

    @classmethod
    def create_level16_registry(cls, verification_fn: Optional[Any] = None) -> "ToolRegistry":
        """Level 16 expanded engineering registry with multi-file and git capabilities."""
        return cls([
            FileGrepTool(),
            DirectoryListTool(),
            FileReadTool(),
            FilePatchTool(),
            FileWriteTool(),
            CommandTool(),
            GitStatusTool(),
            GitCommitTool(),
            GitBranchTool(),
            TestVerifyTool(verification_fn),
        ])
