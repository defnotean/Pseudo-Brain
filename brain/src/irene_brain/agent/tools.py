"""Controlled Tools and Registry for Pseudo-Brain Autonomous Agent."""
from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
        self.tools: List[Tool] = tools or []
        self._name_to_idx: Dict[str, int] = {t.name: i for i, t in enumerate(self.tools)}

    def register(self, tool: Tool) -> int:
        idx = len(self.tools)
        self.tools.append(tool)
        self._name_to_idx[tool.name] = idx
        return idx

    def get_by_index(self, idx: int) -> Tool:
        return self.tools[idx % len(self.tools)]

    def get_by_name(self, name: str) -> Optional[Tool]:
        idx = self._name_to_idx.get(name)
        return self.tools[idx] if idx is not None else None

    def __len__(self) -> int:
        return len(self.tools)
