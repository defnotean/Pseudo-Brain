"""Autonomous Agent Subsystem for Pseudo-Brain.

Bridges natural-language goals, persistent cognitive state, tool invocation,
consequence-gated predictive self-correction, and verifiable task completion.
"""
from __future__ import annotations

from .goal import GoalSpecification, GoalEncoder
from .tools import Tool, ToolRegistry, CommandTool, FileReadTool, FileWriteTool, TestVerifyTool
from .loop import PseudoBrainAgent, AgentStepLog, run_autonomous_task

__all__ = [
    "GoalSpecification",
    "GoalEncoder",
    "Tool",
    "ToolRegistry",
    "CommandTool",
    "FileReadTool",
    "FileWriteTool",
    "TestVerifyTool",
    "PseudoBrainAgent",
    "AgentStepLog",
    "run_autonomous_task",
]
