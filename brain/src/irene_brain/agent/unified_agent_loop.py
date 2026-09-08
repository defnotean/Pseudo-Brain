"""Unified Autonomous Cognitive Agent Loop for Pseudo-Brain.

Integrates:
1. UnifiedPseudoBrain multi-modal recurrent core (Law 1 4.0 KB working cache + episodic store).
2. Decoupled Readout Heads: Language/Code logits, discrete action selection, outcome value prediction, tool gate.
3. CodeExecutionEngine: In-process & subprocess Python code execution, syntax validation, and output capture.
4. Closed-Loop Self-Repair with Contextual Inhibition of Return (IOR) and consequence surprise modulation.
5. Autonomous multi-step tool-calling DAG execution with verification.
"""

from __future__ import annotations

import ast
import io
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from irene_brain.unified.unified_model import UnifiedPseudoBrain, UnifiedCognitiveState, make_unified_model
from irene_brain.agent.tools import Tool, ToolRegistry, ToolResult, CommandTool, FileReadTool, FileWriteTool, FilePatchTool, TestVerifyTool
from irene_brain.agent.goal import GoalSpecification


@dataclass
class ExecutionResult:
    """Result of a Python code execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    elapsed_ms: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class CodeExecutionEngine:
    """Safe, isolated Python code execution engine with timeout and error extraction."""

    def __init__(self, default_timeout: float = 10.0):
        self.default_timeout = default_timeout

    def check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax prior to execution."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def execute_python(
        self,
        code: str,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute Python code in a subprocess and return detailed execution telemetry."""
        timeout_val = timeout if timeout is not None else self.default_timeout

        # Syntax check first
        valid, syntax_err = self.check_syntax(code)
        if not valid:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=syntax_err or "Syntax error",
                return_code=-1,
                elapsed_ms=0.0,
                error_type="SyntaxError",
                error_message=syntax_err,
            )

        t0 = time.perf_counter()
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        try:
            res = subprocess.run(
                [sys.executable, "-c", code],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_val,
                env=run_env,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            success = (res.returncode == 0)

            err_type = None
            err_msg = None
            if not success and res.stderr:
                err_lines = res.stderr.strip().splitlines()
                err_msg = err_lines[-1] if err_lines else "Unknown execution error"
                if ":" in err_msg:
                    err_type = err_msg.split(":")[0].strip()

            return ExecutionResult(
                success=success,
                stdout=res.stdout,
                stderr=res.stderr,
                return_code=res.returncode,
                elapsed_ms=elapsed_ms,
                error_type=err_type,
                error_message=err_msg,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout_val} seconds.",
                return_code=-2,
                elapsed_ms=elapsed_ms,
                error_type="TimeoutExpired",
                error_message=f"Timeout after {timeout_val}s",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-3,
                elapsed_ms=elapsed_ms,
                error_type=type(e).__name__,
                error_message=str(e),
            )


@dataclass
class CognitiveAgentStepLog:
    """Detailed telemetry record for a cognitive agent step."""
    step: int
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    success: bool
    reward: float
    predicted_reward: float
    consequence_surprise: float
    tool_prob: float
    ior_suppression: Dict[str, float]
    output_snippet: str
    code_snippet: Optional[str] = None


@dataclass
class CognitiveTaskReport:
    """Structured report of an autonomous cognitive agent run."""
    goal_id: str
    goal_text: str
    success: bool
    total_steps: int
    iterations: int
    cumulative_reward: float
    steps_log: List[CognitiveAgentStepLog] = field(default_factory=list)
    final_code: Optional[str] = None
    verification_output: Optional[str] = None


class UnifiedCognitiveAgent:
    """Production Autonomous Agent utilizing UnifiedPseudoBrain for Tool-Calling & Self-Repairing Code."""

    def __init__(
        self,
        model: Optional[UnifiedPseudoBrain] = None,
        registry: Optional[ToolRegistry] = None,
        device: Optional[torch.device] = None,
        tier: str = "tier2",
        vocab_size: int = 32000,
    ):
        self.device = device if device is not None else torch.device("cpu")
        if model is None:
            model = make_unified_model(tier=tier, vocab_size=vocab_size)
        self.model = model.to(self.device)
        self.model.eval()

        self.registry = registry if registry is not None else ToolRegistry.create_level16_registry()
        self.code_engine = CodeExecutionEngine()

        # Contextual Inhibition of Return (IOR) memory: tool_name -> suppression strength
        self.ior_memory: Dict[str, float] = {}
        # Consequence history for online adaptation
        self.consequence_history: List[float] = []

    def reset_state(self, batch_size: int = 1) -> UnifiedCognitiveState:
        """Create clean persistent cognitive state for a new autonomous task."""
        self.ior_memory.clear()
        self.consequence_history.clear()
        return self.model.init_state(batch_size=batch_size, device=self.device)

    def _text_to_sensory(self, text: str) -> Tensor:
        """Project input observation / goal text into sensory representation."""
        # Simple deterministic hash embedding to vocab indices for sensory projection
        chars = [ord(c) % self.model.vocab_size for c in text[:128]]
        if not chars:
            chars = [0]
        toks = torch.tensor(chars, dtype=torch.long, device=self.device)
        emb = self.model.embedding(toks).mean(dim=0, keepdim=True)  # [1, embed_dim]
        sensory = self.model.lang_proj(emb)  # [1, proj_dim]
        if self.model.deep_proj is not None:
            sensory = sensory + self.model.deep_proj(sensory)
        return sensory

    def step_cognitive_cycle(
        self,
        observation: str,
        state: UnifiedCognitiveState,
        last_reward: float = 0.0,
    ) -> Tuple[Dict[str, Any], UnifiedCognitiveState]:
        """Execute single recurrent cognitive step through UnifiedPseudoBrain."""
        sensory = self._text_to_sensory(observation)

        with torch.no_grad():
            outputs, next_state = self.model.step(sensory, state, allow_routing=True)

        predicted_reward = float(outputs["predicted_value"].item())
        tool_prob = float(outputs["tool_prob"].item())
        action_logits = outputs["action_logits"].squeeze(0)  # [n_actions]

        # Consequence surprise calculation: delta = r_hat - r
        consequence_surprise = abs(predicted_reward - last_reward)
        self.consequence_history.append(consequence_surprise)

        result = {
            "predicted_reward": predicted_reward,
            "tool_prob": tool_prob,
            "action_logits": action_logits,
            "consequence_surprise": consequence_surprise,
            "language_logits": outputs["logits"],
        }
        return result, next_state

    def update_ior(
        self,
        tool_name: str,
        success: bool,
        consequence_surprise: float,
        decay: float = 0.85,
    ) -> None:
        """Update Contextual Inhibition of Return (IOR) suppression values."""
        # Decay existing suppressions
        for name in list(self.ior_memory.keys()):
            self.ior_memory[name] *= decay
            if self.ior_memory[name] < 0.05:
                del self.ior_memory[name]

        # Apply heavy suppression to failed actions to prevent repetitive cycling
        if not success:
            penalty = 2.0 + 3.0 * min(consequence_surprise, 2.0)
            self.ior_memory[tool_name] = self.ior_memory.get(tool_name, 0.0) + penalty
        else:
            # Gentle discount for completed action to encourage progression
            self.ior_memory[tool_name] = self.ior_memory.get(tool_name, 0.0) + 0.5

    def select_tool(
        self,
        action_logits: Tensor,
        tool_prob_threshold: float = 0.3,
    ) -> Tuple[str, int]:
        """Select best tool considering policy logits and Contextual IOR suppression."""
        tool_names = self.registry.get_tool_names()
        n = min(len(tool_names), action_logits.shape[0])

        scores = action_logits[:n].clone()
        for idx in range(n):
            name = tool_names[idx]
            suppression = self.ior_memory.get(name, 0.0)
            scores[idx] -= suppression

        best_idx = int(scores.argmax().item())
        return tool_names[best_idx], best_idx

    def self_repair_code(
        self,
        buggy_code: str,
        test_script: str,
        cwd: Optional[str] = None,
        max_attempts: int = 4,
        repair_fn: Optional[Callable[[str, ExecutionResult], str]] = None,
    ) -> Tuple[bool, str, List[ExecutionResult]]:
        """Self-repair Python code using execution feedback and Contextual IOR."""
        state = self.reset_state()
        current_code = buggy_code
        history: List[ExecutionResult] = []

        last_reward = 0.0

        for attempt in range(max_attempts):
            # 1. Run the code
            res = self.code_engine.execute_python(current_code + "\n" + test_script, cwd=cwd)
            history.append(res)

            if res.success:
                # Goal satisfied!
                obs = f"Step {attempt}: SUCCESS. All tests passed."
                self.step_cognitive_cycle(obs, state, last_reward=1.0)
                return True, current_code, history

            # 2. Cognitive step with error observation
            err_text = res.stderr if res.stderr else (res.error_message or "Execution failed")
            obs = f"Step {attempt} FAILED with {res.error_type or 'Error'}: {err_text[:100]}"
            cog_res, state = self.step_cognitive_cycle(obs, state, last_reward=-0.5)

            # 3. Contextual IOR: suppress repeating identical erroneous transformation
            action_tag = f"patch_attempt_{attempt}"
            self.update_ior(action_tag, success=False, consequence_surprise=cog_res["consequence_surprise"])

            # 4. Synthesize repair
            if repair_fn is not None:
                current_code = repair_fn(current_code, res)
            else:
                # Heuristic neural patcher: replace common syntax/operator bugs
                if "SyntaxError" in (res.error_type or ""):
                    # Fix unclosed parentheses, colons, or indentation
                    lines = current_code.splitlines()
                    for idx, line in enumerate(lines):
                        if line.strip().startswith("def ") or line.strip().startswith("if ") or line.strip().startswith("for "):
                            if not line.strip().endswith(":"):
                                lines[idx] = line + ":"
                    current_code = "\n".join(lines) + "\n"
                elif "AssertionError" in err_text or "AssertionError" in (res.error_type or ""):
                    # Invert wrong arithmetic or comparison operators
                    if " - " in current_code:
                        current_code = current_code.replace(" - ", " + ", 1)
                    elif " + " in current_code:
                        current_code = current_code.replace(" + ", " - ", 1)
                    elif " * " in current_code:
                        current_code = current_code.replace(" * ", " / ", 1)
                    elif " / " in current_code:
                        current_code = current_code.replace(" / ", " * ", 1)

            last_reward = -0.5

        # Final check
        final_res = self.code_engine.execute_python(current_code + "\n" + test_script, cwd=cwd)
        history.append(final_res)
        return final_res.success, current_code, history

    def run_autonomous_task(
        self,
        goal: GoalSpecification,
        max_steps: int = 10,
        arg_provider: Optional[Callable[[int, str, List[CognitiveAgentStepLog]], Dict[str, Any]]] = None,
    ) -> CognitiveTaskReport:
        """Run complete autonomous task with multi-step tool execution and verification."""
        state = self.reset_state()
        steps_log: List[CognitiveAgentStepLog] = []
        cumulative_reward = 0.0

        observation = f"Goal: {goal.text}"
        last_reward = 0.0

        for step_idx in range(max_steps):
            # 1. Cognitive recurrent step
            cog_res, state = self.step_cognitive_cycle(observation, state, last_reward=last_reward)

            # 2. Tool selection with IOR
            tool_name, tool_idx = self.select_tool(cog_res["action_logits"])
            tool = self.registry.get_by_name(tool_name)

            # 3. Resolve tool arguments
            args: Dict[str, Any] = {}
            if arg_provider is not None:
                args = arg_provider(step_idx, tool_name, steps_log)

            # 4. Execute tool
            try:
                if tool is not None:
                    tool_res = tool.execute(**args)
                else:
                    tool_res = ToolResult(success=False, output="", error=f"Tool {tool_name} not found", reward=-0.5)
            except Exception as e:
                tool_res = ToolResult(success=False, output="", error=f"Tool {tool_name} failed: {e}", reward=-0.2)

            # 5. Reward & IOR update
            step_reward = tool_res.reward
            cumulative_reward += step_reward
            self.update_ior(tool_name, tool_res.success, cog_res["consequence_surprise"])

            # 6. Log telemetry
            snippet = (tool_res.output if tool_res.success else (tool_res.error or ""))[:150]
            log_entry = CognitiveAgentStepLog(
                step=step_idx,
                tool_name=tool_name,
                tool_args=args,
                success=tool_res.success,
                reward=step_reward,
                predicted_reward=cog_res["predicted_reward"],
                consequence_surprise=cog_res["consequence_surprise"],
                tool_prob=cog_res["tool_prob"],
                ior_suppression=dict(self.ior_memory),
                output_snippet=snippet,
            )
            steps_log.append(log_entry)

            # 7. Check goal satisfaction
            if goal.verification_fn is not None and goal.verification_fn():
                return CognitiveTaskReport(
                    goal_id=goal.goal_id,
                    goal_text=goal.text,
                    success=True,
                    total_steps=step_idx + 1,
                    iterations=step_idx + 1,
                    cumulative_reward=cumulative_reward,
                    steps_log=steps_log,
                    verification_output="Task verified successfully.",
                )

            # Update observation for next step
            observation = f"Tool '{tool_name}' returned: {snippet}"
            last_reward = step_reward

        # Check final verification
        is_success = bool(goal.verification_fn and goal.verification_fn())
        return CognitiveTaskReport(
            goal_id=goal.goal_id,
            goal_text=goal.text,
            success=is_success,
            total_steps=max_steps,
            iterations=max_steps,
            cumulative_reward=cumulative_reward,
            steps_log=steps_log,
            verification_output="Task complete." if is_success else "Max steps reached without verification.",
        )
