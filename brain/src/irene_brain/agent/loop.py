"""Persistent Pseudo-Brain Autonomous Agent Cognitive Loop.

Integrates:
1. Persistent Thoughtlet Recurrent Core (h_t)
2. Cognitive Input Gating (g_t)
3. Consequence-Gated Fast Plasticity (P_t)
4. Predictive Future Latent & Outcome Modeling (z_hat, r_hat)
5. Dynamic Trace Resetting upon Consequence Failure
6. Closed-Loop Tool Execution & Verification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .goal import GoalSpecification, GoalEncoder
from .tools import Tool, ToolRegistry, ToolResult, TestVerifyTool


@dataclass
class AgentStepLog:
    """Telemetry log for a single cognitive agent step."""
    step: int
    action_idx: int
    tool_name: str
    tool_args: Dict[str, Any]
    success: bool
    reward: float
    predicted_reward: float
    consequence_surprise: float
    gate_activation: float
    p_norm: float
    output_snippet: str


@dataclass
class AgentTaskReport:
    """Summary report for an autonomous task run."""
    goal_id: str
    goal_text: str
    success: bool
    total_steps: int
    cumulative_reward: float
    steps_log: List[AgentStepLog] = field(default_factory=list)


class AgentCognitiveCore(nn.Module):
    """Neural cognitive policy and world model for the autonomous agent."""

    def __init__(
        self,
        goal_dim: int = 128,
        obs_dim: int = 128,
        thought_dim: int = 384,
        n_tools: int = 4,
        decay: float = 0.98,
        lr: float = 0.25,
    ):
        super().__init__()
        self.goal_dim = goal_dim
        self.obs_dim = obs_dim
        self.thought_dim = thought_dim
        self.n_tools = n_tools
        self.decay = decay
        self.lr = lr

        # 1. Cognitive Input Gate: g_t in [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, thought_dim),
            nn.Sigmoid(),
        )

        # 2. Recurrent Transition Core
        self.cell = nn.GRUCell(obs_dim + goal_dim, thought_dim)

        # 3. Future Latent Predictor
        self.latent_pred = nn.Sequential(
            nn.Linear(thought_dim + n_tools, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim),
        )

        # 4. Consequence Predictor (r_hat)
        self.reward_pred = nn.Sequential(
            nn.Linear(thought_dim + n_tools, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 5. Fast Plasticity Modulator
        self.p_modulator = nn.Linear(thought_dim + 1, n_tools)
        self.p_gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.p_gate[0].bias, -2.0)
        self.scale = nn.Parameter(torch.tensor(1.5))

        # 6. Policy Head
        self.policy_head = nn.Linear(thought_dim, n_tools)

    def forward_step(
        self,
        obs_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        P_t: torch.Tensor,
        consequence_surprise: torch.Tensor,
        last_action: Optional[int] = None,
        last_success: Optional[bool] = None,
        *,
        state_novelty: float = 0.0,
        suppression_decay: float = 1.0,
        consecutive_failures: int = 0,
        specific_suppression: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Runs a single recurrent cognitive step.

        Returns:
            logits: (1, n_tools)
            h_next: (1, thought_dim)
            P_next: (1, n_tools)
            gate_val: scalar cognitive gate intensity
        """
        device = obs_emb.device
        if h_prev is None:
            h_prev = torch.zeros(1, self.thought_dim, device=device)

        inp = torch.cat([obs_emb, goal_emb], dim=-1)

        # Endogenous Cognitive Input Gating
        g_t = self.gate_net(inp)
        h_candidate = self.cell(inp, h_prev)
        h_next = (1.0 - g_t) * h_prev + g_t * h_candidate

        # Consequence-Gated Dynamic Trace Reset
        # When consequence surprise > 0.5 (outcome failure), discount stale policy weights
        reset_gate = torch.sigmoid((consequence_surprise - 0.5) / 0.1)
        effective_decay = self.decay * (1.0 - 0.8 * reset_gate)

        # Plastic Update
        gate = self.p_gate(consequence_surprise)
        delta_P = gate * torch.tanh(self.p_modulator(torch.cat([h_next, consequence_surprise], dim=-1)))
        P_next = effective_decay * P_t + self.lr * delta_P

        # Inhibition of Return & Subgoal Progression:
        # Actively penalize repeating an action that failed, and discount immediate re-execution of completed action
        if specific_suppression is not None:
            suppression = torch.zeros_like(P_next)
            for act_idx, supp_val in specific_suppression.items():
                if 0 <= act_idx < self.n_tools:
                    suppression[0, act_idx] = supp_val
            P_next = P_next + suppression
        elif last_action is not None:
            if last_success is False:
                # State-contingent IOR: decay suppression exponentially if state novelty or repair occurred
                decay = np.exp(-2.0 * max(0.0, state_novelty)) * max(0.0, suppression_decay)
                suppression = torch.zeros_like(P_next)
                suppression[0, last_action] = -4.0 * float(consequence_surprise.item() + 1.0) * float(decay)
                P_next = P_next + suppression
            elif last_success is True:
                discount = torch.zeros_like(P_next)
                discount[0, last_action] = -1.2
                P_next = P_next + discount

        # Combined Policy Logits
        base_logits = self.policy_head(h_next)
        logits = base_logits + self.scale * P_next
        if consecutive_failures >= 2:
            temperature = 1.0 + 0.3 * min(5, consecutive_failures)
            logits = logits / temperature

        gate_val = float(g_t.mean().item())
        return logits, h_next, P_next, gate_val

    def set_tool_bias(self, tool_idx: int, bias: float) -> None:
        """Adjusts the baseline logit bias for a specific tool."""
        with torch.no_grad():
            self.policy_head.bias[tool_idx] += bias

    def predict_future(self, h: torch.Tensor, action: int) -> Tuple[torch.Tensor, float]:
        a_vec = torch.zeros(1, self.n_tools, device=h.device)
        a_vec[0, action] = 1.0
        ha = torch.cat([h, a_vec], dim=-1)
        z_hat = self.latent_pred(ha)
        r_hat = float(self.reward_pred(ha).item())
        return z_hat, r_hat


class PseudoBrainAgent:
    """Fully integrated autonomous agent powered by Pseudo-Brain cognitive core."""

    def __init__(
        self,
        registry: ToolRegistry,
        goal_dim: int = 128,
        obs_dim: int = 128,
        thought_dim: int = 384,
        device_str: str = "cpu",
    ):
        self.registry = registry
        self.device = torch.device(device_str)
        self.goal_encoder = GoalEncoder(embedding_dim=goal_dim).to(self.device)
        self.core = AgentCognitiveCore(
            goal_dim=goal_dim,
            obs_dim=obs_dim,
            thought_dim=thought_dim,
            n_tools=len(registry),
        ).to(self.device)

        # Simple hashing encoder for observation strings
        self.obs_encoder = nn.Sequential(
            nn.Linear(64, obs_dim),
            nn.LayerNorm(obs_dim),
            nn.GELU(),
        ).to(self.device)

    def _encode_observation(
        self,
        text: str,
        last_tool_idx: Optional[int] = None,
        last_success: Optional[bool] = None,
        last_reward: float = 0.0,
    ) -> torch.Tensor:
        import zlib
        words = text.strip().split() if text else ["<empty>"]
        hashes = [int(zlib.crc32(w.lower().encode("utf-8")) & 0xFFFFFFFF) % 60 for w in words[:16]]
        vec = torch.zeros(1, 64, device=self.device)
        for h in hashes:
            vec[0, h] += 1.0

        if last_tool_idx is not None and 0 <= last_tool_idx < 60:
            vec[0, last_tool_idx] += 2.0
        if last_success is not None:
            vec[0, 61] = 2.0 if last_success else -2.0
        vec[0, 62] = float(np.clip(last_reward, -2.0, 2.0))
        return self.obs_encoder(vec)

    def run_task(
        self,
        goal: GoalSpecification,
        max_steps: int = 15,
        default_args: Optional[Dict[str, Dict[str, Any]]] = None,
        arg_provider: Optional[Callable[[int, str, List[AgentStepLog], GoalSpecification], Dict[str, Any]]] = None,
        action_selector: Optional[Callable[[torch.Tensor, List[AgentStepLog]], int]] = None,
    ) -> AgentTaskReport:
        """Executes the autonomous cognitive loop on the given goal."""
        default_args = default_args or {}
        goal_emb = self.goal_encoder(goal.text).to(self.device)

        h_t = None
        P_t = torch.zeros(1, len(self.registry), device=self.device)
        consequence_surprise = torch.zeros(1, 1, device=self.device)
        obs_emb = torch.zeros(1, self.core.obs_dim, device=self.device)
        last_failure_obs_emb: Optional[torch.Tensor] = None

        logs: List[AgentStepLog] = []
        cum_reward = 0.0
        complete = False
        last_action_idx: Optional[int] = None
        last_success: Optional[bool] = None
        last_reward = 0.0
        consecutive_failures = 0

        for step in range(1, max_steps + 1):
            state_novelty = 0.0
            if last_failure_obs_emb is not None and last_success is False:
                state_novelty = float(torch.norm(obs_emb - last_failure_obs_emb).item())

            # 1. Thought / Planning / Policy forward step with outcome history
            with torch.no_grad():
                logits, h_t, P_t, gate_val = self.core.forward_step(
                    obs_emb=obs_emb,
                    goal_emb=goal_emb,
                    h_prev=h_t,
                    P_t=P_t,
                    consequence_surprise=consequence_surprise,
                    last_action=last_action_idx,
                    last_success=last_success,
                    state_novelty=state_novelty,
                    consecutive_failures=consecutive_failures,
                )
                if action_selector is not None:
                    action_idx = action_selector(logits, logs)
                else:
                    action_idx = int(logits.argmax(dim=-1).item())
                _, r_hat = self.core.predict_future(h_t, action_idx)

            # 2. Select & execute tool with dynamic or default arguments
            tool = self.registry.get_by_index(action_idx)
            tool_name = tool.name
            if arg_provider is not None:
                args = arg_provider(step, tool_name, logs, goal)
            else:
                args = default_args.get(tool_name, {})

            tool_res = tool.execute(**args)
            r_true = tool_res.reward
            cum_reward += r_true

            # 3. Compute consequence prediction error (surprise)
            delta_r = abs(r_true - r_hat)
            consequence_surprise = torch.tensor([[delta_r]], device=self.device, dtype=torch.float32)

            # 4. Update observation embedding
            obs_snippet = tool_res.output[:80] if tool_res.output else (tool_res.error or "")[:80]
            obs_emb = self._encode_observation(
                text=tool_res.output + " " + (tool_res.error or ""),
                last_tool_idx=action_idx,
                last_success=tool_res.success,
                last_reward=r_true,
            )

            if tool_res.success is False:
                consecutive_failures += 1
                last_failure_obs_emb = obs_emb.clone()
            else:
                consecutive_failures = 0
                last_failure_obs_emb = None

            last_action_idx = action_idx
            last_success = tool_res.success
            last_reward = r_true

            # 5. Log telemetry
            p_norm = float(P_t.norm().item())
            logs.append(
                AgentStepLog(
                    step=step,
                    action_idx=action_idx,
                    tool_name=tool_name,
                    tool_args=args,
                    success=tool_res.success,
                    reward=r_true,
                    predicted_reward=r_hat,
                    consequence_surprise=delta_r,
                    gate_activation=gate_val,
                    p_norm=p_norm,
                    output_snippet=obs_snippet,
                )
            )

            # 6. Check verification condition
            if goal.is_complete():
                complete = True
                break

        return AgentTaskReport(
            goal_id=goal.goal_id,
            goal_text=goal.text,
            success=complete,
            total_steps=len(logs),
            cumulative_reward=cum_reward,
            steps_log=logs,
        )


def run_autonomous_task(
    goal_text: str,
    verification_fn: Callable[[], bool],
    tools: Optional[List[Tool]] = None,
    default_args: Optional[Dict[str, Dict[str, Any]]] = None,
    max_steps: int = 15,
) -> AgentTaskReport:
    """Helper entrypoint to instantiate and execute an autonomous task."""
    reg = ToolRegistry(tools or [
        FileReadTool(),
        FileWriteTool(),
        CommandTool(),
        TestVerifyTool(verification_fn),
    ])
    agent = PseudoBrainAgent(registry=reg)
    goal = GoalSpecification(
        goal_id="auto_task_01",
        text=goal_text,
        verification_fn=verification_fn,
    )
    return agent.run_task(goal, max_steps=max_steps, default_args=default_args)
