"""Detailed Sub-Stage Latency Profiler for K=32 / C=3 Kernel.

Profiles every sub-stage of the forward pass:
1. Pixel & Sensor Encoding
2. Control & Time Encoding
3. Context Ingestion & Belief Update
4. Thought Seeding & Gating / Refresh
5. Cycle 1 BrainCell Forward
6. Cycle 2 BrainCell Forward (with cross-thought attention)
7. Cycle 3 BrainCell Forward (with cross-thought attention)
8. Thought Routing & Communication
9. Actuator & Button Readout
10. State Validation & Overhead
"""

from __future__ import annotations

import pathlib
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel


def profile_sub_stages(num_warmup: int = 50, num_steps: int = 200):
    config = ThoughtFieldConfig(
        thoughtlets=32,
        cognitive_cycles=3,
        core_width=32,
        attention_heads=4,
        sensor_tokens=4,
        belief_tokens=4,
        working_memory_tokens=4,
        goal_context_tokens=8,
        registers_per_thoughtlet=2,
        brain_cell_blocks=1,
        routed_neighbors=2,
    )
    model = IreneBrainModel(config=config, input_resolution=(32, 32))
    model.eval()

    device = torch.device("cpu")
    pixels = torch.zeros((1, 3, 32, 32), dtype=torch.float32, device=device)
    prev_ctrl = torch.zeros((1, 307), dtype=torch.float32, device=device)
    dt = torch.tensor([0.016], dtype=torch.float32, device=device)
    state = model.initial_state(1)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            out = model(pixels, prev_ctrl, dt, state)
            state = out.next_state

    # Stage Timers
    timings = {
        "sensory_encoding": [],
        "control_time_encoding": [],
        "belief_ingest": [],
        "thought_refresh_seed": [],
        "cycle_1": [],
        "cycle_2": [],
        "cycle_3": [],
        "actuator_readout": [],
        "total_kernel": [],
    }

    width = config.core_width
    thoughtlets = config.thoughtlets
    registers = config.registers_per_thoughtlet

    for _ in range(num_steps):
        t_start = time.perf_counter_ns()

        with torch.no_grad():
            # 1. Sensory
            t0 = time.perf_counter_ns()
            sensors = model.pixel_encoder(pixels)
            t1 = time.perf_counter_ns()
            timings["sensory_encoding"].append((t1 - t0) / 1e6)

            # 2. Control & Time
            t0 = time.perf_counter_ns()
            control_token = model.control_encoder(prev_ctrl).unsqueeze(1)
            time_input = torch.log1p(dt * 1_000.0).unsqueeze(-1)
            time_token = model.time_encoder(time_input).unsqueeze(1)
            action_time_tokens = torch.cat((control_token, time_token), dim=1)
            t1 = time.perf_counter_ns()
            timings["control_time_encoding"].append((t1 - t0) / 1e6)

            # 3. Belief Ingest
            t0 = time.perf_counter_ns()
            ingest_context = torch.cat(
                (sensors, action_time_tokens, state.belief, state.working_memory),
                dim=1,
            )
            belief_proposal = model.ingest_attention(state.belief, ingest_context)
            belief = model.ingest_blend(state.belief, belief_proposal, dt.unsqueeze(-1))
            t1 = time.perf_counter_ns()
            timings["belief_ingest"].append((t1 - t0) / 1e6)

            # 4. Thought Refresh / Seeding
            t0 = time.perf_counter_ns()
            thoughts, thought_ages, _ = model._refresh_thoughts(
                thoughts=state.thoughts,
                sensors=sensors,
                belief=belief,
                elapsed_seconds=dt.unsqueeze(-1),
                thought_age_seconds=state.thought_age_seconds,
                thought_noise=None,
            )
            t1 = time.perf_counter_ns()
            timings["thought_refresh_seed"].append((t1 - t0) / 1e6)

            # Recurrent Cycles
            working_memory = state.working_memory
            goal_context = state.goal_context
            retrieved_memory = torch.zeros(
                (1, thoughtlets, config.retrieved_entries_per_thoughtlet, width),
                dtype=torch.float32,
            )

            # Cycle 1
            t0 = time.perf_counter_ns()
            allow_routing, allow_workspace_writes = model._communication_policy(0)
            belief, working_memory, thoughts, cycle_routing = model.brain_cell(
                belief=belief,
                working_memory=working_memory,
                thoughts=thoughts,
                sensors=sensors,
                action_time_tokens=action_time_tokens,
                goal_context=goal_context,
                retrieved_memory=retrieved_memory,
                elapsed_seconds=dt.unsqueeze(-1),
                allow_routing=allow_routing,
                allow_workspace_writes=allow_workspace_writes,
            )
            t1 = time.perf_counter_ns()
            timings["cycle_1"].append((t1 - t0) / 1e6)

            # Cycle 2
            t0 = time.perf_counter_ns()
            allow_routing, allow_workspace_writes = model._communication_policy(1)
            belief, working_memory, thoughts, cycle_routing = model.brain_cell(
                belief=belief,
                working_memory=working_memory,
                thoughts=thoughts,
                sensors=sensors,
                action_time_tokens=action_time_tokens,
                goal_context=goal_context,
                retrieved_memory=retrieved_memory,
                elapsed_seconds=dt.unsqueeze(-1),
                allow_routing=allow_routing,
                allow_workspace_writes=allow_workspace_writes,
            )
            t1 = time.perf_counter_ns()
            timings["cycle_2"].append((t1 - t0) / 1e6)

            # Cycle 3
            t0 = time.perf_counter_ns()
            allow_routing, allow_workspace_writes = model._communication_policy(2)
            belief, working_memory, thoughts, cycle_routing = model.brain_cell(
                belief=belief,
                working_memory=working_memory,
                thoughts=thoughts,
                sensors=sensors,
                action_time_tokens=action_time_tokens,
                goal_context=goal_context,
                retrieved_memory=retrieved_memory,
                elapsed_seconds=dt.unsqueeze(-1),
                allow_routing=allow_routing,
                allow_workspace_writes=allow_workspace_writes,
            )
            t1 = time.perf_counter_ns()
            timings["cycle_3"].append((t1 - t0) / 1e6)

            # 5. Actuator Readout
            t0 = time.perf_counter_ns()
            action_bundle = model.actuator(
                sensors=sensors,
                belief=belief,
                thoughts=thoughts,
                working_memory=working_memory,
                retrieved_memory=retrieved_memory,
                goal_context=goal_context,
            )
            t1 = time.perf_counter_ns()
            timings["actuator_readout"].append((t1 - t0) / 1e6)

        t_end = time.perf_counter_ns()
        timings["total_kernel"].append((t_end - t_start) / 1e6)

    print("=" * 80)
    print("SUB-STAGE LATENCY PROFILING BREAKDOWN (K=32, C=3):")
    print("=" * 80)
    for stage, times in timings.items():
        arr = np.array(times)
        p50 = np.percentile(arr, 50)
        p95 = np.percentile(arr, 95)
        p99 = np.percentile(arr, 99)
        mean = np.mean(arr)
        pct = (mean / np.mean(timings["total_kernel"])) * 100.0 if stage != "total_kernel" else 100.0
        print(f"  {stage:<25}: mean = {mean:5.2f} ms ({pct:4.1f}%) | p50 = {p50:5.2f} ms | p95 = {p95:5.2f} ms | p99 = {p99:5.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    profile_sub_stages()
