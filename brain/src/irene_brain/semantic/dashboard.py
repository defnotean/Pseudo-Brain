"""Interactive Terminal Telemetry Dashboard for Streaming Cognitive Session.

Visualizes in real-time:
- Recurrent slot activation energy matrix (K slots)
- Fast synaptic latch magnitude (P_t norms)
- Cognitive Input Gate (CIG) salience
- Step latency percentiles (mean, p50, p90, p99) and 60 Hz budget compliance
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch


class CognitiveDashboard:
    """Renders real-time ASCII telemetry for an active StreamingCognitiveSession."""

    def __init__(self, use_color: bool = False):
        self.use_color = use_color

    def render_slot_matrix(
        self,
        thoughts: torch.Tensor,
        P_t: torch.Tensor,
        active_thread: int,
        cols: int = 8,
    ) -> str:
        """Render ASCII grid showing energy of each slot."""
        # thoughts: [K, W], P_t: [K, V]
        K = thoughts.shape[0]
        slot_norms = torch.norm(thoughts, dim=-1).detach().cpu().numpy()
        p_norms = torch.norm(P_t, dim=-1).detach().cpu().numpy()

        max_slot = float(np.max(slot_norms)) if len(slot_norms) > 0 else 1.0
        max_slot = max(1e-6, max_slot)

        lines = ["+--- RECURRENT THOUGHT SLOTS (Energy & Synaptic Latching) ---+"]
        
        # Header
        lines.append(f"Total Slots: {K} | Active Thread: Thread #{active_thread}")
        lines.append("-" * 65)

        for row_start in range(0, K, cols):
            row_end = min(row_start + cols, K)
            row_strs = []
            for i in range(row_start, row_end):
                energy_pct = int((slot_norms[i] / max_slot) * 100)
                is_active = (i == active_thread)
                marker = "*" if is_active else " "
                p_val = p_norms[i]
                
                # Visual bar representation
                bar = "#" * max(1, min(5, energy_pct // 20))
                row_strs.append(f"[{i:2d}{marker}:{bar:<5s} p:{p_val:4.2f}]")
            lines.append(" ".join(row_strs))

        lines.append("-" * 65)
        return "\n".join(lines)

    def render_latency_panel(self, telemetry: Dict[str, Any]) -> str:
        """Render latency statistics and 60 Hz budget compliance."""
        mean_l = telemetry.get("latency_mean_ms", 0.0)
        p50_l = telemetry.get("latency_p50_ms", 0.0)
        p90_l = telemetry.get("latency_p90_ms", 0.0)
        p99_l = telemetry.get("latency_p99_ms", 0.0)
        compliant = telemetry.get("latency_under_16_67ms", True)

        status_str = "MET (<= 16.67 ms, 60 Hz Real-Time)" if compliant else "EXCEEDED (> 16.67 ms)"

        lines = [
            "+--- LATENCY & EMBODIED COMPUTATIONAL BUDGET ---+",
            f"Mean Latency: {mean_l:6.2f} ms | p50: {p50_l:6.2f} ms | p90: {p90_l:6.2f} ms | p99: {p99_l:6.2f} ms",
            f"60 Hz Frame Budget Status: [{status_str}]",
            f"Total Tokens Processed:   {telemetry.get('total_tokens', 0)}",
            "+" + "-" * 47 + "+",
        ]
        return "\n".join(lines)

    def render_full_dashboard(self, session: Any) -> str:
        """Render complete dashboard snapshot from session."""
        telem = session.get_telemetry()
        thoughts = session.state.thoughts[0]
        P_t = session.state.P_t[0]
        active_tid = int(session.state.active_thread.item())

        slot_panel = self.render_slot_matrix(thoughts, P_t, active_tid)
        lat_panel = self.render_latency_panel(telem)

        return f"{slot_panel}\n{lat_panel}\n"
