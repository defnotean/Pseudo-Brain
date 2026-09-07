"""Unit tests for Semantic Cognitive Dashboard."""

import unittest
import torch

from irene_brain.semantic.tokenizer import SemanticTokenizer
from irene_brain.semantic.native_semantic_model import NativeSemanticPseudoBrain
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession
from irene_brain.semantic.dashboard import CognitiveDashboard


class SemanticDashboardTests(unittest.TestCase):
    """Test dashboard formatting and telemetry rendering."""

    def setUp(self):
        self.tokenizer = SemanticTokenizer(max_threads=8)
        self.model = NativeSemanticPseudoBrain(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=32,
            K=8,
            thought_size=32,
            proj_dim=64,
        )
        self.session = StreamingCognitiveSession(model=self.model, tokenizer=self.tokenizer)
        self.dashboard = CognitiveDashboard()

    def test_render_slot_matrix(self):
        """Verify slot matrix produces valid string with correct slot count."""
        thoughts = self.session.state.thoughts[0]
        P_t = self.session.state.P_t[0]
        active_tid = int(self.session.state.active_thread.item())

        panel = self.dashboard.render_slot_matrix(thoughts, P_t, active_tid)
        self.assertIn("RECURRENT THOUGHT SLOTS", panel)
        self.assertIn("Thread #0", panel)
        self.assertIn("[ 0*:", panel)  # Active slot 0 marker

    def test_render_latency_panel(self):
        """Verify latency panel formats statistics correctly."""
        self.session.ingest_text("Hello world", thread_id=0)
        telem = self.session.get_telemetry()
        lat_panel = self.dashboard.render_latency_panel(telem)
        self.assertIn("Mean Latency", lat_panel)
        self.assertIn("60 Hz Frame Budget Status", lat_panel)
        self.assertIn("Total Tokens Processed", lat_panel)

    def test_render_full_dashboard(self):
        """Verify complete dashboard rendering end-to-end."""
        self.session.ingest_text("Alpha beta gamma", thread_id=2)
        full_out = self.dashboard.render_full_dashboard(self.session)
        self.assertIn("Thread #2", full_out)
        self.assertIn("[ 2*:", full_out)
        self.assertIn("LATENCY & EMBODIED COMPUTATIONAL BUDGET", full_out)


if __name__ == "__main__":
    unittest.main()
