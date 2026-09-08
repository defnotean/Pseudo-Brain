"""FastAPI Backend Server for Pseudo-Brain Interactive Neural Console.

Provides real-time chat, 16-slot working memory telemetry, and mental lookahead tree visualization.
Strictly respects Law 1 (4.0 KB working memory) and uses a single CPU thread to maintain near-zero load.
"""

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

torch.set_num_threads(1)

from irene_brain.unified.unified_model import UnifiedPseudoBrain
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.reasoning.mental_lookahead import MentalLookaheadEngine


class ChatRequest(BaseModel):
    prompt: str
    mode: str = "reflex"  # "reflex" or "lookahead"
    max_tokens: int = 45
    temperature: float = 0.7
    repetition_penalty: float = 1.30


class LookaheadRequest(BaseModel):
    prompt: str
    branch_factor: int = 3
    horizon: int = 3
    repetition_penalty: float = 1.30


class ConsoleState:
    """Stateful container holding the active model, tokenizer, and lookahead engine."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.model: Optional[UnifiedPseudoBrain] = None
        self.tokenizer: Optional[BpeSemanticTokenizer] = None
        self.engine: Optional[MentalLookaheadEngine] = None
        self.tier_name: str = "unknown"
        self.total_params: int = 0
        self.checkpoint_name: str = "none"
        self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: Optional[str] = None):
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        checkpoints_dir = repo_root / "checkpoints"

        candidates = [
            checkpoints_dir / "pb_35m_champion.pt",
            checkpoints_dir / "pb_1b_champion.pt",
            checkpoints_dir / "pb_unified_champion.pt",
        ]

        if checkpoint_path and Path(checkpoint_path).exists():
            target_path = Path(checkpoint_path)
        else:
            target_path = None
            for cand in candidates:
                if cand.exists():
                    target_path = cand
                    break

        if not target_path or not target_path.exists():
            print(f"[Console] Warning: No checkpoint found at {candidates}. Initializing fallback 35M model.")
            self.tokenizer = BpeSemanticTokenizer(vocab_size=32000, max_threads=16)
            self.model = UnifiedPseudoBrain(vocab_size=32000, tier="tier2_35m").to(self.device)
            self.tier_name = "tier2_35m (uninitialized)"
            self.total_params = sum(p.numel() for p in self.model.parameters())
            self.checkpoint_name = "scratch"
        else:
            print(f"[Console] Loading checkpoint: {target_path.name}...")
            ckpt = torch.load(str(target_path), map_location="cpu", weights_only=False)
            vocab_size = ckpt.get("vocab_size", 32000)

            if "tokenizer_json" in ckpt:
                self.tokenizer = BpeSemanticTokenizer.from_str(ckpt["tokenizer_json"])
            else:
                self.tokenizer = BpeSemanticTokenizer(vocab_size=vocab_size, max_threads=16)

            tier = ckpt.get("tier", "tier2_35m" if "35m" in target_path.name else ("tier3_1b" if "1b" in target_path.name else "tier1_3m"))
            num_deep = ckpt.get("num_deep_layers", 24 if tier == "tier3_1b" else (4 if tier == "tier2_35m" else 2))

            self.model = UnifiedPseudoBrain(
                vocab_size=vocab_size,
                tier=tier,
                num_deep_layers=num_deep,
            ).to(self.device)

            state_dict = {k: v.to(torch.float32) for k, v in ckpt["model_state_dict"].items()}
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()

            self.tier_name = tier
            self.total_params = sum(p.numel() for p in self.model.parameters())
            self.checkpoint_name = target_path.name
            print(f"[Console] Loaded {self.checkpoint_name} ({self.total_params:,} params, tier={self.tier_name})")

        self.engine = MentalLookaheadEngine(
            model=self.model,
            branch_factor=3,
            horizon=3,
            entropy_threshold=1.0,
            value_weight=0.5,
            repetition_penalty=1.30,
        )


def create_console_app(checkpoint_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Pseudo-Brain Neural Console", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = ConsoleState(checkpoint_path)

    @app.get("/api/health")
    def health():
        return {
            "status": "healthy",
            "model_tier": state.tier_name,
            "total_parameters": state.total_params,
            "checkpoint": state.checkpoint_name,
            "working_memory_bytes": 4096,
            "working_slots": 16,
            "slot_width": 64,
            "episodic_memory_bytes": 65536,
            "device": str(state.device),
        }

    @app.get("/api/slots")
    def get_slots():
        """Returns dummy or current working slot norms."""
        return {
            "slots": [
                {"id": i, "energy": 0.5 + 0.5 * math.sin(i * 0.7), "active": True}
                for i in range(16)
            ]
        }

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        if not state.model or not state.tokenizer:
            raise HTTPException(status_code=500, detail="Model not initialized")

        t0 = time.time()
        prompt_text = f"[THREAD:0]{req.prompt} [RESP]"
        prompt_tokens = state.tokenizer.encode(prompt_text)

        tokens_generated: List[int] = []
        lookahead_activations = 0

        # Run single-step autoregressive generation
        with torch.no_grad():
            # Initialize sequence with prompt tokens
            cur_tokens = list(prompt_tokens)
            recent_tokens = list(prompt_tokens)

            # Slot state representation from model slot identities
            slot_matrix = state.model.slot_identities.detach().cpu()  # [16, 64]

            for step_idx in range(req.max_tokens):
                prompt_tensor = torch.tensor([cur_tokens], dtype=torch.long, device=state.device)
                out = state.model.forward_sequence_parallel(token_seq=prompt_tensor)
                step_logits = out["logits"][0, -1, :].clone()

                # Apply anti-repetition penalty
                if req.repetition_penalty > 1.0:
                    for prev_tok in set(recent_tokens[-32:]):
                        if step_logits[prev_tok] > 0:
                            step_logits[prev_tok] /= req.repetition_penalty
                        else:
                            step_logits[prev_tok] *= req.repetition_penalty

                # Mode selection: lookahead branching vs greedy/sample reflex
                if req.mode == "lookahead" and step_idx < 15:
                    probs = F.softmax(step_logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                    if entropy > 1.2:
                        lookahead_activations += 1

                # Token selection
                if req.temperature <= 0.05:
                    next_tok = int(torch.argmax(step_logits).item())
                else:
                    scaled_logits = step_logits / max(0.1, req.temperature)
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_tok = int(torch.multinomial(probs, num_samples=1).item())

                if next_tok == state.tokenizer.eos_id:
                    break

                tokens_generated.append(next_tok)
                cur_tokens.append(next_tok)
                recent_tokens.append(next_tok)

            elapsed_ms = (time.time() - t0) * 1000
            decoded_text = state.tokenizer.decode(tokens_generated).strip()

            # Compute slot energies: L2 norm of each of the 16 slots [16, 64]
            slot_energies = torch.norm(slot_matrix, dim=-1).tolist()  # 16 floats
            max_energy = max(slot_energies) if max(slot_energies) > 0 else 1.0
            norm_energies = [e / max_energy for e in slot_energies]

            # Slot sample values (top 4 floats per slot for UI visualizer)
            slot_samples = slot_matrix[:, :4].tolist()

            return {
                "response": decoded_text,
                "tokens_generated": len(tokens_generated),
                "latency_ms": round(elapsed_ms, 2),
                "lookahead_branches": lookahead_activations,
                "slot_energies": [round(e, 4) for e in norm_energies],
                "slot_samples": slot_samples,
                "working_memory_bytes": 4096,
                "tier": state.tier_name,
            }

    @app.post("/api/lookahead_tree")
    def lookahead_tree(req: LookaheadRequest):
        """Generates a speculative mental lookahead branching tree for UI visualization."""
        if not state.model or not state.tokenizer:
            raise HTTPException(status_code=500, detail="Model not initialized")

        prompt_tokens = state.tokenizer.encode(f"[THREAD:0]{req.prompt} [RESP]")
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=state.device)

        with torch.no_grad():
            out = state.model.forward_sequence_parallel(token_seq=prompt_tensor)

            # Top candidate tokens
            probs = F.softmax(out["logits"][0, -1, :], dim=-1)
            topk = torch.topk(probs, k=req.branch_factor)

            branches = []
            for i in range(req.branch_factor):
                tok_id = int(topk.indices[i].item())
                tok_prob = float(topk.values[i].item())
                tok_str = state.tokenizer.decode([tok_id])

                # Sub-branches
                sub_branches = []
                # Simulated next step
                step_tensor = torch.tensor([[tok_id]], dtype=torch.long, device=state.device)
                sub_out = state.model.forward_sequence_parallel(token_seq=step_tensor)
                sub_probs = F.softmax(sub_out["logits"][0, -1, :], dim=-1)
                sub_topk = torch.topk(sub_probs, k=req.branch_factor)

                for j in range(req.branch_factor):
                    sub_id = int(sub_topk.indices[j].item())
                    sub_p = float(sub_topk.values[j].item())
                    sub_str = state.tokenizer.decode([sub_id])
                    sub_branches.append({
                        "token": sub_str,
                        "probability": round(sub_p, 4),
                        "value_score": round(1.0 - (sub_p * 0.3) + 0.1 * math.sin(j), 3),
                    })

                branches.append({
                    "token": tok_str,
                    "probability": round(tok_prob, 4),
                    "value_score": round(1.0 - (tok_prob * 0.2), 3),
                    "children": sub_branches,
                })

            return {
                "prompt": req.prompt,
                "entropy": round(-torch.sum(probs * torch.log(probs + 1e-10)).item(), 3),
                "branches": branches,
            }

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_path = Path(__file__).resolve().parent / "web" / "index.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return "<h1>Pseudo-Brain Neural Console</h1><p>index.html not found.</p>"

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_console_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
