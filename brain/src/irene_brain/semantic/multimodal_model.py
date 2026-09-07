"""Multimodal Pseudo-Brain Architecture.

Unifies visual perception (ConvEncoder from Level 12 POMDP / torch_model.py)
and language (SemanticTokenizer) into a shared recurrent state machine
where visual entity tokens and language tokens bind to the same persistent thought slots.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from irene_brain.model.brain_cell import BrainCellCore
from irene_brain.model.sparse_thought_router import SparseThoughtRouter
from irene_brain.model.torch_model import deterministic_thought_identity_codes
from .tokenizer import SemanticTokenizer


# ==============================================================================
# 1. Visual ConvEncoder (Level 12 POMDP / torch_model.py hierarchy)
# ==============================================================================

class ConvEncoder(nn.Module):
    """Visual Convolutional Feature & Entity Extractor.

    Derived from the Level 12 POMDP and torch_model.py pixel encoder architectures.
    Employs a 3-layer convolutional hierarchy with GroupNorm for deterministic,
    batch-size-independent streaming operation (stable for B=1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 192,
        num_entity_tokens: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_entity_tokens = num_entity_tokens

        grid = math.isqrt(num_entity_tokens)
        self.grid_size = grid if grid * grid == num_entity_tokens else 2

        # 3-Stage ConvNet with GroupNorm
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(1, 24)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(1, 48)

        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.GroupNorm(1, 48)

        # Entity spatial grid pooling: (grid_size, grid_size) -> num_entity_tokens
        self.entity_pool = nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))
        self.entity_proj = nn.Sequential(
            nn.Linear(48, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

        # Global scene feature projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_proj = nn.Sequential(
            nn.Linear(48, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        pixels: torch.Tensor,
        as_entities: bool = True,
    ) -> torch.Tensor:
        """Extract visual representations from pixel tensors.

        Args:
            pixels: Image tensor of shape [B, C, H, W]
            as_entities: If True, returns spatial entity tokens [B, S, feature_dim].
                         If False, returns pooled global scene feature [B, feature_dim].
        """
        if pixels.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape {pixels.shape}")

        h = self.pool(F.silu(self.norm1(self.conv1(pixels))))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))  # [B, 48, H', W']

        if as_entities:
            # Pool to spatial grid tokens: [B, 48, grid, grid] -> [B, S, 48]
            grid_feat = self.entity_pool(h)
            B, C, gh, gw = grid_feat.shape
            tokens = grid_feat.permute(0, 2, 3, 1).reshape(B, gh * gw, C)
            return self.entity_proj(tokens)  # [B, S, feature_dim]
        else:
            pooled = self.global_pool(h).flatten(1)  # [B, 48]
            return self.global_proj(pooled)  # [B, feature_dim]


# ==============================================================================
# 2. Multimodal Cognitive State
# ==============================================================================

@dataclass
class MultimodalCognitiveState:
    """Persistent cognitive state across multimodal streaming steps."""
    thoughts: torch.Tensor  # [B, K, thought_size] persistent thought slots
    P_t: torch.Tensor  # [B, K, vocab_size] fast synaptic plasticity latches
    prev_thoughts: torch.Tensor  # [B, K, thought_size] previous thoughts for delta surprise
    active_thread: torch.Tensor  # [B] integer active slot / thread index
    slot_modalities: torch.Tensor  # [B, K] bitmask: 0=empty, 1=language, 2=visual, 3=multimodal


# ==============================================================================
# 3. Multimodal Pseudo-Brain Recurrent Architecture
# ==============================================================================

class MultimodalPseudoBrain(nn.Module):
    """Unified Multimodal Pseudo-Brain Architecture.

    Binds visual entity tokens and language tokens into the same persistent
    recurrent thought slots governed by Cognitive Input Gating (CIG) and
    Consequence-Gated Synaptic Latching (CGSL).
    """

    def __init__(
        self,
        vocab_size: int = 344,
        K: int = 16,
        thought_size: int = 32,
        embed_dim: int = 64,
        proj_dim: int = 128,
        in_channels: int = 3,
        visual_dim: int = 192,
        num_visual_tokens: int = 4,
        routed_neighbors: int = 2,
        plastic_decay: float = 0.9999,
        plastic_lr: float = 0.25,
        use_cgp: bool = True,
        use_routing: bool = True,
        n_actions: int = 5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.K = K
        self.thought_size = thought_size
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.visual_dim = visual_dim
        self.num_visual_tokens = num_visual_tokens
        self.plastic_decay = plastic_decay
        self.plastic_lr = plastic_lr
        self.use_cgp = use_cgp
        self.use_routing = use_routing

        # 1. Language Embedding & Sensory Projection
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lang_proj = nn.Linear(embed_dim, proj_dim)

        # 2. Visual Encoder & Sensory Projection
        self.visual_encoder = ConvEncoder(
            in_channels=in_channels,
            feature_dim=visual_dim,
            num_entity_tokens=num_visual_tokens,
        )
        self.visual_proj = nn.Linear(visual_dim, proj_dim)

        # 3. Shared Recurrent Core across K thought slots
        self.brain_cell = BrainCellCore(input_size=proj_dim, thought_size=thought_size)

        # 4. Cognitive Input Gating (CIG) with T=0.5 sharpening
        self.cig_gate = nn.Sequential(
            nn.Linear(proj_dim + thought_size, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.cig_gate[0].bias, 1.0)

        # 5. Selective Routing (for cross-slot communication)
        if self.use_routing and self.K > 1:
            self.router = SparseThoughtRouter(
                width=thought_size,
                routed_neighbors=min(routed_neighbors, max(1, K - 1)),
                dense_routing=False,
            )
        else:
            self.router = None

        # 6. Consequence-Gated Synaptic Latching (CGSL)
        if self.use_cgp:
            self.plastic_modulator = nn.Linear(thought_size + 16, vocab_size)
            self.surprise_gate = nn.Sequential(
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.surprise_gate[0].bias, -1.0)
            self.plastic_scale = nn.Parameter(torch.tensor(1.0))
            self.surprise_encoder = nn.Sequential(
                nn.Linear(1, 16),
                nn.LayerNorm(16),
                nn.GELU(),
                nn.Linear(16, 16),
            )
        else:
            self.plastic_modulator = None
            self.surprise_gate = None
            self.plastic_scale = None
            self.surprise_encoder = None

        # 7. Slot Readout Heads
        # Language readout
        self.slot_head = nn.Sequential(
            nn.Linear(thought_size, proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, vocab_size),
        )
        # Visual feature reconstruction / entity readout
        self.visual_head = nn.Sequential(
            nn.Linear(thought_size, proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, visual_dim),
        )
        # Action policy readout (POMDP discrete actions: 0=Idle, 1=W, 2=A, 3=S, 4=D)
        self.n_actions = n_actions
        self.action_head = nn.Sequential(
            nn.Linear(thought_size, proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, n_actions),
        )

        # 8. Slot Orthogonality Codes
        self.register_buffer(
            "slot_identities",
            deterministic_thought_identity_codes(thoughtlets=K, width=thought_size),
            persistent=False,
        )

    def init_state(self, batch_size: int, device: torch.device) -> MultimodalCognitiveState:
        """Initialize orthogonal identity state for multimodal streaming."""
        if self.slot_identities.shape[0] != self.K:
            slots = deterministic_thought_identity_codes(thoughtlets=self.K, width=self.thought_size).to(device)
        else:
            slots = self.slot_identities.to(device=device)

        thoughts = slots.unsqueeze(0).expand(batch_size, -1, -1).clone()
        P_t = torch.zeros(batch_size, self.K, self.vocab_size, device=device)
        active_thread = torch.zeros(batch_size, dtype=torch.long, device=device)
        slot_modalities = torch.zeros(batch_size, self.K, dtype=torch.long, device=device)

        return MultimodalCognitiveState(
            thoughts=thoughts,
            P_t=P_t,
            prev_thoughts=thoughts.clone(),
            active_thread=active_thread,
            slot_modalities=slot_modalities,
        )

    def step_sensory_token(
        self,
        x_proj: torch.Tensor,  # [B, proj_dim]
        state: MultimodalCognitiveState,
        slot_idx: Optional[Union[int, torch.Tensor]] = None,
        allow_routing: bool = False,
        modality_tag: int = 1,  # 1 = language, 2 = visual
    ) -> Tuple[torch.Tensor, MultimodalCognitiveState]:
        """Core recurrent step for an arbitrary sensory token (language or visual entity).

        Updates the targeted thought slot (or content-addressed slots) via
        Cognitive Input Gating (CIG) and BrainCellCore.

        Returns:
            logits: Language vocabulary logits [B, vocab_size]
            next_state: Updated MultimodalCognitiveState
        """
        B = x_proj.shape[0]
        dev = x_proj.device

        # Determine target slot
        if slot_idx is not None:
            if isinstance(slot_idx, int):
                target_thread = torch.full((B,), slot_idx, dtype=torch.long, device=dev)
            else:
                target_thread = slot_idx.to(device=dev, dtype=torch.long)
            state.active_thread = target_thread

        # Update modality tracking bitmask for active thread
        batch_idx = torch.arange(B, device=dev)
        current_mod = state.slot_modalities[batch_idx, state.active_thread]
        state.slot_modalities[batch_idx, state.active_thread] = current_mod | modality_tag

        # 1. Expand sensory token across K slots
        x_exp = x_proj.unsqueeze(1).expand(-1, self.K, -1)  # [B, K, proj_dim]

        # 2. Cognitive Input Gating with T=0.5 Sharpening
        gate_in = torch.cat([x_exp, state.thoughts], dim=-1)
        raw_gate = self.cig_gate(gate_in)
        c_gate = raw_gate.clamp(1e-6, 1.0 - 1e-6)
        salience = torch.sigmoid((torch.log(c_gate / (1.0 - c_gate))) / 0.5)

        # Slot-directed masking: when slot_idx is explicit, sharpen attention on targeted slot
        if slot_idx is not None:
            slot_mask = torch.zeros(B, self.K, 1, device=dev)
            slot_mask[batch_idx, state.active_thread, 0] = 1.0
            salience = salience * slot_mask

        # 3. Recurrent Core Update
        t_flat = state.thoughts.reshape(B * self.K, self.thought_size)
        x_flat = x_exp.reshape(B * self.K, -1)
        new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)

        # 4. Selective Routing between slots
        if self.use_routing and self.router is not None and allow_routing:
            messages, _ = self.router(new_t)
            t_candidate = new_t + messages
        else:
            t_candidate = new_t

        # 5. Gated State Update
        next_thoughts = (1.0 - salience) * state.thoughts + salience * t_candidate

        # 6. Fast Synaptic Latching (CGSL)
        if self.use_cgp and self.surprise_encoder is not None:
            delta_slot = torch.norm(next_thoughts - state.prev_thoughts, dim=-1, keepdim=True)
            prev_thoughts_next = next_thoughts.detach()
            surprise_emb = self.surprise_encoder(delta_slot)
            p_gate = self.surprise_gate(surprise_emb)
            raw_delta_P = p_gate * torch.tanh(self.plastic_modulator(torch.cat([next_thoughts, surprise_emb], dim=-1)))

            # Slot-selective plasticity update: preserve quiescent slots from cross-slot leakage or decay
            if slot_idx is not None:
                active_slot_mask = torch.zeros(B, self.K, 1, device=dev)
                active_slot_mask[batch_idx, state.active_thread, 0] = 1.0
                delta_P = active_slot_mask * raw_delta_P
                decay_factor = 1.0 - active_slot_mask * (1.0 - self.plastic_decay)
                next_P_t = decay_factor * state.P_t + self.plastic_lr * delta_P
            else:
                sal_mask = (salience > 0.05).float()
                delta_P = sal_mask * raw_delta_P
                decay_factor = 1.0 - sal_mask * (1.0 - self.plastic_decay)
                next_P_t = decay_factor * state.P_t + self.plastic_lr * delta_P
        else:
            prev_thoughts_next = next_thoughts
            next_P_t = state.P_t


        # 7. Targeted Readout from active slot
        queried_slot = next_thoughts[batch_idx, state.active_thread]  # [B, W]
        slot_logits = self.slot_head(queried_slot)  # [B, V]

        if self.use_cgp and self.plastic_scale is not None:
            queried_P = next_P_t[batch_idx, state.active_thread]  # [B, V]
            total_logits = slot_logits + self.plastic_scale * queried_P
        else:
            total_logits = slot_logits

        next_state = MultimodalCognitiveState(
            thoughts=next_thoughts,
            P_t=next_P_t,
            prev_thoughts=prev_thoughts_next,
            active_thread=state.active_thread,
            slot_modalities=state.slot_modalities,
        )
        return total_logits, next_state

    def step_token(
        self,
        token_ids: torch.Tensor,  # [B]
        state: MultimodalCognitiveState,
        thread_ids: Optional[Union[int, torch.Tensor]] = None,
        allow_routing: bool = False,
    ) -> Tuple[torch.Tensor, MultimodalCognitiveState]:
        """Ingest a single language token into the recurrent state machine."""
        # Check if token is thread token [THREAD:i] (tokens 9 .. 9+K-1 in SemanticTokenizer)
        if thread_ids is None:
            is_thread_tok = (token_ids >= 9) & (token_ids < 9 + self.K)
            if is_thread_tok.any():
                new_tids = (token_ids - 9).clamp(0, self.K - 1)
                thread_ids = torch.where(is_thread_tok, new_tids, state.active_thread)

        emb = self.embedding(token_ids)
        x_proj = self.lang_proj(emb)
        return self.step_sensory_token(
            x_proj=x_proj,
            state=state,
            slot_idx=thread_ids,
            allow_routing=allow_routing,
            modality_tag=1,  # Language
        )

    def step_visual_token(
        self,
        visual_token: torch.Tensor,  # [B, visual_dim]
        state: MultimodalCognitiveState,
        slot_idx: Optional[Union[int, torch.Tensor]] = None,
        allow_routing: bool = False,
    ) -> Tuple[torch.Tensor, MultimodalCognitiveState]:
        """Ingest a single visual entity token into the recurrent state machine."""
        x_proj = self.visual_proj(visual_token)
        return self.step_sensory_token(
            x_proj=x_proj,
            state=state,
            slot_idx=slot_idx,
            allow_routing=allow_routing,
            modality_tag=2,  # Visual
        )

    def ingest_image(
        self,
        pixels: torch.Tensor,  # [B, C, H, W]
        state: MultimodalCognitiveState,
        slot_idx: Optional[int] = None,
        as_entities: bool = True,
        allow_routing: bool = False,
    ) -> Tuple[torch.Tensor, MultimodalCognitiveState]:
        """Extract visual entity tokens and bind them sequentially to the designated thought slot."""
        tokens = self.visual_encoder(pixels, as_entities=as_entities)
        if not as_entities:
            tokens = tokens.unsqueeze(1)  # [B, 1, visual_dim]

        B, S, _ = tokens.shape
        last_logits = None
        curr_state = state

        for s in range(S):
            v_tok = tokens[:, s]  # [B, visual_dim]
            last_logits, curr_state = self.step_visual_token(
                visual_token=v_tok,
                state=curr_state,
                slot_idx=slot_idx,
                allow_routing=allow_routing,
            )

        assert last_logits is not None
        return last_logits, curr_state

    def ingest_text_tokens(
        self,
        token_seq: torch.Tensor,  # [B, T]
        state: MultimodalCognitiveState,
        slot_idx: Optional[int] = None,
        allow_routing: bool = False,
    ) -> Tuple[torch.Tensor, MultimodalCognitiveState]:
        """Ingest a sequence of text token IDs into the designated thought slot."""
        B, T = token_seq.shape
        last_logits = None
        curr_state = state

        for t in range(T):
            tok_t = token_seq[:, t]
            last_logits, curr_state = self.step_token(
                token_ids=tok_t,
                state=curr_state,
                thread_ids=slot_idx,
                allow_routing=allow_routing,
            )

        assert last_logits is not None
        return last_logits, curr_state

    def predict_visual_features(
        self,
        slot_idx: int,
        state: MultimodalCognitiveState,
    ) -> torch.Tensor:
        """Decode predicted visual entity features from the specified thought slot."""
        B = state.thoughts.shape[0]
        slot_t = state.thoughts[:, slot_idx]  # [B, thought_size]
        return self.visual_head(slot_t)  # [B, visual_dim]

    def cross_modal_affinity(
        self,
        slot_idx: int,
        state: MultimodalCognitiveState,
        text_tokens: Optional[torch.Tensor] = None,
        image_pixels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Measure cosine alignment between persistent thought slot and multimodal inputs."""
        slot_t = state.thoughts[0, slot_idx]  # [thought_size]
        results: Dict[str, float] = {}

        if text_tokens is not None:
            if text_tokens.ndim == 1:
                text_tokens = text_tokens.unsqueeze(0)
            emb = self.embedding(text_tokens).mean(dim=1).squeeze(0)  # [embed_dim]
            proj_text = self.lang_proj(emb)  # [proj_dim]
            # Compare via projection
            t_proj = self.slot_head[0](slot_t)  # [proj_dim // 2]
            sim = float(F.cosine_similarity(t_proj, proj_text[:t_proj.shape[0]], dim=0).item())
            results["text_similarity"] = sim

        if image_pixels is not None:
            v_feat = self.visual_encoder(image_pixels, as_entities=False).squeeze(0)  # [visual_dim]
            v_pred = self.visual_head(slot_t)  # [visual_dim]
            sim = float(F.cosine_similarity(v_pred, v_feat, dim=0).item())
            results["visual_similarity"] = sim

        return results

    def forward(
        self,
        token_seq: Optional[torch.Tensor] = None,  # [B, T]
        image_tensor: Optional[torch.Tensor] = None,  # [B, C, H, W]
        state: Optional[MultimodalCognitiveState] = None,
        slot_idx: Optional[int] = None,
        allow_routing: bool = False,
    ) -> Tuple[torch.Tensor, MultimodalCognitiveState]:
        """Sequential multimodal forward unroll supporting interleaved images and text."""
        B = token_seq.shape[0] if token_seq is not None else image_tensor.shape[0]
        dev = token_seq.device if token_seq is not None else image_tensor.device

        if state is None:
            state = self.init_state(B, dev)

        last_logits = None
        if image_tensor is not None:
            last_logits, state = self.ingest_image(
                pixels=image_tensor,
                state=state,
                slot_idx=slot_idx,
                allow_routing=allow_routing,
            )

        if token_seq is not None:
            last_logits, state = self.ingest_text_tokens(
                token_seq=token_seq,
                state=state,
                slot_idx=slot_idx,
                allow_routing=allow_routing,
            )

        assert last_logits is not None
        return last_logits, state

    def get_action_logits(
        self,
        state: MultimodalCognitiveState,
        slot_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Read out discrete action logits from the specified cognitive slot.

        Args:
            state: Active MultimodalCognitiveState
            slot_idx: Thought slot index to query (defaults to state.active_thread)

        Returns:
            action_logits: Tensor of shape [B, n_actions]
        """
        B = state.thoughts.shape[0]
        dev = state.thoughts.device
        batch_idx = torch.arange(B, device=dev)
        if slot_idx is None:
            target_slot = state.active_thread
        else:
            target_slot = torch.full((B,), slot_idx, dtype=torch.long, device=dev)

        slot_t = state.thoughts[batch_idx, target_slot]
        logits = self.action_head(slot_t)

        # Modulate logits with fast synaptic latches P_t if available
        if self.use_cgp and self.plastic_scale is not None and state.P_t.shape[-1] >= self.n_actions:
            p_act = state.P_t[batch_idx, target_slot, : self.n_actions]
            logits = logits + self.plastic_scale * p_act

        return logits


# Architectural Alias
MultimodalPseudoBrainModel = MultimodalPseudoBrain


# ==============================================================================
# 4. Multimodal Streaming Session Engine
# ==============================================================================

class MultimodalStreamingSession:
    """Streaming cognitive session supporting multimodal ingestion and state persistence."""

    def __init__(
        self,
        model: MultimodalPseudoBrain,
        tokenizer: Optional[SemanticTokenizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer or SemanticTokenizer(max_threads=model.K)
        self.device = device or next(model.parameters()).device
        self.model.eval()

        if self.device.type == "cpu":
            try:
                torch.set_num_threads(1)
            except Exception:
                pass

        self.state = self.model.init_state(batch_size=1, device=self.device)
        self.step_latencies_ms: List[float] = []
        self.total_tokens_processed = 0
        self.total_images_processed = 0

    def reset(self) -> None:
        """Reset internal cognitive state to fresh orthogonal slots."""
        self.state = self.model.init_state(batch_size=1, device=self.device)
        self.step_latencies_ms.clear()
        self.total_tokens_processed = 0
        self.total_images_processed = 0

    def ingest_image(
        self,
        image_tensor: torch.Tensor,  # [1, C, H, W] or [C, H, W]
        slot_idx: Optional[int] = None,
        as_entities: bool = True,
    ) -> Dict[str, Any]:
        """Ingest an image into the designated thought slot."""
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits, self.state = self.model.ingest_image(
                pixels=image_tensor,
                state=self.state,
                slot_idx=slot_idx,
                as_entities=as_entities,
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.step_latencies_ms.append(dt_ms)
        self.total_images_processed += 1

        active_slot = int(self.state.active_thread.item())
        mod_status = int(self.state.slot_modalities[0, active_slot].item())

        return {
            "latency_ms": dt_ms,
            "active_slot": active_slot,
            "modality_status": mod_status,
            "is_multimodal_bound": (mod_status == 3),
        }

    def step_token(
        self,
        token_id: int,
        thread_id: Optional[int] = None,
        allow_routing: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Ingest a single language token."""
        t_in = torch.tensor([token_id], dtype=torch.long, device=self.device)
        th_in = torch.tensor([thread_id], dtype=torch.long, device=self.device) if thread_id is not None else None

        t0 = time.perf_counter()
        with torch.no_grad():
            logits, self.state = self.model.step_token(
                token_ids=t_in,
                state=self.state,
                thread_ids=th_in,
                allow_routing=allow_routing,
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.step_latencies_ms.append(dt_ms)
        self.total_tokens_processed += 1
        return logits.squeeze(0), dt_ms

    def ingest_text(
        self,
        text: str,
        thread_id: Optional[int] = None,
        allow_routing: bool = False,
    ) -> List[float]:
        """Ingest a stream of text without generating response tokens."""
        token_ids = self.tokenizer.encode(text, thread_id=thread_id)
        latencies = []
        for tid in token_ids:
            _, dt = self.step_token(tid, thread_id=thread_id, allow_routing=allow_routing)
            latencies.append(dt)
        return latencies

    def generate_response(
        self,
        prompt_text: Optional[str] = None,
        thread_id: Optional[int] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        stop_on_eos: bool = True,
        stop_on_sep: bool = True,
        allow_routing: bool = False,
    ) -> Dict[str, Any]:
        """Generate response tokens autoregressively from current multimodal persistent state."""
        prompt_latencies = []
        last_logits = None

        if prompt_text:
            token_ids = self.tokenizer.encode(prompt_text, thread_id=thread_id)
            for tid in token_ids:
                last_logits, dt = self.step_token(tid, thread_id=thread_id, allow_routing=allow_routing)
                prompt_latencies.append(dt)

        # Ingest [RESP] token marker to switch to response phase
        resp_tok = self.tokenizer.resp_id
        last_logits, dt = self.step_token(resp_tok, thread_id=thread_id, allow_routing=allow_routing)
        prompt_latencies.append(dt)

        generated_token_ids: List[int] = []
        gen_latencies = []

        for _ in range(max_new_tokens):
            if last_logits is None:
                break

            if temperature <= 1e-4:
                next_tok = int(last_logits.argmax().item())
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())

            if stop_on_eos and next_tok == self.tokenizer.eos_id:
                break
            if stop_on_sep and next_tok == self.tokenizer.sep_id:
                break

            generated_token_ids.append(next_tok)
            last_logits, dt = self.step_token(next_tok, thread_id=thread_id, allow_routing=allow_routing)
            gen_latencies.append(dt)

        generated_text = self.tokenizer.decode(generated_token_ids, skip_special=True)
        return {
            "response_text": generated_text,
            "generated_tokens": generated_token_ids,
            "prompt_latencies_ms": prompt_latencies,
            "generation_latencies_ms": gen_latencies,
            "mean_step_latency_ms": float(np.mean(gen_latencies)) if gen_latencies else 0.0,
            "active_thread": int(self.state.active_thread.item()),
            "total_tokens_session": self.total_tokens_processed,
        }

    def get_slot_state(self, slot_idx: int) -> torch.Tensor:
        """Return the persistent thought vector for the specified slot."""
        return self.state.thoughts[0, slot_idx].clone()

    def get_telemetry(self) -> Dict[str, Any]:
        """Compute live session telemetry and latency metrics."""
        lats = np.array(self.step_latencies_ms) if self.step_latencies_ms else np.array([0.0])
        active_tid = int(self.state.active_thread.item())
        thoughts = self.state.thoughts[0]
        P_t = self.state.P_t[0]

        steady_lats = lats[1:] if len(lats) > 1 else lats
        under_budget = bool(np.all(steady_lats <= 16.67)) or bool(
            float(np.percentile(steady_lats, 90)) <= 16.67 and float(np.mean(steady_lats)) <= 16.67
        )
        return {
            "active_thread": active_tid,
            "total_tokens": self.total_tokens_processed,
            "total_images": self.total_images_processed,
            "latency_mean_ms": float(np.mean(lats)),
            "latency_p50_ms": float(np.percentile(lats, 50)),
            "latency_p90_ms": float(np.percentile(lats, 90)),
            "latency_p99_ms": float(np.percentile(lats, 99)),
            "latency_under_16_67ms": under_budget,
            "active_slot_norm": float(torch.norm(thoughts[active_tid]).item()),

            "active_plastic_norm": float(torch.norm(P_t[active_tid]).item()),
            "all_slots_mean_norm": float(torch.norm(thoughts, dim=-1).mean().item()),
            "slot_modalities": self.state.slot_modalities[0].cpu().tolist(),
        }

    def get_action_logits(self, slot_idx: Optional[int] = None) -> torch.Tensor:
        """Query discrete action logits from the active session state."""
        return self.model.get_action_logits(self.state, slot_idx=slot_idx)


# Canonical alias
MultimodalPseudoBrainModel = MultimodalPseudoBrain

__all__ = [
    "ConvEncoder",
    "MultimodalCognitiveState",
    "MultimodalPseudoBrain",
    "MultimodalPseudoBrainModel",
    "MultimodalStreamingSession",
]

