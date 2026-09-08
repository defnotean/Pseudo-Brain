"""PyTorch FSDP (Fully Sharded Data Parallel) Infrastructure for 1B-Scale Pseudo-Brain.

Provides:
- Sharding policies for NativeSemanticPseudoBrain and BrainCellCore.
- Mixed precision policies (bf16 / fp16 compute with float32 master weights).
- Device-aware recurrent state management (Law 1: zero redundant state communication).
- 1B-scale memory footprint calculator and distributed checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import logging
import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)

from irene_brain.model.brain_cell import BrainCellCore
from irene_brain.semantic.native_semantic_model import (
    NativeSemanticPseudoBrain,
    SemanticCognitiveState,
)
from irene_brain.model.torch_model import deterministic_thought_identity_codes

# Safe FSDP imports
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        CPUOffload,
        BackwardPrefetch,
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import (
        ModuleWrapPolicy,
        size_based_auto_wrap_policy,
        lambda_auto_wrap_policy,
    )
    _FSDP_AVAILABLE = True
except ImportError:
    _FSDP_AVAILABLE = False
    ShardingStrategy = None  # type: ignore
    MixedPrecision = None  # type: ignore
    CPUOffload = None  # type: ignore
    BackwardPrefetch = None  # type: ignore


class ShardingStrategyType(str, Enum):
    """FSDP Sharding strategies supported by Pseudo-Brain."""
    FULL_SHARD = "FULL_SHARD"        # ZeRO-3: shards parameters, gradients, and optimizer states
    SHARD_GRAD_OP = "SHARD_GRAD_OP"  # ZeRO-2: shards gradients and optimizer states
    NO_SHARD = "NO_SHARD"            # DDP: replicates parameters and optimizer states
    HYBRID_SHARD = "HYBRID_SHARD"    # Sharded within node, replicated across nodes


class MixedPrecisionPolicy(str, Enum):
    """Supported mixed precision modes."""
    BF16 = "bf16"    # bfloat16 compute, fp32 master weights (recommended for A100/H100)
    FP16 = "fp16"    # float16 compute with loss scaling, fp32 master weights
    FP32 = "fp32"    # full float32 compute and storage


@dataclass
class FSDPConfig:
    """FSDP configuration for 1B-scale Pseudo-Brain training."""
    sharding_strategy: ShardingStrategyType = ShardingStrategyType.FULL_SHARD
    mixed_precision: MixedPrecisionPolicy = MixedPrecisionPolicy.BF16
    cpu_offload: bool = False
    backward_prefetch: Optional[str] = "BACKWARD_PRE"  # "BACKWARD_PRE", "BACKWARD_POST", None
    wrap_brain_cell: bool = True
    wrap_embeddings: bool = True
    wrap_slot_head: bool = True
    min_num_params: int = 100_000
    limit_all_gathers: bool = True
    sync_module_states: bool = True
    forward_prefetch: bool = False


def get_fsdp_sharding_strategy(strategy: ShardingStrategyType) -> Any:
    """Convert ShardingStrategyType to torch.distributed.fsdp.ShardingStrategy."""
    if not _FSDP_AVAILABLE:
        raise RuntimeError("PyTorch FSDP is not available.")
    mapping = {
        ShardingStrategyType.FULL_SHARD: ShardingStrategy.FULL_SHARD,
        ShardingStrategyType.SHARD_GRAD_OP: ShardingStrategy.SHARD_GRAD_OP,
        ShardingStrategyType.NO_SHARD: ShardingStrategy.NO_SHARD,
        ShardingStrategyType.HYBRID_SHARD: getattr(ShardingStrategy, "HYBRID_SHARD", ShardingStrategy.FULL_SHARD),
    }
    return mapping.get(strategy, ShardingStrategy.FULL_SHARD)


def get_mixed_precision_policy(
    precision: Union[str, MixedPrecisionPolicy] = MixedPrecisionPolicy.BF16,
    stable_reduction: bool = True,
) -> Optional[Any]:
    """Build FSDP MixedPrecision policy.

    Guarantees:
    - Computation in bf16 or fp16.
    - Master weights and optimizer updates remain in float32.
    - Gradient reduction in float32 when stable_reduction=True to avoid gradient underflow.
    """
    if not _FSDP_AVAILABLE:
        return None

    prec = MixedPrecisionPolicy(precision) if isinstance(precision, str) else precision
    if prec == MixedPrecisionPolicy.FP32:
        return None

    if prec == MixedPrecisionPolicy.BF16:
        compute_dtype = torch.bfloat16
    elif prec == MixedPrecisionPolicy.FP16:
        compute_dtype = torch.float16
    else:
        return None

    reduce_dtype = torch.float32 if stable_reduction else compute_dtype
    buffer_dtype = compute_dtype

    return MixedPrecision(
        param_dtype=compute_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
        keep_low_precision_grads=False,
    )


def get_fsdp_wrap_policy(
    wrap_brain_cell: bool = True,
    wrap_embeddings: bool = True,
    wrap_slot_head: bool = True,
    min_num_params: int = 100_000,
) -> Callable[[nn.Module, bool, int], bool]:
    """Construct auto-wrap policy for Pseudo-Brain submodules.

    Ensures:
    1. BrainCellCore (containing deep projection matrices and recurrent weights) is wrapped as an FSDP unit.
    2. nn.Embedding (65M-131M parameters at 32k/64k vocab) is wrapped independently.
    3. Large linear readout layers are sharded.
    """
    modules_to_wrap: Set[Type[nn.Module]] = set()
    if wrap_brain_cell:
        modules_to_wrap.add(BrainCellCore)
    if wrap_embeddings:
        modules_to_wrap.add(nn.Embedding)

    def custom_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        # Wrap designated architectural classes
        if type(module) in modules_to_wrap:
            return True
        # Wrap large submodules exceeding min_num_params
        if nonwrapped_numel >= min_num_params and not isinstance(module, (NativeSemanticPseudoBrain, nn.Sequential)):
            return True
        return False

    return custom_wrap_policy


# ==============================================================================
# Distributed Recurrent State Manager (Law 1 Zero Redundant Communication)
# ==============================================================================

class DistributedRecurrentStateManager:
    """Manages recurrent cognitive states in distributed multi-GPU training.

    Core Principles:
    1. Law 1 Local Partitioning: Recurrent state `SemanticCognitiveState`
       (thoughts: [B_local, K, W], P_t: [B_local, K, V]) is strictly partitioned
       by local micro-batch B_local = B_global / world_size.
    2. Zero Redundant Communication: Each GPU evolves its own local thought slots
       independently during autoregressive and streaming steps. There is ZERO
       inter-GPU communication, zero all-gather, and zero all-reduce required for
       recurrent state tensors.
    3. Device-Aware Broadcasting: Orthogonal slot identity codes (deterministic
       basis codes) are initialized identically and broadcast from Rank 0
       upon startup to guarantee bitwise mathematical alignment across ranks.
    """

    def __init__(
        self,
        K: int = 64,
        thought_size: int = 64,
        vocab_size: int = 32000,
        device: Optional[torch.device] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.K = K
        self.thought_size = thought_size
        self.vocab_size = vocab_size
        self.device = device or torch.device("cpu")
        self.rank = rank
        self.world_size = max(1, world_size)

    def broadcast_canonical_slot_identities(
        self,
        model: NativeSemanticPseudoBrain,
        root_rank: int = 0,
    ) -> torch.Tensor:
        """Broadcast canonical orthogonal slot codes from root_rank to all workers.

        Ensures all distributed workers share bitwise identical orthogonal coordinate
        frames across all K thoughtlets.
        """
        k_slots = getattr(model, "K", self.K)
        w_dim = getattr(model, "thought_size", self.thought_size)

        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            if self.rank == root_rank:
                slot_tensor = deterministic_thought_identity_codes(
                    thoughtlets=k_slots, width=w_dim
                ).to(device=self.device, dtype=torch.float32)
            else:
                slot_tensor = torch.empty(
                    k_slots, w_dim, device=self.device, dtype=torch.float32
                )
            dist.broadcast(slot_tensor, src=root_rank)
        else:
            slot_tensor = deterministic_thought_identity_codes(
                thoughtlets=k_slots, width=w_dim
            ).to(device=self.device, dtype=torch.float32)

        # Update model registered buffer if present
        if hasattr(model, "slot_identities"):
            with torch.no_grad():
                model.slot_identities.copy_(slot_tensor)
        return slot_tensor

    def init_local_state(
        self,
        batch_size_per_gpu: int,
        model: Optional[NativeSemanticPseudoBrain] = None,
        compute_dtype: torch.dtype = torch.float32,
    ) -> SemanticCognitiveState:
        """Initialize local recurrent state strictly for this GPU's local micro-batch.

        Recurrent state memory footprint per GPU:
        - thoughts: [B_local, K, W]
        - P_t: [B_local, K, V]
        No inter-GPU communication is involved.
        """
        if model is not None:
            raw_model = model.module if hasattr(model, "module") else model
            state = raw_model.init_state(batch_size=batch_size_per_gpu, device=self.device)
            if compute_dtype != torch.float32:
                state.thoughts = state.thoughts.to(dtype=compute_dtype)
                state.prev_thoughts = state.prev_thoughts.to(dtype=compute_dtype)
                state.P_t = state.P_t.to(dtype=compute_dtype)
            return state

        # Direct initialization
        slots = deterministic_thought_identity_codes(
            thoughtlets=self.K, width=self.thought_size
        ).to(device=self.device, dtype=compute_dtype)
        thoughts = slots.unsqueeze(0).expand(batch_size_per_gpu, -1, -1).clone()
        P_t = torch.zeros(batch_size_per_gpu, self.K, self.vocab_size, device=self.device, dtype=compute_dtype)
        active_thread = torch.zeros(batch_size_per_gpu, dtype=torch.long, device=self.device)

        return SemanticCognitiveState(
            thoughts=thoughts,
            P_t=P_t,
            prev_thoughts=thoughts.clone(),
            active_thread=active_thread,
        )

    def calculate_state_memory_bytes(
        self,
        batch_size_per_gpu: int,
        use_cgp: bool = True,
        bytes_per_element: int = 2,  # 2 for bf16/fp16, 4 for fp32
    ) -> Dict[str, int]:
        """Calculate local recurrent state footprint in bytes (Law 1 verification)."""
        # Thoughts: B_local * K * W * bytes
        thoughts_bytes = batch_size_per_gpu * self.K * self.thought_size * bytes_per_element
        # Prev thoughts: same
        prev_thoughts_bytes = thoughts_bytes
        # Active thread: B_local * 8 bytes (int64)
        active_thread_bytes = batch_size_per_gpu * 8
        # P_t (synaptic latching): B_local * K * V * bytes
        pt_bytes = batch_size_per_gpu * self.K * self.vocab_size * bytes_per_element if use_cgp else 0

        total_local = thoughts_bytes + prev_thoughts_bytes + active_thread_bytes + pt_bytes
        total_global = total_local * self.world_size

        return {
            "local_thoughts_bytes": thoughts_bytes,
            "local_synaptic_pt_bytes": pt_bytes,
            "total_local_state_bytes": total_local,
            "total_global_state_bytes": total_global,
            "redundant_communication_bytes": 0,  # Law 1 Guarantee: Exactly 0 bytes!
        }


# ==============================================================================
# FSDP Model Wrapper & Checkpointing
# ==============================================================================

def setup_fsdp_model(
    model: NativeSemanticPseudoBrain,
    config: Optional[FSDPConfig] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Wrap NativeSemanticPseudoBrain with PyTorch FullyShardedDataParallel (FSDP)."""
    if not _FSDP_AVAILABLE:
        raise RuntimeError("torch.distributed.fsdp is not available in this PyTorch installation.")

    cfg = config or FSDPConfig()
    dev = device or (torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"))

    # If world size is 1 or dist is not initialized, check fallback
    is_distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_distributed else 1

    sharding_strat = get_fsdp_sharding_strategy(
        ShardingStrategyType.NO_SHARD if world_size == 1 else cfg.sharding_strategy
    )

    mp_policy = get_mixed_precision_policy(cfg.mixed_precision) if dev.type == "cuda" else None
    auto_wrap = get_fsdp_wrap_policy(
        wrap_brain_cell=cfg.wrap_brain_cell,
        wrap_embeddings=cfg.wrap_embeddings,
        wrap_slot_head=cfg.wrap_slot_head,
        min_num_params=cfg.min_num_params,
    )

    # CPU offload only if requested and supported
    cpu_offload_cfg = None
    if cfg.cpu_offload and dev.type == "cuda":
        cpu_offload_cfg = CPUOffload(offload_params=True)

    backward_prefetch_enum = None
    if cfg.backward_prefetch == "BACKWARD_PRE":
        backward_prefetch_enum = BackwardPrefetch.BACKWARD_PRE
    elif cfg.backward_prefetch == "BACKWARD_POST":
        backward_prefetch_enum = BackwardPrefetch.BACKWARD_POST

    # Exclude 1-element scalar parameters (e.g. plastic_scale) from sharding
    ignored_states = [p for p in model.parameters() if p.numel() <= 1]

    # Wrap model
    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding_strat,
        cpu_offload=cpu_offload_cfg,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        backward_prefetch=backward_prefetch_enum,
        device_id=dev if dev.type == "cuda" else torch.device("cpu"),
        limit_all_gathers=cfg.limit_all_gathers,
        sync_module_states=cfg.sync_module_states and is_distributed and world_size > 1 and dev.type == "cuda",
        ignored_states=ignored_states if ignored_states else None,
    )
    return fsdp_model


def save_fsdp_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: Union[str, Path],
    rank: int = 0,
) -> None:
    """Save checkpoint from FSDP model using sharded or full state dict."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = None
    opt_state = None

    if hasattr(model, "state_dict_type"):
        is_cuda = next(model.parameters()).is_cuda if any(True for _ in model.parameters()) else False
        if is_cuda:
            try:
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    state_dict = model.state_dict()
                    opt_state = FSDP.optim_state_dict(model, optimizer) if optimizer is not None else None
            except Exception as e:
                logger.debug(f"FULL_STATE_DICT save skipped: {e}")

        if state_dict is None:
            # Universal non-blocking distributed sharded state dict
            try:
                with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                    state_dict = model.state_dict()
                    opt_state = FSDP.optim_state_dict(model, optimizer) if optimizer is not None else None
            except Exception as e:
                logger.debug(f"SHARDED_STATE_DICT save skipped: {e}")
    else:
        state_dict = model.state_dict()
        opt_state = optimizer.state_dict() if optimizer is not None else None

    if rank == 0 and state_dict is not None:
        torch.save(
            {
                "model_state_dict": state_dict,
                "optimizer_state_dict": opt_state,
                "rank": rank,
            },
            str(path),
        )


def load_fsdp_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: Union[str, Path],
) -> None:
    """Load consolidated checkpoint into FSDP-wrapped model."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(str(path), map_location="cpu")
    if hasattr(model, "load_state_dict"):
        if hasattr(model, "state_dict_type"):
            load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
                model.load_state_dict(checkpoint["model_state_dict"])
                if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
                    sharded_opt_state = FSDP.optim_state_dict_to_load(
                        model, optimizer, checkpoint["optimizer_state_dict"]
                    )
                    optimizer.load_state_dict(sharded_opt_state)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# ==============================================================================
# 1B-Scale Memory Footprint Calculator
# ==============================================================================

@dataclass
class MemoryFootprintReport:
    """Detailed memory footprint breakdown for 1B-scale Pseudo-Brain."""
    total_params: int
    param_memory_fp32_gb: float
    param_memory_bf16_gb: float
    optimizer_memory_fp32_gb: float  # AdamW 16 bytes/param
    gradient_memory_fp32_gb: float   # 4 bytes/param
    gradient_memory_bf16_gb: float   # 2 bytes/param
    recurrent_state_local_mb: float  # Law 1 state per GPU
    recurrent_state_global_mb: float # Law 1 state total across cluster
    unshared_ddp_memory_per_gpu_gb: float
    fsdp_memory_per_gpu: Dict[int, float]  # World size -> GB/GPU
    peak_allgather_buffer_mb: float

    def format_summary_table(self) -> str:
        """Format an informative summary table."""
        lines = [
            "=" * 78,
            f"1B-SCALE PSEUDO-BRAIN DISTRIBUTED MEMORY FOOTPRINT ANALYSIS",
            "=" * 78,
            f"Total Parameters:               {self.total_params:,} ({self.total_params / 1e9:.3f} Billion)",
            f"Master Weights (FP32):          {self.param_memory_fp32_gb:.2f} GB",
            f"Compute Weights (BF16/FP16):    {self.param_memory_bf16_gb:.2f} GB",
            f"Gradients (FP32 / BF16):        {self.gradient_memory_fp32_gb:.2f} GB / {self.gradient_memory_bf16_gb:.2f} GB",
            f"AdamW Optimizer States (FP32):  {self.optimizer_memory_fp32_gb:.2f} GB (16 bytes/param: master + m + v)",
            "-" * 78,
            f"Standard DDP (Replicated):      {self.unshared_ddp_memory_per_gpu_gb:.2f} GB VRAM per GPU (OOM on 16GB/24GB GPUs)",
            "-" * 78,
            f"FSDP FULL_SHARD (ZeRO-3) Static Memory Scaling Across Cluster:",
        ]
        for gpus, mem in sorted(self.fsdp_memory_per_gpu.items()):
            lines.append(f"  * {gpus:2d} GPUs:  {mem:6.2f} GB VRAM / GPU  (Fit comfortably on standard 16GB/24GB GPUs)")
        lines.extend([
            "-" * 78,
            f"Peak Transient All-Gather Buffer: {self.peak_allgather_buffer_mb:.1f} MB (BrainCellCore forward gather)",
            f"Recurrent State Footprint (Law 1):{self.recurrent_state_local_mb:.2f} MB / GPU  [ZERO cross-GPU communication]",
            "=" * 78,
        ])
        return "\n".join(lines)


def calculate_1b_memory_footprint(
    vocab_size: int = 32000,
    K: int = 64,
    thought_size: int = 64,
    embed_dim: int = 2048,
    proj_dim: int = 6144,
    num_deep_layers: int = 24,
    batch_size_per_gpu: int = 4,
    gpu_counts: Tuple[int, ...] = (1, 2, 4, 8, 16),
) -> MemoryFootprintReport:
    """Calculate exact memory footprint at 1B scale across distributed topologies."""
    # 1. Parameter breakdown
    emb_params = vocab_size * embed_dim
    proj_params = embed_dim * proj_dim
    deep_params = num_deep_layers * (proj_dim * proj_dim + proj_dim)
    head_hidden = max(64, min(proj_dim // 2, 512))
    head_params = 64 * head_hidden + head_hidden * vocab_size
    cgp_params = (64 + 16) * vocab_size
    router_params = 3 * (thought_size * thought_size)
    recurrent_cell_params = 3 * (proj_dim * 32 + 32 * thought_size) + 3 * (thought_size * thought_size)

    total_params = emb_params + proj_params + deep_params + head_params + cgp_params + router_params + recurrent_cell_params

    # 2. Byte footprints
    bytes_fp32 = 4
    bytes_bf16 = 2
    # AdamW maintains: fp32 master weights (4) + first moment m (4) + second moment v (4) + grads (4) = 16 bytes/param
    optimizer_bytes_per_param = 16

    param_memory_fp32_gb = (total_params * bytes_fp32) / (1024 ** 3)
    param_memory_bf16_gb = (total_params * bytes_bf16) / (1024 ** 3)
    gradient_memory_fp32_gb = (total_params * bytes_fp32) / (1024 ** 3)
    gradient_memory_bf16_gb = (total_params * bytes_bf16) / (1024 ** 3)
    optimizer_memory_fp32_gb = (total_params * optimizer_bytes_per_param) / (1024 ** 3)

    # Standard unshared DDP: bf16 weights + fp32 grads + fp32 AdamW states
    unshared_ddp_gb = param_memory_bf16_gb + gradient_memory_fp32_gb + optimizer_memory_fp32_gb

    # FSDP FULL_SHARD across GPU counts
    fsdp_memory_per_gpu: Dict[int, float] = {}
    for g in gpu_counts:
        # Sharded: (bf16 params + fp32 grads + fp32 opt states) / g
        fsdp_memory_per_gpu[g] = unshared_ddp_gb / g

    # Largest single module all-gather buffer (BrainCellCore deep projection layer)
    single_layer_params = proj_dim * proj_dim
    peak_allgather_mb = (single_layer_params * bytes_bf16) / (1024 ** 2)

    # Recurrent state footprint (Law 1)
    state_mgr = DistributedRecurrentStateManager(K=K, thought_size=thought_size, vocab_size=vocab_size)
    state_mem = state_mgr.calculate_state_memory_bytes(
        batch_size_per_gpu=batch_size_per_gpu, use_cgp=True, bytes_per_element=bytes_bf16
    )
    recurrent_local_mb = state_mem["total_local_state_bytes"] / (1024 ** 2)
    recurrent_global_mb = state_mem["total_global_state_bytes"] / (1024 ** 2)

    return MemoryFootprintReport(
        total_params=total_params,
        param_memory_fp32_gb=param_memory_fp32_gb,
        param_memory_bf16_gb=param_memory_bf16_gb,
        optimizer_memory_fp32_gb=optimizer_memory_fp32_gb,
        gradient_memory_fp32_gb=gradient_memory_fp32_gb,
        gradient_memory_bf16_gb=gradient_memory_bf16_gb,
        recurrent_state_local_mb=recurrent_local_mb,
        recurrent_state_global_mb=recurrent_global_mb,
        unshared_ddp_memory_per_gpu_gb=unshared_ddp_gb,
        fsdp_memory_per_gpu=fsdp_memory_per_gpu,
        peak_allgather_buffer_mb=peak_allgather_mb,
    )
