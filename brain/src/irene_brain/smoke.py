"""Small, bounded proof that Phase 0 contracts work together."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import tomllib
from pathlib import Path

from .data.branching import evaluate_branches
from .data.replay import record_trace, verify_trace
from .environments import MovingShapesEnv
from .runtime.policy import ResourcePolicy
from .types import GenericControl, HidKey


@dataclass(frozen=True, slots=True)
class SmokeSummary:
    seed: int
    replay_steps: int
    replay_sha256: str
    final_state_sha256: str
    branch_sha256: tuple[tuple[str, str], ...]
    gpu_used: bool = False
    capture_used: bool = False
    hid_output_used: bool = False
    background_threads_used: bool = False
    artifacts_written: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_smoke(config_path: str | Path) -> SmokeSummary:
    path = Path(config_path)
    policy = ResourcePolicy.from_toml(path)
    if policy != ResourcePolicy.play_safe():
        raise ValueError("the Phase 0 smoke runner accepts only the play-safe policy")
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    experiment = config["experiment"]
    environment_config = config["environment"]
    steps = experiment["steps"]
    seed = experiment["seed"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 5_000:
        raise ValueError("play-safe smoke steps must be an integer in [1, 5000]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("smoke seed must be a nonnegative integer")
    if environment_config["width"] != 16 or environment_config["height"] != 16:
        raise ValueError("moving_shapes currently requires a 16x16 logical grid")

    environment = MovingShapesEnv(
        tick_period_ns=1_000_000_000 // environment_config["tick_hz"],
        max_ticks=steps + 100,
    )
    environment.reset(seed)
    pattern = (
        (int(HidKey.W),),
        (int(HidKey.D),),
        (int(HidKey.S),),
        (int(HidKey.A),),
        (),
    )
    controls = tuple(
        GenericControl(keys_down=pattern[index % len(pattern)])
        for index in range(steps)
    )
    trace = record_trace(environment, controls)
    verifier = MovingShapesEnv(
        tick_period_ns=environment.tick_period_ns,
        max_ticks=steps + 100,
    )
    verify_trace(verifier, trace, preserve_environment=False)

    environment.restore(trace.root_snapshot)
    branch_root = environment.snapshot()
    branch_controls = {
        "north": (GenericControl(keys_down=(int(HidKey.W),)),) * 8,
        "east": (GenericControl(keys_down=(int(HidKey.D),)),) * 8,
        "south": (GenericControl(keys_down=(int(HidKey.S),)),) * 8,
        "west": (GenericControl(keys_down=(int(HidKey.A),)),) * 8,
    }
    branches = evaluate_branches(environment, branch_root, branch_controls)
    return SmokeSummary(
        seed=seed,
        replay_steps=steps,
        replay_sha256=trace.trace_sha256,
        final_state_sha256=trace.steps[-1].state_sha256,
        branch_sha256=tuple(
            sorted((name, branch.trace_sha256) for name, branch in branches.items())
        ),
    )
