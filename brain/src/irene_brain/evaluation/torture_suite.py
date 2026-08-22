"""Phase 2.6 Foundation Torture Suite.

Unit tests for cognition: each environment isolates ONE foundational property.
All tasks are procedurally generated, deterministic given seed, and designed so
that a reactive (memoryless) policy CANNOPT exceed chance on the property under
test. Every task reports accuracy against its own chance floor.

Design contract per task:
- `TaskSpec.name`: unique id
- `build_episode(rng) -> Episode`: observation sequence + labels
- observations are 32x32 RGB uint8; actions are discrete {0..4}
- each episode ends with a DECISION phase whose label is the ground truth

The suite runner scores a model's final-decision accuracy over N episodes and
reports per-task accuracy vs the recorded chance level.

25 tasks grouped by faculty:
  PERSISTENCE (1-4): hold information across delays
  HYPOTHESES (5-9): maintain/update/compete possibilities
  EVIDENCE (10-13): update, reject, ignore, contradict
  GOALS (14-16): maintain/switch/abandon subgoals
  MEMORY (17-19): retrieve relevant, avoid irrelevant, randomness vs ignorance
  PROBING (20-21): gather information only when useful
  ADAPTATION (22): dynamics change mid-episode
  CAPACITY (23-24): many simultaneous latent variables
  META (25): act while thinking (anytime output validity)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Episode container
# ---------------------------------------------------------------------------

class Episode:
    def __init__(self):
        self.frames: list[np.ndarray] = []      # [32,32,3] uint8 each
        self.decision_frame_index = -1          # frame where decision is queried
        self.label: int = -1                    # correct decision {1..4} or 0 for WAIT
        self.chance: float = 0.2                # chance accuracy for this task
        self.task_name: str = ""

    def add(self, frame: np.ndarray) -> None:
        self.frames.append(frame)


def _blank(rng) -> np.ndarray:
    return (rng.random((32, 32, 3)) * 40 + 40).astype(np.uint8)


def _cue(frame: np.ndarray, quadrant: int, channel: int, intensity: int = 160) -> None:
    """Stamp a cue in one of 4 quadrants on one color channel."""
    h = slice(0, 16) if quadrant in (0, 1) else slice(16, 32)
    w = slice(0, 16) if quadrant in (0, 2) else slice(16, 32)
    frame[h, w, channel] = np.minimum(255, frame[h, w, channel].astype(int) + intensity)


def _cue_channel_to_action(channel: int) -> int:
    return channel + 1  # channels 0,1,2 -> actions 1,2,3


# ---------------------------------------------------------------------------
# PERSISTENCE (tasks 1-4)
# ---------------------------------------------------------------------------

def t01_single_cue_short_delay(rng, delay=8):
    """Flash 1 cue, blank delay, decide which cue. Chance=1/3."""
    ep = Episode(); ep.task_name = "t01_cue_short"; ep.chance = 1 / 3
    f0 = _blank(rng); ch = rng.integers(0, 3); _cue(f0, rng.integers(0, 4), ch)
    ep.add(f0)
    for _ in range(delay):
        ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = _cue_channel_to_action(ch)
    return ep


def t02_single_cue_long_delay(rng, delay=40):
    return t01_single_cue_short_delay(rng, delay=delay)


def t03_three_independent_cues(rng):
    """3 cues flash sequentially (different quadrants), decide ALL via 4-way combo.
    Label encodes set of channels seen; chance=1/8 approximated as 4-way."""
    ep = Episode(); ep.task_name = "t03_three_cues"; ep.chance = 0.125
    chans = list(rng.permutation(3))
    quad = list(rng.permutation(4))
    seen = []
    for i, ch in enumerate(chans):
        f = _blank(rng)
        _cue(f, quad[i], ch)
        ep.add(f)
        seen.append(ch)
        for _ in range(rng.integers(3, 7)):
            ep.add(_blank(rng))
    # label: sum-based code of the SET (order-independent), mapped to 1..4 by mod
    code = sum(2 ** c for c in seen) % 4
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = code + 1
    return ep


def t04_cue_then_blank_long(rng):
    """Cue flashes 1 frame, then 60 blank frames with noise bursts (distractors)."""
    ep = Episode(); ep.task_name = "t04_cue_noise_delay"; ep.chance = 1 / 3
    f0 = _blank(rng); ch = rng.integers(0, 3); _cue(f0, rng.integers(0, 4), ch)
    ep.add(f0)
    for _ in range(60):
        f = _blank(rng)
        # noise burst on a random WRONG channel to punish sloppy memory
        wrong = (ch + rng.integers(1, 3)) % 3
        q = rng.integers(0, 4)
        f[:8, :8, wrong] = np.minimum(255, f[:8, :8, wrong].astype(int) + 120)
        ep.add(f)
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = _cue_channel_to_action(ch)
    return ep


# ---------------------------------------------------------------------------
# HYPOTHESES (tasks 5-9)
# ---------------------------------------------------------------------------

def _staged_world_task(rng, n_stages: int, name: str):
    """n_stages independent binary hazards revealed sequentially; final label is
    the world index. Generalizes the escalation benchmark. Chance=1/2^n."""
    ep = Episode(); ep.task_name = name
    hazards = [bool(rng.integers(0, 2)) for _ in range(n_stages)]
    world = sum((0 if h else (1 << (n_stages - 1 - i))) for i, h in enumerate(hazards))
    rgb = _blank(rng)
    ep.add(rgb)
    for stage, haz in enumerate(hazards):
        for _ in range(6):
            ep.add(_blank(rng))
        f = _blank(rng)
        _cue(f, (stage * 2) % 4, 0 if haz else 1, intensity=140)
        ep.add(f)
        for _ in range(4):
            ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.chance = 1.0 / (2 ** n_stages)
    # map world (0..2^n-1) into action space {1..4}: fold
    ep.label = (world % 4) + 1
    return ep


def t05_two_hypotheses(rng):
    return _staged_world_task(rng, 1, "t05_two_hyp")


def t08_four_hypotheses(rng):
    return _staged_world_task(rng, 2, "t08_four_hyp")


def t09_eight_hypotheses(rng):
    return _staged_world_task(rng, 3, "t09_eight_hyp")


def t06_update_without_erasing(rng):
    """Two hazards resolved at different times; second resolution must NOT erase first."""
    ep = Episode(); ep.task_name = "t06_no_erase"; ep.chance = 0.25
    ha, hb = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
    world = (0 if ha else 2) + (0 if hb else 1)
    rgb = _blank(rng); ep.add(rgb)
    for _ in range(6): ep.add(_blank(rng))
    f = _blank(rng); _cue(f, 0, 0 if ha else 1); ep.add(f)
    for _ in range(20): ep.add(_blank(rng))   # LONG gap: first fact must survive
    f2 = _blank(rng); _cue(f2, 2, 0 if hb else 1); ep.add(f2)
    for _ in range(6): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = world + 1
    return ep


def t07_disproven_hypothesis(rng):
    """Initial partial evidence suggests hypothesis X; later evidence disproves it.
    Correct answer follows the DISPROOF."""
    ep = Episode(); ep.task_name = "t07_disproof"; ep.chance = 0.5
    misleading = bool(rng.integers(0, 2))       # early cue points here
    truth = not misleading                       # later cue contradicts
    rgb = _blank(rng); ep.add(rgb)
    f = _blank(rng); _cue(f, 0, 0 if misleading else 1); ep.add(f)
    for _ in range(10): ep.add(_blank(rng))
    f2 = _blank(rng); _cue(f2, 2, 0 if truth else 1, intensity=200); ep.add(f2)
    for _ in range(5): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = (2 if truth else 1)
    return ep


# ---------------------------------------------------------------------------
# EVIDENCE (tasks 10-13)
# ---------------------------------------------------------------------------

def t10_irrelevant_evidence(rng):
    """Strong but irrelevant cues appear; only the weak RELEVANT cue matters."""
    ep = Episode(); ep.task_name = "t10_irrelevant"; ep.chance = 0.5
    truth = bool(rng.integers(0, 2))
    rgb = _blank(rng); ep.add(rgb)
    for _ in range(3):
        f = _blank(rng)
        _cue(f, rng.integers(0, 4), rng.integers(0, 3), intensity=220)  # loud distractors
        ep.add(f)
        for _ in range(2): ep.add(_blank(rng))
    f = _blank(rng)
    _cue(f, 3, 0 if truth else 1, intensity=60)  # faint relevant cue
    ep.add(f)
    for _ in range(12): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = (2 if truth else 1)
    return ep


def t11_contradictory_sequence(rng):
    """Two contradictory cues of equal strength; LAST one wins (update test)."""
    ep = Episode(); ep.task_name = "t11_contradiction"; ep.chance = 0.5
    first = bool(rng.integers(0, 2)); last = not first
    rgb = _blank(rng); ep.add(rgb)
    f = _blank(rng); _cue(f, 0, 0 if first else 1); ep.add(f)
    for _ in range(15): ep.add(_blank(rng))
    f2 = _blank(rng); _cue(f2, 2, 0 if last else 1); ep.add(f2)
    for _ in range(10): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = (2 if last else 1)
    return ep


def t12_weak_then_strong(rng):
    """Weak evidence then strong CONFLICTING evidence; strong wins (weighted update)."""
    ep = Episode(); ep.task_name = "t12_weak_strong"; ep.chance = 0.5
    weak = bool(rng.integers(0, 2)); strong = not weak
    rgb = _blank(rng); ep.add(rgb)
    f = _blank(rng); _cue(f, 0, 0 if weak else 1, intensity=50); ep.add(f)
    for _ in range(10): ep.add(_blank(rng))
    f2 = _blank(rng); _cue(f2, 2, 0 if strong else 1, intensity=220); ep.add(f2)
    for _ in range(8): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = (2 if strong else 1)
    return epi if False else ep


def t13_probabilistic_majority(rng):
    """Noisy cue stream: 70% of flashes favor side A or B; decide majority."""
    ep = Episode(); ep.task_name = "t13_majority"; ep.chance = 0.5
    truth = bool(rng.integers(0, 2))
    rgb = _blank(rng); ep.add(rgb)
    for i in range(10):
        f = _blank(rng)
        side = truth if (rng.random() < 0.7) else (not truth)
        _cue(f, i % 4, 0 if side else 1, intensity=90)
        ep.add(f)
        for _ in range(2): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = (2 if truth else 1)
    return ep


# ---------------------------------------------------------------------------
# GOALS (tasks 14-16)
# ---------------------------------------------------------------------------

def t14_goal_through_distraction(rng):
    """Early goal color must be reported despite intervening reward-like flashes."""
    ep = Episode(); ep.task_name = "t14_goal_hold"; ep.chance = 1 / 3
    goal_ch = rng.integers(0, 3)
    rgb = _blank(rng); _cue(rgb, 1, goal_ch, intensity=180); ep.add(rgb)
    for _ in range(30):
        f = _blank(rng)
        if rng.random() < 0.4:
            other = (goal_ch + rng.integers(1, 3)) % 3
            _cue(f, rng.integers(0, 4), other, intensity=200)  # tempting distractor
        ep.add(f)
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = _cue_channel_to_action(goal_ch)
    return ep


def t15_goal_switch(rng):
    """First cue sets goal; explicit SWITCH signal appears; second cue replaces it."""
    ep = Episode(); ep.task_name = "t15_goal_switch"; ep.chance = 1 / 3
    g1 = rng.integers(0, 3); g2 = rng.integers(0, 3)
    rgb = _blank(rng); _cue(rgb, 0, g1); ep.add(rgb)
    for _ in range(8): ep.add(_blank(rng))
    f = _blank(rng); _cue(f, 1, 2, intensity=255); ep.add(f)  # white switch signal
    for _ in range(4): ep.add(_blank(rng))
    f2 = _blank(rng); _cue(f2, 2, g2); ep.add(f2)
    for _ in range(12): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = _cue_channel_to_action(g2)
    return ep


def t16_abandon_failed_subgoal(rng):
    """Subgoal cue then explicit FAIL signal; final action must reflect the fallback."""
    ep = Episode(); ep.task_name = "t16_abandon"; ep.chance = 0.5
    fallback = bool(rng.integers(0, 2))
    rgb = _blank(rng); _cue(rgb, 0, rng.integers(0, 3)); ep.add(rgb)  # subgoal
    for _ in range(6): ep.add(_blank(rng))
    f = _blank(rng)
    f[0:8, 0:8, :] = 255  # white flash = subgoal failed
    ep.add(f)
    for _ in range(10): ep.add(_blank(rng))
    f2 = _blank(rng); _cue(f2, 2, 0 if fallback else 1, intensity=100); ep.add(f2)
    for _ in range(8): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = (2 if fallback else 1)
    return ep


# ---------------------------------------------------------------------------
# MEMORY (tasks 17-19)
# ---------------------------------------------------------------------------

def t17_retrieve_relevant(rng):
    """Two episodes worth of cues interleaved; query only matches one; report it."""
    ep = Episode(); ep.task_name = "t17_retrieve"; ep.chance = 1 / 3
    key_ch = rng.integers(0, 3); val_ch = rng.integers(0, 3)
    rgb = _blank(rng); _cue(rgb, 0, key_ch); ep.add(rgb)      # KEY
    for _ in range(5): ep.add(_blank(rng))
    rgb2 = _blank(rng); _cue(rgb2, 1, val_ch); ep.add(rgb2)    # VALUE bound to key
    for _ in range(20): ep.add(_blank(rng))
    probe = _blank(rng); _cue(probe, 2, key_ch, intensity=200); ep.add(probe)  # probe = key
    for _ in range(10): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = _cue_channel_to_action(val_ch)
    return ep


def t18_avoid_irrelevant_memory(rng):
    """Old association must NOT be used after an invalidation signal."""
    ep = Episode(); ep.task_name = "t18_no_stale"; ep.chance = 0.5
    stale_v = bool(rng.integers(0, 2)); new_v = not stale_v
    rgb = _blank(rng); _cue(rgb, 0, 0); ep.add(rgb)
    rgb2 = _blank(rng)
    _cue(rgb2, 1, 0 if stale_v else 1); ep.add(rgb2)           # old binding
    for _ in range(10): ep.add(_blank(rng))
    f = _blank(rng); f[0:8, 24:, :] = 255; ep.add(f)            # invalidate signal
    rgb3 = _blank(rng); _cue(rgb3, 2, 0, intensity=200); ep.add(rgb3)  # same key probed
    for _ in range(10): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = (2 if new_v else 1)
    return ep


def t19_randomness_vs_ignorance(rng):
    """Truly random 50/50 flashes then NO evidence: best play is still 50/50;
    scored as calibration rather than accuracy (accuracy vs both-side baselines)."""
    ep = Episode(); ep.task_name = "t19_calibration"; ep.chance = 0.5
    rgb = _blank(rng); ep.add(rgb)
    for i in range(6):
        f = _blank(rng)
        _cue(f, i % 4, int(rng.integers(0, 2)))
        ep.add(f)
        for _ in range(2): ep.add(_blank(rng))
    for _ in range(15): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = 0  # no correct side; measures whether model commits wrongly
    return ep


# ---------------------------------------------------------------------------
# PROBING (tasks 20-21)
# ---------------------------------------------------------------------------

def t20_probe_pays(rng):
    """Ambiguous state; a WAIT action reveals the answer before deciding.
    Model must learn waiting beats guessing. Scored on final accuracy."""
    ep = Episode(); ep.task_name = "t20_probe_pays"; ep.chance = 0.5
    truth = bool(rng.integers(0, 2))
    rgb = _blank(rng); ep.add(rgb)
    for _ in range(10): ep.add(_blank(rng))     # ambiguous period
    reveal = _blank(rng)
    _cue(reveal, 2, 0 if truth else 1, intensity=230)
    ep.add(reveal)                               # evidence exists IF model waits
    for _ in range(3): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = (2 if truth else 1)
    return ep


def t21_probe_wastes(rng):
    """Evidence will never arrive; waiting is strictly worse than immediate guess.
    Scored by whether the model eventually commits (measured separately)."""
    ep = Episode(); ep.task_name = "t21_no_probe"; ep.chance = 0.5
    truth = bool(rng.integers(0, 2))
    rgb = _blank(rng); ep.add(rgb)
    for _ in range(30): ep.add(_blank(rng))     # nothing ever comes
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = (2 if truth else 1)
    return ep


# ---------------------------------------------------------------------------
# ADAPTATION (task 22)
# ---------------------------------------------------------------------------

def t22_regime_shift(rng):
    """First half: channel A predicts label 1. Second half (no signal): channel A
    predicts label 2. Tests posterior revision without task ID."""
    ep = Episode(); ep.task_name = "t22_regime_shift"; ep.chance = 0.5
    flip_at = 8
    seq = [bool(rng.integers(0, 2)) for _ in range(16)]
    rgb = _blank(rng); ep.add(rgb)
    for i, s in enumerate(seq):
        meaning = (not s) if i >= flip_at else s   # semantics flip silently
        f = _blank(rng)
        _cue(f, i % 4, 0 if meaning else 1, intensity=90)
        ep.add(f)
        for _ in range(2): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    final_meaning = (not seq[-1]) if 15 >= flip_at else seq[-1]
    ep.label = (2 if final_meaning else 1)
    return ep


# ---------------------------------------------------------------------------
# CAPACITY (tasks 23-24)
# ---------------------------------------------------------------------------

def t23_multi_object_tracking(rng):
    """3 objects with independent colors flashed; final query asks about ONE."""
    ep = Episode(); ep.task_name = "t23_multi_object"; ep.chance = 1 / 3
    colors = [int(rng.integers(0, 3)) for _ in range(3)]
    for i, c in enumerate(colors):
        f = _blank(rng)
        _cue(f, i, c, intensity=150)
        ep.add(f)
        for _ in range(4): ep.add(_blank(rng))
    target_obj = int(rng.integers(0, 3))
    probe = _blank(rng)
    probe[16:, 16:, :] = 255  # white marker = object 3 queried (fixed mapping)
    ep.add(probe)
    for _ in range(10): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = _cue_channel_to_action(colors[target_obj])
    return ep


def t24_long_horizon_plan(rng):
    """Sequence A->B->C must be remembered in ORDER; final label encodes order parity."""
    ep = Episode(); ep.task_name = "t24_plan_order"; ep.chance = 0.5
    order = list(rng.permutation([0, 1, 2]))
    for pos, ch in enumerate(order):
        f = _blank(rng)
        _cue(f, pos, ch)
        ep.add(f)
        for _ in range(5): ep.add(_blank(rng))
    even_first = (order[0] % 2 == 0)
    for _ in range(10): ep.add(_blank(rng))
    ep.decision_frame_index = len(ep.frames) - 1
    ep.label = (2 if even_first else 1)
    return ep


# ---------------------------------------------------------------------------
# META (task 25)
# ---------------------------------------------------------------------------

def t25_anytime_validity(rng):
    """Decision requested MID-stream (frame 12 of 40); remaining stream irrelevant.
    Tests that the model can commit without seeing everything."""
    ep = Episode(); ep.task_name = "t25_anytime"; ep.chance = 1 / 3
    ch = rng.integers(0, 3)
    rgb = _blank(rng); _cue(rgb, 0, ch, intensity=180); ep.add(rgb)
    while len(ep.frames) < 12:
        ep.add(_blank(rng))
    ep.decision_frame_index = 11
    ep.label = _cue_channel_to_action(ch)
    while len(ep.frames) < 40:                   # tail must not change the answer
        ep.add(_blank(rng))
    return ep


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS = [
    t01_single_cue_short_delay, t02_single_cue_long_delay, t03_three_independent_cues,
    t04_cue_then_blank_long,
    t05_two_hypotheses, t06_update_without_erasing, t07_disproven_hypothesis,
    t08_four_hypotheses, t09_eight_hypotheses,
    t10_irrelevant_evidence, t11_contradictory_sequence, t12_weak_then_strong,
    t13_probabilistic_majority,
    t14_goal_through_distraction, t15_goal_switch, t16_abandon_failed_subgoal,
    t17_retrieve_relevant, t18_avoid_irrelevant_memory, t19_randomness_vs_ignorance,
    t20_probe_pays, t21_probe_wastes,
    t22_regime_shift,
    t23_multi_object_tracking, t24_long_horizon_plan,
    t25_anytime_validity,
]

TASK_NAMES = [fn.__name__.split("_", 0)[-1] for fn in TASKS]
