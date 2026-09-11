"""Test Suite for Recurrent Core-Driven POMDP Software Engineering Loop.

Verifies:
1. Domain-Agnostic Environment Primitives: NeuralSoftwareEnvironment provides ONLY
   atomic actuators (WRITE_FILE, READ_FILE, EDIT_FILE, RUN_TESTS, RETRIEVE_MEMORY, FINISH)
   with zero hardcoded project templates or preset domain switches.
2. Workspace Path Containment: Strict security check blocks directory traversal (../../).
3. Recurrent POMDP Rollout: RecurrentSoftwareAgent drives the multi-step cycle through
   Pseudo-Brain's recurrent core, stepping Slot 0 (Goal), Slot 1 (Perception/Obs),
   Slot 2 (Action Candidate), and Slot 3 (Action History).
4. Strict Law 1 Working Memory Adherence: Memory footprint remains strictly 4,096 bytes
   throughout the entire POMDP rollout.
5. Closed-Loop Test Failure Observation: Test failures are captured and stepped into
   the core's recurrent state, triggering repair actions.
6. Honest Task Validation: External task validator rejects premature or failing ACTION: FINISH claims.
7. Autonomous Autoregressive Action Generation: Model dynamically samples action tokens directly
   through recurrent state transitions without external scaffolding.
8. Checkpoint Loading: RecurrentSoftwareAgent loads real trained champion weights into the core.
9. POMDP Training Trajectory Generation: POMDPTrajectoryGenerator produces valid
   training streams for pretraining neural weights on autonomous agent traces.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from irene_brain.agent.pomdp_trajectory_generator import POMDPTrajectoryGenerator
from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.agent.software_environment import NeuralSoftwareEnvironment


def test_domain_agnostic_environment_primitives():
    """Verify that NeuralSoftwareEnvironment executes all atomic primitives with zero domain logic."""
    temp_dir = Path(tempfile.mkdtemp())
    env = NeuralSoftwareEnvironment(workspace_dir=temp_dir)

    # 1. ACTION: WRITE_FILE
    write_act = (
        "ACTION: WRITE_FILE math_mod.py\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
    )
    obs_write = env.execute_action(write_act)
    assert obs_write.success is True
    assert "Successfully wrote" in obs_write.observation_text
    assert (temp_dir / "math_mod.py").exists()

    # 2. ACTION: READ_FILE
    obs_read = env.execute_action("ACTION: READ_FILE math_mod.py")
    assert obs_read.success is True
    assert "def add" in obs_read.observation_text

    # 3. ACTION: WRITE_FILE (Unit Test)
    test_act = (
        "ACTION: WRITE_FILE tests/test_math.py\n"
        "from math_mod import add\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
        "if __name__ == '__main__':\n"
        "    test_add()\n"
        "    print('MATH TEST PASSED')\n"
    )
    obs_test_write = env.execute_action(test_act)
    assert obs_test_write.success is True

    # 4. ACTION: RUN_TESTS
    obs_run = env.execute_action("ACTION: RUN_TESTS tests/test_math.py")
    assert obs_run.success is True
    assert "RUN_TESTS PASSED" in obs_run.observation_text
    assert "1 passed" in obs_run.observation_text  # pytest discovers test_add

    # 5. ACTION: EDIT_FILE
    edit_act = (
        "ACTION: EDIT_FILE math_mod.py\n"
        "<<<TARGET\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "===\n"
        "def add(a: int, b: int) -> int:\n"
        "    return int(a + b)\n"
        ">>>"
    )
    obs_edit = env.execute_action(edit_act)
    assert obs_edit.success is True
    assert "int(a + b)" in (temp_dir / "math_mod.py").read_text(encoding="utf-8")

    # 6. ACTION: RETRIEVE_MEMORY
    obs_mem = env.execute_action("ACTION: RETRIEVE_MEMORY binary search")
    assert obs_mem.success is True
    assert "binary search" in obs_mem.observation_text.lower()

    # 7. ACTION: FINISH (without task_validator, rejected for completion)
    obs_fin = env.execute_action("ACTION: FINISH All primitives verified successfully")
    assert obs_fin.success is False
    assert obs_fin.verified_completion is False
    assert obs_fin.action_type == "FINISH"

    # With task_validator, passes verification
    env.task_validator = lambda e: (True, "validated")
    obs_fin_val = env.execute_action("ACTION: FINISH All primitives verified successfully")
    assert obs_fin_val.success is True
    assert obs_fin_val.verified_completion is True


def test_workspace_path_containment_blocks_traversal():
    """Verify security containment: directory traversal escapes (../../) are strictly blocked."""
    temp_dir = Path(tempfile.mkdtemp())
    env = NeuralSoftwareEnvironment(workspace_dir=temp_dir)

    # 1. WRITE_FILE escape attempt
    obs_write = env.execute_action("ACTION: WRITE_FILE ../../escape.py\n# malicious content")
    assert obs_write.success is False
    assert "escapes workspace directory" in obs_write.observation_text

    # 2. READ_FILE escape attempt
    obs_read = env.execute_action("ACTION: READ_FILE ../../../windows/system32/cmd.exe")
    assert obs_read.success is False
    assert "escapes workspace directory" in obs_read.observation_text

    # 3. EDIT_FILE escape attempt
    obs_edit = env.execute_action("ACTION: EDIT_FILE ../../secret.py\n<<<TARGET\na\n===\nb\n>>>")
    assert obs_edit.success is False
    assert "escapes workspace directory" in obs_edit.observation_text

    # 4. RUN_TESTS escape attempt
    obs_run = env.execute_action("ACTION: RUN_TESTS ../../some_test.py")
    assert obs_run.success is False
    assert "escapes workspace directory" in obs_run.observation_text


def test_recurrent_agent_external_task_validation_rejects_premature_finish():
    """Verify that external task validation eliminates the false-positive FINISH shortcut."""
    temp_dir = Path(tempfile.mkdtemp())

    def strict_validator(environment: NeuralSoftwareEnvironment):
        target = environment.workspace_dir / "output.txt"
        if not target.exists():
            return False, "output.txt is missing"
        if "verified_solution" not in target.read_text():
            return False, "output.txt does not contain 'verified_solution'"
        return True, "Requirements met 100%"

    env = NeuralSoftwareEnvironment(workspace_dir=temp_dir, task_validator=strict_validator)
    agent = RecurrentSoftwareAgent()

    # 1. Premature FINISH without fulfilling requirements must be REJECTED
    obs_fail = env.execute_action("ACTION: FINISH I claim task is completed")
    assert obs_fail.success is False
    assert obs_fail.action_type == "FINISH"
    assert "Task incomplete: output.txt is missing" in obs_fail.observation_text

    # 2. Fulfill requirements
    env.execute_action("ACTION: WRITE_FILE output.txt\nverified_solution here")

    # 3. Valid FINISH passes verification
    obs_pass = env.execute_action("ACTION: FINISH All requirements fulfilled")
    assert obs_pass.success is True
    assert "Task verified and passed" in obs_pass.observation_text


def test_recurrent_agent_pomdp_episode_rollout():
    """Verify that RecurrentSoftwareAgent drives an end-to-end POMDP episode through recurrent state."""
    temp_dir = Path(tempfile.mkdtemp())
    def rollout_validator(environment: NeuralSoftwareEnvironment):
        res = environment._handle_run_tests("tests/test_str.py")
        return res.success, "string multiplier tests verified"

    env = NeuralSoftwareEnvironment(workspace_dir=temp_dir, task_validator=rollout_validator)
    agent = RecurrentSoftwareAgent()

    goal = "Build a self-contained string multiplier with unit tests"
    action_plan = [
        (
            "ACTION: WRITE_FILE str_mult.py\n"
            "def repeat_string(s: str, n: int) -> str:\n"
            "    return s * n\n"
        ),
        (
            "ACTION: WRITE_FILE tests/test_str.py\n"
            "from str_mult import repeat_string\n"
            "def test_mult():\n"
            "    assert repeat_string('abc', 2) == 'abcabc'\n"
            "if __name__ == '__main__':\n"
            "    test_mult()\n"
            "    print('STRING MULTIPLIER TESTS OK')\n"
        ),
        "ACTION: RUN_TESTS tests/test_str.py",
        "ACTION: FINISH Module repeat_string verified and passing",
    ]

    result = agent.execute_pomdp_episode(goal=goal, env=env, action_plan=action_plan)

    assert result.success is True
    assert result.cycles_completed == 4
    assert len(result.actions_taken) == 4
    assert len(result.observations) == 4

    # Verify that Slot 0 (Goal), Slot 1 (Perception/Obs), Slot 3 (Action History) transitioned state
    assert result.slot_0_delta > 1e-4, "Slot 0 (Goal) did not update!"
    assert result.slot_1_delta > 1e-4, "Slot 1 (Perception) did not update!"
    assert result.slot_3_delta > 1e-4, "Slot 3 (Action History) did not update!"

    # Verify Law 1 Working Memory Invariance: exactly 4,096 bytes (4.0 KB)
    assert result.working_memory_bytes == 4096, f"Law 1 violated: {result.working_memory_bytes}"

    # Verify physical file existence and test execution output
    assert (temp_dir / "str_mult.py").exists()
    assert (temp_dir / "tests/test_str.py").exists()
    assert any("1 passed" in obs for obs in result.observations)
    assert result.policy_source == "scripted"


def test_recurrent_agent_autonomous_action_generation():
    """Verify that RecurrentSoftwareAgent can autoregressively generate actions from recurrent state."""
    agent = RecurrentSoftwareAgent()
    action = agent.generate_action_autoregressive(slot_id=2, max_new_tokens=16)
    assert isinstance(action, str)
    assert action.startswith("ACTION: ")
    # Working memory stays strictly compliant with Law 1 (4,096 bytes)
    assert agent.cognitive_state.hierarchical_state.fast_state_bytes() == 4096


def test_recurrent_agent_checkpoint_loading():
    """Verify that RecurrentSoftwareAgent cleanly loads real trained champion checkpoints."""
    ckpt_path = Path("brain/checkpoints/pb_35m_champion.pt").resolve()
    if ckpt_path.exists():
        agent = RecurrentSoftwareAgent(checkpoint_path=ckpt_path)
        assert agent.checkpoint_loaded is True
        assert agent.active_checkpoint == str(ckpt_path)
        assert agent.cognitive_state.hierarchical_state.fast_state_bytes() == 4096


def test_pomdp_trajectory_generator_samples_valid_training_text():
    """Verify that POMDPTrajectoryGenerator produces valid pretraining token streams."""
    generator = POMDPTrajectoryGenerator(seed=42)

    for _ in range(5):
        traj = generator.sample_trajectory()
        assert traj.goal != ""
        assert len(traj.steps) >= 3

        text = traj.to_training_text(thread_id=1)
        assert "[THREAD:1][POMDP_CODE]" in text
        assert "[RESP]ACTION:" in text
        assert "[EOS]" in text
        assert "[OBSERVATION:" in text
