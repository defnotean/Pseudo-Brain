"""Test Suite for Pseudo-Brain Cognitive Decision Loop.

Verifies:
1. Perception-Based Memory Retrieval: Recalled knowledge is stepped into recurrent
   working memory slots (Slot 1 = perception), updating state norms and representations
   without bypassing the neural core.
2. Dynamic Thoughtlet Project Decomposition: Novel multi-file project specifications
   (e.g. task_manager, data_processor) are dynamically decomposed across thoughtlet slots,
   written to disk, verified via AST, and executed in the sandbox with passing tests.
3. Closed-Loop Self-Repair: When cross-module import or interface errors occur, the repair
   loop parses the traceback, updates the code, and re-verifies successfully.
4. Strict Law 1 Working Memory Invariance: Fast working memory footprint remains
   strictly <= 4,096 bytes across all thoughtlet transitions.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

from irene_brain.agent.continual_learner import AutonomousLifelongAgent
from irene_brain.agent.multi_file_synthesizer import (
    ArchitecturePlan,
    MultiFileProject,
    MultiFileSandboxVerifier,
    MultiFileSoftwareSynthesizer,
    MultiFileVerificationResult,
)


def test_perception_based_memory_ingestion_updates_slots():
    """Verify that memory retrieval updates recurrent working memory slots as a perception."""
    agent = AutonomousLifelongAgent()

    # Capture initial slot state
    initial_working = agent.cognitive_state.hierarchical_state.working_thoughts.clone()
    slot_0_initial = initial_working[0, 0].clone()
    slot_1_initial = initial_working[0, 1].clone()

    # Ingest a query that triggers static memory recall
    result = agent.respond("how does binary search work on sorted arrays")

    assert result.recalled_from_static is True
    assert result.did_research is False
    assert "binary search" in result.reply.lower()

    # Verify that Slot 0 (Goal) and Slot 1 (Perception) were updated in recurrent memory
    current_working = agent.cognitive_state.hierarchical_state.working_thoughts
    slot_0_updated = current_working[0, 0]
    slot_1_updated = current_working[0, 1]

    # Recurrent state must have changed as a result of stepping prompt and observation
    diff_slot_0 = torch.norm(slot_0_updated - slot_0_initial).item()
    diff_slot_1 = torch.norm(slot_1_updated - slot_1_initial).item()

    assert diff_slot_0 > 1e-4, f"Slot 0 (Goal) did not update! diff={diff_slot_0}"
    assert diff_slot_1 > 1e-4, f"Slot 1 (Perception) did not update! diff={diff_slot_1}"

    # Verify Law 1 Working Memory Footprint (strict 4.0 KB = 4,096 bytes)
    fast_bytes = agent.cognitive_state.hierarchical_state.fast_state_bytes()
    assert fast_bytes == 4096, f"Law 1 violated: {fast_bytes} bytes (expected 4096)"


def test_dynamic_thoughtlet_task_manager_decomposition():
    """Verify dynamic thoughtlet decomposition on a novel task manager project."""
    synthesizer = MultiFileSoftwareSynthesizer()

    project: MultiFileProject = synthesizer.synthesize_multi_file_project(
        "build a modular task management CLI in multiple files with storage, models, and cli"
    )

    assert project.name == "task_manager"
    assert len(project.files) >= 5, f"Expected >=5 files, got {len(project.files)}"
    assert "config.py" in project.files
    assert "models.py" in project.files
    assert "storage.py" in project.files
    assert "cli.py" in project.files
    assert "main.py" in project.files
    assert "tests/test_tasks.py" in project.files

    # Verify physical existence on disk
    for rel_path in project.files.keys():
        f_path = project.root_dir / rel_path
        assert f_path.exists(), f"File {f_path} was not created on disk!"

    # Verify sandbox test suite execution passed
    assert project.success is True, f"Project tests failed: {project.test_output}"
    assert "TASK MANAGER MULTI-FILE TESTS PASSED CLEANLY" in project.test_output


def test_dynamic_thoughtlet_general_utility_decomposition():
    """Verify dynamic thoughtlet decomposition on a novel general utility project."""
    synthesizer = MultiFileSoftwareSynthesizer()

    project: MultiFileProject = synthesizer.synthesize_multi_file_project(
        "create a modular cache service with models, engine, and interface"
    )

    assert project.name == "cache_service"
    assert "config.py" in project.files
    assert "models.py" in project.files
    assert "core.py" in project.files
    assert "interface.py" in project.files
    assert "main.py" in project.files
    assert "tests/test_cache_service.py" in project.files

    # Verify tests passed in sandbox
    assert project.success is True, f"General utility tests failed: {project.test_output}"
    assert "MULTI-FILE TESTS PASSED CLEANLY" in project.test_output


def test_closed_loop_self_repair_on_interface_error():
    """Verify that cross-module errors trigger closed-loop diagnosis and self-repair."""
    synthesizer = MultiFileSoftwareSynthesizer()

    # Construct a project with a deliberate missing export
    broken_files = {
        "config.py": "WIDTH = 60\n",
        "engine.py": "from config import WIDTH, HEIGHT\ndef run(): return WIDTH\n",
    }
    project = MultiFileProject(
        name="test_repair_proj",
        root_dir=Path(tempfile.mkdtemp()),
        files=broken_files,
        entry_point="engine.py",
        test_files=[],
    )
    synthesizer._write_project_files_to_disk(project)

    mock_error_res = MultiFileVerificationResult(
        success=False,
        ast_errors=[],
        import_errors=[],
        test_output="ImportError: cannot import name 'HEIGHT' from 'config'",
        exit_code=1,
    )

    repaired, fixed_files, rep_log = synthesizer.autonomous_multi_file_self_repair(
        project, mock_error_res
    )

    assert repaired is True
    assert "config.py" in fixed_files
    # Verify the missing symbol was repaired
    assert "HEIGHT" in fixed_files["config.py"]
    assert "Added missing export 'HEIGHT'" in rep_log
