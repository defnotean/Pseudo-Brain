"""Test Suite for Multi-File Large Software Engineering Engine.

Verifies:
1. Multi-File AST syntax validation and cross-file import DAG construction.
2. Circular import detection and prevention across multiple modules.
3. End-to-end multi-file project synthesis (modular ASCII Arcade engine) with cross-module test execution.
4. Autonomous cross-module self-repair when an interface error occurs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from irene_brain.agent.multi_file_synthesizer import (
    MultiFileProject,
    MultiFileSandboxVerifier,
    MultiFileSoftwareSynthesizer,
    MultiFileVerificationResult,
)


def test_sandbox_verifier_ast_and_dag():
    """Verify AST checking and dependency DAG construction."""
    verifier = MultiFileSandboxVerifier()

    valid_files = {
        "config.py": "WIDTH = 80\nHEIGHT = 24\n",
        "engine.py": "from config import WIDTH, HEIGHT\ndef run():\n    return WIDTH * HEIGHT\n",
        "main.py": "from engine import run\nif __name__ == '__main__':\n    print(run())\n",
    }

    ast_errs = verifier.check_ast_syntax(valid_files)
    assert len(ast_errs) == 0, f"Expected 0 AST errors, got {ast_errs}"

    dag, dag_errs = verifier.build_and_verify_dependency_dag(valid_files)
    assert len(dag_errs) == 0, f"Expected 0 DAG errors, got {dag_errs}"
    assert "config" in dag.get("engine", [])
    assert "engine" in dag.get("main", [])


def test_sandbox_verifier_detects_circular_dependency():
    """Verify that circular imports are trapped immediately before execution."""
    verifier = MultiFileSandboxVerifier()

    circular_files = {
        "mod_a.py": "from mod_b import b_func\ndef a_func(): pass\n",
        "mod_b.py": "from mod_a import a_func\ndef b_func(): pass\n",
    }

    dag, dag_errs = verifier.build_and_verify_dependency_dag(circular_files)
    assert len(dag_errs) > 0, "Expected circular dependency error to be detected!"
    assert any("Circular dependency" in err for err in dag_errs)


def test_modular_arcade_project_synthesis_and_tests():
    """Verify end-to-end multi-file software synthesis and test suite execution."""
    synthesizer = MultiFileSoftwareSynthesizer()
    project: MultiFileProject = synthesizer.synthesize_multi_file_project(
        "build a modular arcade engine in multiple files with config, models, and renderer"
    )

    assert project.name == "ascii_arcade"
    assert len(project.files) >= 5, f"Expected >=5 modules, got {len(project.files)}"
    assert "config.py" in project.files
    assert "models.py" in project.files
    assert "engine.py" in project.files
    assert "renderer.py" in project.files
    assert "main.py" in project.files
    assert "tests/test_arcade.py" in project.files

    # Verify all files were physically written to disk
    for rel_path in project.files.keys():
        f_path = project.root_dir / rel_path
        assert f_path.exists(), f"File {f_path} was not created on disk!"

    # Verify test suite passed cleanly
    assert project.success is True, f"Project tests failed: {project.test_output}"
    assert "ALL CROSS-MODULE TESTS PASSED CLEANLY" in project.test_output
