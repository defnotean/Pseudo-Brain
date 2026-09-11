"""Execution feedback must reflect the current source and actually run test bodies."""
import os

from irene_brain.agent.procedural_evaluator import make_hidden_task_validator
from irene_brain.agent.software_environment import NeuralSoftwareEnvironment


def test_pytest_failure_is_executed_and_then_repaired(tmp_path):
    env = NeuralSoftwareEnvironment(tmp_path)
    env.execute_action("ACTION: WRITE_FILE answer.py\ndef answer():\n    return 0")
    env.execute_action("ACTION: WRITE_FILE test_answer.py\nfrom answer import answer\ndef test_answer():\n    assert answer() == 1")
    assert not env.execute_action("ACTION: RUN_TESTS").success
    env.execute_action("ACTION: WRITE_FILE answer.py\ndef answer():\n    return 1")
    assert env.execute_action("ACTION: RUN_TESTS test_answer.py").success


def test_missing_explicit_test_path_cannot_fall_back_to_unrelated_pass(tmp_path):
    env = NeuralSoftwareEnvironment(tmp_path)
    env.execute_action("ACTION: WRITE_FILE test_pass.py\nassert True")
    assert not env.execute_action("ACTION: RUN_TESTS missing.py").success
    assert env.execute_action("ACTION: RUN_TESTS").success


def test_directory_runs_only_its_tests(tmp_path):
    env = NeuralSoftwareEnvironment(tmp_path)
    env.execute_action("ACTION: WRITE_FILE tests/test_pass.py\ndef test_ok():\n    assert True")
    env.execute_action("ACTION: WRITE_FILE other/test_fail.py\ndef test_bad():\n    assert False")
    assert env.execute_action("ACTION: RUN_TESTS tests").success


def test_hidden_validator_does_not_reuse_stale_bytecode(tmp_path):
    import py_compile
    env = NeuralSoftwareEnvironment(tmp_path)
    path = tmp_path / "answer.py"
    path.write_text("def answer():\n    return 1\n")
    stamp = 1_700_000_000
    os.utime(path, (stamp, stamp))
    py_compile.compile(str(path), doraise=True)
    path.write_text("def answer():\n    return 0\n")
    os.utime(path, (stamp, stamp))
    validate = make_hidden_task_validator("from answer import answer\nassert answer() == 1\nprint('HIDDEN TESTS PASSED')", "answer.py")
    success, detail = validate(env)
    assert not success
    assert "AssertionError" in detail
