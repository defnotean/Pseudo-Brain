"""Scripted remote integration controls, never autonomous capability scores."""
import pytest
from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment
from irene_brain.evaluation.workspace_sandbox import run_workspace_tests
from irene_brain.evaluation.code_sandbox import check_program


def test_wrong_program_repairs_and_external_finish_is_required():
    def validate(files):
        result = check_program(files.get('maths.py',''),
            'def check(candidate):\n    assert candidate(-7) == -14\n    assert candidate(11) == 22', 'twice')
        return result['passed'], 'Independent checks passed' if result['passed'] else 'Independent checks failed'
    env = IsolatedSoftwareEnvironment(validator=validate)
    assert env.execute_action('ACTION: WRITE_FILE maths.py\ndef twice(x): return x*3').success
    assert env.execute_action('ACTION: WRITE_FILE tests/test_maths.py\nfrom maths import twice\ndef test_positive():\n    assert twice(4) == 8').success
    assert not env.execute_action('ACTION: RUN_TESTS tests').success
    assert not env.execute_action('ACTION: FINISH done').verified_completion
    assert env.execute_action('ACTION: EDIT_FILE maths.py\n<<<TARGET\nx*3\n===\nx*2\n>>>').success
    assert env.execute_action('ACTION: RUN_TESTS').success
    result = env.execute_action('ACTION: FINISH done')
    assert result.success and result.verified_completion
    assert env.execute_action('ACTION: READ_FILE maths.py').observation_text == 'def twice(x): return x*2'
    assert not IsolatedSoftwareEnvironment().execute_action('ACTION: FINISH done').success


def test_unittest_packages_and_readonly_snapshot_isolation():
    files = {'pkg/__init__.py':'','pkg/value.py':'VALUE=19',
             'tests/test_isolation.py':'''import os, unittest
from pkg.value import VALUE
class TestSnapshot(unittest.TestCase):
    def test_value_and_isolation(self):
        self.assertEqual(VALUE,19)
        for path in ('/content','/root','/proc','/dev/nvidia0'):
            self.assertFalse(os.path.exists(path))
        with self.assertRaises(OSError):
            open('/workspace/pkg/value.py','w')
        self.assertLessEqual(set(os.environ),{'PATH','PWD','LC_CTYPE'})
'''}
    result = run_workspace_tests(files)
    assert result['passed'],result
    assert files['pkg/value.py']=='VALUE=19'


@pytest.mark.parametrize('source',[
    '', 'raise SystemExit(0)',
    'def test_bad():\n    assert False',
    'async def test_async():\n    assert False',
    'def test_generator():\n    yield False',
    'import unittest\n@unittest.skip("skip")\ndef test_skip(): pass',
    'import socket\ndef test_socket(): socket.socket()',
], ids=['empty','early_exit','failed_assert','async','generator','skipped','network'])
def test_empty_early_exit_failed_unexecuted_and_network_tests_never_pass(source):
    assert not run_workspace_tests({'test_probe.py':source})['passed']


@pytest.mark.parametrize('action',[
    'ACTION: WRITE_FILE ../escape.py\nx', 'ACTION: WRITE_FILE /escape.py\nx',
    'ACTION: WRITE_FILE a/../escape.py\nx', 'ACTION: WRITE_FILE a\\b.py\nx',
    'ACTION: WRITE_FILE a.py', 'ACTION: WRITE_FILEevil.py\nx',
    'ACTION: READ_FILE a.py\nextra',
    'ACTION: EDIT_FILE a.py\n<<<TARGET\nx\n===\ny\n>>>',
    'ACTION: WRITE_FILE a.py/child.py\nx',
    'ACTION: WRITE_FILE large.py\n'+'x'*65537,
], ids=['parent','absolute','embedded_parent','backslash','missing_body','wrong_verb','extra_body','ambiguous_edit','file_parent','oversize'])
def test_invalid_actions_are_atomic(action):
    env = IsolatedSoftwareEnvironment({'a.py':'xx'})
    before = env.files
    assert not env.execute_action(action).success
    assert env.files==before


def test_snapshot_limits_provider_copy_and_memory_results():
    with pytest.raises(ValueError):
        IsolatedSoftwareEnvironment({str(i):'' for i in range(65)})
    with pytest.raises(ValueError):
        IsolatedSoftwareEnvironment({str(i):'x'*65536 for i in range(17)})
    def validator(files):
        files.clear()
        return False,'Not complete'
    env = IsolatedSoftwareEnvironment({'a.py':'original'},validator=validator,retrieve=lambda query:'Known: '+query)
    assert not env.execute_action('ACTION: FINISH done').success
    assert env.files=={'a.py':'original'}
    assert env.execute_action('ACTION: RETRIEVE_MEMORY topic').observation_text=='Known: topic'
    external = env.files
    external.clear()
    assert env.files=={'a.py':'original'}
