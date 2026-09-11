import ast
import pytest

from irene_brain.data.mbpp_training import adapt_case, structural_code_key


def fixture():
    return {'task_id':601, 'text':'Implement the requested counter interface.',
            'code':'class Counter:\n    """private class documentation"""\n    def __init__(self, value):\n        self.value = value\n\ndef advance(counter, step=1):\n    """private function documentation"""\n    marker = "implementation-secret"\n    counter.value += step\n    return counter.value',
            'test_setup_code':'counter = Counter(10)',
            'test_list':['assert advance(counter) == 11','assert advance(counter, 2) == 13','assert advance(counter) == 14'],
            'challenge_test_list':[]}


def test_prompt_contains_interfaces_without_solution_bodies_or_private_tests():
    row = fixture()
    case = adapt_case(row)
    assert case['reference_solution'] == row['code']
    assert 'class Counter:' in case['initial_prompt']
    assert 'def advance(counter, step=1):' in case['initial_prompt']
    for secret in ('implementation-secret', 'private class documentation', 'private function documentation',
                   'counter.value += step', 'assert advance(counter, 2) == 13'):
        assert secret not in case['initial_prompt']
    interface = ast.parse(case['public_interface'])
    functions = [node for node in ast.walk(interface) if isinstance(node, ast.FunctionDef)]
    assert len(functions) == 2
    assert all(len(node.body) == 1 and isinstance(node.body[0], ast.Expr) and node.body[0].value.value is Ellipsis for node in functions)


def test_setup_and_sequential_assertions_preserved_with_module_level_import():
    case = adapt_case(fixture())
    public = ast.parse(case['initial_files']['test_public.py'])
    private = ast.parse(case['private_check'])
    assert isinstance(public.body[0], ast.ImportFrom) and public.body[0].names[0].name == '*'
    assert [alias.name for alias in public.body[1].names] == ['Counter','advance']
    assert ast.dump(public.body[2]) == ast.dump(ast.parse('counter = Counter(10)').body[0])
    expected = [ast.dump(ast.parse(text).body[0]) for text in fixture()['test_list']]
    assert [ast.dump(node) for node in public.body[-1].body] == expected[:1]
    assert [ast.dump(node) for node in private.body[-1].body] == expected


def test_structural_key_joins_renamed_public_symbols_and_docstrings():
    left = 'def alpha(n):\n    """first doc"""\n    return n + 1'
    right = 'def beta(n):\n    """second doc"""\n    return n + 1'
    assert structural_code_key(left) == structural_code_key(right)
    assert structural_code_key(left) == structural_code_key('"""module documentation"""\n'+right)
    assert structural_code_key(left) != structural_code_key(right.replace('n + 1', 'n + 2'))


def test_nontraining_ids_and_nonassert_tests_are_rejected():
    row = fixture()
    row['task_id'] = 511
    with pytest.raises(ValueError, match='not_official_training_id'):
        adapt_case(row)
    row['task_id'] = 601
    row['test_list'][1] = 'print("not a test")'
    with pytest.raises(ValueError, match='unsupported_test_statement'):
        adapt_case(row)


def test_underscore_public_function_survives_reference_and_test_binding():
    row={'task_id':798,'text':'Sum the array.', 'code':'def _sum(arr):\n    return sum(arr)',
         'test_setup_code':'','test_list':['assert _sum([1,2,3]) == 6','assert _sum([4,5]) == 9','assert _sum([]) == 0'],
         'challenge_test_list':[]}
    case=adapt_case(row)
    for source in (case['initial_files']['test_public.py'],case['private_check']):
        tree=ast.parse(source)
        assert any(isinstance(node,ast.ImportFrom) and any(alias.name=='_sum' for alias in node.names) for node in tree.body)
