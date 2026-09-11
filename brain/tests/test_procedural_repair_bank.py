import ast
from types import SimpleNamespace
import pytest
from irene_brain.data.procedural_repair_bank import solution_family,split_checks,repair_case,partition_tasks


def task(index,value=2):
    name='f_'+str(index)
    return SimpleNamespace(reference_solution=f'def {name}(x): return x*{value}',target_function=name,
        target_module=f'm_{index}.py',task_id=str(index),domain='control',initial_files={},goal='Implement '+name,
        hidden_tests_code=f'from m_{index} import {name}\nassert {name}(2)=={2*value}\ny={name}(-7)\nassert y=={-7*value}\nprint("passed")')


def test_renamed_solutions_stay_in_one_family():
    assert solution_family(task(1))==solution_family(task(901))
    assert solution_family(task(1))!=solution_family(task(2,3))


def test_private_suffix_is_absent_from_public_test():
    public,private=split_checks(task(1).hidden_tests_code)
    assert '-7' not in public and '-7' in private
    assert sum(isinstance(node,ast.Assert) for node in ast.walk(ast.parse(public)))==1
    assert sum(isinstance(node,ast.Assert) for node in ast.walk(ast.parse(private)))==2
    with pytest.raises(ValueError): split_checks('assert True')


def test_family_partition_is_deterministic_and_disjoint():
    tasks=[task(family*100+i,family+1) for family in range(15) for i in range(8)]
    selected,mapping=partition_tasks(tasks,train_per_family=4,development_per_family=2)
    other,other_mapping=partition_tasks(list(reversed(tasks)),train_per_family=4,development_per_family=2)
    assert mapping==other_mapping and selected==other
    assert len(selected['train'])==48 and len(selected['development'])==6
    assert not ({row['family_sha256'] for row in selected['train']} & {row['family_sha256'] for row in selected['development']})


def test_fault_is_explicit_and_private_checks_not_in_initial_context():
    case=repair_case(task(1),repair=True)
    assert [action.kind for action in case['actions']].count('injected_fault')==1
    assert '-7' not in case['initial_prompt'] and '-7' not in case['initial_files']['test_public.py']
    assert case['actions'][-1].text.startswith('ACTION: FINISH')
