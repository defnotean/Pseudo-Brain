"""Real remote sandbox checks for underscore, class, and setup API cases."""
import json
from pathlib import Path
from build_mbpp_training_bank_v1 import load_inputs
from irene_brain.data.mbpp_training import adapt_case
from irene_brain.evaluation.workspace_sandbox import run_workspace_tests

rows = {row['task_id']:row for row in load_inputs()['mbpp_train']}
results = []
for task_id in (601,798,927):
    case = adapt_case(rows[task_id])
    for faulty in (False,True):
        files = {**case['initial_files'], case['target_module']:
                 'raise RuntimeError("Injected training fault")' if faulty else case['reference_solution']}
        for private in (False,True):
            target = '__private_check.py' if private else 'test_public.py'
            snapshot = {**files,'__private_check.py':case['private_check']} if private else files
            result = run_workspace_tests(snapshot,target)
            results.append({'task_id':task_id,'faulty':faulty,'private':private,'result':result})
            Path('api-controls.json').write_text(json.dumps(results,indent=2))
            assert result['passed'] is (not faulty), (task_id,faulty,private)
            if faulty:
                assert 'Injected training fault' in result['output']
print('MBPP_API_CONTROLS',len(results),'passed')
