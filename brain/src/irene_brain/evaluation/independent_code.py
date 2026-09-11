"""Strict full-function HumanEval scoring using the verified remote executor."""
import ast
import hashlib
from pathlib import Path
from irene_brain.evaluation import code_sandbox


def extract_program(response, entry_point):
    code = response.strip()
    if code.startswith('```'):
        lines = code.splitlines()
        if lines[0].strip() not in ('```', '```python') or lines[-1].strip() != '```':
            raise ValueError('Expected one complete Python code block')
        code = '\n'.join(lines[1:-1])
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError('Invalid Python syntax') from exc
    if not any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body):
        raise ValueError('Missing complete entry-point function')
    return code


def score_code_bank(requests, graders, responses):
    """Run controls first; missing and malformed answers remain in denominator."""
    selected = {(row['benchmark'], row['task_id']) for row in requests}
    if len(selected) != len(requests):
        raise ValueError('Duplicate registered task')
    references, outputs = {}, {}
    for rows, destination in ((graders, references), (responses, outputs)):
        for row in rows:
            key = (row['benchmark'], row['task_id'])
            if key not in selected or key in destination:
                raise ValueError('Unexpected or duplicate task')
            destination[key] = row
    if set(references) != selected:
        raise ValueError('Grader bank mismatch')
    tasks = [row for row in requests if row['benchmark'] == 'HumanEval']
    if not tasks:
        raise ValueError('No coding tasks')
    controls = []
    for task in tasks:
        reference = references[('HumanEval', task['task_id'])]
        name = reference['entry_point']
        canonical = code_sandbox.check_program(reference['prompt'] + reference['canonical_solution'], reference['test'], name)
        wrong = code_sandbox.check_program('def ' + name + '(*args, **kwargs): return None', reference['test'], name)
        controls.append({'task_id': task['task_id'], 'canonical_passed': canonical['passed'], 'wrong_passed': wrong['passed']})
        if not canonical['passed'] or wrong['passed']:
            raise RuntimeError('Coding grader control failed: ' + task['task_id'])
    details = []
    for task in tasks:
        key = ('HumanEval', task['task_id'])
        reference, output = references[key], outputs.get(key)
        detail = {'task_id': task['task_id'], 'missing': output is None, 'correct': False}
        if output is None:
            detail['reason'] = 'missing_response'
        else:
            if not isinstance(output.get('response'), str):
                raise ValueError('Malformed response record')
            try:
                code = extract_program(output['response'], reference['entry_point'])
            except ValueError as exc:
                detail['reason'] = str(exc)
            else:
                result = code_sandbox.check_program(code, reference['test'], reference['entry_point'])
                detail.update(correct=result['passed'], execution=result)
        details.append(detail)
    return {'status': 'complete' if all(not row['missing'] for row in details) else 'incomplete',
            'correct': sum(row['correct'] for row in details), 'total': len(details),
            'missing': sum(row['missing'] for row in details), 'details': details, 'controls': controls,
            'sandbox_source_sha256': hashlib.sha256(Path(code_sandbox.__file__).read_bytes()).hexdigest(),
            'grading_limit': 'Same-interpreter test harness is not adversarially tamper-proof'}
