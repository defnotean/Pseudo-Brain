"""Structured, complete exchanges for the separately registered v4 corpus."""
import ast
import re

from irene_brain.data.broad_corpus import assign_components, digest, isolation_keys, normalized


def complete_exchange(row, task):
    if task == 'language':
        messages = row.get('messages')
        if not isinstance(messages, list) or len(messages) < 2 or not all(isinstance(item,dict) for item in messages[:2]):
            raise ValueError('missing_initial_exchange')
        if messages[0].get('role') != 'user' or messages[1].get('role') != 'assistant':
            raise ValueError('unsupported_initial_roles')
        question, answer = messages[0].get('content'), messages[1].get('content')
        key_row = row
    elif task == 'code':
        documentation, answer = row.get('func_documentation_string'), row.get('func_code_string')
        if not isinstance(documentation, str) or not documentation.strip():
            raise ValueError('missing_documentation')
        question = 'Implement this function: ' + documentation.strip()
        key_row = row
    elif task == 'reasoning':
        question, answer = row.get('question'), row.get('answer')
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('missing_answer')
        final = answer.strip().splitlines()[-1]
        answer = re.sub(r'<<[^<>]*>>', '', answer)
        if answer.strip().splitlines()[-1] != final or not re.fullmatch(r'#### -?[0-9,]+(?:\.[0-9]+)?', final):
            raise ValueError('invalid_or_changed_final_answer')
        key_row = {'problem': question}
    else:
        raise ValueError('unknown_domain')
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError('missing_question_or_answer')
    question, answer = question.strip(), answer.strip()
    if any(marker in question or marker in answer for marker in ('[RESP]', '[EOS]', '[THREAD:', '[PAD]')):
        raise ValueError('ambiguous_special_token_boundary')
    text = 'User: ' + question + ' [RESP]' + answer + '[EOS]'
    keys = isolation_keys(key_row, task, text)
    keys.append('question:' + digest(normalized(question)))
    if task == 'code':
        keys.append('documentation:' + digest(normalized(row['func_documentation_string'])))
    return {'text':text, 'text_sha256':digest(text), 'isolation_keys':sorted(set(keys)),
            'question':question, 'answer':answer, 'task':task}


def registered_benchmark_keys(requests):
    """Read question-side fields only; no grader or canonical solution inputs."""
    keys = set()
    for row in requests:
        if row['benchmark'] == 'GSM8K':
            instruction, separator, question = row['question'].partition('\n\n')
            if not separator or not instruction.startswith('Solve the problem.'):
                raise ValueError('unsupported_registered_math_prompt')
            keys.add('question:' + digest(normalized(question)))
        elif row['benchmark'] == 'HumanEval':
            tree = ast.parse(row['original_prompt'])
            functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            target = next(node for node in functions if node.name == row['entry_point'])
            documentation = ast.get_docstring(target)
            if not documentation:
                raise ValueError('missing_registered_code_documentation')
            keys.add('documentation:' + digest(normalized(documentation)))
        else:
            raise ValueError('unknown_registered_benchmark')
    return keys


def quarantine_related(candidates, prior_development, benchmark_keys):
    """Exclude whole connected components, including bridges through old dev."""
    seeds = [dict(row, quarantine_seed=True) for row in prior_development]
    joined = assign_components(candidates + seeds)
    blocked = {row['component_id'] for row in joined
               if row.get('quarantine_seed') or benchmark_keys.intersection(row['isolation_keys'])}
    retained, excluded = [], []
    for row in candidates:
        (excluded if row['component_id'] in blocked else retained).append(row)
    return retained, excluded
