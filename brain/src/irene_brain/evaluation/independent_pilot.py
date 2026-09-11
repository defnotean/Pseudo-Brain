"""Frozen-request decoding and strict numerical grading for the small pilot."""
import json
import re
from decimal import Decimal, InvalidOperation


def load_requests(raw):
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    seen = set()
    for row in rows:
        allowed = {'benchmark', 'task_id', 'question', 'original_prompt', 'entry_point'}
        if set(row) - allowed:
            raise ValueError('Request contains unexpected or grader fields')
        if row.get('benchmark') not in ('HumanEval', 'GSM8K'):
            raise ValueError('Unknown benchmark')
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError('Missing question')
        if not isinstance(row.get('task_id'), str):
            raise ValueError('Missing task id')
        key = (row['benchmark'], row['task_id'])
        if key in seen:
            raise ValueError('Duplicate task')
        seen.add(key)
    if not rows:
        raise ValueError('Empty request bank')
    return rows


def final_number(text):
    """Require a standalone final #### line; never extract incidental numbers."""
    lines = text.strip().splitlines()
    if not lines:
        return None
    match = re.fullmatch(r'\s*####\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*', lines[-1])
    if match is None:
        return None
    try:
        return Decimal(match.group(1).replace(',', ''))
    except InvalidOperation:
        return None


def grade_math(response, reference):
    expected = final_number(reference)
    if expected is None:
        raise ValueError('Invalid numerical reference')
    actual = final_number(response)
    return {'correct': actual is not None and actual == expected,
            'format_valid': actual is not None,
            'predicted': None if actual is None else str(actual),
            'expected': str(expected)}


def score_math_bank(requests, graders, responses):
    """Keep all registered math tasks in the denominator, including missing output."""
    selected = {(row['benchmark'], row['task_id']) for row in requests}
    if len(selected) != len(requests):
        raise ValueError('Duplicate registered task')
    references, outputs = {}, {}
    for rows, destination in ((graders, references), (responses, outputs)):
        for row in rows:
            key = (row['benchmark'], row['task_id'])
            if key not in selected or key in destination:
                raise ValueError('Unexpected or duplicate result task')
            destination[key] = row
    if set(references) != selected:
        raise ValueError('Grader bank does not match registered tasks')
    details = []
    for request in requests:
        key = (request['benchmark'], request['task_id'])
        if key[0] != 'GSM8K':
            continue
        output = outputs.get(key)
        if output is not None and not isinstance(output.get('response'), str):
            raise ValueError('Malformed response')
        grade = grade_math('' if output is None else output['response'], references[key]['answer'])
        details.append({'task_id': key[1], 'missing': output is None, **grade})
    if not details:
        raise ValueError('No registered math tasks')
    return {'status': 'complete' if selected == set(outputs) else 'incomplete',
            'GSM8K': {'correct': sum(row['correct'] for row in details),
                      'total': len(details), 'missing': sum(row['missing'] for row in details),
                      'format_valid': sum(row['format_valid'] for row in details),
                      'details': details},
            'HumanEval': {'status': 'unscored', 'reason': 'Requires verified execution sandbox'}}


def greedy_decode(model, context, prompt, *, recurrent, eos_id, limit=512, parallel_prefill=False):
    """One task per call; fresh recurrent state and immutable prompt throughout."""
    import torch
    if context.ndim != 2 or context.shape[0] != 1 or context.shape[1] < 1:
        raise ValueError('Expected one nonempty task context')
    if limit < 1:
        raise ValueError('Generation limit must be positive')
    if parallel_prefill and not recurrent:
        raise ValueError('Parallel prefill is supported only for the recurrent model')
    initial_prompt = prompt.clone()
    generated = []
    state = None
    with torch.no_grad():
        if recurrent:
            if parallel_prefill:
                from .recurrent_prefill import parallel_recurrent_prefill
                logits, state = parallel_recurrent_prefill(model, context, initial_prompt)
            else:
                state = model.init_state(1)
                for token in context[0]:
                    logits, state = model.step(token.reshape(1), state, initial_prompt)
        else:
            logits = model(context, prompt_tokens=initial_prompt)[:, -1]
        reason = 'token_limit'
        for index in range(limit):
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Nonfinite generation logits')
            token = logits.argmax(-1)
            if token.item() == eos_id:
                reason = 'eos'
                break
            generated.append(token.item())
            if index + 1 == limit:
                break
            if recurrent:
                logits, state = model.step(token, state, initial_prompt)
            else:
                context = torch.cat((context, token[:, None]), dim=1)
                logits = model(context, prompt_tokens=initial_prompt)[:, -1]
    state_bytes = None if state is None else state.numel() * state.element_size()
    return {'token_ids': generated, 'stop_reason': reason,
            'recurrent_state_bytes': state_bytes}


def validate_streaming_parity(model, lengths=(31, 257, 2049)):
    """Gate trained weights on both native and fallback paths before generation."""
    import torch
    device = model.embedding.weight.device
    generator = torch.Generator(device=device).manual_seed(734)
    records = []
    with torch.no_grad():
        for length in lengths:
            ids = torch.randint(1, model.config.vocab_size, (1, length), device=device, generator=generator)
            prompt = ids[:, :7].clone()
            resets = torch.zeros_like(ids)
            resets[:, length // 2] = 1
            expected = model(ids, resets, prompt)
            if not bool(torch.isfinite(expected).all()):
                raise FloatingPointError('Nonfinite parallel parity logits')
            state = model.init_state(1)
            if state.shape != (1, 16, 64) or state.dtype != torch.float32 or state.numel() * state.element_size() != 4096:
                raise RuntimeError('Streaming state budget changed')
            worst = 0.
            for index in range(length):
                state = state * (1 - resets[:, index, None, None])
                actual, state = model.step(ids[:, index], state, prompt)
                if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(state).all()):
                    raise FloatingPointError('Nonfinite streaming parity output')
                worst = max(worst, (expected[:, index] - actual).abs().max().item())
            records.append({'tokens': length, 'max_abs_logits': worst, 'state_bytes': 4096})
            if not worst < 1e-6:
                raise RuntimeError('Trained streaming/parallel parity exceeded 1e-6')
    return records
