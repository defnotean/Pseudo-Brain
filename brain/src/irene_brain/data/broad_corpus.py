"""Frozen diagnostic corpus selection with explicit related-record isolation."""
import hashlib
import unicodedata


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def isolation_keys(row, task, text):
    keys = ['text:' + digest(normalized(text))]
    if task == 'code':
        repo = row.get('repository_name')
        code = row.get('func_code_string')
        if not isinstance(repo, str) or not repo.strip() or not isinstance(code, str):
            raise ValueError('Code needs repository and function source')
        keys += ['repository:' + normalized(repo), 'code:' + digest(normalized(code))]
    else:
        question = row.get('problem') or row.get('prompt')
        if not isinstance(question, str) or not question.strip():
            question = next((m.get('content') for m in row.get('messages', [])
                             if isinstance(m, dict) and m.get('role') == 'user'
                             and isinstance(m.get('content'), str)), None)
        if not question:
            raise ValueError('Missing initial question')
        keys.append('question:' + digest(normalized(question)))
    return keys


def assign_components(records):
    """Join all overlapping keys before assigning a component to either split."""
    parent = {}
    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for record in records:
        keys = record['isolation_keys']
        if not keys:
            raise ValueError('Empty isolation keys')
        for key in keys[1:]:
            left, right = root(keys[0]), root(key)
            parent[max(left, right)] = min(left, right)
    for record in records:
        component = root(record['isolation_keys'][0])
        record['component_id'] = digest(component)
        record['partition'] = ('development' if int(digest('pb-broad-v1|' + component), 16) % 5 == 0 else 'train')
    return records


def assert_isolated(train, development):
    for field in ('component_id', 'text_sha256'):
        if {r[field] for r in train} & {r[field] for r in development}:
            raise ValueError('Cross-split overlap: ' + field)
    left = {k for r in train for k in r['isolation_keys']}
    right = {k for r in development for k in r['isolation_keys']}
    if left & right:
        raise ValueError('Cross-split isolation-key overlap')
