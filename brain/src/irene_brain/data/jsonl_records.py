"""JSONL records are delimited by LF, not Unicode separators inside strings."""
import json


def load_jsonl_bytes(raw):
    lines = raw.split(b'\n')
    if lines and lines[-1] == b'':
        lines.pop()
    if any(not line.strip() for line in lines):
        raise ValueError('Blank JSONL record')
    return [json.loads(line) for line in lines]
