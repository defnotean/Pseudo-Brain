import json
import pytest

from irene_brain.data.jsonl_records import load_jsonl_bytes


@pytest.mark.parametrize('separator',['\u0085','\u2028','\u2029'])
def test_unicode_text_separator_does_not_split_json_record(separator):
    records = [{'text':'first'+separator+'second'},{'text':'another\nline'}]
    raw = ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records).encode()
    assert load_jsonl_bytes(raw) == records


@pytest.mark.parametrize('ending',[b'',b'\n',b'\r\n'])
def test_final_newline_is_optional_and_crlf_is_supported(ending):
    assert load_jsonl_bytes(b'{"x":1}\r\n{"x":2}'+ending) == [{'x':1},{'x':2}]


def test_blank_interior_record_is_rejected():
    with pytest.raises(ValueError,match='Blank'):
        load_jsonl_bytes(b'{"x":1}\n\n{"x":2}\n')


def test_malformed_record_is_not_silently_dropped():
    with pytest.raises(json.JSONDecodeError):
        load_jsonl_bytes(b'{"x":1}\nmalformed\n')
