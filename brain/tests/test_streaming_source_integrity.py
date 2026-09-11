from types import SimpleNamespace
import pytest
import irene_brain.data.streaming_loader as streams


def reader(monkeypatch, rows, **kwargs):
    calls = []
    def load(path, **options):
        calls.append((path, options))
        return list(rows)
    monkeypatch.setattr(streams, '_HF_DATASETS_AVAILABLE', True)
    monkeypatch.setattr(streams, 'datasets', SimpleNamespace(load_dataset=load))
    fallback = SimpleNamespace(sample=lambda: streams.RawDocument('synthetic', streams.TaskType.LANGUAGE))
    result = streams.HFStreamIterator(streams.TaskType.LANGUAGE, 'fixture/data',
        fallback_stream=fallback, **kwargs)
    return result, calls


def test_instruction_is_preserved_before_output(monkeypatch):
    stream, _ = reader(monkeypatch, [{'instruction': 'Add the numbers', 'input': '2 and 3', 'output': '5'}])
    doc = stream.next_document()
    assert doc.text == 'User: Add the numbers 2 and 3 [RESP]5[EOS]'
    assert doc.is_dialogue


def test_messages_precede_generic_output_and_keep_roles(monkeypatch):
    stream, _ = reader(monkeypatch, [{'output': 'wrong field', 'messages': [
        {'role': 'system', 'content': 'Be precise'}, {'role': 'user', 'content': 'Question'},
        {'role': 'assistant', 'content': 'Answer'}]}])
    text = stream.next_document().text
    assert text == 'System: Be precise User: Question [RESP]Answer[EOS]'


def test_restart_does_not_insert_synthetic_row_and_revision_is_forwarded(monkeypatch):
    stream, calls = reader(monkeypatch, [{'text': 'real source'}], revision='frozen-revision', allow_synthetic_fallback=False)
    docs = [stream.next_document() for _ in range(3)]
    assert all(doc.text == 'real source' and doc.metadata['origin'] == 'huggingface' for doc in docs)
    assert all(options['revision'] == 'frozen-revision' for _, options in calls)


def test_unknown_schema_is_skipped_not_serialized(monkeypatch):
    stream, _ = reader(monkeypatch, [{'private_grader_label': 'never input'}, {'text': 'usable'}])
    assert stream.next_document().text == 'usable'


def test_code_subset_and_documentation_are_preserved(monkeypatch):
    stream, calls = reader(monkeypatch, [{'func_documentation_string': 'Return one.',
        'func_code_string': 'def one():\n    return 1', 'repository_name': 'metadata-only'}],
        revision='frozen-code', dataset_config='python', allow_synthetic_fallback=False)
    document = stream.next_document()
    assert calls[0][1]['name'] == 'python'
    assert document.metadata['config'] == 'python'
    assert document.metadata['revision'] == 'frozen-code'
    assert document.text == 'User: Implement this function: Return one. [RESP]def one():\n    return 1[EOS]'


def test_required_source_failure_does_not_fall_back(monkeypatch):
    stream, _ = reader(monkeypatch, [], allow_synthetic_fallback=False)
    with pytest.raises(RuntimeError, match='no usable rows'):
        stream.next_document()
    def fail(*args, **kwargs):
        raise OSError('fixture outage')
    monkeypatch.setattr(streams.datasets, 'load_dataset', fail)
    stream._iterator = None
    with pytest.raises(RuntimeError, match='source unavailable'):
        stream.next_document()


def test_fallback_has_explicit_provenance(monkeypatch):
    stream, _ = reader(monkeypatch, [])
    document = stream.next_document()
    assert document.metadata['origin'] == 'synthetic'
    assert document.metadata['requested_dataset'] == 'fixture/data'
    assert document.metadata['fallback_reason'] == 'no_usable_rows_within_limit'


def test_source_provenance_survives_packing_collation_and_device_transfer(monkeypatch):
    from irene_brain.semantic.tokenizer import SemanticTokenizer
    stream, _ = reader(monkeypatch, [{'instruction': 'Count', 'output': 'one two three four five six seven eight'}])
    packer = streams.SequencePacker(SemanticTokenizer(), seq_len=8)
    packer.append_document(stream.next_document())
    block = packer.emit_block()
    assert sum(record['tokens_in_block'] for record in block.source_records) == 8
    batch = streams.collate_streaming_batch([block]).to('cpu')
    assert batch['source_records'][0][0]['origin'] == 'huggingface'
    assert len(batch.source_records[0][0]['document_sha256']) == 64
    packer.reset()
    assert not packer._source_buffer
