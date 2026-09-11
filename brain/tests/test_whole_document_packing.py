"""Context and isolation checks for stateless-per-block assistant training."""
import re
import pytest
import torch
from irene_brain.data.streaming_loader import RawDocument, SequencePacker, TaskType


class Tokenizer:
    pad_id, eos_id, resp_id = 0, 2, 5

    def encode(self, text):
        special = {'[THREAD:0]': 8, '[RESP]': 5, '[EOS]': 2}
        return [special[p] if p in special else 10 + ord(p)
                for p in re.findall(r'\[THREAD:0\]|\[RESP\]|\[EOS\]|.', text)]


def document(text):
    return RawDocument(text, TaskType.LANGUAGE)


def packer(length=32):
    return SequencePacker(Tokenizer(), seq_len=length, max_threads=1,
                          supervised_only_loss=True, document_policy='whole')


def test_documents_never_cross_blocks_and_padding_has_no_targets():
    p = packer()
    docs = [document('Question' + str(i) + '[RESP]Answer' + str(i)) for i in range(5)]
    for doc in docs:
        p.append_document(doc)
    p.finish()
    found = []
    while p.can_emit_block():
        block = p.emit_block()
        starts = (block.input_ids == 8).nonzero().flatten().tolist()
        for start in starts:
            end = start + (block.input_ids[start:] == 2).nonzero()[0].item() + 1
            found.append(block.input_ids[start:end].tolist())
            assert block.reset_mask[start] == 1
            assert block.labels[end - 2] == 2  # EOS is supervised.
        assert torch.all(block.labels[block.input_ids == 0] == -100)
        assert sum(s['tokens_in_block'] for s in block.source_records) == 32
    expected = [p.tokenizer.encode('[THREAD:0]' + d.text + '[EOS]') for d in docs]
    assert found == expected


def test_oversized_document_fails_without_changing_pending_context():
    p = packer(16)
    p.append_document(document('Q[RESP]A'))
    pending = list(p._token_buffer)
    with pytest.raises(ValueError, match='exceeding seq_len'):
        p.append_document(document('x' * 40 + '[RESP]answer'))
    assert p._token_buffer == pending
    with pytest.raises(ValueError, match='No complete block'):
        p.emit_block()
    p.finish()
    block = p.emit_block()
    assert block.input_ids[:len(pending)].tolist() == pending
    assert torch.all(block.loss_mask[len(pending):] == 0)


def test_packed_model_logits_match_independent_documents():
    from irene_brain.unified.unified_model import make_unified_model
    torch.manual_seed(483)
    model = make_unified_model(tier='tier1', vocab_size=256,
        compensated_state=True, use_pointer_copy=False).eval()
    docs = [document('Qalpha[RESP]Aone'), document('Qbeta[RESP]Atwo')]
    p = packer(48)
    for d in docs:
        p.append_document(d)
    p.finish()
    block = p.emit_block()
    with torch.no_grad():
        result = model(block.input_ids[None], reset_mask=block.reset_mask[None])['logits'][0]
        offset = 0
        for d in docs:
            ids = torch.tensor(p.tokenizer.encode('[THREAD:0]' + d.text + '[EOS]'))
            independent = model(ids[None])['logits'][0]
            assert (result[offset:offset + len(ids)] - independent).abs().max() < 1e-6
            offset += len(ids)


def test_policy_validation_and_legacy_split_are_explicit():
    with pytest.raises(ValueError, match='document_policy'):
        SequencePacker(Tokenizer(), document_policy='unknown')
    legacy = SequencePacker(Tokenizer(), seq_len=8)
    legacy.append_document(document('x' * 24))
    assert legacy.emit_block().input_ids.numel() == 8
    with pytest.raises(ValueError, match='whole'):
        legacy.finish()
