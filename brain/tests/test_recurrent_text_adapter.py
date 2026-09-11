from types import SimpleNamespace

import pytest

from irene_brain.agent.recurrent_text_adapter import RecurrentTextAdapter


class RecordingAgent:
    def __init__(self):
        self.active_prompt_tokens = ["old context"]
        self.ingested = []
        self.resets = 0
        self.tokenizer = SimpleNamespace(encode=lambda text: list(text))

    def reset(self):
        self.resets += 1
        self.active_prompt_tokens = []

    def _ingest_text_into_slot(self, text, slot_id):
        self.ingested.append(text)

    def generate_text_autoregressive(self, max_new_tokens):
        return "raw model output"


def test_oracle_fields_never_enter_generation_context():
    agent = RecordingAgent()
    adapter = RecurrentTextAdapter(agent)
    result = adapter.answer_record({"question_id": "q1", "turns": ["What is 2 + 3?"],
                                    "ground_truth": "SECRET ORACLE LABEL", "reference_solution": "SECRET CODE"})
    assert result == {"question_id": "q1", "answer": "raw model output"}
    assert "SECRET" not in "".join(agent.ingested)
    assert "old context" not in "".join(agent.active_prompt_tokens)
    adapter.answer("A new question")
    assert agent.resets == 2
    assert "2 + 3" not in "".join(agent.active_prompt_tokens)


def test_multiple_turns_are_not_silently_dropped():
    with pytest.raises(ValueError, match="exactly one"):
        RecurrentTextAdapter(RecordingAgent()).answer_record({"question_id": "q", "turns": ["one", "two"]})
