import hashlib
import json

import pytest

from evaluate_general_reasoning import evaluate_records, load_questions, numeric_answer


def test_reasoning_scorer_matches_numeric_format_without_guessing():
    assert numeric_answer("work 17, final #### -1,234.5") == "-1234.5"
    assert numeric_answer("the number is 42") is None
    assert numeric_answer("#### unknown") is None


def test_evaluation_never_forwards_labels_and_keeps_errors():
    class Spy:
        def answer_record(self, record, max_new_tokens):
            assert set(record) == {"question_id", "question"}
            assert "private solution" not in record["question"]
            assert max_new_tokens == 32
            if record["question_id"].endswith("1"):
                raise RuntimeError("deliberate generation failure")
            return {"answer": "#### 2"}
    selected = [(i, {"question": "One plus one?", "answer": "private solution #### 2"}) for i in range(2)]
    rows = []
    report = evaluate_records(Spy(), selected, 32, rows.append)
    assert report["total"] == 2 and report["correct"] == 1
    assert rows[1]["error"] is not None and not rows[1]["correct"]


def test_question_selection_is_reproducible_and_digest_bound(tmp_path):
    path = tmp_path / "test.jsonl"
    content = "\n".join(json.dumps({"question": str(i), "answer": "#### 2"}) for i in range(20)).encode()
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    assert load_questions(path, digest, 4, 42) == load_questions(path, digest, 4, 42)
    with pytest.raises(ValueError, match="SHA-256"):
        load_questions(path, "0" * 64, 4, 42)
