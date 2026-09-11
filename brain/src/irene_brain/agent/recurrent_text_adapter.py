"""Raw-text evaluation interface for the recurrent model, without action heuristics."""
from __future__ import annotations

from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent


class RecurrentTextAdapter:
    """One isolated question per call; no labels, oracle solutions or answer rewriting.

    Providing this adapter does not imply a coding checkpoint is instruction-tuned
    for general questions. It allows that capability to be measured honestly.
    """

    def __init__(self, agent: RecurrentSoftwareAgent):
        self.agent = agent

    def answer(self, question: str, max_new_tokens: int = 256) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty question string is required")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.agent.reset()
        prompt = f"[QUERY]{question}\n"
        self.agent.active_prompt_tokens = self.agent.tokenizer.encode(prompt)
        self.agent._ingest_text_into_slot(prompt, slot_id=0)
        return self.agent.generate_text_autoregressive(max_new_tokens=max_new_tokens)

    def answer_record(self, record: dict, max_new_tokens: int = 256) -> dict:
        """Consume only question text and ID, ignoring any evaluator-only fields."""
        question = record.get("question")
        if question is None:
            turns = record.get("turns")
            if not isinstance(turns, list) or len(turns) != 1:
                raise ValueError("This isolated adapter requires exactly one question turn")
            question = turns[0]
        if "question_id" not in record:
            raise ValueError("question_id is required for result alignment")
        return {"question_id": record["question_id"],
                "answer": self.answer(question, max_new_tokens=max_new_tokens)}
