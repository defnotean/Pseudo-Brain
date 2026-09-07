"""Hugging Face Conversational Dataset Ingestion & Caching Engine.

Fetches multi-turn conversations and instruction-following dialogues directly
from Hugging Face via the Serverless Dataset Viewer API without external heavy dependencies.
Supports:
1. HuggingFaceH4/ultrachat_200k (multi-turn dialogues with user and assistant turns)
2. tatsu-lab/alpaca / vicgalle/alpaca-gpt4 (instruction, input, output)
3. Local disk caching and offline replay
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

logger = logging.getLogger("irene_brain.hf_loader")

_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _REPO_ROOT / "data" / "huggingface"


class HFConversationalLoader:
    """Streams and parses conversational datasets from Hugging Face."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_hf_rows(
        self,
        dataset: str,
        split: str = "train",
        config: str = "default",
        offset: int = 0,
        length: int = 50,
        timeout: int = 10,
    ) -> List[Dict[str, Any]]:
        """Fetch raw rows from Hugging Face Serverless Dataset API."""
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "PseudoBrain-Research/1.0 (Advanced Agentic Recurrent Cognition)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    rows_data = payload.get("rows", [])
                    return [item.get("row", {}) for item in rows_data]
        except Exception as e:
            logger.warning(f"Failed to fetch {dataset} (offset={offset}): {e}")
        return []

    def fetch_ultrachat(
        self,
        num_dialogues: int = 50,
        split: str = "train_sft",
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fetch multi-turn dialogues from HuggingFaceH4/ultrachat_200k."""
        cache_file = self.cache_dir / f"ultrachat_{split}_{num_dialogues}.json"
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if len(data) >= num_dialogues:
                        return data[:num_dialogues]
            except Exception:
                pass

        dialogues: List[Dict[str, Any]] = []
        page_size = min(50, num_dialogues)
        offset = 0

        while len(dialogues) < num_dialogues and offset < num_dialogues * 2:
            batch = self._fetch_hf_rows(
                dataset="HuggingFaceH4/ultrachat_200k",
                split=split,
                config="default",
                offset=offset,
                length=min(page_size, num_dialogues - len(dialogues)),
            )
            if not batch:
                break
            for row in batch:
                messages = row.get("messages", [])
                if messages and len(messages) >= 2:
                    dialogues.append({
                        "dataset": "ultrachat_200k",
                        "dialogue_id": f"ultrachat_{offset}_{len(dialogues)}",
                        "messages": messages,
                    })
            offset += len(batch)
            time.sleep(0.1)

        # Fallback offline generation if internet request timed out or returned few items
        if len(dialogues) < num_dialogues:
            fallback = self._generate_fallback_dialogues(num_dialogues - len(dialogues), prefix="ultrachat_fallback")
            dialogues.extend(fallback)

        if use_cache and dialogues:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(dialogues, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save cache file {cache_file}: {e}")

        return dialogues[:num_dialogues]

    def fetch_alpaca(
        self,
        num_dialogues: int = 50,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fetch instruction-following QA from tatsu-lab/alpaca."""
        cache_file = self.cache_dir / f"alpaca_{num_dialogues}.json"
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if len(data) >= num_dialogues:
                        return data[:num_dialogues]
            except Exception:
                pass

        dialogues: List[Dict[str, Any]] = []
        page_size = min(50, num_dialogues)
        offset = 0

        while len(dialogues) < num_dialogues and offset < num_dialogues * 2:
            batch = self._fetch_hf_rows(
                dataset="tatsu-lab/alpaca",
                split="train",
                config="default",
                offset=offset,
                length=min(page_size, num_dialogues - len(dialogues)),
            )
            if not batch:
                break
            for row in batch:
                instruction = row.get("instruction", "")
                inp = row.get("input", "")
                out = row.get("output", "")
                if instruction and out:
                    prompt = f"{instruction} {inp}".strip() if inp else instruction
                    dialogues.append({
                        "dataset": "alpaca",
                        "dialogue_id": f"alpaca_{offset}_{len(dialogues)}",
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": out},
                        ],
                    })
            offset += len(batch)
            time.sleep(0.1)

        # Fallback offline generation if internet request timed out
        if len(dialogues) < num_dialogues:
            fallback = self._generate_fallback_dialogues(num_dialogues - len(dialogues), prefix="alpaca_fallback")
            dialogues.extend(fallback)

        if use_cache and dialogues:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(dialogues, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save cache file {cache_file}: {e}")

        return dialogues[:num_dialogues]

    def _generate_fallback_dialogues(self, count: int, prefix: str) -> List[Dict[str, Any]]:
        """Generate high-quality conversational turns if network is offline."""
        templates = [
            ("Hello, how are you today?", "I am functioning optimally, ready to think and assist."),
            ("Can you explain how working memory works?", "Working memory temporarily holds and manipulates information for complex cognitive tasks."),
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Remember my favorite color is blue.", "I have noted that your favorite color is blue in my persistent memory."),
            ("What is your favorite topic?", "I enjoy exploring cognitive architectures and recurrent neural dynamics."),
            ("How do you remember facts across turns?", "I latch episodic facts into persistent synaptic weights without needing a token replay buffer."),
            ("Tell me a fact about space.", "A day on Venus is longer than a year on Venus."),
            ("Can we switch tasks for a second?", "Certainly, I can suspend our current topic and switch to another cognitive thread."),
            ("Let's plan a weekend road trip.", "Step 1 is choosing a scenic destination, followed by mapping the route and booking lodging."),
            ("What was my favorite color again?", "Your favorite color is blue, as recorded earlier."),
        ]
        dialogues = []
        for i in range(count):
            user_msg, bot_msg = templates[i % len(templates)]
            dialogues.append({
                "dataset": "synthetic_fallback",
                "dialogue_id": f"{prefix}_{i}",
                "messages": [
                    {"role": "user", "content": f"{user_msg} (ref {i})"},
                    {"role": "assistant", "content": f"{bot_msg}."},
                ],
            })
        return dialogues

    def fetch_combined_corpus(
        self,
        num_ultrachat: int = 30,
        num_alpaca: int = 30,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Combine UltraChat and Alpaca into a diverse unified conversational set."""
        uc = self.fetch_ultrachat(num_dialogues=num_ultrachat, use_cache=use_cache)
        alp = self.fetch_alpaca(num_dialogues=num_alpaca, use_cache=use_cache)
        combined = uc + alp
        return combined
