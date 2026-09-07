"""Lightweight byte-level semantic tokenizer with cognitive control tokens.

Provides complete zero-OOV coverage for UTF-8 text while embedding dedicated
cognitive control tokens for multi-threaded state management:
- [PAD], [BOS], [EOS], [SEP], [QUERY], [RESP], [INTERRUPT], [RESUME]
- [THREAD:0] .. [THREAD:K-1] for thread addressing
- [DEP] for cross-thread dependency markers
"""

from __future__ import annotations

from typing import List, Optional, Union


class SemanticTokenizer:
    """Byte-level tokenizer with dedicated cognitive markers."""

    SPECIAL_TOKENS = [
        "[PAD]",
        "[BOS]",
        "[EOS]",
        "[SEP]",
        "[QUERY]",
        "[RESP]",
        "[INTERRUPT]",
        "[RESUME]",
        "[DEP]",
    ]

    def __init__(self, max_threads: int = 64):
        self.max_threads = max_threads

        # Map special tokens: 0 .. N_special - 1
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(self.SPECIAL_TOKENS)}

        # Thread tokens: [THREAD:0] .. [THREAD:max_threads-1]
        self.thread_tokens = [f"[THREAD:{i}]" for i in range(max_threads)]
        base_thread_idx = len(self.token_to_id)
        for i, tok in enumerate(self.thread_tokens):
            idx = base_thread_idx + i
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

        # Byte tokens: 0 .. 255 mapped after special and thread tokens
        self.byte_offset = len(self.token_to_id)
        self.vocab_size = self.byte_offset + 256

        # Quick special token IDs
        self.pad_id = self.token_to_id["[PAD]"]
        self.bos_id = self.token_to_id["[BOS]"]
        self.eos_id = self.token_to_id["[EOS]"]
        self.sep_id = self.token_to_id["[SEP]"]
        self.query_id = self.token_to_id["[QUERY]"]
        self.resp_id = self.token_to_id["[RESP]"]
        self.interrupt_id = self.token_to_id["[INTERRUPT]"]
        self.resume_id = self.token_to_id["[RESUME]"]
        self.dep_id = self.token_to_id["[DEP]"]

    def thread_id_to_token_id(self, thread_idx: int) -> int:
        """Get the token ID corresponding to a given cognitive thread."""
        tok = f"[THREAD:{thread_idx % self.max_threads}]"
        return self.token_to_id[tok]

    def token_id_to_thread_id(self, token_id: int) -> Optional[int]:
        """If token_id is a thread marker, return the integer thread_id, else None."""
        tok = self.id_to_token.get(token_id)
        if tok and tok.startswith("[THREAD:") and tok.endswith("]"):
            try:
                return int(tok[8:-1])
            except ValueError:
                return None
        return None

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        thread_id: Optional[int] = None,
    ) -> List[int]:
        """Encode a string (with optional thread marker and BOS/EOS) to token IDs."""
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_id)
        if thread_id is not None:
            tokens.append(self.thread_id_to_token_id(thread_id))

        # Check for control tokens in text or parse bytes
        # Support inline special tokens like [QUERY], [RESP], [THREAD:i]
        i = 0
        n = len(text)
        while i < n:
            if text[i] == "[":
                # Check for special token match
                matched = False
                for spec_tok in self.token_to_id:
                    if text.startswith(spec_tok, i):
                        tokens.append(self.token_to_id[spec_tok])
                        i += len(spec_tok)
                        matched = True
                        break
                if matched:
                    continue

            # Fallback to UTF-8 byte encoding
            char_bytes = text[i].encode("utf-8")
            for b in char_bytes:
                tokens.append(self.byte_offset + b)
            i += 1

        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = False) -> str:
        """Decode token IDs back into string."""
        byte_list: List[int] = []
        result_parts: List[str] = []

        def flush_bytes():
            if byte_list:
                result_parts.append(bytes(byte_list).decode("utf-8", errors="replace"))
                byte_list.clear()

        for tid in token_ids:
            if tid >= self.byte_offset and tid < self.vocab_size:
                byte_list.append(tid - self.byte_offset)
            else:
                flush_bytes()
                if not skip_special:
                    tok = self.id_to_token.get(tid, f"[UNK:{tid}]")
                    result_parts.append(tok)

        flush_bytes()
        return "".join(result_parts)

    def __len__(self) -> int:
        return self.vocab_size
