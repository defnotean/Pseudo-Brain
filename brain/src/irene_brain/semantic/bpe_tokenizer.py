"""Standard 32,000 and 64,000 Token BPE Tokenizer for Pseudo-Brain.

Integrates HuggingFace Tokenizers (Byte-level BPE) and SentencePiece backends
while retaining all Pseudo-Brain cognitive control tokens:
- Control tokens: [PAD], [BOS], [EOS], [SEP], [QUERY], [RESP], [INTERRUPT], [RESUME], [DEP], [MASK], [UNK]
- Multi-threaded addressing: [THREAD:0] .. [THREAD:K-1]
- Zero Out-Of-Vocabulary (OOV) byte-fallback coverage for arbitrary UTF-8 text, code, and scientific syntax.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Union

logger = logging.getLogger("bpe_tokenizer")

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

try:
    import sentencepiece as spm
    _SPM_AVAILABLE = True
except ImportError:
    _SPM_AVAILABLE = False


# Standard cognitive special tokens in canonical order
DEFAULT_SPECIAL_TOKENS = [
    "[PAD]",        # 0: Padding
    "[BOS]",        # 1: Beginning of sequence
    "[EOS]",        # 2: End of sequence
    "[SEP]",        # 3: Structural separator
    "[QUERY]",      # 4: Query initiator
    "[RESP]",       # 5: Response initiator
    "[INTERRUPT]",  # 6: Cognitive preemption
    "[RESUME]",     # 7: Thread resumption
    "[DEP]",        # 8: Cross-thread dependency
]


class BpeSemanticTokenizer:
    """Standard 32K/64K BPE Semantic Tokenizer for 1B-Scale Pseudo-Brain.

    Features:
    - Target vocabulary sizes: 32,000 (standard) and 64,000 (extended multi-domain).
    - Preserves backward-compatible token indexing: [THREAD:0] starts at ID 9.
    - Zero Out-Of-Vocabulary (OOV) via ByteLevel BPE.
    - Fast Rust-accelerated encoding & decoding.
    - Pre-trained/cached default vocabulary initialization with instant on-disk caching.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        max_threads: int = 64,
        tokenizer_file: Optional[Union[str, Path]] = None,
        backend: str = "huggingface",
        auto_build_if_missing: bool = True,
    ):
        self.vocab_size = vocab_size
        self.max_threads = max_threads
        self.backend = backend.lower()

        # Build list of cognitive tokens
        self.base_special_tokens = list(DEFAULT_SPECIAL_TOKENS)
        self.thread_tokens = [f"[THREAD:{i}]" for i in range(max_threads)]
        self.post_special_tokens = ["[MASK]", "[UNK]"]

        # Canonical special token sequence
        self.all_special_tokens = (
            self.base_special_tokens + self.thread_tokens + self.post_special_tokens
        )

        # Quick token lookup dictionary
        self._special_token_to_id: Dict[str, int] = {
            tok: idx for idx, tok in enumerate(self.all_special_tokens)
        }
        self._id_to_special_token: Dict[int, str] = {
            idx: tok for idx, tok in enumerate(self.all_special_tokens)
        }

        # Cognitive control token IDs
        self.pad_id: int = self._special_token_to_id["[PAD]"]
        self.bos_id: int = self._special_token_to_id["[BOS]"]
        self.eos_id: int = self._special_token_to_id["[EOS]"]
        self.sep_id: int = self._special_token_to_id["[SEP]"]
        self.query_id: int = self._special_token_to_id["[QUERY]"]
        self.resp_id: int = self._special_token_to_id["[RESP]"]
        self.interrupt_id: int = self._special_token_to_id["[INTERRUPT]"]
        self.resume_id: int = self._special_token_to_id["[RESUME]"]
        self.dep_id: int = self._special_token_to_id["[DEP]"]
        self.thread_token_start: int = len(self.base_special_tokens)  # Typically 9
        self.mask_id: int = self._special_token_to_id["[MASK]"]
        self.unk_id: int = self._special_token_to_id["[UNK]"]

        self._hf_tokenizer: Optional[HFTokenizer] = None
        self._spm_processor: Optional[spm.SentencePieceProcessor] = None

        if tokenizer_file is not None and Path(tokenizer_file).exists():
            self.load(tokenizer_file)
        elif auto_build_if_missing:
            # Default cache location
            cache_dir = Path(__file__).resolve().parents[3] / "data" / "tokenizers"
            cached_path = cache_dir / f"pseudo_brain_bpe_{vocab_size}.json"
            if cached_path.exists():
                self.load(cached_path)
            else:
                self._initialize_default_vocab(cached_path)

    def _initialize_default_vocab(self, cache_save_path: Optional[Path] = None) -> None:
        """Initialize and calibrate default 32K or 64K vocabulary from project corpus."""
        if self.backend in ("huggingface", "hf"):
            if not _HF_AVAILABLE:
                raise ImportError("HuggingFace 'tokenizers' library is not installed.")
            self._init_hf_default(cache_save_path)
        elif self.backend in ("sentencepiece", "spm"):
            if not _SPM_AVAILABLE:
                raise ImportError("sentencepiece library is not installed.")
            self._init_spm_default(cache_save_path)
        else:
            raise ValueError(f"Unknown tokenizer backend: {self.backend}")

    def _get_training_texts(self) -> List[str]:
        """Collect training texts from local corpus and curated conversational vocabulary."""
        texts: List[str] = []
        corpus_candidates = [
            Path(__file__).resolve().parents[3] / "data" / "transformed_hf_conversational_corpus.jsonl",
            Path("/content/data/transformed_hf_conversational_corpus.jsonl"),
            Path("data/transformed_hf_conversational_corpus.jsonl"),
            Path(__file__).resolve().parents[1] / "data" / "transformed_hf_conversational_corpus.jsonl",
        ]
        corpus_path = next((p for p in corpus_candidates if p.exists()), None)
        if corpus_path is not None:
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if "text" in data:
                            texts.append(data["text"])
            except Exception as e:
                logger.warning(f"Could not load corpus from {corpus_path}: {e}")

        # Add curated vocabulary items from vocab.py
        try:
            from irene_brain.semantic.vocab import CONVERSATIONAL_SUBWORDS
            texts.extend(CONVERSATIONAL_SUBWORDS)
        except Exception:
            pass

        # Add baseline conversational templates
        templates = [
            "[THREAD:0]Hello! What is your name? [RESP]I am Pseudo-Brain, an autonomous recurrent cognitive agent.[EOS]",
            "[THREAD:1]Urgent preemption alert. [RESP]Mitigation applied.[EOS]",
            "[THREAD:0]Explain how photosynthesis works. [RESP]Photosynthesis converts sunlight, water, and carbon into glucose.[EOS]",
            "[THREAD:2]Implement a circular queue in Java. [RESP]public class CircularQueue<T> { ... }[EOS]",
            "[THREAD:0]What is the capital of France? [RESP]Paris.[EOS]",
            "[THREAD:3]Multi-threaded state synchronization. [DEP]Sensor telemetry stream.[EOS]",
            "[THREAD:0][MASK] is the process of generating natural language. [RESP]Text generation.[EOS]",
        ]
        texts.extend(templates)
        return texts

    def _init_hf_default(self, cache_save_path: Optional[Path] = None) -> None:
        """Initialize HuggingFace ByteLevel BPE tokenizer to target vocab size."""
        self._hf_tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        self._hf_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._hf_tokenizer.decoder = ByteLevelDecoder()

        texts = self._get_training_texts()
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.all_special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
            min_frequency=1,
        )
        self._hf_tokenizer.train_from_iterator(texts, trainer=trainer)

        # Pad remaining vocabulary up to exact target vocab_size (32,000 or 64,000)
        cur_size = self._hf_tokenizer.get_vocab_size()
        if cur_size < self.vocab_size:
            diff = self.vocab_size - cur_size
            padding = [f"<|reserved_{i}|>" for i in range(diff)]
            self._hf_tokenizer.add_tokens(padding)

        if cache_save_path is not None:
            try:
                cache_save_path.parent.mkdir(parents=True, exist_ok=True)
                self.save(cache_save_path)
            except Exception as e:
                logger.warning(f"Failed to cache tokenizer to {cache_save_path}: {e}")

    def _init_spm_default(self, cache_save_path: Optional[Path] = None) -> None:
        """Initialize SentencePiece BPE model."""
        texts = self._get_training_texts()
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as tf:
            for t in texts:
                tf.write(t.replace("\n", " ") + "\n")
            temp_text_path = tf.name

        model_prefix = str(Path(temp_text_path).with_suffix(""))
        user_defined = ",".join(self.all_special_tokens)
        spm.SentencePieceTrainer.train(
            input=temp_text_path,
            model_prefix=model_prefix,
            vocab_size=min(self.vocab_size, 8000),
            user_defined_symbols=user_defined,
            model_type="bpe",
            byte_fallback=True,
        )
        self._spm_processor = spm.SentencePieceProcessor()
        self._spm_processor.load(f"{model_prefix}.model")

    def thread_id_to_token_id(self, thread_idx: int) -> int:
        """Get the token ID corresponding to a given cognitive thread."""
        tok = f"[THREAD:{thread_idx % self.max_threads}]"
        return self._special_token_to_id[tok]

    def token_id_to_thread_id(self, token_id: int) -> Optional[int]:
        """If token_id is a thread marker, return integer thread_id, else None."""
        if self.thread_token_start <= token_id < self.thread_token_start + self.max_threads:
            return token_id - self.thread_token_start
        return None

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        thread_id: Optional[int] = None,
    ) -> List[int]:
        """Encode string to token IDs with cognitive thread and boundary tokens."""
        prefix_ids: List[int] = []
        if add_bos:
            prefix_ids.append(self.bos_id)
        if thread_id is not None and not text.startswith("[THREAD:"):
            prefix_ids.append(self.thread_id_to_token_id(thread_id))

        if self._hf_tokenizer is not None:
            encoding = self._hf_tokenizer.encode(text)
            body_ids = list(encoding.ids)
        elif self._spm_processor is not None:
            body_ids = self._spm_processor.encode_as_ids(text)
        else:
            raise RuntimeError("Tokenizer has not been initialized.")

        if add_eos:
            body_ids.append(self.eos_id)

        return prefix_ids + body_ids

    def decode(self, token_ids: List[int], skip_special: bool = False) -> str:
        """Decode token IDs back into string."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special)
        elif self._spm_processor is not None:
            return self._spm_processor.decode_ids(token_ids)
        else:
            raise RuntimeError("Tokenizer has not been initialized.")

    def batch_encode(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        thread_ids: Optional[List[Optional[int]]] = None,
    ) -> List[List[int]]:
        """Encode a batch of strings efficiently."""
        if thread_ids is None:
            thread_ids = [None] * len(texts)
        return [
            self.encode(text, add_bos=add_bos, add_eos=add_eos, thread_id=th_id)
            for text, th_id in zip(texts, thread_ids)
        ]

    def batch_decode(
        self,
        batch_ids: List[List[int]],
        skip_special: bool = False,
    ) -> List[str]:
        """Decode a batch of token ID sequences."""
        return [self.decode(ids, skip_special=skip_special) for ids in batch_ids]

    def to_str(self) -> str:
        """Export serialized tokenizer definition as JSON string."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.to_str()
        raise NotImplementedError("to_str only implemented for HuggingFace backend.")

    @classmethod
    def from_str(cls, json_str: str, vocab_size: int = 32000, max_threads: int = 64) -> "BpeSemanticTokenizer":
        """Reconstruct BpeSemanticTokenizer directly from JSON string."""
        tok = cls(vocab_size=vocab_size, max_threads=max_threads, auto_build_if_missing=False)
        tok._hf_tokenizer = HFTokenizer.from_str(json_str)
        tok.vocab_size = tok._hf_tokenizer.get_vocab_size()
        return tok

    def save(self, path: Union[str, Path]) -> None:
        """Save trained tokenizer state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._hf_tokenizer is not None:
            self._hf_tokenizer.save(str(path))
        else:
            raise NotImplementedError("Save only implemented for HuggingFace backend.")

    def load(self, path: Union[str, Path]) -> None:
        """Load tokenizer state from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        if self.backend in ("huggingface", "hf"):
            self._hf_tokenizer = HFTokenizer.from_file(str(path))
            self.vocab_size = self._hf_tokenizer.get_vocab_size()
        elif self.backend in ("sentencepiece", "spm"):
            self._spm_processor = spm.SentencePieceProcessor()
            self._spm_processor.load(str(path))
            self.vocab_size = self._spm_processor.get_piece_size()

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: Optional[int] = None,
    ) -> None:
        """Train tokenizer on a custom text iterator."""
        v_size = vocab_size or self.vocab_size
        self._hf_tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        self._hf_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._hf_tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=v_size,
            special_tokens=self.all_special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
        )
        self._hf_tokenizer.train_from_iterator(iterator, trainer=trainer)
        cur_size = self._hf_tokenizer.get_vocab_size()
        if cur_size < v_size:
            diff = v_size - cur_size
            self._hf_tokenizer.add_tokens([f"<|reserved_{i}|>" for i in range(diff)])
        self.vocab_size = v_size

    def get_vocab(self) -> Dict[str, int]:
        """Return full vocabulary mapping token string to ID."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.get_vocab()
        elif self._spm_processor is not None:
            return {
                self._spm_processor.id_to_piece(i): i
                for i in range(self._spm_processor.get_piece_size())
            }
        return {}

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert single token to ID."""
        if token in self._special_token_to_id:
            return self._special_token_to_id[token]
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(token)
        return None

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert single ID to token string."""
        if token_id in self._id_to_special_token:
            return self._id_to_special_token[token_id]
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.id_to_token(token_id)
        return None

    def __len__(self) -> int:
        return self.vocab_size


def get_bpe_tokenizer(
    vocab_size: int = 32000,
    max_threads: int = 64,
    backend: str = "huggingface",
) -> BpeSemanticTokenizer:
    """Factory creating calibrated 32,000 or 64,000 BPE Semantic Tokenizers."""
    return BpeSemanticTokenizer(
        vocab_size=vocab_size,
        max_threads=max_threads,
        backend=backend,
        auto_build_if_missing=True,
    )
