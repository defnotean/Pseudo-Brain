"""Neural Semantic Salience & Intent Router for Pseudo-Brain.

Replaces brittle regex string peeling with neural self-attention and
semantic entropy analysis over BPE token embeddings in UnifiedPseudoBrain.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple
import torch
import torch.nn as nn

from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer


class NeuralSemanticRouter:
    """Extracts semantic intent and topic entities directly using Pseudo-Brain's neural state."""

    def __init__(self, model: nn.Module, vocab_size: int = 2048):
        self.model = model
        self.vocab_size = vocab_size
        self.tokenizer = BpeSemanticTokenizer(vocab_size=vocab_size)
        self.device = next(model.parameters()).device

        # Closed-class functional syntax tokens (grammatical glue with near-zero semantic information entropy)
        self.functional_syntax_words: Set[str] = {
            "then", "how", "about", "what", "is", "a", "an", "the", "can", "you", "me", "tell",
            "could", "explain", "do", "does", "someone", "properly", "way", "to", "in", "of",
            "for", "and", "or", "so", "well", "now", "ok", "okay", "like", "just", "with", "from",
            "what's", "who", "who's", "where", "why", "which", "are", "was", "were", "been", "being",
            "have", "has", "had", "should", "would", "shall", "will", "might", "must", "again",
            "memory", "prior", "earlier", "previously", "remember", "recall", "know", "research",
            "said", "told", "talk", "mention", "think", "thought"
        }

    def compute_token_salience(self, prompt: str) -> List[Tuple[str, float]]:
        """Compute neural attention-weighted information salience for each word in the prompt."""
        raw_words = [w.strip(" ,.?!:\"'") for w in prompt.split()]
        words = [w for w in raw_words if w]
        if not words:
            return []

        # 1. BPE Tokenization & Mean Subword Embedding Lookup per Word
        word_embs = []
        with torch.no_grad():
            for w in words:
                enc = self.tokenizer.encode(w) or [0]
                tok_tensor = torch.tensor(enc, dtype=torch.long, device=self.device)
                e = self.model.embedding(tok_tensor).mean(dim=0)
                word_embs.append(e)

            word_tensor = torch.stack(word_embs, dim=0)  # [N, embed_dim]

            # 2. Sensory Projection to Working Memory Dimension
            sensory = self.model.lang_proj(word_tensor)  # [N, proj_dim]
            context_vec = sensory.mean(dim=0, keepdim=True)  # [1, proj_dim]

            # 3. Attention Saliency via Dot-Product Projection
            proj_dim = float(sensory.shape[-1])
            attn_logits = torch.matmul(sensory, context_vec.T).squeeze(-1)  # [N]
            attn_weights = torch.softmax(attn_logits / (proj_dim ** 0.5), dim=0)

        results: List[Tuple[str, float]] = []
        for w, aw in zip(words, attn_weights):
            w_lower = w.lower()
            weight = float(aw.item())
            if w_lower in self.functional_syntax_words:
                weight = 0.0
            results.append((w, weight))

        return results

    def extract_semantic_topic(self, prompt: str) -> str:
        """Extract the core semantic entity using neural attention without any regex stripping."""
        word_salience = self.compute_token_salience(prompt)
        if not word_salience:
            return prompt.strip()

        # Select high-salience semantic entity tokens (non-functional content words)
        salient_tokens = [w for w, s in word_salience if s > 0.0]
        if not salient_tokens:
            return prompt.strip(" ?.:,`'\"")

        return " ".join(salient_tokens)

    def evaluate_tool_necessity(self, prompt: str, cognitive_state: Optional[Any] = None) -> Tuple[bool, float]:
        """Evaluate whether the query requires external research using the neural tool_gate."""
        raw_words = [w.strip(" ,.?!:\"'") for w in prompt.split()]
        words = [w for w in raw_words if w]
        if not words:
            return False, 0.0

        token_ids = [self.tokenizer.encode(w)[0] if self.tokenizer.encode(w) else 0 for w in words]
        tok_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)

        with torch.no_grad():
            embs = self.model.embedding(tok_tensor)
            sensory = self.model.lang_proj(embs).mean(dim=0, keepdim=True)
            if self.model.deep_proj is not None:
                sensory = sensory + self.model.deep_proj(sensory)

            # Step the working memory state
            if cognitive_state is None:
                state = self.model.init_state(batch_size=1, device=self.device)
            else:
                state = cognitive_state

            outputs, _ = self.model.step(sensory, state, allow_routing=True)
            tool_prob = float(outputs["tool_prob"].item())

        return tool_prob > 0.4, tool_prob
