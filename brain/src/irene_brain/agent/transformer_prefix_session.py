"""Explicit full-prefix transformer baseline; no fixed-state memory claim."""
from dataclasses import dataclass
import torch


class ContextWindowExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class PrefixGeneration:
    token_ids: tuple[int,...]
    stop_reason: str
    state_bytes: None = None


class TransformerPrefixSession:
    """Recomputes the full prefix, retaining all action/observation tokens.

This is the comparison baseline only. No KV optimization or recurrent 4 KB state
is provided, and its generation timing must not be called a fair speed benchmark.
"""
    def __init__(self,model,*,max_prefix_tokens=32768):
        if getattr(model,'architecture',None)!='comparison_transformer_v1':
            raise ValueError('Prefix session requires the transformer comparison model')
        if type(max_prefix_tokens) is not int or max_prefix_tokens<1:
            raise ValueError('Positive prefix bound required')
        self.model=model.eval()
        self.max_prefix_tokens=max_prefix_tokens
        self._prompt=None
        self._prefix=None

    def _tokens(self,tokens):
        if not isinstance(tokens,torch.Tensor) or tokens.ndim!=2 or tokens.shape[0]!=1 or tokens.shape[1]<1:
            raise ValueError('Expected nonempty [1,tokens] input')
        if tokens.dtype!=torch.long or tokens.device!=self.model.embedding.weight.device:
            raise ValueError('Tokens must be long on the model device')
        if not ((tokens>=0)&(tokens<self.model.config.vocab_size)).all():
            raise ValueError('Token is outside the vocabulary')
        return tokens

    @property
    def state_bytes(self):
        return None

    @property
    def prefix_tokens(self):
        return 0 if self._prefix is None else self._prefix.shape[1]

    @property
    def prefix(self):
        if self._prefix is None: raise RuntimeError('Start a task first')
        return self._prefix.clone()

    def start(self,prompt_tokens):
        prompt=self._tokens(prompt_tokens)
        if prompt.shape[1]>self.max_prefix_tokens:
            raise ContextWindowExceeded('Initial prompt exceeds transformer prefix bound')
        self._prompt=prompt.clone()
        self._prefix=prompt.clone()

    @torch.no_grad()
    def ingest(self,new_tokens):
        if self._prefix is None: raise RuntimeError('Start a task first')
        tokens=self._tokens(new_tokens)
        if self.prefix_tokens+tokens.shape[1]>self.max_prefix_tokens:
            raise ContextWindowExceeded('Transformer prefix limit reached; no truncation')
        combined=torch.cat((self._prefix,tokens),dim=1)
        logits=self.model(combined,prompt_tokens=self._prompt)[:,-1]
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite transformer logits')
        self._prefix=combined
        return logits

    @torch.no_grad()
    def generate(self,prefix_tokens,*,eos_id,max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens<1:
            raise ValueError('Positive generation limit required')
        if type(eos_id) is not int or not 0<=eos_id<self.model.config.vocab_size:
            raise ValueError('Valid EOS token required')
        logits=self.ingest(prefix_tokens)
        generated=[]
        for _ in range(max_new_tokens):
            if self.prefix_tokens==self.max_prefix_tokens:
                return PrefixGeneration(tuple(generated),'context_limit')
            token=logits.argmax(dim=-1)
            value=int(token.item())
            logits=self.ingest(token[:,None])
            if value==eos_id:
                return PrefixGeneration(tuple(generated),'eos')
            generated.append(value)
        return PrefixGeneration(tuple(generated),'token_limit')
