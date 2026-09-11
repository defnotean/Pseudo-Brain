"""Opt-in bounded episodes connecting the recurrent session and isolated tools."""
from dataclasses import asdict, dataclass
import hashlib
import json

import torch

from irene_brain.agent.parallel_depth_session import ParallelDepthSession
from irene_brain.agent.transformer_prefix_session import ContextWindowExceeded,TransformerPrefixSession


def observation_frame(feedback):
    """A fixed data frame, without inferred phases or repeated task history.

Escape bracket characters inside the JSON data so tool output cannot introduce
literal tokenizer control markers. This is framing, not prompt-injection proof.
"""
    data = json.dumps(asdict(feedback),ensure_ascii=True,sort_keys=True,separators=(',',':'))
    data = data.replace('[','\\u005b').replace(']','\\u005d')
    return '\n[OBSERVATION: '+data+']\n'


@dataclass(frozen=True)
class ParallelEpisodeResult:
    success: bool
    stop_reason: str
    cycles: int
    state_bytes: int | None
    initial_prompt_sha256: str
    trace: tuple[dict, ...]
    prefix_tokens: int | None = None


class ParallelDepthSoftwareAgent:
    """No action plan, repair rules, parser correction, or success fallback."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _create_session(self):
        return ParallelDepthSession(self.model)

    def execute_episode(self, initial_prompt, environment, *, max_cycles=8,
                        max_action_tokens=512, max_observation_tokens=4096,
                        max_prompt_tokens=4096):
        for value in (max_cycles,max_action_tokens,max_observation_tokens,max_prompt_tokens):
            if type(value) is not int or value<1:
                raise ValueError('Episode limits must be positive integers')
        if not isinstance(initial_prompt,str) or not initial_prompt:
            raise ValueError('A nonempty initial task prompt is required')
        device = self.model.embedding.weight.device
        def encode(text):
            tokens = self.tokenizer.encode(text)
            if not tokens:
                raise ValueError('Text encoded to no tokens')
            return torch.tensor([tokens],dtype=torch.long,device=device)
        prompt = encode(initial_prompt)
        if prompt.shape[1]>max_prompt_tokens:
            raise ValueError('Initial prompt exceeds the registered token limit')
        prefix = encode('[RESP]')
        if prefix.tolist()!=[[self.tokenizer.resp_id]]:
            raise ValueError('Tokenizer response marker does not match its declared ID')
        session = self._create_session()
        session.start(prompt)
        prompt_sha = hashlib.sha256(initial_prompt.encode()).hexdigest()
        trace = []
        def finish(success,reason):
            return ParallelEpisodeResult(success,reason,len(trace),session.state_bytes,prompt_sha,tuple(trace),getattr(session,'prefix_tokens',None))
        for cycle in range(max_cycles):
            try:
                generated = session.generate(prefix,eos_id=self.tokenizer.eos_id,max_new_tokens=max_action_tokens)
            except ContextWindowExceeded:
                return finish(False,'context_limit')
            action = self.tokenizer.decode(list(generated.token_ids),skip_special=False)
            entry = {'cycle':cycle+1,'raw_action':action,'token_ids':list(generated.token_ids),
                     'generation_stop':generated.stop_reason,'executed':False,'state_bytes':session.state_bytes}
            trace.append(entry)
            if generated.stop_reason!='eos':
                # Do not execute or repair an incomplete action, or fabricate EOS.
                reasons={'token_limit':'action_token_limit','context_limit':'context_limit'}
                return finish(False,reasons.get(generated.stop_reason,'invalid_generation_stop'))
            feedback = environment.execute_action(action)
            entry.update(executed=True,feedback=asdict(feedback))
            frame = observation_frame(feedback)
            observation = encode(frame)
            entry['observation_tokens'] = observation.shape[1]
            if observation.shape[1]>max_observation_tokens:
                return finish(False,'observation_token_limit')
            try:
                session.ingest(observation)
            except ContextWindowExceeded:
                return finish(False,'context_limit')
            entry['observation_ingested'] = True
            if feedback.action_type=='FINISH' and feedback.success and feedback.verified_completion:
                return finish(True,'verified_completion')
        return finish(False,'cycle_limit')


class TransformerSoftwareAgent(ParallelDepthSoftwareAgent):
    """Same actuator/feedback rules, full-prefix baseline with explicit memory use."""
    def __init__(self,model,tokenizer,*,max_prefix_tokens=32768):
        super().__init__(model,tokenizer)
        self.max_prefix_tokens=max_prefix_tokens

    def _create_session(self):
        return TransformerPrefixSession(self.model,max_prefix_tokens=self.max_prefix_tokens)
