import pytest
import torch
from irene_brain.agent.transformer_prefix_session import TransformerPrefixSession,ContextWindowExceeded
from irene_brain.agent.parallel_depth_episode import TransformerSoftwareAgent
from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment
from irene_brain.unified.parallel_depth_model import ParallelDepthConfig
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def setup():
    torch.manual_seed(1988)
    model=ComparisonTransformer(ParallelDepthConfig(width=64,vocab_size=128,pointer=True)).eval()
    prompt=torch.randint(10,128,(1,9))
    return model,prompt


@torch.no_grad()
def test_multiple_turns_match_explicit_full_prefix_and_consume_eos():
    model,prompt=setup()
    session=TransformerPrefixSession(model)
    session.start(prompt)
    history=prompt.clone()
    for turn in range(3):
        obs=torch.randint(10,128,(1,7))
        history=torch.cat((history,obs),1)
        expected=model(history,prompt_tokens=prompt)[:,-1]
        torch.testing.assert_close(session.ingest(obs),expected,atol=0,rtol=0)
        prefix=torch.tensor([[5]])
        history=torch.cat((history,prefix),1)
        logits=model(history,prompt_tokens=prompt)[:,-1]
        eos=int(logits.argmax().item()) if turn==1 else 2
        output=[]
        stop='token_limit'
        for _ in range(5):
            token=logits.argmax(-1)
            history=torch.cat((history,token[:,None]),1)
            logits=model(history,prompt_tokens=prompt)[:,-1]
            if int(token.item())==eos:
                stop='eos'
                break
            output.append(int(token.item()))
        result=session.generate(prefix,eos_id=eos,max_new_tokens=5)
        assert result.token_ids==tuple(output) and result.stop_reason==stop
        assert torch.equal(session.prefix,history)
        assert session.state_bytes is None and result.state_bytes is None
        assert torch.equal(session._prompt,prompt)


def test_bounds_are_explicit_and_start_discards_old_context():
    model,prompt=setup()
    session=TransformerPrefixSession(model,max_prefix_tokens=11)
    session.start(prompt)
    result=session.generate(torch.tensor([[5]]),eos_id=2,max_new_tokens=8)
    assert session.prefix_tokens<=11
    assert result.stop_reason in ('context_limit','eos')
    before=session.prefix
    with pytest.raises(ContextWindowExceeded): session.ingest(torch.ones(1,12,dtype=torch.long))
    assert torch.equal(session.prefix,before)
    session.start(prompt)
    assert torch.equal(session.prefix,prompt)
    prompt.zero_()
    assert not torch.equal(session.prefix,prompt)


def test_shared_episode_loop_reports_transformer_prefix_without_4kb_claim():
    class Tokenizer:
        resp_id=5
        eos_id=2
        def encode(self,text): return [5] if text=='[RESP]' else [11,12,13]
        def decode(self,tokens,skip_special=False): return 'malformed output'
    model,_=setup()
    agent=TransformerSoftwareAgent(model,Tokenizer(),max_prefix_tokens=6)
    result=agent.execute_episode('goal',IsolatedSoftwareEnvironment(),max_action_tokens=9)
    assert not result.success and result.stop_reason=='context_limit'
    assert result.state_bytes is None and result.prefix_tokens<=6
