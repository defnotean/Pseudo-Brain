"""Mechanical multi-turn parity checks, explicitly not capability evaluation."""
import torch

from irene_brain.agent.parallel_depth_session import ParallelDepthSession


def reconstructed(state):
    return state[:,:8].double()+state[:,8:].double()


@torch.no_grad()
def validate_multiturn_parity(model, encoded, *, resp_id, eos_id, generation_limit=16):
    device=model.embedding.weight.device
    ids=encoded['input_ids'].to(device)
    prompt=encoded['prompt_tokens'].to(device)
    resets=encoded['reset_mask'].to(device)
    spans=encoded['action_spans']
    if ids.shape[0]!=1 or not spans or generation_limit<1:
        raise ValueError('One nonempty trajectory and positive generation limit required')
    if not torch.equal(ids[:,:prompt.shape[1]],prompt) or resets.sum()!=1 or resets[0,0]!=1:
        raise ValueError('Trajectory must begin with the immutable prompt and one reset')
    boundaries={prompt.shape[1]-1}
    for span in spans:
        boundaries.update((span['eos_position'],span['observation_end']-1))
    features=model.features_parallel(ids,resets)
    state=model.init_state(1)
    expected_states={}
    report={'status':'running','tokens':ids.shape[1],'all_position_logit_error':0.,
            'boundary_logit_error':0.,'boundary_state_error':0.,'generated_state_error':0.,
            'generated_turns':[],'state_bytes':4096,
            'scope':'Mechanical parity with executed teacher frames and fixed diagnostic observations; not a task score.'}
    for offset in range(0,ids.shape[1],128):
        end=min(offset+128,ids.shape[1])
        parallel=model.readout(features[:,offset:end],ids[:,offset:end],prompt)
        assert torch.isfinite(parallel).all(), 'Nonfinite parallel logits'
        maximum=torch.zeros((),dtype=torch.float64,device=device)
        for index in range(offset,end):
            logits,state=model.step(ids[:,index],state,prompt)
            assert torch.isfinite(logits).all() and torch.isfinite(state).all(), 'Nonfinite streaming output'
            assert state.shape==(1,16,64) and state.dtype==torch.float32
            maximum=torch.maximum(maximum,(logits-parallel[:,index-offset]).abs().max())
            if index in boundaries: expected_states[index]=state.clone()
        report['all_position_logit_error']=max(report['all_position_logit_error'],maximum.item())
        assert report['all_position_logit_error']<1e-6
    session=ParallelDepthSession(model)
    session.start(prompt)
    def check_boundary(index,logits=None):
        actual=session.state
        assert session.state_bytes==4096 and torch.isfinite(actual).all()
        error=(reconstructed(actual)-reconstructed(expected_states[index])).abs().max().item()
        report['boundary_state_error']=max(report['boundary_state_error'],error)
        assert error<1e-12
        if logits is not None:
            expected=model.readout(features[:,index:index+1],ids[:,index:index+1],prompt)[:,0]
            assert torch.isfinite(logits).all() and torch.isfinite(expected).all()
            error=(logits-expected).abs().max().item()
            report['boundary_logit_error']=max(report['boundary_logit_error'],error)
            assert error<1e-6
    check_boundary(prompt.shape[1]-1)
    for span in spans:
        logits=session.ingest(ids[:,span['start']:span['eos_position']+1])
        check_boundary(span['eos_position'],logits)
        logits=session.ingest(ids[:,span['eos_position']+1:span['observation_end']])
        check_boundary(span['observation_end']-1,logits)
    del features,expected_states
    # Fresh generated-token probe. Fixed feedback is a diagnostic input, not a
    # claim that these generated outputs caused those environment observations.
    session.start(prompt)
    state=model.init_state(1)
    for token in prompt.unbind(1): _,state=model.step(token,state,prompt)
    prefix=torch.tensor([[resp_id]],dtype=torch.long,device=device)
    for span in spans[:3]:
        logits,state=model.step(prefix[:,0],state,prompt)
        generated=[]
        stop='token_limit'
        for _ in range(generation_limit):
            assert torch.isfinite(logits).all()
            token=logits.argmax(dim=-1)
            logits,state=model.step(token,state,prompt)
            if int(token.item())==eos_id:
                stop='eos'
                break
            generated.append(int(token.item()))
        actual=session.generate(prefix,eos_id=eos_id,max_new_tokens=generation_limit)
        assert actual.token_ids==tuple(generated) and actual.stop_reason==stop
        observation=ids[:,span['eos_position']+1:span['observation_end']]
        for token in observation.unbind(1): logits,state=model.step(token,state,prompt)
        actual_logits=session.ingest(observation)
        assert torch.isfinite(logits).all() and torch.isfinite(state).all()
        assert (actual_logits-logits).abs().max().item()<1e-6
        error=(reconstructed(session.state)-reconstructed(state)).abs().max().item()
        report['generated_state_error']=max(report['generated_state_error'],error)
        assert error<1e-12 and session.state_bytes==4096
        report['generated_turns'].append({'token_ids':generated,'stop_reason':stop})
    report['status']='passed'
    return report
