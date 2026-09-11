"""Parallel ingestion of new tokens from an existing fixed-size recurrent state.

Opt-in continuation primitive for future action/observation integration. Earlier
tokens are not inputs; the immutable initial prompt remains the pointer source.
The established initial-prefill helper and frozen decoders are unchanged.
"""
import torch
from irene_brain.parallel.triton_scan import triton_scan


@torch.no_grad()
def parallel_recurrent_ingest(model, token_ids, state, prompt_tokens=None, reset_mask=None):
    if getattr(model,'architecture',None)!='parallel_depth_v1' or model.embedding.weight.dtype!=torch.float64:
        raise ValueError('Continuation ingestion requires the FP64 parallel-depth model')
    if token_ids.ndim!=2 or min(token_ids.shape)<1 or token_ids.dtype!=torch.long:
        raise ValueError('New tokens must be a nonempty [batch,tokens] long tensor')
    if state.shape!=(token_ids.shape[0],16,64) or state.dtype!=torch.float32:
        raise ValueError('Carried state must be [batch,16,64] float32')
    if state.device!=token_ids.device or token_ids.device!=model.embedding.weight.device:
        raise ValueError('Tokens, state and model must share a device')
    if not torch.isfinite(state).all():
        raise ValueError('Carried state must be finite')
    if reset_mask is not None:
        if reset_mask.shape!=token_ids.shape or reset_mask.device!=token_ids.device:
            raise ValueError('Reset mask must match the new tokens')
        if not ((reset_mask==0)|(reset_mask==1)).all():
            raise ValueError('Reset mask must be binary')
    previous=state[:,:8].double()+state[:,8:].double()
    x=model.embedding(token_ids)
    final_states=[]
    for index,layer in enumerate(model.layers):
        a,b=layer.coefficients(x)
        if reset_mask is not None:
            a=a*(1-reset_mask.to(a.dtype).unsqueeze(-1))
        hidden=triton_scan(a,b,h0=previous[:,index])
        final_states.append(hidden[:,-1])
        x=layer.output(x,hidden)
    values=torch.stack(final_states,dim=1)
    high=values.float()
    low=(values-high.double()).float()
    next_state=torch.cat((high,low),dim=1)
    logits=model.readout(x[:,-1:],token_ids[:,-1:],prompt_tokens)[:,0]
    if not torch.isfinite(next_state).all() or not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite continuation output')
    return logits,next_state
