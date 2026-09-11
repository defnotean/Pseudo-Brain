"""Parallel initial-context ingestion without a generated-token history cache.

Experimental opt-in path; frozen v3/v4 benchmark decoders remain unchanged.
The returned state is the ordinary16x64 float32 tensor accepted by model.step.
Sequence features are transient prefill workspace, not persistent inference state.
"""
import torch
from irene_brain.parallel.triton_scan import triton_scan


@torch.no_grad()
def parallel_recurrent_prefill(model, token_ids, prompt_tokens=None, reset_mask=None):
    if getattr(model,'architecture',None) != 'parallel_depth_v1':
        raise ValueError('Parallel prefill requires the parallel-depth recurrent model')
    if model.embedding.weight.dtype != torch.float64:
        raise ValueError('Parallel prefill requires the FP64 parity model')
    if token_ids.ndim != 2 or token_ids.shape[1] == 0 or token_ids.dtype != torch.long:
        raise ValueError('Initial context must be a nonempty [batch,tokens] long tensor')
    if reset_mask is not None and reset_mask.shape != token_ids.shape:
        raise ValueError('Reset mask must match the initial context')
    x = model.embedding(token_ids)
    final_states = []
    for layer in model.layers:
        a,b = layer.coefficients(x)
        if reset_mask is not None:
            a = a*(1-reset_mask.to(a.dtype).unsqueeze(-1))
        hidden = triton_scan(a,b)
        final_states.append(hidden[:,-1])
        x = layer.output(x,hidden)
    values = torch.stack(final_states,dim=1)
    high = values.float()
    low = (values-high.double()).float()
    state = torch.cat((high,low),dim=1)
    logits = model.readout(x[:,-1:],token_ids[:,-1:],prompt_tokens)[:,0]
    return logits,state
