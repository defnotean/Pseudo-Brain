"""Shared action-only objective for the registered recurrent/transformer pair."""
import torch
from torch.nn import functional as F
from irene_brain.data.sequence_buckets import bucket_training_example


def action_trajectory_loss(model, encoded, *, chunk_size=256):
    if chunk_size<1:
        raise ValueError('Readout chunk size must be positive')
    ids,labels=encoded['input_ids'],encoded['labels']
    prompt,resets=encoded['prompt_tokens'],encoded['reset_mask']
    if ids.ndim!=2 or ids.shape[0]!=1 or labels.shape!=ids.shape or resets.shape!=ids.shape:
        raise ValueError('Expected one whole trajectory with matching labels and resets')
    if not 0<ids.shape[1]<=8192 or prompt.ndim!=2 or prompt.shape[0]!=1 or not 0<prompt.shape[1]<=ids.shape[1]:
        raise ValueError('Invalid trajectory or immutable prompt length')
    if any(value.dtype!=torch.long for value in (ids,labels,prompt,resets)):
        raise ValueError('Trajectory inputs must be long tensors')
    if any(value.device!=model.embedding.weight.device for value in (ids,labels,prompt,resets)):
        raise ValueError('Model and trajectory must share a device')
    if not torch.equal(ids[:,:prompt.shape[1]],prompt):
        raise ValueError('Pointer source must equal the initial trajectory prompt')
    if resets[0,0]!=1 or torch.count_nonzero(resets[:,1:])!=0:
        raise ValueError('Only task start may reset recurrent context')
    if model.architecture=='parallel_depth_v1':
        ids,labels=bucket_training_example(ids,labels,max_length=8192)
        resets=torch.zeros_like(ids)
        resets[0,0]=1
    elif model.architecture!='comparison_transformer_v1':
        raise ValueError('Unsupported paired model architecture')
    if model.architecture=='comparison_transformer_v1':
        # Native transformer gradients vary with readout batching shape. Keep the
        # full reference readout here; this uses more transient training memory.
        # Frozen foundation training still uses its original language_loss path.
        logits=model(ids,reset_mask=resets,prompt_tokens=prompt)
        loss=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100)
    else:
        loss=model.language_loss(ids,labels,reset_mask=resets,prompt_tokens=prompt,chunk_size=chunk_size)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite action-trajectory loss')
    return loss
