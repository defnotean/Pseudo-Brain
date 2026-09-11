"""Explicit length buckets for stateless-per-example causal training/readout.

This is not a streaming prefill API: states advanced through trailing padding
must not be reused as the real document's final state.
"""
import torch
from torch.nn import functional as F


def bucket_training_example(input_ids, labels, *, multiple=256, max_length=32768, pad_id=0):
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError('Inputs and next-token labels must have the same [batch,time] shape')
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError('Inputs and labels must be integer token tensors')
    if multiple < 1 or max_length < 1 or max_length % multiple:
        raise ValueError('Positive maximum length must be divisible by the bucket multiple')
    length = input_ids.shape[1]
    if not 0 < length <= max_length:
        raise ValueError('Document length is outside the supported context bound')
    padded_length = ((length + multiple - 1) // multiple) * multiple
    padding = padded_length - length
    if padding == 0:
        return input_ids, labels
    return F.pad(input_ids, (0, padding), value=pad_id), F.pad(labels, (0, padding), value=-100)
