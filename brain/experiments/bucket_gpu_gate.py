"""Run the registered GPU equivalence or isolated-cache timing diagnostic."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from irene_brain.data.sequence_buckets import bucket_training_example
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def example(length):
    generator = torch.Generator(device='cuda').manual_seed(909 + length)
    ids = torch.randint(1, 32000, (1, length), generator=generator, device='cuda')
    labels = torch.roll(ids, -1, 1)
    labels[:, :7] = -100
    labels[:, -1] = -100
    return ids, labels, ids[:, :7].clone()


def equivalence():
    records = []
    for name, cls in [('recurrent', ParallelDepthModel), ('transformer', ComparisonTransformer)]:
        torch.manual_seed(909)
        model = cls().cuda()
        for length in (31, 257, 2049):
            ids, labels, prompt = example(length)
            padded, targets = bucket_training_example(ids, labels)
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                ordinary_logits = model(ids, prompt_tokens=prompt)
                padded_logits = model(padded, prompt_tokens=prompt)[:, :length]
                logit_difference = (ordinary_logits - padded_logits).abs().max().item()
            del ordinary_logits, padded_logits
            ordinary = model.language_loss(ids, labels, prompt_tokens=prompt)
            ordinary.backward()
            grads = {key: parameter.grad.clone() for key, parameter in model.named_parameters()}
            model.zero_grad(set_to_none=True)
            bucketed = model.language_loss(padded, targets, prompt_tokens=prompt)
            bucketed.backward()
            maximum, squared_error, squared_reference = 0., 0., 0.
            for key, parameter in model.named_parameters():
                assert torch.isfinite(parameter.grad).all(), key
                difference = parameter.grad - grads[key]
                maximum = max(maximum, difference.abs().max().item())
                squared_error += difference.double().square().sum().item()
                squared_reference += grads[key].double().square().sum().item()
            relative = (squared_error / max(squared_reference, 1e-300)) ** .5
            loss_difference = abs(ordinary.item() - bucketed.item())
            if name == 'recurrent':
                passed = max(logit_difference, loss_difference, maximum) < 1e-8
            else:
                passed = logit_difference < 1e-3 and loss_difference < 1e-5 and relative < torch.finfo(torch.bfloat16).eps
            row = dict(model=name, length=length, padded_length=padded.shape[1],
                       loss_difference=loss_difference, logit_difference=logit_difference,
                       gradient_max_difference=maximum, gradient_relative_l2=relative, passed=passed)
            records.append(row)
            print('BUCKET_EQUIVALENCE', json.dumps(row), flush=True)
            del grads, ordinary, bucketed
        del model
        torch.cuda.empty_cache()
    return {'passed': all(row['passed'] for row in records), 'records': records}


def timing(bucketed):
    torch.manual_seed(909)
    model = ParallelDepthModel().cuda()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for phase in ('cold', 'warm'):
        for length in (1025, 1031, 1057, 1111, 1173, 1259, 1281, 1347):
            ids, labels, prompt = example(length)
            if bucketed:
                ids, labels = bucket_training_example(ids, labels)
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.monotonic()
            loss = model.language_loss(ids, labels, prompt_tokens=prompt)
            loss.backward()
            torch.cuda.synchronize()
            row = dict(phase=phase, length=length, padded_length=ids.shape[1],
                       seconds=time.monotonic()-start, loss=loss.item())
            records.append(row)
            print('BUCKET_TIMING', json.dumps(row), flush=True)
    return {'arm': 'bucketed' if bucketed else 'exact', 'records': records,
            'totals': {phase: sum(row['seconds'] for row in records if row['phase'] == phase) for phase in ('cold', 'warm')},
            'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
            'cubin_files': len(list(Path(os.environ['TRITON_CACHE_DIR']).rglob('*.cubin')))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('equivalence', 'exact', 'bucketed'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    report = equivalence() if args.mode == 'equivalence' else timing(args.mode == 'bucketed')
    report.update(torch=torch.__version__, gpu=torch.cuda.get_device_name())
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
