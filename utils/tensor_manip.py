# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
import numpy as np

def rebatch(idx_0, idx_det):
    # Rebuild the batch dimension : (N, ...) is turned into (batch_dim, nb_max, ...) 
    # with zero padding for batch elements with fewer people.
    values, counts = torch.unique(idx_0, sorted=True, return_counts=True)
    #print(idx_0)
    if not len(values) == values.max() + 1:
        # Abnormal jumps in the idx_0: some images in the batch did not produce any inputs.
        jumps = (values - torch.concat([torch.Tensor([-1]).to(values.device), values])[:-1]) - 1
        offsets = torch.cumsum(jumps.int(), dim=0)

        # Correcting idx_0 to account for missing batch elements
        # This is actually wrong: in the case where we have 2 consecutive images without ppl, this will fail.
        # But two consecutive jumps has proba so close to 0 that I consider it 'impossible'.
        offsets = [c * [o] for o, c in [(offsets[i], counts[i]) for i in range(offsets.shape[0])]]
        offsets = torch.Tensor([e for o in offsets for e in o]).to(jumps.device).int()
        idx_0 = idx_0 - offsets
        idx_det_0 = idx_det[0] - offsets
    else:
        idx_det_0 = idx_det[0]
    return counts, idx_det_0

def pad(x, padlen, dim):
    assert x.shape[dim] <= padlen, "Incoherent dimensions"
    if not dim == 1:
        raise NotImplementedError("Not implemented for this dim.")
    padded = torch.concat([x, x.new_zeros((x.shape[0], padlen - x.shape[dim],) + x.shape[2:])], dim=dim) 
    mask = torch.concat([x.new_ones((x.shape[0], x.shape[dim])), x.new_zeros((x.shape[0], padlen - x.shape[dim]))], dim=dim)
    return padded, mask

def pad_to_max(x_central, counts):
    """Pad so that each batch images has the same number of x_central queries.
    Mask is used in attention to remove the fact queries. """
    max_count = counts.max()
    xlist = torch.split(x_central, tuple(counts), dim=0)
    xlist2 = [x.unsqueeze(0) for x in xlist]
    xlist3 = [pad(x, max_count, dim=1) for x in xlist2]
    xlist4, mask = [x[0] for x in xlist3], [x[1] for x in xlist3]
    x_central, mask = torch.concat(xlist4, dim=0), torch.concat(mask, dim=0)
    return x_central, mask

def recursive_apply(obj, func, kwargs=None):
    if isinstance(obj, dict):
        return {k: recursive_apply(v, func, kwargs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_apply(v, func, kwargs) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_apply(v, func, kwargs) for v in obj)
    elif isinstance(obj, torch.Tensor):
        return func(obj, **kwargs) if kwargs is not None else func(obj)
    elif isinstance(obj, np.ndarray):
        return func(obj, **kwargs) if kwargs is not None else func(obj)
    else:
        return obj
    
def recursive_stack(dicts):
    if isinstance(dicts[0], dict):
        return {k: recursive_stack([d[k] for d in dicts]) for k in dicts[0]}
    elif isinstance(dicts[0], list):
        return [recursive_stack([d[i] for d in dicts]) for i in range(len(dicts[0]))]
    elif isinstance(dicts[0], torch.Tensor):
        return torch.stack(dicts, dim=0)
    elif isinstance(dicts[0], np.ndarray):
        return np.stack(dicts)
    else:
        raise TypeError(f"Unsupported type: {type(dicts[0])}")