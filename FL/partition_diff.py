import numpy as np
import torch

def flatten_diff(diff):
    flat_diff = []
    shape_info = {}
    for name, param in diff.items():
        flat_diff.append(param.view(-1).cpu().numpy())
        shape_info[name] = param.shape
    flat_diff = np.concatenate(flat_diff)
    return flat_diff, shape_info

def unflatten_diff(flat_diff, shape_info):
    diff = {}
    offset = 0
    for name, shape in shape_info.items():
        size = np.prod(shape)
        param = flat_diff[offset:offset+size].reshape(shape)
        diff[name] = torch.tensor(param)
        offset += size
    return diff

def split_flat_diff(flat_diff, num_blocks):
    return np.array_split(flat_diff, num_blocks)

