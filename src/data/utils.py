import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from torch.nn.utils.rnn import pad_sequence

import numpy as np

def torch_dtype(dtype):
    if dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16


def convert_to_item(dataset, feats):
    """Convert to batchable item format."""
    feats = [torch.from_numpy(f).float() for f in feats]
    feats = torch.cat([f.view(1, *f.size()) for f in feats], dim=0)
    labels = dataset.graph
    item = {
        "key": dataset.key,
        "label": labels,
        "data": dataset.data,
        "feats": feats,
        "time": dataset.time,
        "interv": dataset.interv,
        "number_of_nodes": labels.shape[0],
    }
    return item


def collate(args, batch):
    """Overwrite default_collate to handle jagged tensors across graphs of different sizes."""
    keys = ["label", "data", "key", "feats", "time", "interv", "number_of_nodes"]
    batch = {key:[item[key] for item in batch if key in item] for key in keys}
    new_batch = {}
    for key, val in batch.items():
        if key in ["data", "interv"]:
            max_samples = max([data.shape[0] for data in val])
            max_nodes = max([data.shape[1] for data in val])
            padded_data = []
            for data in val:
                curr_samples = min(max_samples, data.shape[0])
                curr_nodes = min(max_nodes, data.shape[1])
                step = max(1, data.shape[0] // curr_samples)
                padded = np.zeros((max_samples, max_nodes))
                padded[:curr_samples, :curr_nodes] = data[::step, :curr_nodes][:curr_samples]
                padded_data.append(padded)
            padded_data = [torch.from_numpy(data).float() for data in padded_data]
            new_batch[key] = torch.stack(padded_data, dim=0)
            new_batch[f"{key}_sample_lengths"] = torch.tensor([data.shape[0] for data in val])
            new_batch[f"{key}_node_lengths"] = torch.tensor([data.shape[1] for data in val])
        elif not torch.is_tensor(val[0]) or val[0].ndim == 0:
            new_batch[key] = default_collate(val)
        # don't collate this; adjust based on train/val
        elif key in ["feats", "label"]:
            # each is [N, N] so require 2D padding
            max_nodes = max([len(v) for v in val])
            for i, v in enumerate(val):
                pad = max_nodes - len(v)
                if pad > 0:
                    val[i] = F.pad(v, (0, pad, 0, pad))
            new_batch[key] = torch.stack(val, dim=0)
        else:
            new_batch[f"{key}_len"] = torch.tensor([len(v) for v in val])
            padded = pad_sequence(val, batch_first=True)
            new_batch[key] = padded
    return new_batch

