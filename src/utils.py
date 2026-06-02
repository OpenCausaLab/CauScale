import os
import sys
import csv
import pickle
import random

import numpy as np
import torch
from datetime import datetime


def save_pickle(fp, data):
    with open(fp, "wb+") as f:
        pickle.dump(data, f)


def read_pickle(fp):
    with open(fp, "rb") as f:
        data = pickle.load(f)
    return data


def read_csv(fp, fieldnames=None, delimiter=',', str_keys=[]):
    data = []
    with open(fp) as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        # iterate and append
        for item in reader:
            data.append(item)
    return data

# -------- general

def get_timestamp():
    return datetime.now().strftime('%H:%M:%S')


def printt(*args, **kwargs):
    print(get_timestamp(), *args, **kwargs)


def get_suffix(metric):
    suffix = "model_best_"
    suffix = suffix + "{global_step}_{epoch}_{"
    suffix = suffix + metric + ":.3f}_{val_loss:.3f}"
    return suffix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
