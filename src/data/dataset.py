import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from .utils import convert_to_item
from utils import read_csv
from sklearn.covariance import LedoitWolf

def compute_features(x, normalize=False, remove_diag=False):
    """
    Implementation depends on matrix size
    x: (num_vars, num_samples)
    """
    Nan_flag = False
    has_nan = np.isnan(x).any()
    has_inf = np.isinf(x).any()
    if has_nan or has_inf:
        Nan_flag = True
    x = np.nan_to_num(x)
    x = np.clip(x, -1e6, 1e6)

    if x.shape[0] < 100:
        return np.linalg.pinv(np.cov(x), rcond=1e-10), Nan_flag
    lw = LedoitWolf()
    lw.fit(x.T)
    invcovs = lw.get_precision()
    if remove_diag:
        np.fill_diagonal(invcovs, 0)
    if normalize:
        abs_invcovs = np.abs(invcovs)
        max_abs = np.max(abs_invcovs)
        invcovs = abs_invcovs / max_abs
    return invcovs, Nan_flag

def standardize_data(data):
    # input: (num_samples, num_vars)
    mean = data.mean(0, keepdims=True)
    std = data.std(0, keepdims=True)
    return (data - mean) / np.where(std == 0, 1.0, std)

class InterventionalDataset(Dataset):
    def __init__(self, fp_data, fp_graph, fp_regime, args, sample_size):
        """
        fp_data:   path to .npy observation matrix, shape (num_samples, num_vars)
        fp_graph:  path to .npy adjacency matrix, shape (num_vars, num_vars)
        fp_regime: path to .csv or .npy intervention targets (which nodes were intervened per sample)
        """
        super().__init__()

        # --- load data and graph ---
        self.key = fp_graph
        self.sample_size = sample_size
        self.time = 0  # placeholder for later
        try:
            self.data = np.load(fp_data)
        except Exception as e:
            print(f"error on {fp_data}:\n{e}")
            exit()
        self.graph = torch.from_numpy(np.load(fp_graph)).long()
        self.num_vars = self.data.shape[1]
        self.num_edges = self.graph.sum()

        # --- load interventions ---
        # self.interv: (num_samples, num_vars) binary matrix; interv[i, j] = 1 if node j was intervened in sample i
        if fp_regime.endswith('.npy'):
            self.interv = np.load(fp_regime).astype(np.float32)
        else:
            # parse CSV regime file: each line lists intervened node indices (comma-separated)
            # empty line = observational sample
            if fp_regime is not None and len(fp_regime) > 0 and fp_regime != "None":
                with open(fp_regime) as f:
                    lines = [line.strip() for line in f.readlines()]
            else:
                lines = [()] * len(self.data)
            regimes = [tuple(sorted(int(x) for x in line.split(",")))
                    if len(line) > 0 else () for line in lines]
            assert len(regimes) == len(self.data), f"{len(regimes)} {len(self.data)}"
            interv = np.zeros((len(self.data), len(self.graph)))
            for i, regime in enumerate(regimes):
                for node in regime:
                    interv[i, node] = 1
            self.interv = interv.astype(np.float32)

        # --- compute graph prior feature (precision matrix) ---
        if args.status == "inference": # NOTE: kept to reproduce the main experiment
            np.random.seed(42)
            random.seed(42)
            torch.manual_seed(42)
        idxs = np.random.choice(len(self.data), (args.feature_sample_size,),
                replace=False)
        self.feature, NaN_flag = compute_features(self.data[idxs].T, normalize=False, remove_diag=False)
        if NaN_flag:
            print("Warning: NAN in data:", fp_data)
        if args.feature_map == "zero":
            # Ablation: no prior (zero matrix), to measure contribution of the graph prior
            self.feature = np.zeros_like(self.feature) + 0.2
        elif args.feature_map == "noised_invcov":
            # Robustness experiment: noised precision matrix
            # noise_level=0.0 is identical to "invcov"; noise_level=1.0 → noise std = matrix std
            noise_level = getattr(args, "invcov_noise_level", 0.0)
            if noise_level > 0.0:
                noise = np.random.randn(*self.feature.shape).astype(self.feature.dtype)
                noise = (noise + noise.T) / 2.0  # keep symmetry of precision matrix
                scale = float(np.std(self.feature))
                self.feature = self.feature + noise_level * scale * noise
        elif args.feature_map != "invcov":
            raise ValueError(f"Invalid feature map: {args.feature_map}")

        # --- subsample and normalize ---
        step = max(1, self.data.shape[0] // self.sample_size)
        self.data = self.data[::step, :]
        self.interv = self.interv[::step, :]
        if args.normalize_data:
            self.data = standardize_data(self.data)

        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_features(self):
        return self.feature

class MetaDataset(Dataset):
    """
    Dataset of datasets.

    When load_data_while_use=True, each InterventionalDataset is constructed
    on-the-fly in __getitem__ instead of upfront, to avoid loading all graphs
    into memory at once.
    """
    def __init__(self, data_file, args, splits_to_load=None, sample_size=None):
        super().__init__()
        # read raw data
        self.args = args
        self.data_file = data_file
        data_to_load = read_csv(self.data_file)
        self.splits = defaultdict(list)
        self.data = []
        # create individual Dataset objects
        for item in tqdm(data_to_load, ncols=40):
            split = item["split"]
            if splits_to_load is not None and split not in splits_to_load:
                continue
            self.splits[split].append(len(self.data))
            if not self.args.load_data_while_use:
                self.data.append(InterventionalDataset(item["fp_data"],
                                                        item["fp_graph"],
                                                        item["fp_regime"],
                                                        args,
                                                        sample_size))
            else:
                self.data.append((item["fp_data"], item["fp_graph"], item["fp_regime"], args, sample_size))
            if args.debug and len(self.data) > 100:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.args.load_data_while_use:
            dataset = self.data[idx]
        else:
            fp_data, fp_graph, fp_regime, args, sample_size = self.data[idx]
            dataset = InterventionalDataset(fp_data, fp_graph, fp_regime, args, sample_size)
        corrs = dataset.get_features()
        return convert_to_item(dataset, corrs)

