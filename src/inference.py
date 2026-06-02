import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl

from args import parse_args
from data import InferenceDataModule
from model import CauScale
from utils import printt, save_pickle, set_seed
from eval_results import evaluate_results


def main():
    printt("Inference")
    args = parse_args()
    printt(args)
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    # data loaders
    printt("Loading data...")
    data = InferenceDataModule(args)
    set_seed(args.seed)
    printt("Loading model...")
    model = CauScale(args)

    kwargs = {"accelerator": "gpu" if args.gpu >= 0 else "cpu"}
    if args.gpu >= 0:
        kwargs["devices"] = [args.gpu]
    tester = pl.Trainer(num_nodes=1,
                        enable_checkpointing=False,
                        logger=False,
                        **kwargs)

    best_path = args.checkpoint_path
    results = tester.predict(model, data, ckpt_path=best_path)

    # post-process results
    results_dict = defaultdict(list)
    for batch in results:
        for k, v in batch.items():
            if type(v) is list:
                results_dict[k].extend(v)
            else:
                results_dict[k].append(v)
    # organize by data setting
    fields = {"auc": "auroc", "mAP": "auprc", "time": "time",
              "true": "true", "pred": "pred", "feats": "feats", "labels": "labels"}
    key_to_metrics = defaultdict(lambda: defaultdict(list))
    for i, key in enumerate(results_dict["key"]):
        for dst, src in fields.items():
            key_to_metrics[key][dst].append(results_dict[src][i])
    key_to_metrics = dict(key_to_metrics)
    results_dir = Path(args.results_prefix).parent
    if getattr(args, 'keep_raw_pkl', False):
        results_dir.mkdir(parents=True, exist_ok=True)
        save_pickle(args.results_prefix + ".pkl", key_to_metrics)
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluate_results(
        key_to_metrics, args.results_prefix,
        break_cycles=getattr(args, "break_cycles", False),
    )


if __name__ == "__main__":
    main()

