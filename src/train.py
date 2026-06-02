import os
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from args import parse_args
from data import DataModule, InferenceDataModule
from model import CauScale
from utils import printt, get_suffix, save_pickle, set_seed
from eval_results import evaluate_results


def main():
    printt("Train")
    args = parse_args()
    printt(args)
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    set_seed(args.seed)
    printt("Loading model...")
    model = CauScale(args)

    args.save_path = os.path.join(args.save_path, args.output_name)

    # save args
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(args.args_file, "w+") as f:
        yaml.dump(args.__dict__, f)

    # train loop
    mode = "max"
    if "loss" in args.metric:
        mode = "min"

    checkpoint_kwargs = {
        "save_top_k": 1,
        "monitor": args.metric,
        "mode": mode,
        "filename": get_suffix(args.metric),
        "dirpath": args.save_path,
        "save_last": True,
    }
    # checkpoint_path is a PTH to resume training
    cb_checkpoint = ModelCheckpoint(**checkpoint_kwargs)

    cb_earlystop = EarlyStopping(
            monitor=args.metric,
            patience=args.patience,
            mode=mode,
    )
    callbacks = [
            RichProgressBar(),
            cb_checkpoint,
            cb_earlystop,
    ]
    if args.no_tqdm:
        callbacks[0].disable()

    device_ids = [args.gpu + i for i in range(args.num_gpu)]

    trainer_kwargs = {
        "max_epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "accumulate_grad_batches": args.accumulate_batches,
        "gradient_clip_val": 1.,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "callbacks": callbacks,
        "log_every_n_steps": args.log_frequency,
        "fast_dev_run": args.debug,
        "logger": False,
        "devices": device_ids if args.num_gpu > 0 else None,
        "accelerator": "gpu" if args.num_gpu > 0 else "cpu",
        "strategy": "ddp" if args.num_gpu > 0 else None,
        "num_nodes": getattr(args, 'num_nodes', 1),
    }

    printt("Initializing trainer...")
    trainer = pl.Trainer(**trainer_kwargs)
    start_time = datetime.now()

    # data loaders
    printt("Loading data paths...")
    data = DataModule(args)
    print(f"Train batches: {len(data.train_dataloader())}")
    print(f"Val batches: {len(data.val_dataloader())}")

    # restore full training from checkpoint
    fit_kwargs = {}
    if os.path.exists(args.checkpoint_path):
        if dist.is_initialized():
            dist.barrier()
        fit_kwargs["ckpt_path"] = args.checkpoint_path
    trainer.fit(model, data, **fit_kwargs)

    if args.num_gpu <= 1 or not dist.is_initialized() or dist.get_rank() == 0:
        end_time = datetime.now()
        duration = end_time - start_time
        hours = duration.total_seconds() / 3600
        print(f"Execution time: {hours:.4f} hours")
        with open(os.path.join(args.save_path, "info.txt"), 'a', encoding='utf-8') as f:
            f.write(f"start {start_time}, end {end_time}, duration {hours:.4f} h\n")
            f.write(str(args))
            f.write(f"\nbest ckpt path: {cb_checkpoint.best_model_path}")

        # set eval mode
        model.eval()

        if not args.debug:
            best_path = cb_checkpoint.best_model_path
            printt(best_path)

        # run inference on test split
        if not args.debug:
            tester = pl.Trainer(devices=[args.gpu],
                                num_nodes=1,
                                enable_checkpointing=False,
                                logger=False,
                                accelerator="gpu" if args.num_gpu > 0 else "cpu")

            model = model.load_from_checkpoint(best_path)

            if args.inference_data_file:
                args.data_file = args.inference_data_file
                args.batch_size = 1
                data = InferenceDataModule(args)
            results = tester.predict(model, data)
            results_dict = defaultdict(list)
            for batch in results:
                for k, v in batch.items():
                    if type(v) is list:
                        results_dict[k].extend(v)
                    else:
                        results_dict[k].append(v)
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
            evaluate_results(key_to_metrics, args.results_prefix)


if __name__ == "__main__":
    main()

