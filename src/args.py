import os
from pathlib import Path
import yaml
import argparse
import torch
from utils import printt

def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

def get_parser():
    parser = argparse.ArgumentParser("CauScale: Neural Causal Discovery at Scale")

    # ======== Setup ========
    parser.add_argument("--debug",
                        action="store_true",
                        help="use small dataset and run fast dev mode (for debugging)")
    parser.add_argument("--seed",
                        type=int, default=0,
                        help="random seed")
    parser.add_argument("--no_tqdm",
                        action="store_true",
                        help="disable progress bar (useful when running as background job)")
    parser.add_argument("--status",
                    type=str, default="train",
                    help="run mode: 'train' or 'inference'")
    parser.add_argument("--config_file",
                        type=str, default=None,
                        help="YAML file")
                        
    # ======== output ========
    parser.add_argument("--args_file",
                        type=str, default="args.yaml",
                        help="dump arguments for reproducibility")
    parser.add_argument("--save_pred",
                        action="store_true",
                        help="save predictions on test set")
    parser.add_argument("--keep_raw_pkl",
                    action="store_true",
                    help="keep intermediate raw predictions file")
    parser.add_argument("--results_prefix",
                        type=str, default="results",
                        help="path prefix for output files")
    parser.add_argument("--save_path",
                        type=str, default="",
                        help="root to model checkpoints")
    parser.add_argument("--output_name",
                    type=str, default=None,
                    help="output subdirectory name under save_path")

    # ======== data ========
    parser.add_argument("--inference_data_file",
                    type=str, default="",
                    help="inference data file")
    parser.add_argument("--data_file",
                        type=str, default="",
                        help="path to training data CSV file")
    parser.add_argument("--normalize_data",
                        action="store_true",
                        help="normalize data")
    parser.add_argument("--num_workers",
                        type=int, default=0,
                        help="data loader workers")
    parser.add_argument("--batch_size",
                        type=int, default=16,
                        help="number of graphs per batch")
    parser.add_argument("--sample_size",
                    type=int, default=None,
                    help="number of observations to sample per graph")
    parser.add_argument("--load_data_while_use",
                    action="store_true",
                    help="load data on-the-fly instead of preloading into memory")
    parser.add_argument("--feature_sample_size",
                        type=int, default=500,
                        help="number of samples used to compute the precision matrix feature") # TODO: you may make it larger
    parser.add_argument("--feature_map",
                    type=str, default="invcov",
                    choices=["invcov", "zero", "noised_invcov"],
                    help="graph prior feature: 'invcov' (precision matrix), 'zero' (no prior, ablation), 'noised_invcov' (noised precision matrix)")
    parser.add_argument("--invcov_noise_level",
                        type=float, default=0.0,
                        help=(
                            "relative noise level injected into the precision matrix "
                            "when feature_map=noised_invcov. "
                            "noise std = invcov_noise_level * std(precision_matrix). "
                            "0.0 = clean precision matrix (no noise)."
                        ))

    # ====== training ======
    parser.add_argument("--log_frequency",
                        type=int, default=5,
                        help="log metrics every [n] batches")
    parser.add_argument("--epochs",
                        type=int, default=200,
                        help="max epochs to train")
    parser.add_argument("--min_epochs",
                        type=int, default=0,
                        help="min epochs to train")
    parser.add_argument("--patience",
                        type=int, default=50,
                        help="lack of validation improvement for [n] epochs")
    parser.add_argument("--metric",
                        type=str, default="Val/loss",
                        help="validation metric to monitor for early stopping and checkpointing")
    parser.add_argument("--accumulate_batches",
                        type=int, default=1,
                        help="accumulate gradient")
    parser.add_argument("--lr",
                        type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--weight_decay",
                        type=float, default=1e-6,
                        help="L2 regularization weight")
    parser.add_argument("--limit_train_batches",
                    type=int, default=200,
                    help="max number of batches per training epoch")
    parser.add_argument("--limit_val_batches",
                    type=int, default=50,
                    help="max number of batches per validation epoch")

    # ======== Hardware =======
    parser.add_argument("--gpu",
                        type=int, default=0,
                        help="GPU id")
    parser.add_argument("--num_gpu",
                        type=int, default=1,
                        help="number of GPUs")
    parser.add_argument("--num_nodes",
                        type=int, default=1,
                        help="number of machines for distributed training")

    # ======== model =======
    parser.add_argument("--checkpoint_path",
                        type=str, default="",
                        help="checkpoint path for fine-tuning or inference")
    parser.add_argument("--n_heads",
                        type=int,
                        default=8,
                        help="number of attention heads")
    parser.add_argument("--embed_dim",
                        type=int,
                        default=64,
                        help="transformer embedding dimension")
    parser.add_argument("--ffn_embed_dim",
                        type=int,
                        default=512,
                        help="transformer FFN hidden size")
    parser.add_argument("--transformer_num_layers",
                        type=int,
                        default=4,
                        help="number of 2d transformer blocks")
    parser.add_argument("--dropout",
                        type=float, default=0.1,
                        help="dropout probability")
    parser.add_argument("--use_self_defined_mask",
                    action="store_true",
                    help="use self-defined mask for symmetric") # TODO: remove this after validated
    parser.add_argument("--scale_graph_rows",
                    action="store_true",
                    help="enable sequence-length-aware row attention scaling in the graph stream")
    parser.add_argument("--scale_graph_cols",
                    action="store_true",
                    help="enable sequence-length-aware column attention scaling in the graph stream")
    parser.add_argument("--scale_data_cols",
                    action="store_true",
                    help="enable sequence-length-aware column attention scaling in the data stream")
    parser.add_argument("--attn_shape",
                    type=str, default="hnij", choices=["hnij", "rhnij"],
                    help="attention weight layout: 'hnij' aggregates across samples, 'rhnij' keeps per-sample weights")
    parser.add_argument("--head_dim",
                    type=int, default=-1,
                    help="attention head dimension; -1 defaults to embed_dim // n_heads")
    parser.add_argument("--disable_reduction_unit",
                        action="store_true",
                        help="disable sample reduction unit (ablation)")
    # ======== post-processing =======
    parser.add_argument("--break_cycles",
                        action="store_true",
                        help=(
                            "if set, apply confidence-guided greedy cycle breaking "
                            "to the thresholded prediction before computing ALL "
                            "evaluation metrics (SHD, OA, AID). "
                            "When not set, metrics are computed on the raw "
                            "thresholded prediction; AID returns NaN for cyclic graphs."
                        ))
    return parser


def parse_args():
    # First pass: get config_file only
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            config = yaml.safe_load(f)
        flat = {}
        def _flatten(d):
            for k, v in d.items():
                if type(v) is dict:
                    _flatten(v)
                else:
                    flat[k] = v
        _flatten(config)
        parser.set_defaults(**flat)

    # Second pass: CLI args override YAML defaults
    args = parser.parse_args()

    if args.config_file is not None:
        for k, yaml_val in flat.items():
            cli_val = getattr(args, k, None)
            if cli_val != yaml_val:
                printt(f"\033[33mWARNING: --{k} overridden by command line: {yaml_val} -> {cli_val}\033[0m")

    process_args(args)
    return args


def process_args(args):
    # prepend output root
    if args.status == "train":
        args.args_file = os.path.join(args.save_path, args.output_name, args.args_file)
        if not os.path.exists(os.path.join(args.save_path, args.output_name)):
            os.makedirs(os.path.join(args.save_path, args.output_name), exist_ok=True)
    else:
        results_dir = Path(args.results_prefix).parent
        args.args_file = str(results_dir / args.args_file)
        results_dir.mkdir(parents=True, exist_ok=True)

