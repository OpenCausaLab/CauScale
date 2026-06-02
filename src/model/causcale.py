"""
CauScale: Neural Causal Discovery at Scale.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.classification import BinaryAccuracy

from .modules import TwoStreamEncoder, TopLayer


def get_params_groups(model, args):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [
        {"params": regularized, "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": not_regularized, "weight_decay": 0., "lr": args.lr},
    ]

class CauScale(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.encoder = TwoStreamEncoder(args)
        self.top_layer = TopLayer(
                embed_dim=self.args.embed_dim * 2,
                output_dim=3
            )

        # validation meters
        self.auroc = BinaryAUROC()
        self.auprc = BinaryAveragePrecision()
        self.acc = BinaryAccuracy()
        self.save_hyperparameters()

    def forward(self, batch):
        """
        Used on predict_dataloader
        """
        start = time.time()  # keep track of GPU time
        output = self.encoder(batch)
        results = self.compute_metrics_per_graph(output, batch,
                save_preds=True)
        end = time.time()  # keep track of GPU time
        # run with batch_size=1 for accurate timing
        results["time"] = batch["time"] + (end - start)
        return results

    def symmetrize(self, output, batch, reduce=True):
        """
            P(i->j) = 1 - P(j->i)
            reduce: bool  True to reduce batch, False to preserve graphs
        """
        # symmetrize output
        # select upper and lower triangular, skip diagonal
        # NOTE: tril is NOT same as triu of transpose
        output = output.permute(0, 3, 1, 2)  # (B, N, N, d) -> (B, d, N, N): move embed_dim to front for triu
        forward_edge = torch.triu(output, 1)
        backward_edge = torch.triu(output.transpose(2, 3), 1)
        forward_edge = forward_edge.permute(0, 2, 3, 1)   # (B, d, N, N) -> (B, N, N, d)
        backward_edge = backward_edge.permute(0, 2, 3, 1) # (B, d, N, N) -> (B, N, N, d)
        # select away padding as well as wrong direction
        # backward_mask should == forward_mask
        if ("use_self_defined_mask" in self.args and self.args.use_self_defined_mask): # TODO: 
            forward_mask = []
            for i in range(len(batch["number_of_nodes"])):
                node_num = batch["number_of_nodes"][i]
                single_forward_mask = torch.ones(node_num, node_num, dtype=torch.bool)
                single_forward_mask = torch.triu(single_forward_mask, 1).bool()
                single_forward_mask = torch.cat([single_forward_mask, torch.zeros(node_num, output[i].shape[1] - node_num).bool()], dim=1)
                single_forward_mask = torch.cat([single_forward_mask, torch.zeros(output[i].shape[1] - node_num, output[i].shape[1]).bool()], dim=0)
                forward_mask.append(single_forward_mask)
            # print("in inference")
            forward_mask = torch.stack(forward_mask, dim=0)
            backward_mask = forward_mask
        else:
            # # new padding method
            forward_mask = forward_edge[...,0] != 0.  # remove embed_dim dimension
            backward_mask = backward_edge[...,0] != 0.
        if reduce:
            forward_edge = forward_edge[forward_mask]
            backward_edge = backward_edge[backward_mask]
            assert len(forward_edge) == len(backward_edge)
            logits = torch.cat([forward_edge, backward_edge], dim=1)
            edge_pred = self.top_layer(logits)  # (B * N*(N-1)/2, 2d) -> (B * N*(N-1)/2, 3)
        else:
            edge_pred = []
            for i in range(len(forward_edge)):
                forward_i = forward_edge[i][forward_mask[i]]
                backward_i = backward_edge[i][backward_mask[i]]
                logits = torch.cat([forward_i, backward_i], dim=-1)
                edge_pred.append(self.top_layer(logits))

        # get label corresponding to same entries
        label = batch["label"]
        forward_label = torch.triu(label, 1)
        backward_label = torch.triu(label.transpose(1, 2), 1)
        if reduce:
            forward_label = forward_label[forward_mask]
            backward_label = backward_label[backward_mask] * 2
            # forward/backward should be mutually exclusive
            joint_label = forward_label + backward_label  # {0, 1, 2}
        else:
            joint_label = []
            for i in range(len(forward_edge)):
                forward_i = forward_label[i][forward_mask[i]]
                backward_i = backward_label[i][backward_mask[i]]
                joint_label.append(torch.cat([forward_i, backward_i]))

        return edge_pred, joint_label


    def compute_losses(self, output, batch):
        losses = {}
        pred, true = self.symmetrize(output, batch)            
        losses["loss"] = F.cross_entropy(pred, true)
        return losses

    def compute_metrics_per_graph(self, output, batch, save_preds=False):
        """
            Split up individual graphs from batch
        """
        auroc, auprc, acc = [], [], []
        if save_preds:
            pred_list, true_list = [], []
            feat_list, graph_list = [], []

        # do not reduce over batch
        pred, true = self.symmetrize(output, batch, reduce=False)
        for i, (p, t, f, gt) in enumerate(zip(pred, true, batch["feats"], batch["label"])):
            # select P(forward) and P(backward), remove P(no edge)
            # corresponding to dim=1, dim=2, and dim=0
            p = F.softmax(p, dim=-1)[:,1:].t().reshape(-1)
            p = p.cpu()
            t = t.cpu()
            assert p.shape == t.shape
            auroc.append(self.auroc(p, t).item())
            auprc.append(self.auprc(p, t).item())
            acc.append(self.acc(p, t).item())
            if any(np.isnan(auroc)) or any(np.isnan(auprc)) or any(np.isnan(acc)):
                print("NAN in validation step")
                print(batch["label"])
                print(output)
                print(p, t)
                exit()
            if save_preds:
                pred_list.append(p.tolist())
                true_list.append(t.tolist())
                feat_list.append(f)
                graph_list.append(gt)

        outputs = {}
        outputs["auroc"] = auroc
        outputs["auprc"] = auprc
        outputs["acc"] = acc
        outputs["key"] = batch["key"]
        if save_preds:
            outputs["pred"] = pred_list
            outputs["true"] = true_list
            outputs["feats"] = feat_list
            outputs["labels"] = graph_list
        return outputs

    def training_step(self, batch, batch_idx):
        output = self.encoder(batch)
        losses = self.compute_losses(output, batch)
        for k, v in losses.items():
            if not torch.is_tensor(v) or v.numel() == 1:
                self.log(f"Train/{k}", v.item(),
                    batch_size=len(output), sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        # try:
        output = self.encoder(batch)
        losses = self.compute_losses(output, batch)
        # except torch.cuda.OutOfMemoryError:
        #     print("cuda out of memory")
        #     torch.cuda.empty_cache()
        #     return
        for k, v in losses.items():
            if not torch.is_tensor(v) or v.numel() == 1:
                self.log(f"Val/{k}", v.item(),
                    batch_size=len(output), sync_dist=True)
        # metrics
        results = self.compute_metrics_per_graph(output, batch,
                save_preds=False)
        for k, v in results.items():
            if k != "key":
                self.log(f"Val/{k}", np.mean(v),
                    batch_size=len(output), sync_dist=True)

    def configure_optimizers(self):
        param_groups = get_params_groups(self, self.args)
        optimizer = AdamW(param_groups)
        return [optimizer]