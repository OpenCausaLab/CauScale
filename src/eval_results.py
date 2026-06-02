import os
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
from gadjid import ancestor_aid, oset_aid, parent_aid
import torch
import utils

def to_2d(a):
    # a = [forward_upper_tri, backward_upper_tri]: a[i,j] for i<j is score for i→j, a[j,i] for j→i
    # len(a) = n*(n-1)  =>  n = (1 + sqrt(1 + 4*len(a))) / 2  (positive root of n^2 - n - len(a) = 0)
    n = round((1 + (1 + 4 * len(a)) ** 0.5) / 2)
    assert n * (n - 1) == len(a), f"unexpected input length {len(a)}"
    halfway = len(a) // 2
    mask = np.tri(n, k=-1, dtype=bool).T
    g1, g2 = np.zeros((n, n)), np.zeros((n, n))
    g1[mask] = a[:halfway]
    g2[mask] = a[halfway:]
    return g1 + g2.T, a[:halfway], a[halfway:]


def from_key_to_info(key):
    if "sergio" in key:
        parts = key.split("/")
        n = int(parts[-2][1:])
        e = int(parts[-1].split("_")[-2][1:])
        return n, e, "sergio"
    if "synthetic" in key:
        k = key[len("synthetic_"):].split("_structural")[0]
        n = int(k.split("_")[0][1:])
        e = int(k.split("_")[1][1:])
        return n, e, "synthetic_" + k.split("_", 3)[-1]
    raise ValueError(f"Unknown key: {key}")


def from_path_to_info(fp, key_extracted=False, specific_data="Nan"):
    key = fp.split("/")[-2] if not key_extracted else fp
    if "sergio" in fp or "sergio" in specific_data:
        # key = "sergio/test/nYY/topology_eZ_mode"
        key = "/".join(fp.split("/")[-6:-2])
        n = int(fp.split("/")[-4][1:])
        e = int(fp.split("/")[-3].split("_")[-2][1:])
        return n, e, key
    if "synthetic" in fp or "synthetic" in specific_data or "structural" in fp:
        k = key[5:].rsplit("_", 1)[0]
        key = "synthetic_" + k
        n, e, _ = from_key_to_info("synthetic_" + k)
        return n, e, key
    raise ValueError(f"Unknown dataset: {fp}")

def shd_metric(pred, target):
    """
    Calculates the structural hamming distance

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    shd

    """
    true_labels = target
    predictions = pred

    diff = true_labels - predictions

    rev = (((diff + diff.T) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn + fp + rev

def check_cycle(adj_matrix):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    is_dag = nx.is_directed_acyclic_graph(G)
    has_cycles = not is_dag
    return has_cycles, G


def break_cycles_greedy(pred_bin_2d, pred_scores_2d=None):
    """
    Remove edges from a cyclic predicted graph until it becomes a DAG.

    When ``pred_scores_2d`` (raw model output probabilities) is provided, the
    edge with the *lowest confidence score* in each detected cycle is removed
    first.  This is principled: the model is least certain about that edge, so
    it is the natural candidate for removal.  When scores are unavailable an
    arbitrary cycle edge is removed instead.

    Parameters
    ----------
    pred_bin_2d : np.ndarray, shape (n, n), dtype int
        Binary adjacency matrix that may contain cycles.
    pred_scores_2d : np.ndarray or None, shape (n, n)
        Raw (continuous) model scores for each directed edge.

    Returns
    -------
    dag : np.ndarray, shape (n, n), dtype int
        Acyclic adjacency matrix derived from ``pred_bin_2d``.
    n_removed : int
        Number of edges removed to break all cycles.
    """
    G = nx.from_numpy_array(pred_bin_2d, create_using=nx.DiGraph)
    n_removed = 0
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break
        if pred_scores_2d is not None:
            # Remove the edge in the cycle with the lowest model confidence.
            min_score = float("inf")
            min_edge = cycle[0]
            for u, v in cycle:
                if pred_scores_2d[u, v] < min_score:
                    min_score = pred_scores_2d[u, v]
                    min_edge = (u, v)
        else:
            min_edge = cycle[0]
        G.remove_edge(*min_edge)
        n_removed += 1
    return nx.to_numpy_array(G, dtype=np.int8), n_removed


def compute_aid_metrics(true_2d, pred_dag_2d):
    """
    Compute AID family metrics (Henckel et al. 2024) between the true DAG and
    a *guaranteed-acyclic* predicted graph.

    The caller is responsible for ensuring ``pred_dag_2d`` is a valid DAG
    (use ``break_cycles_greedy`` first if needed).  If the graph is still
    cyclic despite the caller's intent, NaN is returned for all metrics.

    Parameters
    ----------
    true_2d : np.ndarray, shape (n, n)
        Ground-truth DAG adjacency matrix.
    pred_dag_2d : np.ndarray, shape (n, n)
        Binary, acyclic predicted adjacency matrix.

    Returns
    -------
    anc_norm, oset_norm, par_norm : float
        Normalised AID scores (lower = better; 0 = perfect).

    Reference: Henckel, Würtzen & Weichwald, arXiv:2402.08616 (2024).
    """
    Gtrue = true_2d.astype(np.int8)
    Gguess = pred_dag_2d.astype(np.int8)
    direction = "from row to column"
    try:
        anc_norm, _ = ancestor_aid(Gtrue, Gguess, edge_direction=direction)
    except Exception:
        anc_norm = float("nan")
    try:
        oset_norm, _ = oset_aid(Gtrue, Gguess, edge_direction=direction)
    except Exception:
        oset_norm = float("nan")
    try:
        par_norm, _ = parent_aid(Gtrue, Gguess, edge_direction=direction)
    except Exception:
        par_norm = float("nan")
    return anc_norm, oset_norm, par_norm


def _orientation_accuracy(true_f, true_b, eval_f, eval_b):
    """Orientation accuracy on edges that exist in the true graph."""
    true_mask = (true_f + true_b) > 0
    true_direction = true_f[true_mask] > true_b[true_mask]
    pred_forward = eval_f[true_mask] > eval_b[true_mask]
    pred_backward = eval_f[true_mask] < eval_b[true_mask]
    correct = (
        (true_direction == pred_forward)[true_direction].sum()
        + (~true_direction == pred_backward)[~true_direction].sum()
    )
    return correct / len(true_direction)


def compute_additional_metrics(true, pred, threshold=0.5, break_cycles=False):
    """Compute all evaluation metrics for one (true, pred) pair.

    Parameters
    ----------
    true : array-like, 1-D
        Ground-truth adjacency flattened as [forward_upper, backward_upper].
    pred : array-like, 1-D
        Raw model scores in the same format.
    threshold : float
        Binarisation threshold (default 0.5).
    break_cycles : bool
        If True, apply confidence-guided greedy cycle breaking before computing
        SHD, OA, and AID.  AID returns NaN when the graph is cyclic and this
        is False.

    Returns
    -------
    shd, edge_acc, cycle, anc_aid, oset_aid_val, par_aid, n_removed
    """
    true = np.array(true, dtype=int)
    pred = np.array(pred)
    true, true_f, true_b = to_2d(true)
    pred, pred_f, pred_b = to_2d(pred)

    pred_bin = (pred > threshold).astype(int)
    cycle, _ = check_cycle(pred_bin)

    n_removed = 0
    if break_cycles and cycle:
        eval_bin, n_removed = break_cycles_greedy(pred_bin, pred_scores_2d=pred)
    else:
        eval_bin = pred_bin

    shd = shd_metric(eval_bin, true)
    edge_acc = _orientation_accuracy(true_f, true_b, pred_f, pred_b)

    if not cycle or break_cycles:
        anc_aid, oset_aid_val, par_aid = compute_aid_metrics(true, eval_bin)
    else:
        anc_aid = oset_aid_val = par_aid = float("nan")

    return shd, edge_acc, cycle, anc_aid, oset_aid_val, par_aid, n_removed

def evaluate_results(file_or_dict, name, break_cycles=False, verbose=True):
    """Compute metrics per dataset key and save a CSV.

    Parameters
    ----------
    file_or_dict : str or dict
        Path to the pickle file produced by inference, or a pre-loaded results dict.
    name : str
        Output CSV is written to ``{name}.csv``.
    break_cycles : bool
        Apply confidence-guided greedy cycle breaking before computing SHD,
        OA, and AID.  AUC/PRC are unaffected (computed on raw scores).
    """
    results = utils.read_pickle(file_or_dict) if isinstance(file_or_dict, str) else file_or_dict
    per_key = defaultdict(lambda: defaultdict(list))
    for k, v in results.items():
        _, _, k = from_path_to_info(fp=k)
        for metric in ["auc", "mAP"]:
            per_key[k][metric].extend(v[metric])
        for true, pred in zip(v["true"], v["pred"]):
            shd, OA, cycle, anc_aid, oset_aid_val, par_aid, n_removed = \
                compute_additional_metrics(true, pred, break_cycles=break_cycles)
            per_key[k]["cycle"].append(float(cycle))
            per_key[k]["shd"].append(shd)
            per_key[k]["OA"].append(OA)
            per_key[k]["ancestor_aid"].append(anc_aid)
            per_key[k]["oset_aid"].append(oset_aid_val)
            per_key[k]["parent_aid"].append(par_aid)
            per_key[k]["aid_edges_removed"].append(n_removed)
        per_key[k]["time"].extend(v["time"])

    for k, v in per_key.items():
        for metric, vals in v.items():
            if isinstance(vals[0], torch.Tensor):
                vals = [val.numpy() for val in vals]
            if metric in {"ancestor_aid", "oset_aid", "parent_aid", "aid_edges_removed"}:
                per_key[k][metric] = np.nanmean(vals), np.nanstd(vals)
            else:
                per_key[k][metric] = np.mean(vals), np.std(vals)

    all_vals = defaultdict(list)
    for metric in ["auc", "mAP", "shd", "OA", "cycle", "time",
                   "ancestor_aid", "oset_aid", "parent_aid", "aid_edges_removed"]:
        for k, v in sorted(per_key.items()):
            nodes, edges, mechanism = from_key_to_info(k)
            all_vals["nodes"].append(nodes)
            all_vals["edges"].append(edges)
            all_vals["mechanism"].append(mechanism)
            all_vals["metric"].append(metric)
            all_vals["mean"].append(v[metric][0])
            all_vals["std"].append(v[metric][1])
            all_vals["model"].append("ours")

    df = pd.DataFrame.from_dict(all_vals)
    df.to_csv(f'{name}.csv', index=False)
    if verbose:
        print(df.to_string())
        time_by_size = df[df['metric'] == 'time'].groupby(['nodes', 'edges'])['mean'].mean()
        print(time_by_size)
        print(name)

if __name__ == "__main__":
    test_graph_with_cycle = np.array([
        [0, 1, 0],
        [0, 0, 1], 
        [1, 0, 0]
    ])
    
    test_graph_without_cycle = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])
    print(check_cycle(test_graph_with_cycle))
    print(check_cycle(test_graph_without_cycle))