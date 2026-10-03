"""Measure model complexity: trainable parameters, forward FLOPs and measured training cost.

Writes `manuscript/t_complexity.csv`, consumed by `build_manuscript.py`. Requires torch and
rdkit; the generated CSV is committed so the manuscript rebuilds without them.

FLOPs are counted by `torch.profiler` with `with_flops=True`, which accounts for the matrix
multiplications (including those the GRU cell decomposes into). The elementwise terms it does
not count are negligible here: for the graph model they are O(N^2 H) against an O(N^2 H^2) edge
gate. One forward pass over a single molecule is reported, since the graph model's cost depends
on molecule size while the descriptor models' does not.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import graph_mpnn as gm                                     # noqa: E402
import mfmn                                                 # noqa: E402
from graph_features import EDGE_FEATURE_DIM, NODE_FEATURE_DIM   # noqa: E402

FACTORS = ["size", "lipophilicity", "ionization", "polarity", "binding", "electronic", "structural"]
CLVD_DIMS = dict(size=3, lipophilicity=2, ionization=2, polarity=3, binding=1, electronic=4, structural=4)
FU_DIMS = dict(size=3, lipophilicity=2, ionization=15, polarity=3, binding=13, electronic=6, structural=4)
CLVD_CTX, FU_CTX = 199, 1419


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flops(fn):
    """Forward FLOPs of a zero-argument call, via the profiler's matmul accounting."""
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as pr:
            fn()
    return sum(e.flops for e in pr.key_averages() if e.flops)


def molecule_sizes():
    """Median and maximum heavy-atom count over the labeled pKa pools."""
    g = pd.read_parquet(ROOT / "data" / "pka" / "aux_xtb_graph.parquet", columns=["status", "n_heavy"])
    n = g.loc[g.status == "ok", "n_heavy"].to_numpy(int)
    return int(np.median(n)), int(n.max())


def graph_flops(model, n_atoms):
    x = torch.randn(1, n_atoms, NODE_FEATURE_DIM)
    adj = torch.randn(1, n_atoms, n_atoms, EDGE_FEATURE_DIM)
    am = (torch.rand(1, n_atoms, n_atoms) > 0.5).float()
    nm = torch.ones(1, n_atoms)
    si = torch.tensor([0])
    model.eval()
    return flops(lambda: model.forward_finetune(x, adj, am, nm, si))


def mfmn_flops(model, ctx_dim, dims):
    c = torch.randn(1, ctx_dim)
    fr = {f: torch.randn(1, dims[f]) for f in FACTORS}
    model.eval()
    return flops(lambda: model(c, fr))


def training_cost(target):
    """Measured seconds per seed from the run log shipped with the results."""
    p = ROOT / "results" / "final_models" / target / "run_log.csv"
    if not p.exists():
        return None, None, None
    d = pd.read_csv(p)
    return float(d.seconds.mean()), float(d.seconds.min()), float(d.seconds.max())


def main():
    n_med, n_max = molecule_sizes()
    rows = []

    for target, site in (("pKa_Acidic", True), ("pKa_Basic", False)):
        m = gm.GraphMPNN(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim=128, n_layers=4,
                         dropout=0.1, site_readout=site).add_finetune_head(1)
        mean_s, min_s, max_s = training_cost(target)
        rows.append(dict(target=target,
                         model="GraphMPNN" + ("" if site else " (no site readout)"),
                         family="graph", params=n_params(m),
                         flops_median_mol=graph_flops(m, n_med), flops_max_mol=graph_flops(m, n_max),
                         n_atoms_median=n_med, n_atoms_max=n_max,
                         sec_per_seed_mean=mean_s, sec_per_seed_min=min_s, sec_per_seed_max=max_s))

    for target, cls, ctx_dim, dims in (
            ("CL", mfmn.MFMN_DynGate, CLVD_CTX, CLVD_DIMS),
            ("VDss", mfmn.MFMN_InteractAux, CLVD_CTX, CLVD_DIMS),
            ("Fu", mfmn.MFMN_DynGate, FU_CTX, FU_DIMS)):
        m = cls(ctx_dim, dims, FACTORS, targets=(target,))
        fl = mfmn_flops(m, ctx_dim, dims)
        mean_s, min_s, max_s = training_cost(target)
        rows.append(dict(target=target, model=cls.__name__.replace("_", "-"), family="descriptor",
                         params=n_params(m), flops_median_mol=fl, flops_max_mol=fl,
                         n_atoms_median=np.nan, n_atoms_max=np.nan,
                         sec_per_seed_mean=mean_s, sec_per_seed_min=min_s, sec_per_seed_max=max_s))

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "t_complexity.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[["target", "model", "params", "flops_median_mol", "flops_max_mol", "sec_per_seed_mean"]].to_string(index=False))
    print(f"\nmolecule size: median {n_med}, max {n_max} heavy atoms")
    print("wrote", HERE / "t_complexity.csv")


if __name__ == "__main__":
    main()
