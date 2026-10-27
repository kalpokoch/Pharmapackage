"""10-seed ablation for pKa_Basic: final architecture (GraphMPNN, site_readout=False) with the xTB features removed.

Protocol is notebook 02's, unchanged (official train split, seeds 42-51, same optimiser / early stopping / scaling);
the only difference is that the xTB-derived inputs are zeroed so the input dimensions stay identical:
  node: xTB atom charge, xTB attached-H charge sum, xTB attached-H charge max   (the last 4 columns minus is_site)
  edge: xTB Wiberg bond order                                                    (the last column)
The is_site flag is not an xTB quantity and is kept.

Writes results/ablation/pKa_Basic/no_site_no_xtb/{per_seed_preds.npy, per_seed_metrics.csv}.
Usage: python ablation_pka_basic_no_site_no_xtb.py   (GPU recommended; ~85-90 min)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from graph_mpnn import GraphMPNN
from graph_features import mol_to_graph, batch_graphs, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from basic_site import find_basic_site
import data_loader
from metrics import eval_preds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = "pKa_Basic"
VARIANT = "no_site_no_xtb"
OUT = ROOT / "results" / "ablation" / TARGET / VARIANT

HIDDEN_DIM, N_LAYERS, DROPOUT = 128, 4, 0.1
EPOCHS, LR, BATCH, PATIENCE, VAL_FRAC = 300, 5e-4, 64, 40, 0.15
N_SEEDS = 10

XTB_NODE_COLS = [NODE_FEATURE_DIM - 4, NODE_FEATURE_DIM - 3, NODE_FEATURE_DIM - 2]  # is_site (-1) is kept
XTB_EDGE_COL = EDGE_FEATURE_DIM - 1


def strip_xtb(g):
    nf, ef, am, sidx = g
    nf, ef = nf.copy(), ef.copy()
    nf[:, XTB_NODE_COLS] = 0.0
    ef[:, :, XTB_EDGE_COL] = 0.0
    return nf, ef, am, sidx


def to_device_batch(graph_list):
    nodes, adj, adj_mask, node_mask, site_idx = batch_graphs(graph_list)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=DEVICE)
    return t(nodes), t(adj), t(adj_mask), t(node_mask), t(site_idx, torch.int64)


def predict_chunked(model, graphs, batch=BATCH):
    preds = []
    with torch.no_grad():
        for st in range(0, len(graphs), batch):
            nodes, adj, adj_mask, node_mask, site_idx = to_device_batch(graphs[st:st + batch])
            preds.append(model.forward_finetune(nodes, adj, adj_mask, node_mask, site_idx).cpu().numpy())
    return np.concatenate(preds)


def fit_predict_gnn(graphs, y_all, train_pos, test_pos, seed):
    y_mu, y_sd = y_all[train_pos].mean(), y_all[train_pos].std()
    y_s = (y_all - y_mu) / y_sd

    torch.manual_seed(seed)
    model = GraphMPNN(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim=HIDDEN_DIM,
                      n_layers=N_LAYERS, dropout=DROPOUT, site_readout=False).add_finetune_head(1).to(DEVICE)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_pos))
    n_val = max(1, int(VAL_FRAC * len(train_pos)))
    val_pos, fit_pos = train_pos[perm[:n_val]], train_pos[perm[n_val:]]

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        ep_perm = rng.permutation(len(fit_pos))
        for st in range(0, len(fit_pos), BATCH):
            idx = fit_pos[ep_perm[st:st + BATCH]]
            nodes, adj, adj_mask, node_mask, site_idx = to_device_batch([graphs[i] for i in idx])
            pred = model.forward_finetune(nodes, adj, adj_mask, node_mask, site_idx)
            target_t = torch.tensor(y_s[idx], dtype=torch.float32, device=DEVICE)
            loss = nn.functional.smooth_l1_loss(pred, target_t, beta=0.5)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        sched.step()
        model.eval()
        vpred = predict_chunked(model, [graphs[i] for i in val_pos])
        vl = np.abs(vpred - y_s[val_pos]).mean()
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    pred_s = predict_chunked(model, [graphs[i] for i in test_pos])
    return pred_s * y_sd + y_mu


def main():
    print("device:", DEVICE, flush=True)
    graph_ok, val_col, split_col = data_loader.load_pka_official_split(TARGET)
    graphs, keep_rows = [], []
    for i, row in graph_ok.iterrows():
        g = mol_to_graph(row["canon_smiles"], row, site_finder=find_basic_site)
        if g is not None:
            graphs.append(strip_xtb(g))
            keep_rows.append(i)
    graph_ok = graph_ok.iloc[keep_rows].reset_index(drop=True)
    y_all = graph_ok[val_col].values.astype(np.float64)
    train_pos = np.where(graph_ok[split_col].values == "train")[0]
    test_pos = np.where(graph_ok[split_col].values == "test")[0]
    print(f"train={len(train_pos)} test={len(test_pos)}", flush=True)

    # same test rows / order as results/final_models/pKa_Basic
    ref = pd.read_csv(ROOT / "results" / "final_models" / TARGET / "ensemble_predictions.csv")
    assert len(ref) == len(test_pos), (len(ref), len(test_pos))
    assert np.allclose(ref.iloc[:, 1].values.astype(float), y_all[test_pos]), "test row order differs from final model"

    preds, rows, t0 = [], [], time.time()
    for s in range(N_SEEDS):
        seed = 42 + s
        p = fit_predict_gnn(graphs, y_all, train_pos, test_pos, seed)
        preds.append(p)
        m = eval_preds(y_all[test_pos], p, False)
        rows.append({"target": TARGET, "seed_idx": s, "seed": seed, **m, "variant": VARIANT})
        print(f"[seed {seed}] R2={m['R2']:.4f} MAE={m['MAE']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)  # incremental save so a crash keeps finished seeds
        np.save(OUT / "per_seed_preds.partial.npy", np.vstack(preds))

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "per_seed_preds.npy", np.vstack(preds))
    pd.DataFrame(rows).to_csv(OUT / "per_seed_metrics.csv", index=False)
    (OUT / "per_seed_preds.partial.npy").unlink(missing_ok=True)
    ens = eval_preds(y_all[test_pos], np.mean(np.vstack(preds), axis=0), False)
    print("ENSEMBLE:", ens, flush=True)


if __name__ == "__main__":
    main()
