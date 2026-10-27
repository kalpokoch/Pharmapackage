"""Direct MFMN vs conventional MLP on the same descriptors (CL and VDss).

Question: does the seven-factor design help beyond what an ordinary, parameter-matched dense network gets from the
same inputs? The MLP receives exactly MFMN's information -- the shared context (PCA RDKit+Mordred + MACCS) and all seven
factor blocks -- concatenated into one vector, and is trained with MFMN's recipe (AdamW, cosine schedule, SmoothL1,
batch 128, 15% validation split drawn with the same RandomState(seed), early stopping, 10 seeds 42-51, official split).

Variants (hidden width chosen so the trainable parameter count matches the MFMN final model for that target):
  mlp_2L      two hidden layers, no auxiliary loss
  mlp_3L      three hidden layers, no auxiliary loss
  mlp_2L_aux  two hidden layers + the same seven auxiliary-property heads and weight as MFMN (isolates the factor
              architecture from the auxiliary supervision)
Phase "matched": MFMN's hyperparameters (lr 3e-3, wd 1e-2, dropout 0.2, MFMN's aux weight).
Phase "tuned":   the MLP's own lr / weight decay / dropout (/ aux weight) chosen by 5-fold CV on the TRAINING split only
                 (features refit inside each fold); the test split is not used for any choice.

Usage: python mlp_vs_mfmn.py <CL|VDss> <matched|tuned|compare>
Writes results/mlp_vs_mfmn/<target>/<variant>/{per_seed_preds.npy, per_seed_metrics.csv} and, for `compare`,
results/mlp_vs_mfmn/t_mlp_vs_mfmn.csv.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "manuscript"))
from descriptor_features import build_shared_context, build_factor_blocks, build_aux_targets, FACTOR_NAMES
from mfmn import MFMN, MFMN_InteractAux, MFMN_DynGate, count_params
from metrics import eval_log_preds
import data_loader
import analysis  # reuse the paired-bootstrap helpers so the statistics match the existing tables

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = ROOT / "results" / "mlp_vs_mfmn"
EPOCHS, BATCH, VAL_FRAC, PATIENCE = 400, 128, 0.15, 50
SEEDS = list(range(42, 52))
CFG = {  # MFMN final recipe per target (notebooks 03 / 04)
    "VDss": dict(cls=MFMN_InteractAux, split="VD_train_test", log_col="lgVD", aux_w=2.0, lr=3e-3, wd=1e-2),
    "CL": dict(cls=MFMN_DynGate, split="CL_train_test", log_col="lgCL", aux_w=0.8, lr=3e-3, wd=1e-2),
}
VARIANTS = {"mlp_2L": (2, False), "mlp_3L": (3, False), "mlp_2L_aux": (2, True)}


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout, aux_names=()):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 1)
        self.aux = nn.ModuleDict({f: nn.Linear(hidden, 1) for f in aux_names})

    def forward(self, x):
        h = self.trunk(x)
        return self.out(h).squeeze(-1), {f: m(h).squeeze(-1) for f, m in self.aux.items()}


def n_params(in_dim, hidden, n_layers, n_aux):
    p = in_dim * hidden + hidden + (n_layers - 1) * (hidden * hidden + hidden) + (hidden + 1) + n_aux * (hidden + 1)
    return p


def matched_width(in_dim, n_layers, n_aux, target):
    return min(range(4, 400), key=lambda h: abs(n_params(in_dim, h, n_layers, n_aux) - target))


def make_data(train_ids, test_ids):
    ctx_tr, ctx_te = build_shared_context(train_ids, test_ids)
    bl_tr, bl_te = build_factor_blocks(train_ids, test_ids)
    Xtr = np.hstack([ctx_tr] + [bl_tr[f] for f in FACTOR_NAMES]).astype(np.float32)
    Xte = np.hstack([ctx_te] + [bl_te[f] for f in FACTOR_NAMES]).astype(np.float32)
    raw = build_aux_targets(train_ids)
    mu = {f: np.nanmean(raw[f]) for f in FACTOR_NAMES}
    sd = {f: (np.nanstd(raw[f]) or 1.0) for f in FACTOR_NAMES}
    aux = {f: np.where(np.isnan(raw[f]), 0.0, (raw[f] - mu[f]) / sd[f]).astype(np.float32) for f in FACTOR_NAMES}
    mask = {f: (~np.isnan(raw[f])).astype(np.float32) for f in FACTOR_NAMES}
    return Xtr, Xte, aux, mask


def fit_predict(Xtr, y_log, aux, mask, Xte, seed, n_layers, use_aux, hp, target_params):
    """Same loop as notebooks 03/04, with an MLP in place of MFMN. Returns predictions in log10 units."""
    y_mu, y_sd = y_log.mean(), y_log.std()
    y_s = ((y_log - y_mu) / y_sd).astype(np.float32)
    T = lambda a: torch.tensor(a, device=DEVICE)
    n_all = len(y_s)
    rng = np.random.RandomState(seed)
    perm0 = rng.permutation(n_all)
    n_val = max(1, int(VAL_FRAC * n_all))
    val_i, fit_i = perm0[:n_val], perm0[n_val:]

    names = FACTOR_NAMES if use_aux else ()
    hidden = matched_width(Xtr.shape[1], n_layers, len(names), target_params)
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], hidden, n_layers, hp["dropout"], names).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    Xf, Xv = T(Xtr[fit_i]), T(Xtr[val_i])
    yf, yv = T(y_s[fit_i]), T(y_s[val_i])
    auxf = {f: T(aux[f][fit_i]) for f in names}
    maskf = {f: T(mask[f][fit_i]) for f in names}
    n_fit = len(fit_i)
    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_fit, device=DEVICE)
        for st in range(0, n_fit, BATCH):
            b = perm[st:st + BATCH]
            opt.zero_grad()
            out, ap = model(Xf[b])
            loss = nn.functional.smooth_l1_loss(out, yf[b], beta=0.5)
            for f in names:
                m = maskf[f][b]
                if m.sum() > 0:
                    loss = loss + hp["aux_w"] * (((ap[f] - auxf[f][b]) ** 2) * m).sum() / m.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.l1_loss(model(Xv)[0], yv).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(T(Xte))[0].cpu().numpy()
    return pred_s * y_sd + y_mu, hidden, count_params(model)


def load_target(target):
    c = CFG[target]
    _, _, sub = data_loader.load_full_pool(target)
    tr = sub[sub[c["split"]] == "train"]
    te = sub[sub[c["split"]] == "test"]
    ref = pd.read_csv(ROOT / "results" / "final_models" / target / "ensemble_predictions.csv")
    assert np.allclose(ref.y_true.values, te[c["log_col"]].values), "test row order differs from final model"
    return tr["row_idx"].values, tr[c["log_col"]].values.astype(np.float32), te["row_idx"].values, te[c["log_col"]].values


def mfmn_param_target(target, Xtr_dims):
    ctx_dim, dims = Xtr_dims
    return count_params(CFG[target]["cls"](ctx_dim, dims, FACTOR_NAMES, targets=[target]))


def run_variants(target, hp_for, tag):
    """hp_for(variant) -> hyperparameter dict. Trains 10 seeds per variant on the official split and saves."""
    tr_ids, y_tr, te_ids, y_te = load_target(target)
    Xtr, Xte, aux, mask = make_data(tr_ids, te_ids)
    ctx_dim = build_shared_context(tr_ids, te_ids)[0].shape[1]
    dims = {f: v.shape[1] for f, v in build_factor_blocks(tr_ids, te_ids)[0].items()}
    tgt_params = mfmn_param_target(target, (ctx_dim, dims))
    print(f"[{target}] MFMN final params = {tgt_params}; input dim = {Xtr.shape[1]}", flush=True)
    for var, (n_layers, use_aux) in VARIANTS.items():
        hp = hp_for(var)
        if hp is None:
            continue
        name = f"{tag}_{var}"
        preds, rows, t0 = [], [], time.time()
        for s in SEEDS:
            p, hidden, npar = fit_predict(Xtr, y_tr, aux, mask, Xte, s, n_layers, use_aux, hp, tgt_params)
            preds.append(p)
            m = eval_log_preds(y_te, p)
            rows.append({"target": target, "seed": s, **m, "variant": name, "hidden": hidden, "params": npar, **hp})
        d = OUT / target / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "per_seed_preds.npy", np.vstack(preds))
        pd.DataFrame(rows).to_csv(d / "per_seed_metrics.csv", index=False)
        e = eval_log_preds(y_te, np.mean(preds, axis=0))
        print(f"[{target}] {name}: hidden={hidden} params={npar} (target {tgt_params}) hp={hp} "
              f"ENS GMFE={e['GMFE']:.3f} w2f={e['within_2fold']:.3f} R2={e['R2']:.3f} MAE={e['MAE']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)


def tune(target, var):
    """5-fold CV on the training split; features rebuilt inside each fold. Returns best hp dict."""
    n_layers, use_aux = VARIANTS[var]
    tr_ids, y_tr, te_ids, _ = load_target(target)
    ctx_dim = build_shared_context(tr_ids, te_ids)[0].shape[1]
    dims = {f: v.shape[1] for f, v in build_factor_blocks(tr_ids, te_ids)[0].items()}
    tgt_params = mfmn_param_target(target, (ctx_dim, dims))
    folds = np.array_split(np.random.RandomState(0).permutation(len(tr_ids)), 5)
    fold_data = []
    for k in range(5):
        va = folds[k]
        fi = np.concatenate([folds[j] for j in range(5) if j != k])
        Xf, Xv, aux, mask = make_data(tr_ids[fi], tr_ids[va])
        fold_data.append((Xf, Xv, aux, mask, y_tr[fi], y_tr[va]))
    grid = [dict(lr=lr, wd=wd, dropout=dp, aux_w=aw)
            for lr, wd, dp, aw in itertools.product(
                [1e-3, 3e-3], [1e-3, 1e-2], [0.1, 0.2, 0.3], ([0.8, 2.0] if use_aux else [0.0]))]
    res = []
    for hp in grid:
        oof = []
        for Xf, Xv, aux, mask, yf, yv in fold_data:
            p, _, _ = fit_predict(Xf, yf, aux, mask, Xv, 42, n_layers, use_aux, hp, tgt_params)
            oof.append(np.abs(p - yv))
        mae = float(np.concatenate(oof).mean())
        res.append({**hp, "cv_mae": mae})
        print(f"[{target}] tune {var} {hp} cv_mae={mae:.4f}", flush=True)
    res = pd.DataFrame(res).sort_values("cv_mae")
    d = OUT / target
    d.mkdir(parents=True, exist_ok=True)
    res.to_csv(d / f"tuning_{var}.csv", index=False)
    best = res.iloc[0]
    return dict(lr=float(best.lr), wd=float(best.wd), dropout=float(best.dropout), aux_w=float(best.aux_w))


def compare(target):
    log, names = True, analysis.METRICS[target]
    ref = pd.read_csv(ROOT / "results" / "final_models" / target / "ensemble_predictions.csv")
    y = ref.y_true.to_numpy(float)
    fin = np.load(ROOT / "results" / "final_models" / target / "per_seed_preds.npy").astype(float)
    idx = analysis.boot_idx(len(y))
    bf = analysis.boot_metrics(y, fin.mean(0), idx, names, log)
    ens_f = {m: v[0] for m, v in analysis.metric_matrix(y, fin.mean(0)[None], names, log).items()}
    rows = []
    cands = [("MFMN base", ROOT / "results" / "ablation" / target / "mfmn_base" / "per_seed_preds.npy", None)]
    for d in sorted(p for p in (OUT / target).glob("*_mlp_*") if p.is_dir()):
        cands.append((d.name, d / "per_seed_preds.npy", d / "per_seed_metrics.csv"))
    for name, path, mpath in cands:
        v = np.load(path).astype(float)
        pv = v.mean(0)
        em = {m: x[0] for m, x in analysis.metric_matrix(y, pv[None], names, log).items()}
        ps = analysis.metric_matrix(y, v, names, log)
        bv = analysis.boot_metrics(y, pv, idx, names, log)
        extra = {}
        if mpath is not None:
            mm = pd.read_csv(mpath)
            extra = {"hidden": int(mm.hidden.iloc[0]), "params": int(mm.params.iloc[0]),
                     "lr": mm.lr.iloc[0], "wd": mm.wd.iloc[0], "dropout": mm.dropout.iloc[0], "aux_w": mm.aux_w.iloc[0]}
        for m in names:
            lo, hi = analysis.ci(bf[m] - bv[m])  # MFMN final minus this variant
            fav = (lo > 0) if analysis.HIGHER[m] else (hi < 0)
            unfav = (hi < 0) if analysis.HIGHER[m] else (lo > 0)
            rows.append(dict(target=target, variant=name, metric=m, ensemble=em[m], seed_mean=ps[m].mean(),
                             seed_sd=ps[m].std(ddof=1), mfmn_final=ens_f[m], diff_mfmn_minus_variant=ens_f[m] - em[m],
                             diff_ci_lo=lo, diff_ci_hi=hi,
                             verdict="MFMN better (CI excludes 0)" if fav else (
                                 "variant better (CI excludes 0)" if unfav else "CI includes 0"), **extra))
    df = pd.DataFrame(rows)
    out = OUT / f"t_mlp_vs_mfmn_{target}.csv"
    df.to_csv(out, index=False)
    print(df[df.metric.isin(["GMFE", "within_2fold", "R2", "MAE"])][
        ["variant", "metric", "ensemble", "seed_mean", "seed_sd", "diff_mfmn_minus_variant", "diff_ci_lo", "diff_ci_hi",
         "verdict"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    target, phase = sys.argv[1], sys.argv[2]
    base = dict(lr=CFG[target]["lr"], wd=CFG[target]["wd"], dropout=0.2, aux_w=CFG[target]["aux_w"])
    if phase == "matched":
        run_variants(target, lambda v: dict(base) if VARIANTS[v][1] else {**base, "aux_w": 0.0}, "matched")
    elif phase == "tuned":
        best = {v: tune(target, v) for v in ("mlp_2L", "mlp_2L_aux")}
        json.dump(best, open(OUT / target / "tuned_hparams.json", "w"), indent=1)
        run_variants(target, lambda v: best.get(v), "tuned")
    elif phase == "compare":
        compare(target)
