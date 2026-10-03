"""Compute all numbers for the Results section from the saved 10-seed predictions in ../results/.
Bootstrap: compound-level resampling with replacement, B=10,000, percentile 95% CI, fixed RNG seed.
Paired comparisons resample the same compounds for both models."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
RES = OUT.parent / "results"
FIN = RES / "final_models"
BAS = RES / "baselines"
ABL = RES / "ablation"
B, RNG_SEED = 10_000, 2026

TARGETS = ["pKa_Acidic", "pKa_Basic", "CL", "VDss", "Fu"]
LOG_TARGETS = {"CL", "VDss", "Fu"}
METRICS = {t: (["R2", "MAE", "RMSE", "GMFE", "within_2fold"] if t in LOG_TARGETS
               else ["R2", "MAE", "RMSE", "within_1_pKa_unit"]) for t in TARGETS}
HIGHER = {"R2": True, "MAE": False, "RMSE": False, "GMFE": False, "within_2fold": True, "within_1_pKa_unit": True}
BASELINES = {"RandomForest": "Random Forest", "XGBoost": "XGBoost", "SVM": "SVM"}
# 10-seed ablation variants (slug = folder under results/ablation/<target>/, label used in the manuscript)
ABLATION = {
    "CL": [("mfmn_base", "MFMN (base)"), ("interactaux", "MFMN-InteractAux")],
    "VDss": [("mfmn_base", "MFMN (base)"), ("dyngate", "MFMN-DynGate")],
    "Fu": [("dyngate_unweighted", "MFMN-DynGate, no loss weight")],
    "pKa_Acidic": [("no_xtb", "w/o xTB features"), ("no_edge_gating", "w/o edge gating"),
                   ("no_site_readout", "w/o site readout")],
    # pKa_Basic final = GraphMPNN without the site readout; the full GraphMPNN and its ablations are compared against it
    "pKa_Basic": [("full_graphmpnn", "GraphMPNN (full)"), ("no_xtb", "full, w/o xTB features"),
                  ("no_edge_gating", "full, w/o edge gating")],
}

def metric_matrix(y, P_, names, log):
    """y: [n]; P_: [k, n] predictions; returns dict metric -> [k]."""
    err = P_ - y[None, :]
    ss_res = (err ** 2).sum(1)
    ss_tot = ((y - y.mean()) ** 2).sum()
    out = {"R2": 1 - ss_res / ss_tot, "MAE": np.abs(err).mean(1), "RMSE": np.sqrt((err ** 2).mean(1))}
    if log:
        out["GMFE"] = 10 ** np.abs(err).mean(1)
        out["within_2fold"] = (np.abs(err) <= np.log10(2)).mean(1)
    else:
        out["within_1_pKa_unit"] = (np.abs(err) <= 1).mean(1)
    return {m: out[m] for m in names}


def boot_idx(n, seed=RNG_SEED):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def boot_metrics(y, p, idx, names, log):
    """Bootstrap distribution of each metric for one prediction vector p."""
    res = {m: np.empty(B) for m in names}
    for s in range(0, B, 1000):
        ii = idx[s:s + 1000]
        yy, pp = y[ii], p[ii]
        err = pp - yy
        ss_tot = ((yy - yy.mean(1, keepdims=True)) ** 2).sum(1)
        vals = {"R2": 1 - (err ** 2).sum(1) / ss_tot, "MAE": np.abs(err).mean(1), "RMSE": np.sqrt((err ** 2).mean(1))}
        if log:
            vals["GMFE"] = 10 ** np.abs(err).mean(1)
            vals["within_2fold"] = (np.abs(err) <= np.log10(2)).mean(1)
        else:
            vals["within_1_pKa_unit"] = (np.abs(err) <= 1).mean(1)
        for m in names:
            res[m][s:s + 1000] = vals[m]
    return res


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


rows_main, rows_paper, rows_base, rows_abl = [], [], [], []
fig_data = {}
for t in TARGETS:
    log = t in LOG_TARGETS
    names = METRICS[t]
    f = pd.read_csv(FIN / t / "ensemble_predictions.csv")
    y = f.y_true.to_numpy(float)
    p_ens = f.y_pred_ensemble.to_numpy(float)
    seeds = np.load(FIN / t / "per_seed_preds.npy").astype(float)
    assert seeds.shape == (10, len(y)) and np.allclose(seeds.mean(0), p_ens, atol=1e-5), t
    summ = pd.read_csv(FIN / t / "ensemble_summary.csv").set_index("metric")
    ens = {m: v[0] for m, v in metric_matrix(y, p_ens[None], names, log).items()}
    per_seed = metric_matrix(y, seeds, names, log)
    idx = boot_idx(len(y))
    bd = boot_metrics(y, p_ens, idx, names, log)
    fig_data[t] = {"y": y.tolist(), "p": p_ens.tolist()}
    for m in names:
        # cross-check against the pipeline's own summary
        if m in summ.index:
            assert abs(summ.loc[m, "ensemble"] - ens[m]) < 1e-6, (t, m)
        lo, hi = ci(bd[m])
        rows_main.append(dict(target=t, metric=m, n_test=len(y), ensemble=ens[m], ci_lo=lo, ci_hi=hi,
                              seed_mean=per_seed[m].mean(), seed_sd=per_seed[m].std(ddof=1)))
        pap = summ.loc[m, "paper"] if m in summ.index else np.nan
        if pd.notna(pap):
            better = ens[m] > pap if HIGHER[m] else ens[m] < pap
            rows_paper.append(dict(target=t, metric=m, ensemble=ens[m], ci_lo=lo, ci_hi=hi, paper=pap,
                                   delta=ens[m] - pap, ensemble_better=bool(better),
                                   paper_inside_ci=bool(lo <= pap <= hi),
                                   seed_mean=per_seed[m].mean(), seed_mean_better=bool(
                                       per_seed[m].mean() > pap if HIGHER[m] else per_seed[m].mean() < pap)))
    # baselines
    for key, label in BASELINES.items():
        b = pd.read_csv(BAS / t / key / "ensemble_predictions.csv")
        assert (b.id.astype(str).values == f.id.astype(str).values).all() and np.allclose(b.y_true, y)
        pb = b.y_pred_ensemble.to_numpy(float)
        bm = {m: v[0] for m, v in metric_matrix(y, pb[None], names, log).items()}
        bb = boot_metrics(y, pb, idx, names, log)
        psm = pd.read_csv(BAS / t / key / "per_seed_metrics.csv")
        for m in names:
            d = bd[m] - bb[m]  # proposed - baseline, same resampled compounds
            lo, hi = ci(d)
            fav = (lo > 0) if HIGHER[m] else (hi < 0)
            unfav = (hi < 0) if HIGHER[m] else (lo > 0)
            vlo, vhi = ci(bb[m])
            rows_base.append(dict(target=t, model=label, metric=m, value=bm[m], value_ci_lo=vlo, value_ci_hi=vhi, proposed=ens[m],
                                  diff_proposed_minus_baseline=ens[m] - bm[m], diff_ci_lo=lo, diff_ci_hi=hi,
                                  verdict="proposed better (CI excludes 0)" if fav else (
                                      "baseline better (CI excludes 0)" if unfav else "CI includes 0"),
                                  n_seeds=len(psm),
                                  seed_sd=psm[m].std(ddof=1) if (m in psm and len(psm) > 1) else np.nan))
    # ablation (10 seeds, same test rows as the final model)
    for slug, label in ABLATION[t]:
        vs = np.load(ABL / t / slug / "per_seed_preds.npy").astype(float)
        assert vs.shape == seeds.shape
        pv = vs.mean(0)
        vm = {m: v[0] for m, v in metric_matrix(y, pv[None], names, log).items()}
        vps = metric_matrix(y, vs, names, log)
        vb = boot_metrics(y, pv, idx, names, log)
        for m in names:
            lo, hi = ci(bd[m] - vb[m])
            fav = (lo > 0) if HIGHER[m] else (hi < 0)
            unfav = (hi < 0) if HIGHER[m] else (lo > 0)
            rows_abl.append(dict(target=t, variant=label, metric=m, ensemble=vm[m],
                                 seed_mean=vps[m].mean(), seed_sd=vps[m].std(ddof=1), final=ens[m],
                                 diff_final_minus_variant=ens[m] - vm[m], diff_ci_lo=lo, diff_ci_hi=hi,
                                 verdict="final better (CI excludes 0)" if fav else (
                                     "variant better (CI excludes 0)" if unfav else "CI includes 0")))
    if t in ("CL", "VDss"):
        fig_data[t]["ablation_seeds"] = {lab: metric_matrix(y, np.load(ABL / t / s / "per_seed_preds.npy").astype(float),
                                                            names, log)["GMFE"].tolist() for s, lab in ABLATION[t]}
        fig_data[t]["ablation_seeds"]["final"] = per_seed["GMFE"].tolist()

pd.DataFrame(rows_main).to_csv(OUT / "t_main.csv", index=False)
pd.DataFrame(rows_paper).to_csv(OUT / "t_paper.csv", index=False)
pd.DataFrame(rows_base).to_csv(OUT / "t_baselines.csv", index=False)
pd.DataFrame(rows_abl).to_csv(OUT / "t_ablation.csv", index=False)
json.dump(fig_data, open(OUT / "fig_data.json", "w"))
print("done")
