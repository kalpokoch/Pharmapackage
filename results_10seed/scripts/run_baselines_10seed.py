#!/usr/bin/env python
"""
run_baselines_10seed.py -- UNTUNED baselines (library defaults) on the official train/test split of every target, then the comparison
with the retrained proposed models in retrain_10seed/finals/.

Protocol (same as baselines_10seed_ensemble_vs_proposed.ipynb): features = ablation_study/baselines/features/<target>__<view>.npz
(CL/VDss/Fu 'model_inputs', pKa 'desc'); StandardScaler fit on the training split; RandomForestRegressor(random_state=seed) and
XGBRegressor(random_state=seed) for seeds 42..51; SVR() has no randomness so it is fit once.  No hyperparameter is set (n_jobs only
controls parallelism and does not change the result).  The test split is only used for prediction.

  python retrain_10seed/run_baselines_10seed.py --stage train      # fit + save every seed (resumable)
  python retrain_10seed/run_baselines_10seed.py --stage compare    # tables + figures vs the retrained finals (needs finals/<target>)
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ablation_10seed" / "_deps"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "ablation_10seed"))
sys.path.insert(0, str(ROOT / "mfmn_benchmark_package" / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

OUT = HERE / "baselines"
FEAT = ROOT / "mfmn_benchmark_package" / "ablation_study" / "baselines" / "features"
TARGETS = ["CL", "VDss", "Fu", "pKa_Acidic", "pKa_Basic"]
VIEW = {"CL": "model_inputs", "VDss": "model_inputs", "Fu": "model_inputs", "pKa_Acidic": "desc", "pKa_Basic": "desc"}
IS_LOG = {t: t in ("CL", "VDss", "Fu") for t in TARGETS}
YFILE = {"CL": "y_test_log.npy", "VDss": "y_test_log.npy", "Fu": "y_test.npy", "pKa_Acidic": "y_test.npy", "pKa_Basic": "y_test.npy"}
MODELS = {"RandomForest": "Random Forest", "XGBoost": "XGBoost", "SVM": "SVM"}
SEEDS = list(range(42, 52))
LOGM = ["R2", "MAE", "RMSE", "GMFE", "within_2fold", "within_3fold", "within_5fold"]
PKAM = ["R2", "MAE", "RMSE", "within_1_pKa_unit", "within_2_pKa_unit"]
HIGHER = lambda m: m == "R2" or m.startswith("within_")
metric_cols = lambda t: LOGM if IS_LOG[t] else PKAM


def load(t):
    f = np.load(FEAT / f"{t}__{VIEW[t]}.npz", allow_pickle=True)
    return f["X_train"], f["y_train"], f["X_test"], f["y_test"], f["ids_test"].astype(str)


def fit_one(model, t, seed):
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr, Xte, yte, _ = load(t)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    t0 = time.time()
    if model == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(random_state=seed)               # library defaults
    elif model == "XGBoost":
        from xgboost import XGBRegressor
        m = XGBRegressor(random_state=seed, n_jobs=1)              # library defaults (n_jobs = parallelism only)
    else:
        from sklearn.svm import SVR
        m = SVR()                                                  # library defaults
    p = np.asarray(m.fit(Xtr, ytr).predict(Xte)).ravel()
    return p, time.time() - t0


def save_model_outputs(t, model, seeds, preds, secs):
    from run_finals import save_parity            # same publication style as the finals
    import run_finals as F
    import run_ablation_10seed as R
    d = OUT / t / model
    d.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, y, ids = load(t)
    fm_ = R.metric_fn(t)
    cols = metric_cols(t)
    np.save(d / "per_seed_preds.npy", np.vstack(preds))
    np.save(d / YFILE[t], y)
    (d / "seeds_done.json").write_text(json.dumps(seeds))
    rows, cum = [], []
    for s, p, sec in zip(seeds, preds, secs):
        sd = d / f"seed_{s}"
        sd.mkdir(exist_ok=True)
        m = fm_(y, p)
        df = pd.DataFrame({"id": ids, "y_true": y, "y_pred": p, "error_pred_minus_true": p - y, "abs_error": np.abs(p - y)})
        if IS_LOG[t]:
            df["obs_linear"], df["pred_linear"] = 10 ** y, 10 ** p
        df.to_csv(sd / "predictions.csv", index=False)
        pd.DataFrame([dict(target=t, model=MODELS[model], seed=s, n_test=len(y), **m, seconds=sec)]).to_csv(sd / "metrics.csv", index=False)
        rows.append(dict(target=t, model=MODELS[model], seed_idx=s - 42, seed=s, **{c: m[c] for c in cols}, seconds=sec))
    pd.DataFrame(rows).to_csv(d / "per_seed_metrics.csv", index=False)
    P = np.vstack(preds)
    for k in range(1, len(P) + 1):
        m = fm_(y, P[:k].mean(0))
        cum.append(dict(target=t, model=MODELS[model], k_seeds=k, **{c: m[c] for c in cols}))
    pd.DataFrame(cum).to_csv(d / "cumulative_ensemble_metrics.csv", index=False)
    ens = P.mean(0)
    pd.DataFrame({"id": ids, "y_true": y, "y_pred_ensemble": ens, "abs_error": np.abs(ens - y)}).to_csv(d / "ensemble_predictions.csv", index=False)
    F.style()
    m_ens = fm_(y, ens)
    save_parity(str(d / f"parity_ensemble"), y, ens, t, m_ens,
                f"{MODELS[model]}, {len(seeds)}-seed ensemble" if len(seeds) > 1 else f"{MODELS[model]} (single fit)")


def train(args):
    import run_ablation_10seed as R
    R.metric_fn("CL")
    jobs = []
    for t in (args.targets or TARGETS):
        for model in MODELS:
            seeds = [SEEDS[0]] if model == "SVM" else SEEDS
            jobs += [(t, model, s) for s in seeds]
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(fit_one)(m, t, s) for t, m, s in jobs)
    by = {}
    for (t, m, s), (p, sec) in zip(jobs, res):
        by.setdefault((t, m), []).append((s, p, sec))
    for (t, m), lst in by.items():
        lst.sort(key=lambda a: a[0])
        save_model_outputs(t, m, [a[0] for a in lst], [a[1] for a in lst], [a[2] for a in lst])
        print("saved", t, m, len(lst), "seeds", flush=True)
    print(f"train stage done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ comparison
def metric_values(y, p, t):
    e = p - y
    sse, sst = (e ** 2).sum(-1), ((y - y.mean(-1, keepdims=True)) ** 2).sum(-1)
    out = {"R2": 1 - sse / sst, "MAE": np.abs(e).mean(-1), "RMSE": np.sqrt((e ** 2).mean(-1))}
    if IS_LOG[t]:
        out["GMFE"] = 10 ** np.abs(e).mean(-1)
        for k in (2, 3, 5):
            out[f"within_{k}fold"] = (np.abs(e) <= np.log10(k)).mean(-1)
    else:
        for k in (1, 2):
            out[f"within_{k}_pKa_unit"] = (np.abs(e) <= k).mean(-1)
    return out


def compare(args, B=10_000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run_finals as F
    import run_ablation_10seed as R
    R.metric_fn("CL")
    F.style()
    tabs = OUT / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    PAPER = F.paper
    summ, comp = [], []
    for t in (args.targets or TARGETS):
        fdir = HERE / "finals" / t
        if not (fdir / "seeds_done.json").exists() or len(json.loads((fdir / "seeds_done.json").read_text())) < 10:
            print(f"skip {t}: retrained final model not complete yet")
            continue
        y = np.load(fdir / YFILE[t])
        prop = np.load(fdir / "per_seed_preds.npy")
        cols = metric_cols(t)
        entries = {"Proposed": prop}
        for model in MODELS:
            d = OUT / t / model
            if (d / "per_seed_preds.npy").exists():
                assert np.array_equal(np.load(d / YFILE[t]), y), f"{t}/{model}: y_test differs from the proposed model's"
                entries[MODELS[model]] = np.load(d / "per_seed_preds.npy")
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(y), size=(B, len(y)))
        # bootstrap draws (vectorised in chunks)
        draws = {name: {c: [] for c in cols} for name in entries}
        for a in range(0, B, 2000):
            ii = idx[a:a + 2000]
            for name, P in entries.items():
                v = metric_values(y[ii], P.mean(0)[ii], t)
                for c in cols:
                    draws[name][c].append(v[c])
        draws = {n: {c: np.concatenate(v) for c, v in dd.items()} for n, dd in draws.items()}
        for name, P in entries.items():
            per = pd.DataFrame([{c: metric_values(y, p, t)[c] for c in cols} for p in P])
            ens = metric_values(y, P.mean(0), t)
            r = dict(target=t, model=name, n_seeds=len(P), n_test=len(y))
            for c in cols:
                r[f"{c}_ens"] = float(ens[c]); r[f"{c}_seed_mean"] = float(per[c].mean())
                r[f"{c}_seed_sd"] = float(per[c].std(ddof=1)) if len(P) > 1 else np.nan
            summ.append(r)
            if name == "Proposed":
                continue
            for c in cols:
                dd = draws[name][c] - draws["Proposed"][c]
                better_prop = (dd < 0) if HIGHER(c) else (dd > 0)      # proposed model better than the baseline
                comp.append(dict(target=t, baseline=name, metric=c, higher_is_better=HIGHER(c),
                                 baseline_ens=float(ens[c]), proposed_ens=float(metric_values(y, prop.mean(0), t)[c]),
                                 diff_baseline_minus_proposed=float(ens[c] - metric_values(y, prop.mean(0), t)[c]),
                                 ci95_lo=float(np.percentile(dd, 2.5)), ci95_hi=float(np.percentile(dd, 97.5)),
                                 ci_flag="excludes_zero" if (np.percentile(dd, 2.5) > 0 or np.percentile(dd, 97.5) < 0) else "overlaps_zero",
                                 p_proposed_better=float(better_prop.mean()), boot_B=B,
                                 baseline_seed_mean=float(per[c].mean()),
                                 baseline_seed_sd=float(per[c].std(ddof=1)) if len(P) > 1 else np.nan))
        # figure: one panel per metric; bars = ensemble, dots = single seeds
        show = ["R2", "MAE", "RMSE", "GMFE", "within_2fold"] if IS_LOG[t] else ["R2", "MAE", "RMSE", "within_2_pKa_unit"]
        order = ["SVM", "Random Forest", "XGBoost", "Proposed"]
        colors = {"SVM": "#999999", "Random Forest": "#56B4E9", "XGBoost": "#E69F00", "Proposed": "#0072B2"}
        fig, axes = plt.subplots(1, len(show), figsize=(1.7 * len(show) + 0.8, 2.7))
        pp = PAPER(t)
        for ax, c in zip(axes, show):
            for i, name in enumerate(order):
                if name not in entries:
                    continue
                P = entries[name]
                ens = metric_values(y, P.mean(0), t)[c]
                ax.bar(i, ens, color=colors[name], width=0.65, zorder=2)
                if len(P) > 1:
                    pv = [metric_values(y, p, t)[c] for p in P]
                    ax.scatter(np.full(len(pv), i) + np.linspace(-0.18, 0.18, len(pv)), pv, s=4, color="k", zorder=3, linewidths=0)
            key = {"within_2fold": "within_2fold"}.get(c, c)
            if c in pp:
                ax.axhline(pp[c], color="#D55E00", ls="--", lw=0.9, zorder=4)
            ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace("Random Forest", "RF").replace("Proposed", "Prop.") for o in order], rotation=45, ha="right")
            ax.set_title({"R2": r"R$^2$", "within_2fold": "within 2-fold", "within_2_pKa_unit": r"within 2 pK$_a$"}.get(c, c), fontsize=9)
            if HIGHER(c) and c != "R2":
                ax.set_ylim(0, 1.0)
        fig.suptitle(f"{t}: untuned baselines vs proposed (bars: ensemble; dots: single seeds; dashed: published benchmark)", fontsize=7.5)
        fig.savefig(tabs / f"comparison_{t}.png", dpi=600)
        fig.savefig(tabs / f"comparison_{t}.pdf")
        plt.close(fig)
    pd.DataFrame(summ).to_csv(tabs / "baselines_and_proposed_summary.csv", index=False)
    pd.DataFrame(comp).to_csv(tabs / "baseline_vs_proposed_bootstrap.csv", index=False)
    # paper-ready table: ensemble R2 / MAE / RMSE (+GMFE) per target x model
    s = pd.DataFrame(summ)
    keep = ["target", "model", "n_seeds"] + [f"{m}_ens" for m in ("R2", "MAE", "RMSE", "GMFE", "within_2fold", "within_2_pKa_unit") if f"{m}_ens" in s.columns]
    s[keep].round(4).to_csv(tabs / "baselines_vs_proposed_table.csv", index=False)
    print("comparison written to", tabs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "compare"], required=True)
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (train if args.stage == "train" else compare)(args)


if __name__ == "__main__":
    main()
