#!/usr/bin/env python
"""
run_finals.py -- retrain the five FINAL models with seeds 42..51 into retrain_10seed/finals/<target>/ and save, for every seed,
a CSV of per-compound predictions, a metrics CSV and a publication-style parity plot; then 10-seed aggregates and a summary figure.

Training code is the original code that produced the stored arrays (10seed_training_and_analysis.ipynb cells 1,3 (CL/VDss),
1,3,6 + data prep of 7 (Fu), 1,9 + data prep of 10 (pKa)), executed verbatim through ablation_10seed/run_ablation_10seed.py
(see that file / manifest.json).  Nothing is read from the old final arrays except in the optional retrained_vs_stored.csv.

  python retrain_10seed/run_finals.py --workers 8 --pka-concurrency 2          # everything, resumable
  python retrain_10seed/run_finals.py --finalize-only                          # rebuild aggregates + figures from saved seeds
"""
import argparse, csv, datetime, json, logging, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ablation_10seed"))
import run_ablation_10seed as R          # sets thread env, adds ablation_10seed/_deps (rdkit) to sys.path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FIN = HERE / "finals"
TARGETS = R.TARGETS
SEEDS = R.SEEDS_ALL
IS_LOG = {t: t in R.LOG_TARGETS for t in TARGETS}
FINAL_MODEL = {
    "CL": ("final_clvd", "MFMN_DynGate (aux_weight 0.8)"),
    "VDss": ("final_clvd", "MFMN_InteractAux (aux_weight 2.0)"),
    "Fu": ("final_fu", "MFMN_DynGate + 1.25x low-fu loss weight"),
    "pKa_Acidic": ("pka_none", "GraphMPNN"),
    "pKa_Basic": ("pka_none", "GraphMPNN"),
}
UNITS = {"CL": r"log$_{10}$ CL (L h$^{-1}$ kg$^{-1}$)", "VDss": r"log$_{10}$ VD$_{ss}$ (L kg$^{-1}$)",
         "Fu": r"log$_{10}$ F$_u$", "pKa_Acidic": r"acidic pK$_a$", "pKa_Basic": r"basic pK$_a$"}
C_ENS, C_SEED, C_PAPER = "#0072B2", "#7F7F7F", "#D55E00"
log = logging.getLogger("finals")


def style():
    cands = ["Times New Roman", "Liberation Serif", "STIXGeneral"]
    avail = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in cands if c in avail), "DejaVu Serif")
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": [chosen], "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })
    return chosen


def paper(t):
    from metrics import PAPER            # package src is on sys.path after R.metric_fn() has been called
    return PAPER[t]


# ----------------------------------------------------------------------------------------------------- plots
def draw_parity(ax, y, p, t, m, tag):
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    pad = 0.04 * (hi - lo)
    lims = [lo - pad, hi + pad]
    band = float(np.log10(2)) if IS_LOG[t] else 1.0
    ax.scatter(y, p, s=6, alpha=0.55, c=C_ENS, edgecolors="none", rasterized=True)
    ax.plot(lims, lims, "k-", lw=0.8, label="identity")
    ax.plot(lims, [l + band for l in lims], "k:", lw=0.6, label=("2-fold" if IS_LOG[t] else r"$\pm$1 pK$_a$"))
    ax.plot(lims, [l - band for l in lims], "k:", lw=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed " + UNITS[t]); ax.set_ylabel("Predicted " + UNITS[t])
    lines = [rf"R$^2$ = {m['R2']:.3f}", f"MAE = {m['MAE']:.3f}", f"RMSE = {m['RMSE']:.3f}"]
    if IS_LOG[t]:
        lines.append(f"GMFE = {m['GMFE']:.3f}")
    lines.append(rf"$n$ = {len(y)}")
    ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=7.5)
    ax.text(0.97, 0.03, tag, transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), frameon=False, handlelength=1.6)


def save_parity(base, y, p, t, m, tag, dpi=600):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    draw_parity(ax, y, p, t, m, tag)
    fig.savefig(f"{base}.png", dpi=dpi)
    fig.savefig(f"{base}.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------- saving
def tdir(t):
    return FIN / t


def seed_done(t):
    raw = tdir(t) / "raw"
    return sorted(int(p.stem.split("_")[1]) for p in raw.glob("seed_*.npy")) if raw.exists() else []


def save_seed_outputs(t, seed, res):
    import numpy as np
    d = tdir(t)
    sd = d / f"seed_{seed}"
    sd.mkdir(parents=True, exist_ok=True)
    (d / "raw").mkdir(exist_ok=True)
    raw = d / "raw" / f"seed_{seed}.npy"
    if raw.exists():
        raise RuntimeError(f"refusing to overwrite completed seed {raw}")
    y, p = res["y_test"], res["pred"]
    ids = res.get("ids")
    if ids is None:
        ids = np.arange(len(y)).astype(str)
    fm_ = R.metric_fn(t)
    m = fm_(y, p)
    df = pd.DataFrame({"id": ids, "y_true": y, "y_pred": p, "error_pred_minus_true": p - y, "abs_error": np.abs(p - y)})
    if IS_LOG[t]:
        df["obs_linear"], df["pred_linear"] = 10 ** y, 10 ** p
    df.to_csv(sd / "predictions.csv", index=False)
    pd.DataFrame([dict(target=t, model=FINAL_MODEL[t][1], seed=seed, n_test=len(y), **m, seconds=res["seconds"],
                       device=res["device"])]).to_csv(sd / "metrics.csv", index=False)
    save_parity(str(sd / f"parity_seed_{seed}"), y, p, t, m, f"seed {seed}")
    np.save(d / "raw" / f"seed_{seed}.npy", p)
    np.save(d / R.FINAL_FILES[t], y)
    seeds = seed_done(t)
    preds = np.vstack([np.load(d / "raw" / f"seed_{s}.npy") for s in seeds])
    np.save(d / "per_seed_preds.npy", preds)
    (d / "seeds_done.json").write_text(json.dumps(seeds))
    R.write_metrics(d, t, FINAL_MODEL[t][1], seeds, preds, y)
    pd.DataFrame({"id": ids}).to_csv(d / "test_ids.csv", index=False)
    f = d / "run_log.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["seed", "seconds", "device", "status", "timestamp"])
        w.writerow([seed, f"{res['seconds']:.1f}", res["device"], "ok", R.now()])
    return m


def finalize(t):
    """10-seed aggregates + ensemble predictions + summary figure for a target (needs >=1 seed)."""
    seeds = seed_done(t)
    if not seeds:
        return
    d = tdir(t)
    preds = np.load(d / "per_seed_preds.npy")
    y = np.load(d / R.FINAL_FILES[t])
    ids = pd.read_csv(d / "test_ids.csv")["id"].astype(str).values
    fm_ = R.metric_fn(t)
    cols = R.metric_cols(t)
    ens = preds.mean(0)
    m_ens = fm_(y, ens)
    per = pd.DataFrame([fm_(y, p) for p in preds])[cols]
    ep = pd.DataFrame({"id": ids, "y_true": y, "y_pred_ensemble": ens, "y_pred_seed_sd": preds.std(0, ddof=1) if len(seeds) > 1 else np.nan})
    ep["abs_error"] = np.abs(ens - y)
    ep.to_csv(d / "ensemble_predictions.csv", index=False)
    pp = paper(t)
    rows = []
    for c in cols:
        r = dict(target=t, model=FINAL_MODEL[t][1], metric=c, n_seeds=len(seeds), ensemble=m_ens[c], seed_mean=per[c].mean(),
                 seed_sd=per[c].std(ddof=1) if len(seeds) > 1 else np.nan, seed_min=per[c].min(), seed_max=per[c].max())
        pk = {"within_2fold": "within_2fold"}.get(c, c)
        r["paper"] = pp.get(pk, np.nan) if c in ("R2", "MAE", "RMSE", "GMFE", "within_2fold") else np.nan
        rows.append(r)
    pd.DataFrame(rows).to_csv(d / "ensemble_summary.csv", index=False)
    # summary figure: (a) ensemble parity, (b) per-seed R2, (c) per-seed MAE
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    ax = fig.add_subplot(gs[0])
    draw_parity(ax, y, ens, t, m_ens, f"{len(seeds)}-seed ensemble")
    ax.text(-0.28, 1.02, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    for k, (met, lab, panel) in enumerate((("R2", r"R$^2$", "(b)"), ("MAE", "MAE", "(c)"))):
        a = fig.add_subplot(gs[k + 1])
        a.scatter(seeds, per[met].values, s=14, color=C_SEED, zorder=3, label="single seed")
        a.axhline(m_ens[met], color=C_ENS, lw=1.2, label=f"{len(seeds)}-seed ensemble")
        if met in pp:
            a.axhline(pp[met], color=C_PAPER, lw=1.0, ls="--", label="published benchmark")
        a.set_xlabel("Seed"); a.set_ylabel(lab)
        a.set_xticks(seeds[::2] if len(seeds) > 5 else seeds)
        a.text(-0.28, 1.02, panel, transform=a.transAxes, fontsize=9, fontweight="bold")
        if k == 0:
            a.legend(frameon=False, loc="best")
    fig.savefig(d / f"summary_{t}.png", dpi=600)
    fig.savefig(d / f"summary_{t}.pdf")
    plt.close(fig)


def retrained_vs_stored():
    """Compare the retrained finals with the ORIGINAL stored arrays (seed_analysis_results/): shows the environment drift."""
    rows = []
    for t in TARGETS:
        d = tdir(t)
        if not (d / "per_seed_preds.npy").exists():
            continue
        new, y = np.load(d / "per_seed_preds.npy"), np.load(d / R.FINAL_FILES[t])
        old = np.load(R.ROOT / "seed_analysis_results" / t / "per_seed_preds.npy")
        yo = np.load(R.ROOT / "seed_analysis_results" / t / R.FINAL_FILES[t])
        same_y = bool(np.array_equal(y, yo))
        fm_ = R.metric_fn(t)
        n = min(len(new), len(old))
        for c in R.metric_cols(t):
            rn = [fm_(y, p)[c] for p in new[:n]]
            ro = [fm_(yo, p)[c] for p in old[:n]]
            rows.append(dict(target=t, metric=c, n_seeds_compared=n, y_test_identical=same_y,
                             retrained_seed_mean=np.mean(rn), stored_seed_mean=np.mean(ro),
                             retrained_ensemble=fm_(y, new[:n].mean(0))[c], stored_ensemble=fm_(yo, old[:n].mean(0))[c],
                             seed42_retrained=rn[0], seed42_stored=ro[0]))
    pd.DataFrame(rows).to_csv(FIN / "retrained_vs_stored.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pka-concurrency", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()
    FIN.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(HERE / "run_finals.log")])
    log.info("start args=%s", vars(args))
    R.metric_fn("CL")                      # puts package src on sys.path (metrics module)
    chosen = style()
    targets = args.targets or TARGETS
    seeds = R.parse_seeds(args.seeds)
    if not args.finalize_only:
        R.stage_shm()
        todo = [(t, x) for t in targets for x in seeds if x not in seed_done(t)]
        order = {t: i for i, t in enumerate(TARGETS)}
        todo.sort(key=lambda a: (0 if not FINAL_MODEL[a[0]][0].startswith("pka") else 1, a[1], order[a[0]]))
        log.info("%d trainings pending: %s", len(todo), {t: [x for tt, x in todo if tt == t] for t in targets})
        if todo:
            groups = {R.RUNNER_GROUP[FINAL_MODEL[t][0]] for t, _ in todo}
            pools = R.make_pools(args.workers, groups)
            st = dict(ok=0, fail=0)

            def handle(tag, res):
                t, x = tag
                if isinstance(res, Exception):
                    st["fail"] += 1
                    log.error("FAIL %s seed %d: %r\n%s", t, x, res, getattr(res, "tb", ""))
                    return
                try:
                    m = save_seed_outputs(t, x, res)
                    st["ok"] += 1
                    log.info("DONE %s seed %d R2=%.4f MAE=%.4f (%.0fs) [%d ok, %d failed, %d left]", t, x, m["R2"], m["MAE"],
                             res["seconds"], st["ok"], st["fail"], len(todo) - st["ok"] - st["fail"])
                    if len(seed_done(t)) == len(SEEDS):
                        finalize(t)
                except Exception as e:
                    st["fail"] += 1
                    log.error("SAVE FAIL %s seed %d: %r\n%s", t, x, e, traceback.format_exc())

            tasks = [((t, x), FINAL_MODEL[t][0], (t, "final", x, args.device)) for t, x in todo]
            R.run_bounded(pools, tasks, args.pka_concurrency, handle)
            for p in pools.values():
                p.shutdown()
            log.info("training finished: %d ok, %d failed", st["ok"], st["fail"])
    for t in targets:
        finalize(t)
    retrained_vs_stored()
    man = dict(generated=R.now(), seeds=SEEDS, font=chosen, models={t: FINAL_MODEL[t][1] for t in TARGETS},
               runners={t: FINAL_MODEL[t][0] for t in TARGETS},
               sources=dict(CL_VDss="seed_analysis_results/10seed_training_and_analysis.ipynb cells 1,3 (fit_predict_cl_vdss, CONFIGS)",
                            Fu="same notebook cells 1,3,6 (fit_predict_fu, 1.25x low-fu weight) + data prep lines of cell 7",
                            pKa="same notebook cells 1,9 (fit_predict_gnn) + data prep lines of cell 10"),
               sha256={str(p.relative_to(ROOT)): R.sha256(p) for p in R.HASH_FILES if p.exists()})
    (FIN / "manifest.json").write_text(json.dumps(man, indent=2))
    log.info("end")


if __name__ == "__main__":
    main()
