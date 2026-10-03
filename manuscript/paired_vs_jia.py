"""Paired head-to-head against Jia et al.'s own per-compound predictions (106 compounds).

The published benchmark's per-compound predictions for all five endpoints exist in sheet
`106_cmp_test_set` of the paper's Supporting Information, so on those 106 compounds the two
models can be compared with a paired test rather than against a reported point estimate.
The 106 compounds are in the official test split of every endpoint, so neither model trained
on them. This is the only subset on which a paired comparison is possible: the SI carries no
per-compound predictions for the remaining test compounds.

Reuses the bootstrap of analysis.py (B = 10,000, same RNG seed, paired resampling) and the
house style of figures.py. Writes manuscript/t_paired_vs_jia.csv and fig3_paired_vs_jia.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import downstream as ds                                     # noqa: E402
import analysis                                             # noqa: E402
import figures                                              # noqa: E402
import matplotlib.pyplot as plt                             # noqa: E402

ENDPOINTS = ["pKa_Acidic", "pKa_Basic", "CL", "VDss", "Fu"]
LOG_ENDPOINTS = {"CL", "VDss", "Fu"}
HEADLINE = {"pKa_Acidic": "MAE", "pKa_Basic": "MAE", "CL": "GMFE", "VDss": "GMFE", "Fu": "GMFE"}


def main():
    paired = ds.load_paired_predictions()
    idx = analysis.boot_idx(ds.N_TEST)
    assert idx.shape == (analysis.B, ds.N_TEST)

    rows, per_compound = [], []
    for ep in ENDPOINTS:
        df, seeds = paired[ep]
        log = ep in LOG_ENDPOINTS
        names = analysis.METRICS[ep]
        y = df.y_obs.to_numpy(float)
        preds = {"jia": df.y_jia.to_numpy(float), "ours": df.y_ours.to_numpy(float)}

        point = {a: {m: v[0] for m, v in analysis.metric_matrix(y, p[None], names, log).items()}
                 for a, p in preds.items()}
        boot = {a: analysis.boot_metrics(y, p, idx, names, log) for a, p in preds.items()}
        seed_vals = analysis.metric_matrix(y, seeds, names, log)   # our 10 seeds, one metric each

        for m in names:
            lo, hi = analysis.ci(boot["ours"][m] - boot["jia"][m])
            v = ds.verdict(lo, hi, analysis.HIGHER[m])
            rows.append(dict(
                target=ep, metric=m, n_test=ds.N_TEST,
                jia=point["jia"][m], jia_ci_lo=analysis.ci(boot["jia"][m])[0], jia_ci_hi=analysis.ci(boot["jia"][m])[1],
                ours=point["ours"][m], ours_ci_lo=analysis.ci(boot["ours"][m])[0], ours_ci_hi=analysis.ci(boot["ours"][m])[1],
                diff_ours_minus_jia=point["ours"][m] - point["jia"][m], diff_ci_lo=lo, diff_ci_hi=hi,
                verdict=v, seed_mean=seed_vals[m].mean(), seed_sd=seed_vals[m].std(ddof=1)))

        err = pd.DataFrame({
            "target": ep, "PubChem_CID": df.PubChem_CID, "Name": df.Name,
            "y_obs": y, "y_jia": preds["jia"], "y_ours": preds["ours"],
            "abs_err_jia": np.abs(preds["jia"] - y), "abs_err_ours": np.abs(preds["ours"] - y)})
        per_compound.append(err)

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "t_paired_vs_jia.csv", index=False)
    out_dir = ROOT / "results" / "paired_vs_jia"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(per_compound, ignore_index=True).to_csv(out_dir / "per_compound_errors.csv", index=False)
    out.to_csv(out_dir / "paired_summary.csv", index=False)

    make_figure(pd.concat(per_compound, ignore_index=True), out)

    print(f"\nPaired head-to-head vs Jia et al., n = {ds.N_TEST} compounds (official test split of every endpoint)\n")
    print(f"{'endpoint':11} {'metric':18} {'Jia':>8} {'ours':>8} {'diff':>9}  {'95% CI':>18}   verdict")
    for _, r in out.iterrows():
        print(f"{r.target:11} {r.metric:18} {r.jia:8.3f} {r.ours:8.3f} {r.diff_ours_minus_jia:+9.3f}  "
              f"[{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}]   {r.verdict}")
    res = out[out.verdict != "CI includes 0"]
    print(f"\n{len(res)} of {len(out)} metric-endpoint pairs are resolved; "
          f"all in favour of: {sorted(set(res.verdict))}")
    print("wrote", HERE / "t_paired_vs_jia.csv", "and", HERE / "fig3_paired_vs_jia.png")


def make_figure(per, summary):
    """Fig. 6: per-compound absolute error, Jia et al. vs this work, one panel per endpoint."""
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.95))
    for ax, ep in zip(axes, ENDPOINTS):
        d = per[per.target == ep]
        x, y = d.abs_err_jia.to_numpy(float), d.abs_err_ours.to_numpy(float)
        hi = float(max(x.max(), y.max())) * 1.08
        ax.plot([0, hi], [0, hi], color=figures.INK, linewidth=0.8, zorder=2)
        better = y < x
        ax.scatter(x[better], y[better], s=7, color=figures.BLUE, alpha=0.75, linewidths=0, zorder=3)
        ax.scatter(x[~better], y[~better], s=7, color=figures.ORANGE, alpha=0.75, linewidths=0, zorder=3)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_title(f"{figures.TITLE[ep]}\n{int(better.sum())}/{len(d)} closer", color=figures.INK, fontsize=7)
        ax.set_xlabel("|error| Jia et al.", fontsize=6.5)
        figures.grid(ax)
    axes[0].set_ylabel("|error| this work", fontsize=6.5)
    fig.text(0.5, -0.06, "Per-compound absolute error on the 106 compounds for which the benchmark publishes "
                         "predictions; pKa in pKa units, CL/VDss/Fu in log$_{10}$ units. Points below the diagonal "
                         "favour this work.", ha="center", fontsize=6, color=figures.MUTED)
    fig.tight_layout(w_pad=0.7)
    fig.savefig(HERE / "fig3_paired_vs_jia.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
