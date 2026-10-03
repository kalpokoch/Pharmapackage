"""Experiment A: analytic exposure propagation (AUC / Cmax) for the 106-compound test set.

Holding a one-compartment IV exposure model fixed, compares three sources of CL and VDss --
observed, Jia et al.'s QSAR predictions, and this work's 10-seed ensembles -- and asks
whether our parameter predictions propagate to smaller fold errors in AUC and Cmax.

Assumptions: one-compartment linear kinetics, dose linearity, 1 mg/kg IV dose. This
quantifies how parameter error propagates into exposure error, not absolute profile
accuracy; its numbers are not comparable with any result obtained by scoring against
observed concentration-time profiles.

Bootstrap (B, RNG seed, paired resampling, verdict vocabulary) is reused from analysis.py.
Writes results/downstream/* and manuscript/{t_downstream.csv,fig5_downstream.png}.
"""
import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import downstream as ds                                     # noqa: E402
import analysis                                             # noqa: E402  (bootstrap + CI helpers)
import figures                                              # noqa: E402  (house rcParams + palette)
import matplotlib.pyplot as plt                             # noqa: E402

OUT_RES = ROOT / "results" / "downstream"
NAMES = ds.METRIC_NAMES
ENDPOINTS = ["AUC", "Cmax"]
ARMS = ["jia", "ours"]
LABEL = {"jia": "Jia et al. QSAR", "ours": "This work (10-seed)"}


def arm_exposures(test, ens, t_hours):
    """AUC and Cmax for each arm, with the parameters each arm supplies."""
    cl = {"observed": test[ds.CL_OBS].to_numpy(float), "jia": test[ds.CL_JIA].to_numpy(float), "ours": ens["CL"]}
    vd = {"observed": test[ds.VD_OBS].to_numpy(float), "jia": test[ds.VD_JIA].to_numpy(float), "ours": ens["VDss"]}
    exposure = {}
    for arm in ("observed", "jia", "ours"):
        auc, cmax = ds.auc_cmax(cl[arm], vd[arm], t_hours)
        exposure[arm] = {"AUC": auc, "Cmax": cmax}
    return cl, vd, exposure


def main():
    OUT_RES.mkdir(parents=True, exist_ok=True)
    ds.fetch_si()
    test = ds.load_test_set()
    ens, seeds = ds.load_ours(test)
    t_hours, n_blank, n_zero = ds.infusion_hours(test)
    cl, vd, exposure = arm_exposures(test, ens, t_hours)
    n = len(test)
    assert n == ds.N_TEST

    # ---------------- per-compound table ----------------
    per = pd.DataFrame({
        "ID_trend": test.ID_trend, "PubChem_CID": test.PubChem_CID, "Name": test.Name_trend,
        "SMILES": test.SMILES, "InfusionTime_min": test[ds.T_MIN],
        "CL_observed": cl["observed"], "CL_jia": cl["jia"], "CL_ours": cl["ours"],
        "VDss_observed": vd["observed"], "VDss_jia": vd["jia"], "VDss_ours": vd["ours"],
    })
    for ep in ENDPOINTS:
        for arm in ("observed", "jia", "ours"):
            per[f"{ep}_{arm}"] = exposure[arm][ep]
        for arm in ARMS:
            per[f"{ep}_folderror_{arm}"] = ds.fold_error(exposure["observed"][ep], exposure[arm][ep])
    per.to_csv(OUT_RES / "exposure_per_compound.csv", index=False)

    # ---------------- metrics, bootstrap CIs, paired differences ----------------
    idx = analysis.boot_idx(n)                       # same B and RNG seed as the rest of the manuscript
    assert idx.shape == (analysis.B, n)
    summary_rows, paired_rows = [], []
    for ep in ENDPOINTS:
        obs_log = np.log10(exposure["observed"][ep])
        boot = {}
        for arm in ARMS:
            pred_log = np.log10(exposure[arm][ep])
            point = ds.exposure_metrics(exposure["observed"][ep], exposure[arm][ep], NAMES)
            boot[arm] = analysis.boot_metrics(obs_log, pred_log, idx, NAMES, log=True)
            # per-seed spread of the `ours` arm
            seed_vals = {m: [] for m in NAMES}
            if arm == "ours":
                for s in range(seeds["CL"].shape[0]):
                    a_s, c_s = ds.auc_cmax(seeds["CL"][s], seeds["VDss"][s], t_hours)
                    ms = ds.exposure_metrics(exposure["observed"][ep], {"AUC": a_s, "Cmax": c_s}[ep], NAMES)
                    for m in NAMES:
                        seed_vals[m].append(ms[m])
            for m in NAMES:
                lo, hi = analysis.ci(boot[arm][m])
                sv = np.array(seed_vals[m], float)
                summary_rows.append(dict(
                    endpoint=ep, arm=arm, metric=m, value=point[m], value_ci_lo=lo, value_ci_hi=hi,
                    seed_mean=sv.mean() if sv.size else np.nan,
                    seed_sd=sv.std(ddof=1) if sv.size > 1 else np.nan,
                    n_seeds=int(sv.size) if sv.size else np.nan))
        for m in NAMES:
            d = boot["ours"][m] - boot["jia"][m]     # paired: same resampled compounds in both arms
            lo, hi = analysis.ci(d)
            point_j = ds.exposure_metrics(exposure["observed"][ep], exposure["jia"][ep], NAMES)[m]
            point_o = ds.exposure_metrics(exposure["observed"][ep], exposure["ours"][ep], NAMES)[m]
            paired_rows.append(dict(
                endpoint=ep, metric=m, jia=point_j, ours=point_o, diff_ours_minus_jia=point_o - point_j,
                diff_ci_lo=lo, diff_ci_hi=hi,
                verdict=ds.verdict(lo, hi, ds.HIGHER_IS_BETTER[m])))
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    summary.to_csv(OUT_RES / "exposure_summary.csv", index=False)
    paired.to_csv(OUT_RES / "exposure_paired.csv", index=False)

    # manuscript-ready table, column convention of the existing t_*.csv files
    t_rows = []
    for _, r in summary.iterrows():
        d = paired[(paired.endpoint == r.endpoint) & (paired.metric == r.metric)].iloc[0]
        t_rows.append(dict(endpoint=r.endpoint, arm=LABEL[r.arm], metric=r.metric, value=r.value,
                           value_ci_lo=r.value_ci_lo, value_ci_hi=r.value_ci_hi, ours=d.ours,
                           diff_ours_minus_jia=d.diff_ours_minus_jia, diff_ci_lo=d.diff_ci_lo,
                           diff_ci_hi=d.diff_ci_hi, verdict=d.verdict, n_test=n,
                           seed_mean=r.seed_mean, seed_sd=r.seed_sd))
    pd.DataFrame(t_rows).to_csv(HERE / "t_downstream.csv", index=False)

    # ---------------- supplementary checks ----------------
    checks = []
    # 1. AUC fold error is identically CL fold error under this model (AUC = D/CL)
    max_dev = float(np.max(np.abs(per.AUC_folderror_ours.to_numpy() - ds.fold_error(cl["observed"], cl["ours"]))))
    max_dev_j = float(np.max(np.abs(per.AUC_folderror_jia.to_numpy() - ds.fold_error(cl["observed"], cl["jia"]))))
    assert max(max_dev, max_dev_j) < 1e-12
    checks.append(f"AUC fold error equals CL fold error exactly (max |deviation| = {max(max_dev, max_dev_j):.2e}); "
                  "the AUC column therefore carries no information beyond the CL column. Cmax depends on CL, V and "
                  "infusion time jointly and is the genuinely new quantity.")

    # 2. infusion-time sensitivity: recompute Cmax treating every compound as a bolus
    bolus = {}
    for arm in ("observed", "jia", "ours"):
        _, cmax_b = ds.auc_cmax(cl[arm], vd[arm], np.zeros(n))
        bolus[arm] = cmax_b
    bolus_rows = []
    for arm in ARMS:
        pt = ds.exposure_metrics(bolus["observed"], bolus[arm], NAMES)
        bolus_rows.append(dict(endpoint="Cmax_bolus", arm=arm, **pt))
    bolus_df = pd.DataFrame(bolus_rows)
    bolus_df.to_csv(OUT_RES / "exposure_cmax_bolus_sensitivity.csv", index=False)
    bd = {r.arm: r for _, r in bolus_df.iterrows()}
    as_is = {r.arm: r for _, r in summary[(summary.endpoint == "Cmax")].pivot(index="arm", columns="metric",
                                                                             values="value").reset_index().iterrows()}
    checks.append(
        "Infusion-time sensitivity (Cmax recomputed as if every compound were an IV bolus): "
        f"GMFE jia {as_is['jia'].GMFE:.3f} -> {bd['jia'].GMFE:.3f}, ours {as_is['ours'].GMFE:.3f} -> "
        f"{bd['ours'].GMFE:.3f}; within-2-fold jia {as_is['jia'].within_2fold:.3f} -> {bd['jia'].within_2fold:.3f}, "
        f"ours {as_is['ours'].within_2fold:.3f} -> {bd['ours'].within_2fold:.3f}.")

    # 3. worst compounds by Cmax fold error under `ours`
    worst = per.nlargest(10, "Cmax_folderror_ours")[[
        "ID_trend", "PubChem_CID", "Name", "InfusionTime_min", "CL_observed", "CL_ours",
        "VDss_observed", "VDss_ours", "Cmax_observed", "Cmax_ours", "Cmax_folderror_ours", "Cmax_folderror_jia"]]
    worst.to_csv(OUT_RES / "exposure_worst_cmax.csv", index=False)

    # 4. Fu
    checks.append("Fu does not enter the one-compartment model: AUC depends on CL alone and Cmax on CL, VDss and "
                  "infusion time. Fu is carried in the per-compound table for reference only. Under a PBPK model it "
                  "would matter through tissue partitioning, which this model does not represent.")

    # ---------------- figure ----------------
    make_figure(per)

    # ---------------- report ----------------
    write_readme(per, summary, paired, checks, n_blank, n_zero, worst)

    # ---------------- stdout summary ----------------
    print(f"\nExperiment A -- analytic exposure propagation, one-compartment IV, 1 mg/kg, n = {n} compounds")
    print(f"infusion time: {n_blank} blank (treated as bolus), {n_zero} exactly zero, "
          f"range {test[ds.T_MIN].min():g}-{test[ds.T_MIN].max():g} min")
    for ep in ENDPOINTS:
        print(f"\n{ep}")
        for arm in ARMS:
            g = summary[(summary.endpoint == ep) & (summary.arm == arm)].set_index("metric")
            print(f"  {LABEL[arm]:<22} GMFE {g.loc['GMFE','value']:.3f} "
                  f"[{g.loc['GMFE','value_ci_lo']:.3f}, {g.loc['GMFE','value_ci_hi']:.3f}]   "
                  f"within-2-fold {g.loc['within_2fold','value']:.3f} "
                  f"[{g.loc['within_2fold','value_ci_lo']:.3f}, {g.loc['within_2fold','value_ci_hi']:.3f}]")
        for m in ("GMFE", "within_2fold"):
            d = paired[(paired.endpoint == ep) & (paired.metric == m)].iloc[0]
            print(f"  paired diff (ours - jia) {m:<13} {d.diff_ours_minus_jia:+.4f} "
                  f"[{d.diff_ci_lo:+.4f}, {d.diff_ci_hi:+.4f}]  -> {d.verdict}")
    print("\nNot comparable with the paper's reported within-2-fold figures for AUC/Cmax: those score predicted "
          "concentration profiles against observed profiles, this scores analytic exposure against "
          "observed-parameter-derived exposure.")
    print("wrote", OUT_RES, "and", HERE / "t_downstream.csv", HERE / "fig5_downstream.png")


def make_figure(per):
    """Fig. 5: per-compound fold error, Jia et al. vs this work, in the house style."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    for ax, ep in zip(axes, ENDPOINTS):
        x = per[f"{ep}_folderror_jia"].to_numpy(float)
        y = per[f"{ep}_folderror_ours"].to_numpy(float)
        hi = float(max(x.max(), y.max())) * 1.15
        ax.fill_between([1, 2], 1, 2, color=figures.GRID, alpha=0.55, linewidth=0, zorder=0)
        ax.axhline(2, color=figures.MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
        ax.axvline(2, color=figures.MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
        ax.plot([1, hi], [1, hi], color=figures.INK, linewidth=0.8, zorder=2)
        better = y < x
        ax.scatter(x[better], y[better], s=13, color=figures.BLUE, alpha=0.8, linewidths=0, zorder=3,
                   label=f"this work closer ({int(better.sum())})")
        ax.scatter(x[~better], y[~better], s=13, color=figures.ORANGE, alpha=0.8, linewidths=0, zorder=3,
                   label=f"Jia et al. closer ({int((~better).sum())})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1, hi); ax.set_ylim(1, hi)
        ax.set_xlabel("fold error, Jia et al. QSAR parameters")
        ax.set_ylabel("fold error, this work's parameters")
        ax.set_title(ep, color=figures.INK)
        ax.legend(loc="lower right", frameon=False, fontsize=6.2)
        figures.grid(ax)
    fig.text(0.5, -0.03, "One-compartment IV model, 1 mg/kg dose, n = 106 compounds; shaded square = both within "
                         "2-fold; points below the diagonal favour this work", ha="center", fontsize=6,
             color=figures.MUTED)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(HERE / "fig5_downstream.png")
    plt.close(fig)


def write_readme(per, summary, paired, checks, n_blank, n_zero, worst):
    piv = summary.pivot_table(index=["endpoint", "arm"], columns="metric", values="value")
    lines = [
        "# Downstream exposure propagation (Experiment A)",
        "",
        "Does replacing Jia et al.'s QSAR-predicted CL and VDss with ours reduce the fold error in predicted AUC and",
        "Cmax, with the downstream exposure model held fixed? Produced by `manuscript/downstream_analysis.py`",
        "(`python manuscript/downstream_analysis.py`); deterministic and idempotent.",
        "",
        "## What this is, and is not",
        "",
        "A **one-compartment IV model** with **dose linearity** and a **1 mg/kg dose** turns CL and VDss into AUC and",
        "Cmax in closed form: `AUC = D / CL`, and `Cmax = D / V` for a bolus or `(R0 / CL) * (1 - exp(-k T))` for an",
        "infusion of duration `T` with `R0 = D / T`, `k = CL / V`. Both the paper and this work normalise every",
        "profile to 1 mg/kg, so parameter error can be propagated to exposure error analytically, with no training",
        "data and no fitted downstream model.",
        "",
        "This measures **how step-1 parameter error propagates into exposure error**, with the downstream model held",
        "constant across arms. It is **not** a measurement of absolute concentration-profile accuracy. The only",
        "legitimate comparison here is `ours` vs `jia` within this experiment. These numbers must **not** be placed",
        "beside the paper's reported within-2-fold figures for AUC and Cmax: those come from the authors'",
        "hierarchical-ML and PBPK pipelines scored against *observed concentration profiles*, whereas these score an",
        "analytic exposure against *observed-parameter-derived* exposure. The two are not on the same scale.",
        "",
        "## Data and arms",
        "",
        "106 compounds, sheet `106_cmp_test_set` of the paper's Supporting Information",
        "(`jm5c00340_si_001.xlsx`; CC BY-NC-ND, not redistributed here -- `src/downstream.py` downloads it from the",
        "Europe PMC open-access mirror into `data/external/` and verifies its SHA-256).",
        "",
        "| Arm | CL | VDss |",
        "|---|---|---|",
        "| `observed` (reference) | `CL_final(L/hour/kg)` | `VD_final(L/kg)` |",
        "| `jia` | `pred_CL_(L/hour/kg)` | `pred_VD_(L/kg)` |",
        "| `ours` | 10-seed ensemble, `results/final_models/CL/` | `results/final_models/VDss/` |",
        "",
        "`observed` is the reference that defines what perfect parameter prediction would give under this model; it is",
        "not a scored arm.",
        "",
        "## Assertions that passed",
        "",
        "- SI workbook SHA-256 matches the recorded digest.",
        "- 106 unique compounds; all CL and VDss columns finite and positive.",
        "- All 106 are in the official **test** split for CL, VDss *and* Fu (`*_train_test == \"test\"` in sheet",
        "  `Fu_VDss_CL_modeling_set`), so none was trained on.",
        "- Join: the `id` column of `results/final_models/{CL,VDss}/ensemble_predictions.csv` is **`row_idx` of",
        "  `data/cl_vdss/modeling_index.parquet`**, which carries `PUBCHEM_CID`; that CID is the join key to the SI.",
        "  It is neither `ID_trend` nor `PubChem_CID` directly (see \"Deviations\" below).",
        "- Exactly 106 compounds survive the join for CL and for VDss.",
        "- Our back-transformed `10 ** y_true` reproduces the SI's `CL_final` / `VD_final` (rtol 1e-3; observed",
        "  agreement ~1e-9).",
        "- Per-seed arrays average to the stored ensemble prediction (atol 1e-5).",
        f"- Infusion time: {n_blank} blank/NaN (would be treated as a bolus) and {n_zero} exactly zero, so no compound",
        "  needed bolus imputation; the column is complete.",
        "",
        "## Results",
        "",
        "Fold error = `max(pred/obs, obs/pred)`; GMFE = `exp(mean |ln(pred/obs)|)`; R2/MAE/RMSE on log10 exposure.",
        "95% CIs: compound-level bootstrap, B = 10,000, the same resampling protocol and RNG seed as",
        "`manuscript/analysis.py`. Differences use a **paired** bootstrap: the same resampled compounds in both arms.",
        "",
        "| Endpoint | Arm | GMFE | within-2-fold | within-3-fold | median FE | R2 (log10) |",
        "|---|---|---|---|---|---|---|",
    ]
    for ep in ENDPOINTS:
        for arm in ARMS:
            r = piv.loc[(ep, arm)]
            lines.append(f"| {ep} | {LABEL[arm]} | {r.GMFE:.3f} | {r.within_2fold:.3f} | {r.within_3fold:.3f} | "
                         f"{r.median_FE:.3f} | {r.R2:.3f} |")
    lines += ["", "Paired differences (ours - jia):", "",
              "| Endpoint | Metric | Difference | 95% CI | Verdict |", "|---|---|---|---|---|"]
    for _, d in paired.iterrows():
        if d.metric in ("GMFE", "within_2fold", "within_3fold", "median_FE", "R2"):
            lines.append(f"| {d.endpoint} | {d.metric} | {d.diff_ours_minus_jia:+.4f} | "
                         f"[{d.diff_ci_lo:+.4f}, {d.diff_ci_hi:+.4f}] | {d.verdict} |")
    lines += ["", "Per-seed spread of the `ours` arm (each of the 10 seeds propagated separately) is in",
              "`exposure_summary.csv` (`seed_mean`, `seed_sd`), to show whether the downstream result depends on",
              "ensembling the way the step-1 CL and Fu results do.", "", "## Supplementary checks", ""]
    lines += [f"{i}. {c}" for i, c in enumerate(checks, 1)]
    lines += ["", "Ten compounds with the largest Cmax fold error under `ours` are in `exposure_worst_cmax.csv`; the",
              "largest is "
              f"{worst.iloc[0]['Name']} (fold error {worst.iloc[0]['Cmax_folderror_ours']:.1f}; Jia et al. "
              f"{worst.iloc[0]['Cmax_folderror_jia']:.1f}).", "",
              "## Experiment B", "",
              "**Skipped: no observed concentration-time data is present.** The paper's SI workbook contains nine",
              "sheets and none has a time or concentration column; the 8,715 training and 1,351 test concentration",
              "points appear only as rendered plots in the Figures S2-S5 PDF. Experiment B therefore cannot be run",
              "without data obtained from the authors or from the Lombardo-cited primary literature. No concentration",
              "value was reconstructed, digitized or simulated.",
              "",
              "## Outputs", "",
              "- `exposure_per_compound.csv` -- one row per compound: identifiers, infusion time, CL and VDss for all",
              "  three arms, AUC and Cmax for all three arms, and fold errors for `jia` and `ours`.",
              "- `exposure_summary.csv` -- arm x metric x {point estimate, CI, seed mean, seed SD}.",
              "- `exposure_paired.csv` -- metric x {difference, CI, verdict}.",
              "- `exposure_cmax_bolus_sensitivity.csv` -- Cmax metrics with every compound treated as a bolus.",
              "- `exposure_worst_cmax.csv` -- the 10 worst Cmax fold errors under `ours`.",
              "- `../../manuscript/t_downstream.csv`, `../../manuscript/fig5_downstream.png`.", ""]
    (OUT_RES / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
