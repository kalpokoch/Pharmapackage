"""Publication figures for the Results section (IEEE double-column width 7.16 in)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": "#6b6b66", "axes.linewidth": 0.6, "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600, "savefig.bbox": "tight",
})
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3dc"
MODEL_COLOR = {"Proposed": BLUE, "Random Forest": ORANGE, "XGBoost": AQUA, "SVM": YELLOW}
TITLE = {"pKa_Acidic": "pKa (acidic)", "pKa_Basic": "pKa (basic)", "CL": "CL", "VDss": "VDss", "Fu": "Fu"}
MODEL_NAME = {"pKa_Acidic": "GraphMPNN", "pKa_Basic": "GraphMPNN", "CL": "MFMN-DynGate",
              "VDss": "MFMN-InteractAux", "Fu": "MFMN-DynGate (LW)"}
TARGETS = ["pKa_Acidic", "pKa_Basic", "CL", "VDss", "Fu"]
LOG = {"CL", "VDss", "Fu"}
UNITS = {"pKa_Acidic": "pKa", "pKa_Basic": "pKa", "CL": r"$\log_{10}$ CL", "VDss": r"$\log_{10}$ VDss",
         "Fu": r"$\log_{10}$ Fu"}


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    fd = json.load(open(OUT / "fig_data.json"))
    main = pd.read_csv(OUT / "t_main.csv")
    paper = pd.read_csv(OUT / "t_paper.csv")
    base = pd.read_csv(OUT / "t_baselines.csv")

    # ---------------- Fig. 1: parity plots ----------------
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.75))
    for ax, t in zip(axes, TARGETS):
        y, p = np.array(fd[t]["y"]), np.array(fd[t]["p"])
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        band = np.log10(2) if t in LOG else 1.0
        xs = np.array([lo, hi])
        ax.fill_between(xs, xs - band, xs + band, color=GRID, alpha=0.7, linewidth=0)
        ax.plot(xs, xs, color=MUTED, linewidth=0.7)
        ax.scatter(y, p, s=3.5, color=BLUE, alpha=0.55, linewidths=0, rasterized=True)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        r = main[(main.target == t)].set_index("metric")
        txt = f"$R^2$ = {r.loc['R2','ensemble']:.3f}\nMAE = {r.loc['MAE','ensemble']:.3f}"
        if t in LOG:
            txt += f"\nGMFE = {r.loc['GMFE','ensemble']:.3f}"
        if t == "Fu":
            ax.text(0.97, 0.04, txt, transform=ax.transAxes, va="bottom", ha="right", fontsize=6.3, color=INK)
        else:
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.3, color=INK)
        ax.set_title(f"{TITLE[t]} (n = {len(y)})", color=INK)
        ax.set_xlabel(f"Observed {UNITS[t]}", fontsize=7)
        if t == "pKa_Acidic":
            ax.set_ylabel("Predicted (10-seed ensemble)", fontsize=7)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig1_parity.png")
    plt.close(fig)

    # ---------------- Fig. 2: relative difference vs published benchmark ----------------
    HIGHER = {"R2": True, "MAE": False, "RMSE": False, "GMFE": False, "within_2fold": True}
    LABEL = {"R2": "$R^2$", "MAE": "MAE", "RMSE": "RMSE", "GMFE": "GMFE", "within_2fold": "W2F"}
    rows = []
    for _, r in paper.iterrows():
        s = 1 if HIGHER[r.metric] else -1
        rel = s * (r.ensemble - r.paper) / r.paper * 100
        a, b = s * (r.ci_lo - r.paper) / r.paper * 100, s * (r.ci_hi - r.paper) / r.paper * 100
        rows.append((r.target, r.metric, rel, min(a, b), max(a, b)))
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    ylabels, ypos, y0 = [], [], 0
    for t in TARGETS:
        sub = [x for x in rows if x[0] == t]
        for (tt, m, rel, a, b) in sub:
            ax.plot([a, b], [y0, y0], color=BLUE, linewidth=1.2, solid_capstyle="round")
            ax.scatter([rel], [y0], s=16, color=BLUE, zorder=3, edgecolors="white", linewidths=0.6)
            ylabels.append(f"{TITLE[t]} – {LABEL[m]}"); ypos.append(y0); y0 += 1
        y0 += 0.7
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=6.6); ax.invert_yaxis()
    ax.set_xlabel("Relative difference from published benchmark (%)\n(positive = proposed better)")
    grid(ax, "x")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_vs_paper.png")
    plt.close(fig)

    # ---------------- Fig. 3: proposed vs untuned baselines (headline error metric) ----------------
    HEAD = {"pKa_Acidic": "MAE", "pKa_Basic": "MAE", "CL": "GMFE", "VDss": "GMFE", "Fu": "GMFE"}
    models = ["Proposed", "Random Forest", "XGBoost", "SVM"]
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.9))
    for ax, t in zip(axes, TARGETS):
        m = HEAD[t]
        pr = main[(main.target == t) & (main.metric == m)].iloc[0]
        vals = [(pr.ensemble, pr.ci_lo, pr.ci_hi)]
        for mod in models[1:]:
            b = base[(base.target == t) & (base.model == mod) & (base.metric == m)].iloc[0]
            vals.append((b.value, b.value_ci_lo, b.value_ci_hi))
        x = np.arange(len(models))
        for i, (v, lo, hi) in enumerate(vals):
            ax.bar(i, v, width=0.68, color=MODEL_COLOR[models[i]], edgecolor="white", linewidth=1.0)
            ax.plot([i, i], [lo, hi], color=INK, linewidth=0.7)
        ax.set_xticks(x); ax.set_xticklabels(["Prop.", "RF", "XGB", "SVM"], fontsize=6.5)
        if m == "GMFE":
            ax.set_ylim(1.0, max(v[2] for v in vals) * 1.05)
        else:
            ax.set_ylim(0, max(v[2] for v in vals) * 1.08)
        ax.set_title(f"{TITLE[t]}", color=INK)
        ax.set_ylabel(m + (" (pKa units)" if m == "MAE" else ""), fontsize=7)
        grid(ax, "y")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[k]) for k in models]
    fig.legend(handles, ["Proposed", "Random Forest", "XGBoost", "SVM"],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.8)
    fig.savefig(OUT / "fig4_baselines.png")
    plt.close(fig)

    # ---------------- Fig. 4: CL/VDss ablation, per-seed GMFE ----------------
    abl = pd.read_csv(OUT / "t_ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=False)
    for ax, t in zip(axes, ["CL", "VDss"]):
        seeds = fd[t]["ablation_seeds"]
        order = ["MFMN (base)"] + [k for k in seeds if k not in ("MFMN (base)", "final")] + ["final"]
        labels = {"final": MODEL_NAME[t] + "\n(final)"}
        colors = {"MFMN (base)": ORANGE, "final": BLUE}
        rng = np.random.default_rng(0)
        for i, k in enumerate(order):
            c = colors.get(k, AQUA)
            v = np.array(seeds[k])
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color=c, alpha=0.75, linewidths=0)
            if k == "final":
                e = main[(main.target == t) & (main.metric == "GMFE")].ensemble.iloc[0]
            else:
                e = abl[(abl.target == t) & (abl.variant == k) & (abl.metric == "GMFE")].ensemble.iloc[0]
            ax.plot([i - 0.28, i + 0.28], [e, e], color=INK, linewidth=1.3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels.get(k, k).replace("MFMN-", "MFMN-\n").replace("MFMN (base)", "MFMN\n(base)")
                            for k in order], fontsize=6.2)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_title(t, color=INK)
        grid(ax, "y")
    axes[0].set_ylabel("GMFE (lower is better)", fontsize=7)
    fig.text(0.5, -0.04, "dots: individual seeds (n = 10); bar: 10-seed ensemble", ha="center", fontsize=6,
             color=MUTED)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "fig5_ablation.png")
    plt.close(fig)
    print("figures done")


if __name__ == "__main__":
    main()
