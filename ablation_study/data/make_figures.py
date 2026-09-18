"""
Publication figures: one two-panel chart per target (R^2 left, headline trailing metric right --
MAE in pKa units for the two pKa targets, GMFE fold-error for CL/VDss/Fu, matching how each
target's benchmark comparison is already framed throughout this project). Horizontal bars,
sorted by R^2, colored by model-family (fixed mapping across every chart so e.g. "GNN" is
always the same color everywhere), paper benchmark shown as a dashed reference line (not a
bar -- it's a target to clear, not a competing model), chosen model's bar bold-outlined and
annotated. Palette is Okabe-Ito (Okabe & Ito 2008) -- the standard colorblind-safe categorical
palette for scientific publication figures, applied in a fixed order (never cycled/reassigned
per chart).
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("all_models_comparison.csv")

# Fixed categorical color assignment, consistent across every target's chart.
FAMILY_COLOR = {
    "Classical": "#7F7F7F",       # neutral gray -- the baseline every target starts from
    "MFMN (base/tuned)": "#0072B2",   # blue
    "MFMN_InteractAux": "#56B4E9",    # sky blue
    "MFMN_Attention": "#E69F00",      # orange
    "ChemBERTa": "#CC79A7",           # reddish purple
    "Evidential": "#D55E00",          # vermillion
    "GraphMPNN (GNN)": "#009E73",     # bluish green
    "MFMN_DynGate": "#F0B400",        # gold (darker than Okabe-Ito yellow for legibility on white)
    "Rejected extra": "#B0B0B0",      # light gray -- post-hoc ideas that failed
    "FINAL (chosen)": "#1B9E4B",      # strong green, bold-outlined separately
}


def classify(model):
    m = model.lower()
    if "final" in m:
        return "FINAL (chosen)"
    if "classical" in m:
        return "Classical"
    if "attention" in m:
        return "MFMN_Attention"
    if "chemberta" in m:
        return "ChemBERTa"
    if "evidential" in m:
        return "Evidential"
    if "graphmpnn" in m or "gnn" in m:
        return "GraphMPNN (GNN)"
    if "dyngate" in m:
        return "MFMN_DynGate"
    if "blend" in m or "amin" in m:
        return "Rejected extra"
    if "interactaux" in m:
        return "MFMN_InteractAux"
    if "mfmn base" in m or "mfmn +" in m:
        return "MFMN (base/tuned)"
    return "Classical"


PKA_TARGETS = {"pKa_Acidic", "pKa_Basic"}

for target in df.target.unique():
    sub = df[df.target == target].copy()
    paper_row = sub[sub.model.str.contains("Paper")].iloc[0]
    models = sub[~sub.model.str.contains("Paper")].copy()
    models["family"] = models.model.apply(classify)

    is_pka = target in PKA_TARGETS
    metric2 = "MAE" if is_pka else "GMFE"
    metric2_label = "MAE (pKa units)" if is_pka else "GMFE (fold-error)"
    lower_is_better_2 = True  # both MAE and GMFE: lower is better

    # Only keep rows that have both metrics plotted (some CV-only rows lack R2 or the 2nd metric)
    plot_df = models.dropna(subset=["R2"]).copy()
    plot_df = plot_df.sort_values("R2", ascending=True)  # ascending so best ends up at top in barh

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(plot_df) + 1.5))
    y = np.arange(len(plot_df))
    colors = [FAMILY_COLOR[f] for f in plot_df.family]
    is_final = plot_df.model.str.contains("FINAL").values

    # ---- Panel 1: R^2 ----
    ax = axes[0]
    bars = ax.barh(y, plot_df.R2, color=colors, edgecolor="none", height=0.65)
    for i, (bar, final, proto) in enumerate(zip(bars, is_final, plot_df.protocol)):
        if proto == "CV":
            bar.set_hatch("///")
            bar.set_edgecolor("white")
            bar.set_linewidth(0.5)
        if final:
            bar.set_edgecolor("black")
            bar.set_linewidth(2.2)
    if not np.isnan(paper_row.get("R2", np.nan)):
        ax.axvline(paper_row.R2, color="black", linestyle="--", linewidth=1.3, zorder=0)
        ax.text(paper_row.R2, len(plot_df) - 0.3, " paper", fontsize=8, va="bottom", ha="left", style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.model, fontsize=9)
    ax.set_xlabel("R² (higher is better)")
    ax.set_title(f"{target}: R² across model families", fontsize=11, loc="left")
    for i, v in enumerate(plot_df.R2):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- Panel 2: headline trailing metric ----
    ax2 = axes[1]
    plot_df2 = models.dropna(subset=[metric2]).copy()
    plot_df2 = plot_df2.sort_values(metric2, ascending=False)  # ascending=False -> worst at bottom, best (lowest) at top of barh reversed
    plot_df2 = plot_df2.iloc[::-1]
    y2 = np.arange(len(plot_df2))
    colors2 = [FAMILY_COLOR[classify(m)] for m in plot_df2.model]
    is_final2 = plot_df2.model.str.contains("FINAL").values
    bars2 = ax2.barh(y2, plot_df2[metric2], color=colors2, edgecolor="none", height=0.65)
    for bar, final, proto in zip(bars2, is_final2, plot_df2.protocol):
        if proto == "CV":
            bar.set_hatch("///")
            bar.set_edgecolor("white")
            bar.set_linewidth(0.5)
        if final:
            bar.set_edgecolor("black")
            bar.set_linewidth(2.2)
    if not pd.isna(paper_row.get(metric2, np.nan)):
        ax2.axvline(paper_row[metric2], color="black", linestyle="--", linewidth=1.3, zorder=0)
        ax2.text(paper_row[metric2], len(plot_df2) - 0.3, " paper", fontsize=8, va="bottom", ha="left", style="italic")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(plot_df2.model, fontsize=9)
    ax2.set_xlabel(f"{metric2_label} (lower is better)")
    ax2.set_title(f"{target}: {metric2} across model families", fontsize=11, loc="left")
    for i, v in enumerate(plot_df2[metric2]):
        ax2.text(v + 0.01 * plot_df2[metric2].max(), i, f"{v:.3f}", va="center", fontsize=7.5)
    ax2.spines[["top", "right"]].set_visible(False)

    # shared legend
    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()
              if f in set(plot_df.family) | set(plot_df2.model.apply(classify))]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(f"{target}: model comparison vs. published benchmark", fontsize=13, y=1.04, fontweight="bold")
    fig.text(0.5, -0.10, "Hatched bars = cross-validation estimate (development pool); solid bars = one-shot official held-out test-set result. Not directly comparable across protocols.",
             ha="center", fontsize=8, style="italic", color="#444444")
    plt.tight_layout()
    outpath = f"../figures/{target}_comparison.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {outpath} ({len(plot_df)} models in R2 panel, {len(plot_df2)} in {metric2} panel)")

print("\nDone.")
