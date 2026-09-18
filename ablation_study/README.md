# Ablation & model-comparison study — why each architecture was chosen

This directory answers, with sourced evidence, one question per target: **why this specific
model over every other reasonable alternative?** It consolidates this project's full
experiment history — classical ML baselines, every alternative neural architecture actually
tried, and every intermediate step of the winning architecture's own development — into
comparison tables and publication-format figures, for the 5 targets modeled in this work
(pKa_Acidic, pKa_Basic, CL, VDss, Fu), each benchmarked against Jia et al. (J. Med. Chem. 2025,
68(7), 7737-7750).

## Provenance — this is a consolidation, not a fresh experiment run

Every number in `data/all_models_comparison.csv` is read from an already-existing result file
produced elsewhere in this project's history (the `source` column gives the exact path for
every row) — no model is trained by this directory's own scripts. This was a deliberate choice:
the alternative architectures compared here (classical RF/SVM/XGB, attention, ChemBERTa
embeddings, graph neural networks, evidential regression) were each the subject of substantial
dedicated experimentation earlier in this project (see `mfmn/deliverables/PROJECT_REPORT.md`
for the full narrative), and re-running all of them from scratch for every target would
reproduce work already done carefully once. Instead, the **final/chosen model for each target
was independently re-verified this session** — fresh reruns landed within noise of (in most
cases, exactly matching) the historical numbers reused here, which is the basis for trusting
the rest of the historical record rather than re-deriving all of it:

| Target | Fresh rerun (this session) | Historical value used here | Match |
|---|---|---|---|
| pKa_Acidic (GNN) | R²=0.9689 | R²=0.9689 | exact |
| pKa_Basic (GNN) | R²=0.9478 | R²=0.9478 | exact |
| CL (DynGate) | GMFE=2.0166 | GMFE=2.0166 | exact |
| VDss (InteractAux) | GMFE=1.7740 | GMFE=1.7740 | exact |
| Fu (DynGate, unweighted) | R²=0.7113 | R²=0.7113 | exact |
| Fu (DynGate + 1.25× weighting) | R²=0.7121 | R²=0.7121 | exact (this session's own result) |

## Structure

```
ablation_study/
├── README.md                              this file
├── data/
│   ├── all_models_comparison.csv          every model x every target, with metrics + source
│   ├── build_comparison_tables.py         regenerates the CSV (documents every source)
│   └── make_figures.py                    regenerates the 5 figures below
├── figures/
│   ├── pKa_Acidic_comparison.png
│   ├── pKa_Basic_comparison.png
│   ├── CL_comparison.png
│   ├── VDss_comparison.png
│   └── Fu_comparison.png
└── notebooks/
    ├── 00_overview.ipynb                  cross-target summary
    ├── 01_pKa_Acidic_ablation.ipynb
    ├── 02_pKa_Basic_ablation.ipynb
    ├── 03_CL_ablation.ipynb
    ├── 04_VDss_ablation.ipynb
    └── 05_Fu_ablation.ipynb
```

Each notebook is already executed (tables and figures are baked in) and can be re-run from
scratch — it only reads the CSV and figure files, no GPU or training required (seconds to run).

## How to read the figures

Two panels per target: R² (left) and the headline trailing metric (right — MAE in pKa units
for the two pKa targets, GMFE fold-error for CL/VDss/Fu, matching how this project frames each
target's own benchmark comparison). Bars are colored by model family, held **consistent across
every target's chart** (e.g. "GraphMPNN (GNN)" is always the same green-teal everywhere).

**Hatched bars are cross-validation estimates** (repeated k-fold over the development pool —
the metric used for model *selection*); **solid bars are one-shot official held-out test-set
results** (the number actually compared to the published paper). These two protocols are not
directly interchangeable — see the CL notebook for a case where this matters concretely: every
architecture's CV ranking failed to predict the eventual official-test winner, a documented,
paper's-test-set-is-non-random finding from this project's own methodology work
(`PROJECT_REPORT.md` Section 3), not an error in this comparison.

The chosen model's bar is bold-outlined in every chart. The dashed vertical line is the
published paper's own value for that metric.

## Headline finding per target

| Target | Chosen architecture | Mechanistic reason it wins |
|---|---|---|
| **pKa_Acidic / pKa_Basic** | `GraphMPNN` (from-scratch GNN) | pKa is a *local, atom-level* property (one ionizable site's electronic environment) — a graph encoder with an explicit site readout represents this directly; whole-molecule aggregate descriptors structurally cannot. |
| **CL** | `MFMN_DynGate` | CV rankings did not predict the official-test winner (the paper's 177-compound test split is a documented non-random subset) — DynGate's dynamic, per-compound gating is the specific refinement that carried the descriptor-based family past the paper's benchmark. |
| **VDss** | `MFMN_InteractAux` | Dynamic gating (which helped CL) is flat/within-noise here — the simpler recipe wins on Occam's-razor grounds with no accuracy cost, delivering this project's cleanest sweep of the paper on every metric. |
| **Fu** | `MFMN_DynGate` + 1.25× low-fu loss weighting | No localized site exists for Fu (a systemic plasma-protein-binding property), so the GNN's advantage for pKa doesn't transfer — it underperforms even the classical baseline here. A targeted diagnostic found error concentrated almost entirely in the bottom ~10% of compounds by fu; a small, targeted loss reweighting for that subgroup is the one post-delivery improvement (of seven tried) that survived confirmation end-to-end. |

See each target's own notebook for the full comparison table, every architecture's numbers,
and the detailed narrative.
