# PK/pKa benchmark-paper reproductions — 5 targets

Self-contained reproduction of this project's current best result for
each of the 5 targets modeled in this work, evaluated in the exact configuration used to
compare against the published benchmark (Jia, X. et al. *"Application of Machine Learning and
Mechanistic Modeling to Predict Intravenous Pharmacokinetic Profiles in Humans."* J. Med.
Chem. 2025, 68(7), 7737-7750 — CL/VDss/Fu; this project's established pKa benchmark
comparison — pKa_Acidic/pKa_Basic). Each notebook fits on the **official train split only**
and predicts on the **official held-out test split only** — never an internal CV estimate —
using a multi-seed ensemble, matching the actual final-confirmation scripts this project used.

## Results (10-seed retrained, official test split)

All five proposed models were retrained with 10 seeds (42-51) and scored on the official held-out test split
(ensemble of the 10 seeds). Predictions, per-seed metrics, untuned baselines and the 10-seed ablation of all five targets
are in [`results/`](results/README.md); the Results-section draft and the scripts that build it are in `manuscript/`.

| Target | Architecture | n_test | Our result | Paper | Verdict |
|---|---|---|---|---|---|
| pKa_Acidic | GraphMPNN | 776 | R²=0.969 MAE=0.424 | R²=0.94 MAE=0.61 | beats paper |
| pKa_Basic | GraphMPNN (no site readout) | 815 | R²=0.950 MAE=0.406 | R²=0.91 MAE=0.67 | beats paper |
| CL | MFMN_DynGate | 177 | R²=0.497 MAE=0.305 GMFE=2.018 w2f=0.605 | R²=0.48 MAE=0.31 GMFE=2.00 w2f=0.64 | beats paper on R²/MAE/RMSE; narrowly trails on GMFE/w2f |
| VDss | MFMN_InteractAux | 177 | R²=0.648 MAE=0.247 GMFE=1.764 w2f=0.684 | R²=0.60 MAE=0.28 GMFE=1.88 w2f=0.62 | beats paper |
| Fu | MFMN_DynGate + 1.25x low-fu weighting | 633 | R²=0.712 MAE=0.319 GMFE=2.086 w2f=0.613 | R²=0.69 MAE=0.30 GMFE=2.01 w2f=0.60 | beats paper on R²/w2f |

("w2f" = within-2-fold.)

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Model and feature code (flat package; see "Code layout" below). |
| `data/` | Input descriptors, graphs, labels and official train/test splits. |
| `notebooks/` | One notebook per target that trains the final model with 10 seeds and scores the official test split. |
| `results/` | 10-seed predictions and metrics: final models, untuned baselines, ablation variants. |
| `manuscript/` | Results-section draft (`Results_section_draft.docx`), its tables (`t_*.csv`) and figures (`fig*.png`), and the scripts that rebuild them. |
| `Diagram/` | Architecture diagrams of GraphMPNN and MFMN. |

## Quick start

```bash
pip install torch rdkit pandas numpy scikit-learn matplotlib jupyter nbformat pyarrow
cd notebooks
jupyter notebook   # run a notebook top to bottom to train its final model (writes to ../outputs/)
```

Notebooks 01 and 05 contain their saved outputs; notebooks 02-04 were updated (pKa_Basic now trains the
no-site-readout model; CL/VDss now use 10 seeds) and have no saved outputs until re-run. The reference 10-seed numbers
are in `results/`.

To rebuild the Results draft from `results/` (needs `python-docx` in addition to the packages above):

```bash
cd manuscript
python analysis.py    # tables t_*.csv and fig_data.json (bootstrap CIs, B = 10,000)
python figures.py     # fig1-fig4
python build_docx.py  # Results_section_draft.docx
```

Each notebook is standalone (adds `../src` to `sys.path` itself) and runs on GPU automatically
if available, falling back to CPU otherwise. Runtimes on a single modern GPU: pKa notebooks
(01/02) take ~85-90 minutes each (10-seed GNN ensemble on 5,300-5,700 training compounds);
CL/VDss/Fu (03/04/05) take well under 5 minutes each (dense descriptor-based nets, no
graph message-passing).

## Why 5 different architectures, not one

This project found that the single best architecture differs by target — there is no one
model that wins everywhere:
- **pKa_Acidic / pKa_Basic**: `GraphMPNN`, a from-scratch edge-gated message-passing GNN
  operating directly on molecular graphs (`src/graph_mpnn.py`, `src/graph_features.py`). For
  pKa_Basic the final model omits the ionizable-site readout (`site_readout=False`), which
  scored best on every metric in the 10-seed ablation.
- **CL**: `MFMN_DynGate` — a factorized-descriptor model (7 mechanistic factors, each
  privileged to its own named descriptors) with per-compound input-conditioned gating and
  auxiliary supervision (`aux_weight=0.8`).
- **VDss**: `MFMN_InteractAux` — the same factorized-descriptor family without dynamic
  gating (gating was flat/within-noise for VDss specifically), `aux_weight=2.0`.
- **Fu**: `MFMN_DynGate` again, but with a different descriptor context (raw
  variance/correlation-pruned RDKit+Mordred+MACCS+FCFP6, no PCA), **no** auxiliary
  supervision loss (found to hurt here, since the legitimate real-property side-signals used
  for CL/VDss aren't available/valid for this target), and a **1.25× training-loss weight**
  for the bottom ~10% of compounds by fu (see the post-delivery investigation below — this was
  added after the original delivery and is a confirmed, official-test-verified improvement).

The full experimental history behind why each of these specific recipes won is in
`mfmn/deliverables/PROJECT_REPORT.md` in the parent project (not included in this repository).

## Data

`data/` (~81MB total), split by which target(s) need it:
- `data/pka/` (19MB) — `aux_index.parquet` (SMILES + labels + official train/test split
  flags for all 5 targets), plus xTB graph tables for pKa_Acidic/pKa_Basic.
- `data/cl_vdss/` (16MB) — the 1,501-compound CL/VDss pool: RDKit/Mordred/MACCS descriptors,
  auxiliary predicted-pKa/fu features, xTB electronic descriptors, official train/test splits.
- `data/fu/` (47MB) — RDKit/Mordred/MACCS/FCFP6 descriptors, row-trimmed to exactly Fu's
  4,675-compound pool (down from the full ~13,216-compound aux universe this project also
  covers, since only Fu's own rows are ever accessed by the feature-building code — this
  trimming changes nothing numerically, just file size).

## Post-delivery investigation: the search for a better Fu recipe (H11-H17)

After this package was first delivered, a structured, disciplined investigation (CV-first,
official 633-compound test set touched only for one-shot locked-in confirmations, never for
iteration) tried seven independent ideas to close Fu's gap to the paper. **One survived and
is now baked into notebook 05**; six are documented here as negative results, per this
project's own convention (see `mfmn/deliverables/PROJECT_REPORT.md`'s H1/H3/H5 entries for
the same pattern) — a thorough search is worth recording even when most branches fail.

**Adopted — H15, low-fu loss reweighting.** A diagnostic (H14: 5-fold×3-repeat OOF over the
full labeled pool) found error is *not* spread evenly — it is concentrated almost entirely in
the bottom ~10% by fu (fu<0.006): MAE≈0.67 and GMFE≈4.5 in that subgroup alone, vs ≈0.3/≈2.0
everywhere else. Upweighting that subgroup's training loss by 1.25× (`w=1.25` if fu<0.006,
else `w=1.0`, normalized as a weighted mean) was screened across 7 weights (1.0-4.0): higher
weights fixed the tail further but cost real overall accuracy; 1.25× was the one point with a
CV-confirmed (3-repeat, noise-exceeding) low-fu improvement at negligible overall cost. The
locked-in one-shot official-test run **improved every single metric** over the unweighted
baseline (R² 0.7113→0.7121, MAE 0.3204→0.3193, RMSE 0.4358→0.4352, GMFE 2.0913→2.0860,
within-2-fold 0.6019→0.6130) — the only idea in this whole investigation whose CV signal
actually held up on the real test set. Note: the 10-seed ablation (`results/ablation/Fu/`) reproduces these exact
numbers, but none of these differences is resolved by a paired bootstrap (all 95% CIs include zero),
so the improvement is consistent in direction but within test-set uncertainty.

**Rejected:**
1. **Logit-transform the target** — motivated by the same low-fu shrinkage pattern (textbook
   signature of an unbounded log10 space for a [0,1]-bounded fraction). A wash in CV (deltas
   within noise); the shrinkage is real but not a recoverable parameterization artifact.
2. **Isotonic calibration** of DynGate's predictions, nested to avoid leakage. Made every
   metric slightly worse — per-fu-bin noise dwarfs the per-bin bias.
3. **CV-tuned classical-model blend** (DynGate + RF/SVM/XGB consensus). Looked like a robust
   win in 3-repeat CV, but the pre-registered official-test run (alpha locked in from CV
   evidence) came back worse on 4/6 metrics — the official DynGate is a 10-seed ensemble on
   more data than any CV fold saw, so it's disproportionately stronger than the CV screen's
   single-seed comparison predicted. (A post-hoc alpha found by sweeping against the test set
   looked better, but selecting it that way is exactly the test-set-overfitting failure mode
   this project's discipline exists to prevent — not adopted.)
4. **Cross-fitted residual learner** (Ridge / XGBoost) predicting DynGate's residual from
   ensemble-uncertainty, chemical-space (Tanimoto AD-) distance, MolLogP, and TPSA. A direct
   signed-residual correlation check (run before building any model) found all four features
   correlate with *error magnitude* (r=0.15-0.26) but essentially *not at all* with *error
   direction* (signed residual, |r|<0.05 for all four) — they predict "this will be hard," not
   "which way to correct it." No model family can extract a correction from that; both Ridge
   and XGBoost made things flat-to-worse, confirming the mechanism rather than just the outcome.
5. **AMIN-Fu architecture ladder** — explicit, mechanistically-named pairwise interactions
   (lipophilicity×polarity, ×binding, polarity×binding) added to DynGate's factor embeddings,
   plus learned gates and a third-order lipophilicity×polarity×binding term. The simplest
   variant (3 ungated pairwise products, +384 params, +0.24%) looked like a clean broad win in
   a single-seed screen — but a proper 3-repeat confirmation showed every delta was smaller
   than its own standard deviation: the apparent win was driven almost entirely by one
   fold-seed and reversed in the other two repeats. Learned gating and the third-order term
   were each, independently, clean regressions even in the single-seed screen (added
   flexibility without enough data to support it — the same failure mode multi-head attention
   showed earlier in this project for Fu specifically).

## Code layout

`src/` is a flat package — every module sits at the same level and imports its siblings
directly. Notable files:
- `graph_mpnn.py`, `graph_features.py`, `acidic_site.py`, `basic_site.py` — the pKa GNN stack
  (`GraphMPNN(site_readout=False)` gives the pKa_Basic final model).
- `mfmn.py` — trimmed to the 4 classes actually used here (`MFMN`, `MFMN_Interact`,
  `MFMN_InteractAux`, `MFMN_DynGate`) out of the full project's ~10 architecture variants.
- `descriptor_features.py` — CL/VDss's feature pipeline (shared PCA'd context + 7 factor blocks).
- `aux_descriptor_features.py` — Fu's feature pipeline (raw+FCFP6 context + 7 factor blocks,
  trimmed to the exact recipe Fu's final model uses).
- `data_loader.py`, `metrics.py` — minimal data access and evaluation (R²/MAE/RMSE, GMFE,
  within-N-fold/pKa-unit accuracy), plus the published `PAPER` benchmark numbers.
