# 10-seed retrained results (2026-10)

Everything here was produced by retraining all five proposed models with **10 seeds (42-51)** on the official train split and
scoring the **official test split** (same protocol as the notebooks in `../notebooks`, but 10 seeds for every target).
"ensemble" = metrics of the mean prediction over the 10 seeds; "seed mean" = average of the per-seed metrics.

| Target | Model | n_test | R2 (ens) | MAE (ens) | GMFE (ens) | within-2-fold (ens) |
|---|---|---|---|---|---|---|
| CL | MFMN_DynGate (aux 0.8) | 177 | 0.497 | 0.305 | 2.018 | 0.605 |
| VDss | MFMN_InteractAux (aux 2.0) | 177 | 0.648 | 0.247 | 1.764 | 0.684 |
| Fu | MFMN_DynGate + 1.25x low-fu weight | 633 | 0.712 | 0.319 | 2.086 | 0.613 |
| pKa_Acidic | GraphMPNN | 776 | 0.969 | 0.424 | - | - |
| pKa_Basic | GraphMPNN | 815 | 0.948 | 0.416 | - | - |

Full per-metric tables (seed mean/sd/min/max and paper values): `finals/<target>/ensemble_summary.csv`.
`finals/retrained_vs_stored.csv` compares these retrained runs with the previously stored results.

## Contents
- `finals/` - retrained final models: per-target summaries, per-seed metrics, ensemble predictions, summary figures, `manifest.json`.
- `baselines/` - untuned Random Forest / XGBoost / SVM vs the proposed models (CL, VDss, Fu; same test rows). See `baselines/README.md`.
- `ablation_10seed/` - 10-seed ablation tables and `report.md` (see status below).
- `scripts/` - the scripts that produced these results (`run_finals.py`, `run_baselines_10seed.py`, `run_ablation_10seed.py`,
  `analyze_ablation_10seed.py`, `run_pipeline_rest.sh`). They assume the full project layout and are included for provenance,
  not as standalone runnable code.

## Baselines vs proposed (10-seed ensemble, official test)
| Target | Proposed | Random Forest | XGBoost | SVM |
|---|---|---|---|---|
| CL (GMFE / w2f) | 2.018 / 0.605 | **2.010 / 0.638** | 2.266 / 0.497 | 2.042 / 0.627 |
| VDss (GMFE / w2f) | **1.764 / 0.684** | 1.848 / 0.661 | 2.026 / 0.588 | 1.919 / 0.610 |
| Fu (GMFE / w2f) | **2.086 / 0.613** | 2.391 / 0.488 | 2.316 / 0.531 | 2.284 / 0.534 |

On **CL**, the untuned Random Forest is on par with or ahead of the proposed model on GMFE and within-2-fold; the proposed model
leads on VDss and Fu. Paired-bootstrap confidence intervals are in `baselines/tables/baseline_vs_proposed_bootstrap.csv`.

## Ablation status (partial)
Completed (10 seeds, passed the seed-42 reproduction check): **CL** (MFMN base, InteractAux) and **VDss** (MFMN base, DynGate).

| Target | Variant | R2 (ens) | GMFE (ens) | within-2-fold (ens) |
|---|---|---|---|---|
| CL | final (DynGate) | 0.497 | 2.019 | 0.605 |
| CL | MFMN base | 0.468 | 2.057 | 0.571 |
| CL | InteractAux | 0.490 | 2.036 | 0.605 |
| VDss | final (InteractAux) | 0.648 | 1.764 | 0.684 |
| VDss | MFMN base | 0.571 | 1.872 | 0.655 |
| VDss | DynGate | 0.647 | 1.771 | 0.695 |

**Not included yet:**
- **Fu** ablation variants - blocked: the seed-42 smoke test did not reproduce the stored numbers within tolerance.
- **pKa** ablations (no xTB / no edge gating / no site readout) - the original code for these rows is not in the project, so the
  variants were reconstructed and are unverified; the 10-seed runs are still in progress. They will be added when complete.

The ablation tables in `ablation_10seed/` were generated before the final retraining and compare against the previously stored
final models; the retrained finals agree with those to within ~1e-4 on CL/VDss (see `finals/retrained_vs_stored.csv`).
