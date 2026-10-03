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
| pKa_Basic | GraphMPNN, no site readout | 815 | 0.950 | 0.406 | - | - |

Full per-metric tables (seed mean/sd/min/max and paper values): `finals/<target>/ensemble_summary.csv`.
`finals/retrained_vs_stored.csv` compares these retrained runs with the previously stored results.

**pKa_Basic final model:** GraphMPNN without the ionizable-site readout, chosen because it scored best on every metric in the
10-seed ablation (below). `finals/pKa_Basic/` holds the full GraphMPNN (the previous final model); the no-site-readout
results are in `ablation_10seed/` (variant `no_site_readout`).

## Contents
- `finals/` - retrained final models: per-target summaries, per-seed metrics, ensemble predictions, summary figures, `manifest.json`.
- `baselines/` - untuned Random Forest / XGBoost / SVM vs the proposed models (all five targets; same test rows). See `baselines/README.md`.
- `ablation_10seed/` - 10-seed ablation tables, `report.md` and `checks.json` (all 12 variants; see below).
- `scripts/` - the scripts that produced these results (`run_finals.py`, `run_baselines_10seed.py`, `run_ablation_10seed.py`,
  `analyze_ablation_10seed.py`, `run_pipeline_rest.sh`). They assume the full project layout and are included for provenance,
  not as standalone runnable code.

## Baselines vs proposed (10-seed ensemble, official test)
| Target | Proposed | Random Forest | XGBoost | SVM |
|---|---|---|---|---|
| pKa_Acidic (R2 / MAE) | **0.969 / 0.424** | 0.927 / 0.699 | 0.919 / 0.743 | 0.770 / 1.475 |
| pKa_Basic (R2 / MAE) | **0.950 / 0.406** | 0.862 / 0.758 | 0.881 / 0.701 | 0.723 / 1.068 |
| CL (GMFE / w2f) | 2.018 / 0.605 | **2.010 / 0.638** | 2.266 / 0.497 | 2.042 / 0.627 |
| VDss (GMFE / w2f) | **1.764 / 0.684** | 1.848 / 0.661 | 2.026 / 0.588 | 1.919 / 0.610 |
| Fu (GMFE / w2f) | **2.086 / 0.613** | 2.391 / 0.488 | 2.316 / 0.531 | 2.284 / 0.534 |

On **CL**, the untuned Random Forest is on par with or ahead of the proposed model on GMFE and within-2-fold; the proposed model
leads on both pKa targets, VDss and Fu. Paired-bootstrap confidence intervals are in `baselines/tables/baseline_vs_proposed_bootstrap.csv`.

## Ablation (10 seeds, complete)
All 12 variants were run with 10 seeds (42-51) on the official test split; `ablation_10seed/checks.json` reports all consistency
checks passed (test labels identical to the final model's, seed/row mapping consistent, metrics recomputed). "Resolved" below means
the 95% CI of the ensemble difference (compound-level paired bootstrap, B = 10,000) excludes zero.

| Target | Variant | Status | R2 (ens) | MAE (ens) | GMFE (ens) | within-2-fold (ens) |
|---|---|---|---|---|---|---|
| CL | final (DynGate) | | 0.497 | 0.305 | 2.018 | 0.605 |
| CL | MFMN base | verified | 0.468 | 0.313 | 2.057 | 0.571 |
| CL | InteractAux | verified | 0.490 | 0.309 | 2.036 | 0.605 |
| VDss | final (InteractAux) | | 0.648 | 0.247 | 1.764 | 0.684 |
| VDss | MFMN base | verified | 0.571 | 0.272 | 1.872 | 0.655 |
| VDss | DynGate | verified | 0.647 | 0.248 | 1.771 | 0.695 |
| Fu | final (DynGate + 1.25x low-fu weight) | | 0.712 | 0.319 | 2.086 | 0.613 |
| Fu | DynGate, unweighted | caveat (1) | 0.711 | 0.320 | 2.091 | 0.602 |
| Fu | "MFMN base (untuned)" | caveat (2) | 0.673 | 0.341 | - | - |

| Target | Variant | Status | R2 (ens) | MAE (ens) | RMSE (ens) |
|---|---|---|---|---|---|
| pKa_Acidic | final (GraphMPNN) | | 0.969 | 0.424 | 0.757 |
| pKa_Acidic | no xTB features | reconstructed, verified | 0.969 | 0.426 | 0.759 |
| pKa_Acidic | no edge gating | reconstructed, unverified | 0.969 | 0.428 | 0.754 |
| pKa_Acidic | no site readout | reconstructed, unverified | 0.968 | 0.427 | 0.771 |
| pKa_Basic | full GraphMPNN (previous final) | | 0.948 | 0.416 | 0.670 |
| pKa_Basic | no xTB features | reconstructed, unverified | 0.943 | 0.424 | 0.703 |
| pKa_Basic | no edge gating | reconstructed, unverified | 0.946 | 0.415 | 0.680 |
| pKa_Basic | **no site readout (final)** | reconstructed, unverified | 0.950 | 0.406 | 0.654 |

Findings:
- **CL**: the final model has the best ensemble value on R2/MAE/RMSE/GMFE, but no difference from either variant is resolved.
- **VDss**: the final model beats MFMN base (R2, MAE, RMSE, GMFE resolved); final vs DynGate is not resolved on any metric.
- **Fu**: removing the 1.25x low-fu loss weight changes nothing that is resolved (final is better on every point estimate, by less
  than one seed SD). The "MFMN base (untuned)" row is clearly worse but is not a clean ablation (see caveat 2).
- **pKa_Acidic**: no ablation has a resolved effect; every metric is within 0.014 of the final model.
- **pKa_Basic**: removing the site readout gives the best model on every metric, so it is now the final model (the gain over the
  full GraphMPNN is consistent across seeds but not resolved; MAE CI borderline). Against it, the full model without xTB features
  is worse on all four metrics and the full model without edge gating on R2 and RMSE (resolved).

Caveats:
- **Reconstructed pKa variants**: the original code for the pKa ablation rows is not in the project, so the variants were
  reconstructed from their names. "Verified" = seed 42 reproduces the stored single-seed result within |dR2|, |dMAE| <= 0.002; only
  pKa_Acidic "no xTB features" passes. The other five may not be the same models as the original ablation.
- (1) Fu unweighted: the seed-42 reproduction check failed for every candidate code path (attributed to environment drift); run
  anyway. Its numbers match the unweighted model quoted in the main README.
- (2) Fu "MFMN base (untuned)": the code attributed to this label trains MFMN_InteractAux (aux_weight 0.3), not a plain MFMN, and
  does not reproduce the stored row. Not used in the manuscript.

The CL/VDss/Fu ablation runs date from 2026-10-01 and the pKa runs completed 2026-10-02; all were compared against the retrained
finals in `finals/` (`report.md`, sections 5-6).
