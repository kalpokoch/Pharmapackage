# Results (10 seeds, official test split)

Every model was trained on the official train split with 10 seeds (42-51) and scored once on the official held-out test
split. "Ensemble" = metric of the mean prediction over the 10 seeds; "seed mean ± SD" = average and sample SD of the
per-seed metrics. `manuscript/analysis.py` computes every number in the Results draft from the files in this folder.

## Final models

| Target | Model | n_test | R² | MAE | RMSE | GMFE | within-2-fold |
|---|---|---|---|---|---|---|---|
| pKa_Acidic | GraphMPNN | 776 | 0.969 | 0.424 | 0.757 | - | - |
| pKa_Basic | GraphMPNN, no site readout | 815 | 0.950 | 0.406 | 0.654 | - | - |
| CL | MFMN_DynGate (aux_weight 0.8) | 177 | 0.497 | 0.305 | 0.412 | 2.018 | 0.605 |
| VDss | MFMN_InteractAux (aux_weight 2.0) | 177 | 0.648 | 0.247 | 0.328 | 1.764 | 0.684 |
| Fu | MFMN_DynGate + 1.25x low-fu loss weight | 633 | 0.712 | 0.319 | 0.435 | 2.086 | 0.613 |

pKa MAE/RMSE are in pKa units; CL/VDss/Fu are modeled in log10 space. The pKa_Basic final model omits the
ionizable-site readout (`GraphMPNN(..., site_readout=False)`); it scored best on every metric in the ablation below.

## Baselines (untuned, library defaults)

| Target | Proposed | Random Forest | XGBoost | SVM |
|---|---|---|---|---|
| pKa_Acidic (R² / MAE) | **0.969 / 0.424** | 0.927 / 0.699 | 0.919 / 0.743 | 0.770 / 1.475 |
| pKa_Basic (R² / MAE) | **0.950 / 0.406** | 0.862 / 0.758 | 0.881 / 0.701 | 0.723 / 1.067 |
| CL (GMFE / w2f) | 2.018 / 0.605 | **2.010 / 0.638** | 2.266 / 0.497 | 2.042 / 0.627 |
| VDss (GMFE / w2f) | **1.764 / 0.684** | 1.848 / 0.661 | 2.026 / 0.588 | 1.919 / 0.610 |
| Fu (GMFE / w2f) | **2.086 / 0.613** | 2.391 / 0.488 | 2.316 / 0.531 | 2.284 / 0.534 |

Random Forest: 10-seed ensemble (seeds 42-51). XGBoost: seeds 42-51, but with default settings there is no row/column
subsampling, so all 10 fits are identical. SVM (`SVR()`): deterministic, fit once. Features are standardized on the
training split. On CL the untuned Random Forest is on par with the proposed model; the proposed model leads elsewhere.

## Ablation

| Target | Variant | R² | MAE | RMSE | GMFE |
|---|---|---|---|---|---|
| CL | final (MFMN_DynGate) | 0.497 | 0.305 | 0.412 | 2.018 |
| CL | MFMN base | 0.468 | 0.313 | 0.424 | 2.057 |
| CL | MFMN_InteractAux | 0.490 | 0.309 | 0.415 | 2.036 |
| VDss | final (MFMN_InteractAux) | 0.648 | 0.247 | 0.328 | 1.764 |
| VDss | MFMN base | 0.571 | 0.272 | 0.362 | 1.872 |
| VDss | MFMN_DynGate | 0.647 | 0.248 | 0.329 | 1.771 |
| Fu | final (DynGate + 1.25x low-fu weight) | 0.712 | 0.319 | 0.435 | 2.086 |
| Fu | DynGate, no loss weight | 0.711 | 0.320 | 0.436 | 2.091 |
| pKa_Acidic | final (GraphMPNN) | 0.969 | 0.424 | 0.757 | - |
| pKa_Acidic | no xTB features | 0.969 | 0.426 | 0.759 | - |
| pKa_Acidic | no edge gating | 0.969 | 0.428 | 0.754 | - |
| pKa_Acidic | no site readout | 0.968 | 0.427 | 0.771 | - |
| pKa_Basic | final (GraphMPNN, no site readout) | 0.950 | 0.406 | 0.654 | - |
| pKa_Basic | full GraphMPNN | 0.948 | 0.416 | 0.670 | - |
| pKa_Basic | full, no xTB features | 0.943 | 0.424 | 0.703 | - |
| pKa_Basic | full, no edge gating | 0.946 | 0.415 | 0.680 | - |

Paired-bootstrap confidence intervals for every difference are in `manuscript/t_ablation.csv` (and for the baselines in
`manuscript/t_baselines.csv`).

## Layout

- `final_models/<target>/` - `ensemble_predictions.csv` (id, observed, ensemble prediction, seed SD, absolute error),
  `ensemble_summary.csv` (ensemble and seed statistics per metric, with the published value), `per_seed_metrics.csv`,
  `per_seed_preds.npy` (10 x n_test, rows = seeds 42-51, same row order as `ensemble_predictions.csv`).
- `baselines/<target>/<RandomForest|XGBoost|SVM>/` - `ensemble_predictions.csv`, `per_seed_metrics.csv` (same test rows
  and order as the final model).
- `ablation/<target>/<variant>/` - `per_seed_preds.npy`, `per_seed_metrics.csv` (same test rows and order as the final
  model). `ablation/pKa_Basic/full_graphmpnn/` is the full GraphMPNN (the previous pKa_Basic final model).
