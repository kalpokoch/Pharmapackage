# Downstream exposure propagation (Experiment A)

Does replacing Jia et al.'s QSAR-predicted CL and VDss with ours reduce the fold error in predicted AUC and
Cmax, with the downstream exposure model held fixed? Produced by `manuscript/downstream_analysis.py`
(`python manuscript/downstream_analysis.py`); deterministic and idempotent.

## What this is, and is not

A **one-compartment IV model** with **dose linearity** and a **1 mg/kg dose** turns CL and VDss into AUC and
Cmax in closed form: `AUC = D / CL`, and `Cmax = D / V` for a bolus or `(R0 / CL) * (1 - exp(-k T))` for an
infusion of duration `T` with `R0 = D / T`, `k = CL / V`. Both the paper and this work normalise every
profile to 1 mg/kg, so parameter error can be propagated to exposure error analytically, with no training
data and no fitted downstream model.

This measures **how step-1 parameter error propagates into exposure error**, with the downstream model held
constant across arms. It is **not** a measurement of absolute concentration-profile accuracy. The only
legitimate comparison here is `ours` vs `jia` within this experiment. These numbers must **not** be placed
beside the paper's reported within-2-fold figures for AUC and Cmax: those come from the authors'
hierarchical-ML and PBPK pipelines scored against *observed concentration profiles*, whereas these score an
analytic exposure against *observed-parameter-derived* exposure. The two are not on the same scale.

## Data and arms

106 compounds, sheet `106_cmp_test_set` of the paper's Supporting Information
(`jm5c00340_si_001.xlsx`; CC BY-NC-ND, not redistributed here -- `src/downstream.py` downloads it from the
Europe PMC open-access mirror into `data/external/` and verifies its SHA-256).

| Arm | CL | VDss |
|---|---|---|
| `observed` (reference) | `CL_final(L/hour/kg)` | `VD_final(L/kg)` |
| `jia` | `pred_CL_(L/hour/kg)` | `pred_VD_(L/kg)` |
| `ours` | 10-seed ensemble, `results/final_models/CL/` | `results/final_models/VDss/` |

`observed` is the reference that defines what perfect parameter prediction would give under this model; it is
not a scored arm.

## Assertions that passed

- SI workbook SHA-256 matches the recorded digest.
- 106 unique compounds; all CL and VDss columns finite and positive.
- All 106 are in the official **test** split for CL, VDss *and* Fu (`*_train_test == "test"` in sheet
  `Fu_VDss_CL_modeling_set`), so none was trained on.
- Join: the `id` column of `results/final_models/{CL,VDss}/ensemble_predictions.csv` is **`row_idx` of
  `data/cl_vdss/modeling_index.parquet`**, which carries `PUBCHEM_CID`; that CID is the join key to the SI.
  It is neither `ID_trend` nor `PubChem_CID` directly (see "Deviations" below).
- Exactly 106 compounds survive the join for CL and for VDss.
- Our back-transformed `10 ** y_true` reproduces the SI's `CL_final` / `VD_final` (rtol 1e-3; observed
  agreement ~1e-9).
- Per-seed arrays average to the stored ensemble prediction (atol 1e-5).
- Infusion time: 0 blank/NaN (would be treated as a bolus) and 0 exactly zero, so no compound
  needed bolus imputation; the column is complete.

## Results

Fold error = `max(pred/obs, obs/pred)`; GMFE = `exp(mean |ln(pred/obs)|)`; R2/MAE/RMSE on log10 exposure.
95% CIs: compound-level bootstrap, B = 10,000, the same resampling protocol and RNG seed as
`manuscript/analysis.py`. Differences use a **paired** bootstrap: the same resampled compounds in both arms.

| Endpoint | Arm | GMFE | within-2-fold | within-3-fold | median FE | R2 (log10) |
|---|---|---|---|---|---|---|
| AUC | Jia et al. QSAR | 1.969 | 0.642 | 0.858 | 1.647 | 0.433 |
| AUC | This work (10-seed) | 1.953 | 0.613 | 0.830 | 1.579 | 0.411 |
| Cmax | Jia et al. QSAR | 1.872 | 0.632 | 0.802 | 1.597 | 0.572 |
| Cmax | This work (10-seed) | 1.774 | 0.689 | 0.840 | 1.452 | 0.615 |

Paired differences (ours - jia):

| Endpoint | Metric | Difference | 95% CI | Verdict |
|---|---|---|---|---|
| AUC | R2 | -0.0219 | [-0.1089, +0.0583] | CI includes 0 |
| AUC | GMFE | -0.0160 | [-0.1213, +0.0927] | CI includes 0 |
| AUC | within_2fold | -0.0283 | [-0.1038, +0.0472] | CI includes 0 |
| AUC | within_3fold | -0.0283 | [-0.0755, +0.0189] | CI includes 0 |
| AUC | median_FE | -0.0686 | [-0.2486, +0.1479] | CI includes 0 |
| Cmax | R2 | +0.0430 | [-0.0426, +0.1218] | CI includes 0 |
| Cmax | GMFE | -0.0973 | [-0.2285, +0.0344] | CI includes 0 |
| Cmax | within_2fold | +0.0566 | [-0.0377, +0.1509] | CI includes 0 |
| Cmax | within_3fold | +0.0377 | [-0.0283, +0.1038] | CI includes 0 |
| Cmax | median_FE | -0.1445 | [-0.4006, +0.0092] | CI includes 0 |

Per-seed spread of the `ours` arm (each of the 10 seeds propagated separately) is in
`exposure_summary.csv` (`seed_mean`, `seed_sd`), to show whether the downstream result depends on
ensembling the way the step-1 CL and Fu results do.

## Supplementary checks

1. AUC fold error equals CL fold error exactly (max |deviation| = 3.55e-15); the AUC column therefore carries no information beyond the CL column. Cmax depends on CL, V and infusion time jointly and is the genuinely new quantity.
2. Infusion-time sensitivity (Cmax recomputed as if every compound were an IV bolus): GMFE jia 1.872 -> 1.889, ours 1.774 -> 1.759; within-2-fold jia 0.632 -> 0.604, ours 0.689 -> 0.698.
3. Fu does not enter the one-compartment model: AUC depends on CL alone and Cmax on CL, VDss and infusion time. Fu is carried in the per-compound table for reference only. Under a PBPK model it would matter through tissue partitioning, which this model does not represent.

Ten compounds with the largest Cmax fold error under `ours` are in `exposure_worst_cmax.csv`; the
largest is ceftaroline fosamil (fold error 13.6; Jia et al. 15.1).

## Experiment B

**Skipped: no observed concentration-time data is present.** The paper's SI workbook contains nine
sheets and none has a time or concentration column; the 8,715 training and 1,351 test concentration
points appear only as rendered plots in the Figures S2-S5 PDF. Experiment B therefore cannot be run
without data obtained from the authors or from the Lombardo-cited primary literature. No concentration
value was reconstructed, digitized or simulated.

## Outputs

- `exposure_per_compound.csv` -- one row per compound: identifiers, infusion time, CL and VDss for all
  three arms, AUC and Cmax for all three arms, and fold errors for `jia` and `ours`.
- `exposure_summary.csv` -- arm x metric x {point estimate, CI, seed mean, seed SD}.
- `exposure_paired.csv` -- metric x {difference, CI, verdict}.
- `exposure_cmax_bolus_sensitivity.csv` -- Cmax metrics with every compound treated as a bolus.
- `exposure_worst_cmax.csv` -- the 10 worst Cmax fold errors under `ours`.
- `../../manuscript/t_downstream.csv`, `../../manuscript/fig6_downstream.png`.
