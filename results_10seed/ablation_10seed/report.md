# Ablation with 10 seeds - report

Generated 2026-10-02T19:44:07 by `analyze_ablation_10seed.py`. Facts only; no recommendations about what to claim.

## 1. Status of all 12 variants

| target | variant | slug | status | detail |
|---|---|---|---|---|
| CL | MFMN (base) | mfmn_base | RUN (10 seeds) |  |
| CL | MFMN_InteractAux | interactaux | RUN (10 seeds) |  |
| VDss | MFMN (base) | mfmn_base | RUN (10 seeds) |  |
| VDss | MFMN_DynGate | dyngate | RUN (10 seeds) |  |
| Fu | MFMN_DynGate (unweighted) | dyngate_unweighted | RUN WITH CAVEAT (10/10 seeds) | seed-42 gate FAILED for every candidate (candidate A and B give identical results); running candidate A (primary) anyway at the user's instruction; see the Fu control in the report for environment drift |
| Fu | MFMN base (untuned) | mfmn_base_untuned | RUN WITH CAVEAT (10/10 seeds) | no candidate (C1 existing script; C2/C3 reconstructed plain-MFMN) reproduces the CSV row within the gate; running C1 because the project's own records attribute the label to that script. It trains MFMN_InteractAux (aux_weight 0.3, pca32 context), not plain MFMN, and does NOT reproduce the CSV row |
| pKa_Acidic | GraphMPNN, no xTB features | no_xtb | RECONSTRUCTED, seed-42 gate passed (10/10 seeds) | reconstructed; seed 42 reproduces the CSV row within the gate |
| pKa_Acidic | GraphMPNN, no edge gating | no_edge_gating | RECONSTRUCTED, seed-42 gate FAILED (10/10 seeds) | reconstructed; seed 42 does NOT reproduce the CSV row within the gate (see candidates_tested) |
| pKa_Acidic | GraphMPNN, no site readout | no_site_readout | RECONSTRUCTED, seed-42 gate FAILED (10/10 seeds) | reconstructed; seed 42 does NOT reproduce the CSV row within the gate (see candidates_tested) |
| pKa_Basic | GraphMPNN, no xTB features | no_xtb | RECONSTRUCTED, seed-42 gate FAILED (10/10 seeds) | reconstructed; seed 42 does NOT reproduce the CSV row within the gate (see candidates_tested) |
| pKa_Basic | GraphMPNN, no edge gating | no_edge_gating | RECONSTRUCTED, seed-42 gate FAILED (10/10 seeds) | reconstructed; seed 42 does NOT reproduce the CSV row within the gate (see candidates_tested) |
| pKa_Basic | GraphMPNN, no site readout | no_site_readout | RECONSTRUCTED, seed-42 gate FAILED (10/10 seeds) | reconstructed; seed 42 does NOT reproduce the CSV row within the gate (see candidates_tested) |


## 2. Smoke test: seed 42 vs `proposed_model_ablation.csv`

Gate: |dR2| <= 0.002 and |dMAE| <= 0.002 and y_test identical to the final model's. The latest smoke run of each (variant, code path) is shown (tag in the last column). Candidates for the same variant are listed in priority order; `fuC2`/`fuC3` and every pKa variant are RECONSTRUCTED code paths.


Candidate code paths: `clvd` = `ablation/ablation_official_test_CL_VDss_Fu.ipynb` cells 1,3,5 with `src/mfmn.py`; `fuA` = `MFMN_v2.ipynb` cells 1,13 + `mfmn/experiments/expv2_aux_h7_dyngate.py::fit_predict_dyngate` with `RECIPE['Fu']`; `fuB` = notebook `05_Fu_MFMN_DynGate.ipynb` cells 1-4 with the low-fu multiplier set to 1.0 (only change); `fuC1` = `mfmn/experiments/expv2_aux_targets_final.py::fit_predict` defaults (class `MFMN_InteractAux`, aux_weight 0.3); `fuC2`/`fuC3` = RECONSTRUCTED plain-`MFMN` versions of the original `fit_predict_dyngate` loop (C2: pca32 context, dropout 0.2, wd 1e-2; C3: `RECIPE['Fu']`); `pka_*` = original `fit_predict_gnn` (10seed notebook cell 9) with a RECONSTRUCTED modification (descriptions in section 8).


## 3. Seconds per seed (full run, 10 concurrent workers sharing one MIG slice; smoke-test timings in section 2)

| target | variant | seeds_logged | sec_mean | sec_min | sec_max | failed | device |
|---|---|---|---|---|---|---|---|
| CL | mfmn_base | 10 | 9.0 | 4.4 | 16.1 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| CL | interactaux | 10 | 70.4 | 51.9 | 104.3 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| VDss | mfmn_base | 10 | 4.9 | 4.4 | 5.4 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| VDss | dyngate | 10 | 78.9 | 53.3 | 113.1 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| Fu | dyngate_unweighted | 10 | 74.0 | 44.6 | 157.8 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| Fu | mfmn_base_untuned | 10 | 218.7 | 100.9 | 288.7 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| pKa_Acidic | no_xtb | 10 | 997.4 | 642.5 | 1363.5 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| pKa_Acidic | no_edge_gating | 10 | 455.1 | 289.8 | 551.7 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| pKa_Acidic | no_site_readout | 10 | 1041.8 | 691.5 | 1374.1 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| pKa_Basic | no_xtb | 10 | 1055.4 | 808.2 | 1389.2 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| pKa_Basic | no_edge_gating | 10 | 496.7 | 399.1 | 578.5 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| pKa_Basic | no_site_readout | 10 | 1160.8 | 585.9 | 1337.4 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |


## 4. Seeds completed per variant

| target | variant | status | n_seeds | seeds |
|---|---|---|---|---|
| CL | mfmn_base | RUN | 10 | 42 43 44 45 46 47 48 49 50 51 |
| CL | interactaux | RUN | 10 | 42 43 44 45 46 47 48 49 50 51 |
| VDss | mfmn_base | RUN | 10 | 42 43 44 45 46 47 48 49 50 51 |
| VDss | dyngate | RUN | 10 | 42 43 44 45 46 47 48 49 50 51 |
| Fu | dyngate_unweighted | RUN_WITH_CAVEAT | 10 | 42 43 44 45 46 47 48 49 50 51 |
| Fu | mfmn_base_untuned | RUN_WITH_CAVEAT | 10 | 42 43 44 45 46 47 48 49 50 51 |
| pKa_Acidic | no_xtb | RECONSTRUCTED_VERIFIED | 10 | 42 43 44 45 46 47 48 49 50 51 |
| pKa_Acidic | no_edge_gating | RECONSTRUCTED_UNVERIFIED | 10 | 42 43 44 45 46 47 48 49 50 51 |
| pKa_Acidic | no_site_readout | RECONSTRUCTED_UNVERIFIED | 10 | 42 43 44 45 46 47 48 49 50 51 |
| pKa_Basic | no_xtb | RECONSTRUCTED_UNVERIFIED | 10 | 42 43 44 45 46 47 48 49 50 51 |
| pKa_Basic | no_edge_gating | RECONSTRUCTED_UNVERIFIED | 10 | 42 43 44 45 46 47 48 49 50 51 |
| pKa_Basic | no_site_readout | RECONSTRUCTED_UNVERIFIED | 10 | 42 43 44 45 46 47 48 49 50 51 |


## 5. Results (only variants that were run)

Seed mean ± SD (ddof=1) over seeds and the metrics of the 10-seed ensemble; `diff` = variant - final, paired by seed index.

| target | variant | name | status | n_seeds | R2_mean_pm_sd | MAE_mean_pm_sd | RMSE_mean_pm_sd | R2_ens | MAE_ens | diff_mean_R2 | diff_in_final_sd_R2 | flag_R2 | diff_mean_MAE | diff_in_final_sd_MAE | flag_MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CL | mfmn_base | MFMN (base) | RUN | 10 | 0.364 ± 0.076 | 0.348 ± 0.022 | 0.463 ± 0.027 | 0.4677 | 0.3133 | -0.0761 | -3.8300 | overlaps_zero | 0.0237 | 3.2100 | overlaps_zero |
| CL | interactaux | MFMN_InteractAux | RUN | 10 | 0.426 ± 0.036 | 0.329 ± 0.013 | 0.440 ± 0.014 | 0.4897 | 0.3088 | -0.0148 | -0.7400 | overlaps_zero | 0.0053 | 0.7100 | overlaps_zero |
| CL | final | FINAL | final | 10 | 0.441 ± 0.020 | 0.324 ± 0.007 | 0.435 ± 0.008 | 0.4967 | 0.3050 |  |  | reference |  |  | reference |
| VDss | mfmn_base | MFMN (base) | RUN | 10 | 0.528 ± 0.029 | 0.287 ± 0.009 | 0.379 ± 0.012 | 0.5713 | 0.2724 | -0.0803 | -2.7800 | excludes_zero | 0.0280 | 2.4100 | excludes_zero |
| VDss | dyngate | MFMN_DynGate | RUN | 10 | 0.609 ± 0.009 | 0.260 ± 0.004 | 0.346 ± 0.004 | 0.6465 | 0.2482 | 0.0002 | 0.0100 | overlaps_zero | 0.0018 | 0.1500 | overlaps_zero |
| VDss | final | FINAL | final | 10 | 0.609 ± 0.029 | 0.259 ± 0.012 | 0.345 ± 0.013 | 0.6478 | 0.2466 |  |  | reference |  |  | reference |
| Fu | dyngate_unweighted | MFMN_DynGate (unweighted) | RUN_WITH_CAVEAT | 10 | 0.680 ± 0.009 | 0.337 ± 0.007 | 0.459 ± 0.007 | 0.7113 | 0.3204 | -0.0009 | -0.1000 | overlaps_zero | 0.0000 | 0.0100 | overlaps_zero |
| Fu | mfmn_base_untuned | MFMN base (untuned) | RUN_WITH_CAVEAT | 10 | 0.631 ± 0.003 | 0.361 ± 0.004 | 0.493 ± 0.002 | 0.6728 | 0.3408 | -0.0500 | -5.8500 | excludes_zero | 0.0244 | 4.2800 | excludes_zero |
| Fu | final | FINAL | final | 10 | 0.681 ± 0.009 | 0.337 ± 0.006 | 0.458 ± 0.006 | 0.7121 | 0.3193 |  |  | reference |  |  | reference |
| pKa_Acidic | no_xtb | GraphMPNN, no xTB features | RECONSTRUCTED_VERIFIED | 10 | 0.963 ± 0.002 | 0.488 ± 0.017 | 0.822 ± 0.021 | 0.9687 | 0.4256 | -0.0002 | -0.1000 | overlaps_zero | 0.0043 | 0.2200 | overlaps_zero |
| pKa_Acidic | no_edge_gating | GraphMPNN, no edge gating | RECONSTRUCTED_UNVERIFIED | 10 | 0.964 ± 0.002 | 0.484 ± 0.017 | 0.814 ± 0.027 | 0.9691 | 0.4277 | 0.0005 | 0.2500 | overlaps_zero | 0.0003 | 0.0100 | overlaps_zero |
| pKa_Acidic | no_site_readout | GraphMPNN, no site readout | RECONSTRUCTED_UNVERIFIED | 10 | 0.963 ± 0.002 | 0.486 ± 0.017 | 0.830 ± 0.025 | 0.9677 | 0.4269 | -0.0009 | -0.4200 | overlaps_zero | 0.0026 | 0.1300 | overlaps_zero |
| pKa_Acidic | final | FINAL | final | 10 | 0.963 ± 0.002 | 0.484 ± 0.020 | 0.820 ± 0.025 | 0.9689 | 0.4241 |  |  | reference |  |  | reference |
| pKa_Basic | no_xtb | GraphMPNN, no xTB features | RECONSTRUCTED_UNVERIFIED | 10 | 0.931 ± 0.004 | 0.483 ± 0.014 | 0.772 ± 0.024 | 0.9427 | 0.4238 | -0.0064 | -2.4000 | excludes_zero | 0.0109 | 1.0400 | overlaps_zero |
| pKa_Basic | no_edge_gating | GraphMPNN, no edge gating | RECONSTRUCTED_UNVERIFIED | 10 | 0.935 ± 0.004 | 0.469 ± 0.015 | 0.747 ± 0.023 | 0.9463 | 0.4147 | -0.0020 | -0.7500 | overlaps_zero | -0.0027 | -0.2600 | overlaps_zero |
| pKa_Basic | no_site_readout | GraphMPNN, no site readout | RECONSTRUCTED_UNVERIFIED | 10 | 0.940 ± 0.002 | 0.461 ± 0.016 | 0.720 ± 0.013 | 0.9504 | 0.4060 | 0.0028 | 1.0400 | overlaps_zero | -0.0116 | -1.1000 | excludes_zero |
| pKa_Basic | final | FINAL | final | 10 | 0.937 ± 0.003 | 0.472 ± 0.010 | 0.736 ± 0.016 | 0.9478 | 0.4162 |  |  | reference |  |  | reference |


### Interpretation notes

- Seeds share one fixed test set, so the seed-level tests capture training randomness only. The compound-level bootstrap captures test-set sampling. Both are reported and not merged.
- Pairing 'by seed index' is a convenience: the variant and the final model are different architectures, so seed 42 does not mean the same random numbers across them. It is paired by seed index, nothing stronger.
- There are many comparisons (variants x metrics). The text here uses 'CI excludes zero' / 'CI overlaps zero' and reports the Holm-adjusted p-values (`p_holm`, over the variants of the same target and metric, from the exact Wilcoxon p-values; `p_holm_ttest` from the paired t-test).
- Bootstrap: B = 10,000, `np.random.default_rng(0)`, one resampled-index matrix per target shared by every variant and the final model, ensemble metrics recomputed inside each draw; 'ens_p_variant_better' is the share of draws in which the variant's metric is better than the final's.


### Component effects per target: ensemble-level 95% CI of (variant - final)


**CL** (n_test=177)

| variant | metric | n_seeds | diff_mean | diff_in_final_sd | n_seeds_variant_better | p_wilcoxon | p_holm | ens_diff | ens_ci95_lo | ens_ci95_hi | ens_p_variant_better | ens_ci_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mfmn_base | R2 | 10 | -0.0761 | -3.8307 | 2 | 0.0098 | 0.0195 | -0.0289 | -0.0767 | 0.0219 | 0.1277 | overlaps_zero |
| mfmn_base | MAE | 10 | 0.0237 | 3.2100 | 1 | 0.0098 | 0.0195 | 0.0083 | -0.0068 | 0.0237 | 0.1423 | overlaps_zero |
| mfmn_base | RMSE | 10 | 0.0280 | 3.6292 | 2 | 0.0098 | 0.0195 | 0.0117 | -0.0084 | 0.0317 | 0.1277 | overlaps_zero |
| mfmn_base | GMFE | 10 | 0.1204 | 3.3572 | 1 | 0.0098 | 0.0195 | 0.0389 | -0.0322 | 0.1131 | 0.1423 | overlaps_zero |
| mfmn_base | within_2fold | 10 | -0.0175 | -0.9781 | 3 | 0.2324 | 0.4648 | -0.0339 | -0.0847 | 0.0169 | 0.0698 | overlaps_zero |
| mfmn_base | within_3fold | 10 | -0.0345 | -2.1552 | 1 | 0.0059 | 0.0117 | -0.0395 | -0.0791 | 0.0000 | 0.0143 | overlaps_zero |
| mfmn_base | within_5fold | 10 | -0.0198 | -1.9138 | 2 | 0.0391 | 0.0781 | 0.0000 | -0.0282 | 0.0282 | 0.4187 | overlaps_zero |
| interactaux | R2 | 10 | -0.0148 | -0.7446 | 3 | 0.2754 | 0.2754 | -0.0070 | -0.0260 | 0.0127 | 0.2411 | overlaps_zero |
| interactaux | MAE | 10 | 0.0053 | 0.7150 | 2 | 0.4316 | 0.4316 | 0.0038 | -0.0039 | 0.0118 | 0.1705 | overlaps_zero |
| interactaux | RMSE | 10 | 0.0056 | 0.7235 | 3 | 0.2754 | 0.2754 | 0.0028 | -0.0048 | 0.0112 | 0.2411 | overlaps_zero |
| interactaux | GMFE | 10 | 0.0264 | 0.7354 | 2 | 0.4316 | 0.4316 | 0.0177 | -0.0181 | 0.0554 | 0.1705 | overlaps_zero |
| interactaux | within_2fold | 10 | 0.0124 | 0.6942 | 5 | 0.2500 | 0.4648 | 0.0000 | -0.0282 | 0.0339 | 0.4223 | overlaps_zero |
| interactaux | within_3fold | 10 | -0.0181 | -1.1306 | 3 | 0.1055 | 0.1055 | -0.0169 | -0.0395 | 0.0000 | 0.0000 | overlaps_zero |
| interactaux | within_5fold | 10 | 0.0011 | 0.1094 | 5 | 0.7422 | 0.7422 | 0.0056 | -0.0113 | 0.0226 | 0.6093 | overlaps_zero |


- `interactaux`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, GMFE, within_2fold, within_3fold, within_5fold.

- `mfmn_base`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, GMFE, within_2fold, within_3fold, within_5fold.


**VDss** (n_test=177)

| variant | metric | n_seeds | diff_mean | diff_in_final_sd | n_seeds_variant_better | p_wilcoxon | p_holm | ens_diff | ens_ci95_lo | ens_ci95_hi | ens_p_variant_better | ens_ci_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mfmn_base | R2 | 10 | -0.0803 | -2.7848 | 0 | 0.0020 | 0.0039 | -0.0766 | -0.1221 | -0.0367 | 0.0001 | excludes_zero |
| mfmn_base | MAE | 10 | 0.0280 | 2.4145 | 1 | 0.0039 | 0.0078 | 0.0258 | 0.0113 | 0.0408 | 0.0001 | excludes_zero |
| mfmn_base | RMSE | 10 | 0.0339 | 2.6739 | 0 | 0.0020 | 0.0039 | 0.0339 | 0.0164 | 0.0516 | 0.0001 | excludes_zero |
| mfmn_base | GMFE | 10 | 0.1205 | 2.4876 | 1 | 0.0039 | 0.0078 | 0.1082 | 0.0467 | 0.1751 | 0.0001 | excludes_zero |
| mfmn_base | within_2fold | 10 | -0.0384 | -1.3741 | 1 | 0.0078 | 0.0156 | -0.0282 | -0.0791 | 0.0226 | 0.1103 | overlaps_zero |
| mfmn_base | within_3fold | 10 | -0.0520 | -2.9693 | 1 | 0.0039 | 0.0078 | -0.0621 | -0.1017 | -0.0282 | 0.0000 | excludes_zero |
| mfmn_base | within_5fold | 10 | -0.0164 | -1.0218 | 2 | 0.0488 | 0.0977 | -0.0056 | -0.0169 | 0.0000 | 0.0000 | overlaps_zero |
| dyngate | R2 | 10 | 0.0002 | 0.0064 | 5 | 0.9219 | 0.9219 | -0.0013 | -0.0152 | 0.0124 | 0.4247 | overlaps_zero |
| dyngate | MAE | 10 | 0.0018 | 0.1533 | 4 | 0.4922 | 0.4922 | 0.0017 | -0.0035 | 0.0069 | 0.2650 | overlaps_zero |
| dyngate | RMSE | 10 | 0.0001 | 0.0083 | 5 | 0.9219 | 0.9219 | 0.0006 | -0.0057 | 0.0069 | 0.4247 | overlaps_zero |
| dyngate | GMFE | 10 | 0.0069 | 0.1430 | 4 | 0.4922 | 0.4922 | 0.0068 | -0.0146 | 0.0282 | 0.2650 | overlaps_zero |
| dyngate | within_2fold | 10 | 0.0096 | 0.3435 | 6 | 0.5566 | 0.5566 | 0.0113 | -0.0169 | 0.0395 | 0.7314 | overlaps_zero |
| dyngate | within_3fold | 10 | 0.0051 | 0.2905 | 4 | 0.7422 | 0.7422 | 0.0113 | -0.0169 | 0.0395 | 0.7351 | overlaps_zero |
| dyngate | within_5fold | 10 | 0.0023 | 0.1409 | 3 | 0.9375 | 0.9375 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | overlaps_zero |


- `dyngate`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, GMFE, within_2fold, within_3fold, within_5fold.

- `mfmn_base`: ensemble CI excludes zero for: R2, MAE, RMSE, GMFE, within_3fold; overlaps zero for: within_2fold, within_5fold.


**Fu** (n_test=633)

| variant | metric | n_seeds | diff_mean | diff_in_final_sd | n_seeds_variant_better | p_wilcoxon | p_holm | ens_diff | ens_ci95_lo | ens_ci95_hi | ens_p_variant_better | ens_ci_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dyngate_unweighted | R2 | 10 | -0.0009 | -0.1008 | 5 | 0.9219 | 0.9219 | -0.0008 | -0.0059 | 0.0044 | 0.3802 | overlaps_zero |
| dyngate_unweighted | MAE | 10 | 0.0000 | 0.0055 | 4 | 0.9219 | 0.9219 | 0.0011 | -0.0022 | 0.0043 | 0.2595 | overlaps_zero |
| dyngate_unweighted | RMSE | 10 | 0.0006 | 0.0998 | 5 | 0.9219 | 0.9219 | 0.0006 | -0.0032 | 0.0045 | 0.3802 | overlaps_zero |
| dyngate_unweighted | GMFE | 10 | 0.0002 | 0.0081 | 4 | 0.9219 | 0.9219 | 0.0053 | -0.0103 | 0.0209 | 0.2595 | overlaps_zero |
| dyngate_unweighted | within_2fold | 10 | -0.0013 | -0.0858 | 5 | 0.8457 | 0.8457 | -0.0111 | -0.0300 | 0.0063 | 0.1026 | overlaps_zero |
| dyngate_unweighted | within_3fold | 10 | -0.0062 | -0.8673 | 2 | 0.0840 | 0.0840 | 0.0063 | -0.0079 | 0.0205 | 0.7818 | overlaps_zero |
| dyngate_unweighted | within_5fold | 10 | 0.0017 | 0.2097 | 6 | 0.4922 | 0.4922 | 0.0063 | -0.0016 | 0.0158 | 0.8998 | overlaps_zero |
| mfmn_base_untuned | R2 | 10 | -0.0500 | -5.8493 | 0 | 0.0020 | 0.0039 | -0.0393 | -0.0656 | -0.0129 | 0.0012 | excludes_zero |
| mfmn_base_untuned | MAE | 10 | 0.0244 | 4.2793 | 0 | 0.0020 | 0.0039 | 0.0214 | 0.0067 | 0.0361 | 0.0016 | excludes_zero |
| mfmn_base_untuned | RMSE | 10 | 0.0346 | 5.6430 | 0 | 0.0020 | 0.0039 | 0.0287 | 0.0095 | 0.0476 | 0.0012 | excludes_zero |
| mfmn_base_untuned | GMFE | 10 | 0.1253 | 4.4028 | 0 | 0.0020 | 0.0039 | 0.1056 | 0.0326 | 0.1793 | 0.0016 | excludes_zero |
| mfmn_base_untuned | within_2fold | 10 | -0.0427 | -2.8956 | 0 | 0.0020 | 0.0039 | -0.0363 | -0.0711 | -0.0016 | 0.0193 | excludes_zero |
| mfmn_base_untuned | within_3fold | 10 | -0.0256 | -3.6025 | 1 | 0.0039 | 0.0078 | -0.0126 | -0.0411 | 0.0158 | 0.1827 | overlaps_zero |
| mfmn_base_untuned | within_5fold | 10 | -0.0130 | -1.5634 | 1 | 0.0059 | 0.0117 | -0.0142 | -0.0332 | 0.0047 | 0.0611 | overlaps_zero |


- `dyngate_unweighted`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, GMFE, within_2fold, within_3fold, within_5fold.

- `mfmn_base_untuned`: ensemble CI excludes zero for: R2, MAE, RMSE, GMFE, within_2fold; overlaps zero for: within_3fold, within_5fold.


**pKa_Acidic** (n_test=776)

| variant | metric | n_seeds | diff_mean | diff_in_final_sd | n_seeds_variant_better | p_wilcoxon | p_holm | ens_diff | ens_ci95_lo | ens_ci95_hi | ens_p_variant_better | ens_ci_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no_xtb | R2 | 10 | -0.0002 | -0.1034 | 4 | 0.5566 | 0.7500 | -0.0001 | -0.0013 | 0.0010 | 0.3968 | overlaps_zero |
| no_xtb | MAE | 10 | 0.0043 | 0.2194 | 4 | 0.2754 | 0.8262 | 0.0015 | -0.0078 | 0.0108 | 0.3749 | overlaps_zero |
| no_xtb | RMSE | 10 | 0.0026 | 0.1078 | 4 | 0.4922 | 0.7500 | 0.0018 | -0.0116 | 0.0161 | 0.3968 | overlaps_zero |
| no_xtb | within_1_pKa_unit | 10 | -0.0017 | -0.1605 | 5 | 0.6250 | 1.0000 | 0.0026 | -0.0090 | 0.0142 | 0.6257 | overlaps_zero |
| no_xtb | within_2_pKa_unit | 10 | 0.0012 | 0.2103 | 4 | 0.8203 | 1.0000 | -0.0013 | -0.0039 | 0.0000 | 0.0000 | overlaps_zero |
| no_edge_gating | R2 | 10 | 0.0005 | 0.2457 | 6 | 0.3750 | 0.7500 | 0.0002 | -0.0011 | 0.0018 | 0.6189 | overlaps_zero |
| no_edge_gating | MAE | 10 | 0.0003 | 0.0141 | 5 | 0.9219 | 0.9844 | 0.0036 | -0.0066 | 0.0135 | 0.2458 | overlaps_zero |
| no_edge_gating | RMSE | 10 | -0.0062 | -0.2507 | 6 | 0.3750 | 0.7500 | -0.0030 | -0.0199 | 0.0148 | 0.6189 | overlaps_zero |
| no_edge_gating | within_1_pKa_unit | 10 | -0.0014 | -0.1358 | 3 | 0.6523 | 1.0000 | -0.0026 | -0.0142 | 0.0090 | 0.2872 | overlaps_zero |
| no_edge_gating | within_2_pKa_unit | 10 | 0.0006 | 0.1168 | 4 | 1.0000 | 1.0000 | 0.0000 | -0.0064 | 0.0064 | 0.4212 | overlaps_zero |
| no_site_readout | R2 | 10 | -0.0009 | -0.4200 | 2 | 0.1602 | 0.4805 | -0.0012 | -0.0027 | 0.0003 | 0.0601 | overlaps_zero |
| no_site_readout | MAE | 10 | 0.0026 | 0.1342 | 4 | 0.4922 | 0.9844 | 0.0029 | -0.0090 | 0.0146 | 0.3213 | overlaps_zero |
| no_site_readout | RMSE | 10 | 0.0103 | 0.4195 | 2 | 0.1602 | 0.4805 | 0.0140 | -0.0039 | 0.0301 | 0.0601 | overlaps_zero |
| no_site_readout | within_1_pKa_unit | 10 | 0.0003 | 0.0247 | 6 | 0.9219 | 1.0000 | -0.0039 | -0.0155 | 0.0077 | 0.2210 | overlaps_zero |
| no_site_readout | within_2_pKa_unit | 10 | 0.0005 | 0.0934 | 4 | 1.0000 | 1.0000 | -0.0026 | -0.0077 | 0.0026 | 0.0970 | overlaps_zero |


- `no_edge_gating`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, within_1_pKa_unit, within_2_pKa_unit.

- `no_site_readout`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, within_1_pKa_unit, within_2_pKa_unit.

- `no_xtb`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, within_1_pKa_unit, within_2_pKa_unit.


**pKa_Basic** (n_test=815)

| variant | metric | n_seeds | diff_mean | diff_in_final_sd | n_seeds_variant_better | p_wilcoxon | p_holm | ens_diff | ens_ci95_lo | ens_ci95_hi | ens_p_variant_better | ens_ci_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no_xtb | R2 | 10 | -0.0064 | -2.3951 | 1 | 0.0137 | 0.0273 | -0.0051 | -0.0098 | -0.0015 | 0.0011 | excludes_zero |
| no_xtb | MAE | 10 | 0.0109 | 1.0409 | 3 | 0.1309 | 0.3164 | 0.0075 | -0.0029 | 0.0183 | 0.0823 | overlaps_zero |
| no_xtb | RMSE | 10 | 0.0362 | 2.3271 | 1 | 0.0137 | 0.0273 | 0.0323 | 0.0101 | 0.0575 | 0.0011 | excludes_zero |
| no_xtb | within_1_pKa_unit | 10 | -0.0096 | -1.0915 | 0 | 0.0020 | 0.0059 | -0.0074 | -0.0184 | 0.0025 | 0.0564 | overlaps_zero |
| no_xtb | within_2_pKa_unit | 10 | -0.0036 | -0.6785 | 4 | 0.1934 | 0.4922 | -0.0012 | -0.0037 | 0.0000 | 0.0000 | overlaps_zero |
| no_edge_gating | R2 | 10 | -0.0020 | -0.7463 | 4 | 0.1055 | 0.1055 | -0.0016 | -0.0046 | 0.0011 | 0.1384 | overlaps_zero |
| no_edge_gating | MAE | 10 | -0.0027 | -0.2616 | 7 | 0.3223 | 0.3223 | -0.0016 | -0.0111 | 0.0082 | 0.6319 | overlaps_zero |
| no_edge_gating | RMSE | 10 | 0.0114 | 0.7297 | 4 | 0.1055 | 0.1055 | 0.0100 | -0.0076 | 0.0276 | 0.1384 | overlaps_zero |
| no_edge_gating | within_1_pKa_unit | 10 | -0.0031 | -0.3498 | 5 | 0.3223 | 0.6445 | 0.0012 | -0.0110 | 0.0135 | 0.5330 | overlaps_zero |
| no_edge_gating | within_2_pKa_unit | 10 | -0.0010 | -0.1872 | 2 | 0.6406 | 0.6406 | 0.0012 | -0.0025 | 0.0061 | 0.6055 | overlaps_zero |
| no_site_readout | R2 | 10 | 0.0028 | 1.0435 | 9 | 0.0039 | 0.0117 | 0.0026 | -0.0002 | 0.0055 | 0.9633 | overlaps_zero |
| no_site_readout | MAE | 10 | -0.0116 | -1.1038 | 8 | 0.1055 | 0.3164 | -0.0103 | -0.0203 | -0.0003 | 0.9782 | excludes_zero |
| no_site_readout | RMSE | 10 | -0.0164 | -1.0534 | 9 | 0.0039 | 0.0117 | -0.0169 | -0.0359 | 0.0016 | 0.9633 | overlaps_zero |
| no_site_readout | within_1_pKa_unit | 10 | 0.0038 | 0.4338 | 7 | 0.4316 | 0.6445 | 0.0074 | -0.0049 | 0.0196 | 0.8710 | overlaps_zero |
| no_site_readout | within_2_pKa_unit | 10 | 0.0023 | 0.4446 | 6 | 0.1641 | 0.4922 | 0.0025 | -0.0025 | 0.0074 | 0.7735 | overlaps_zero |


- `no_edge_gating`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, within_1_pKa_unit, within_2_pKa_unit.

- `no_site_readout`: ensemble CI excludes zero for: MAE; overlaps zero for: R2, RMSE, within_1_pKa_unit, within_2_pKa_unit.

- `no_xtb`: ensemble CI excludes zero for: R2, RMSE; overlaps zero for: MAE, within_1_pKa_unit, within_2_pKa_unit.


## 6. Is the final model the best variant on each metric?

By seed mean, and by the 10-seed ensemble metric, among the variants that were run for that target.

| target | metric | best_by_seed_mean | final_is_best_by_seed_mean | best_by_ensemble | final_is_best_by_ensemble |
|---|---|---|---|---|---|
| CL | R2 | final | True | final | True |
| CL | MAE | final | True | final | True |
| CL | RMSE | final | True | final | True |
| CL | GMFE | final | True | final | True |
| CL | within_2fold | interactaux | False | final | True |
| CL | within_3fold | final | True | final | True |
| CL | within_5fold | interactaux | False | interactaux | False |
| VDss | R2 | dyngate | False | final | True |
| VDss | MAE | final | True | final | True |
| VDss | RMSE | final | True | final | True |
| VDss | GMFE | final | True | final | True |
| VDss | within_2fold | dyngate | False | dyngate | False |
| VDss | within_3fold | dyngate | False | dyngate | False |
| VDss | within_5fold | dyngate | False | final | True |
| Fu | R2 | final | True | final | True |
| Fu | MAE | final | True | final | True |
| Fu | RMSE | final | True | final | True |
| Fu | GMFE | final | True | final | True |
| Fu | within_2fold | final | True | final | True |
| Fu | within_3fold | final | True | dyngate_unweighted | False |
| Fu | within_5fold | dyngate_unweighted | False | dyngate_unweighted | False |
| pKa_Acidic | R2 | no_edge_gating | False | no_edge_gating | False |
| pKa_Acidic | MAE | final | True | final | True |
| pKa_Acidic | RMSE | no_edge_gating | False | no_edge_gating | False |
| pKa_Acidic | within_1_pKa_unit | no_site_readout | False | no_xtb | False |
| pKa_Acidic | within_2_pKa_unit | no_xtb | False | final | True |
| pKa_Basic | R2 | no_site_readout | False | no_site_readout | False |
| pKa_Basic | MAE | no_site_readout | False | no_site_readout | False |
| pKa_Basic | RMSE | no_site_readout | False | no_site_readout | False |
| pKa_Basic | within_1_pKa_unit | no_site_readout | False | no_site_readout | False |
| pKa_Basic | within_2_pKa_unit | no_site_readout | False | no_site_readout | False |


- CL: the final model is NOT the best variant by seed mean on: within_2fold, within_5fold.
- CL: the final model is NOT the best variant by ensemble metric on: within_5fold.

- VDss: the final model is NOT the best variant by seed mean on: R2, within_2fold, within_3fold, within_5fold.
- VDss: the final model is NOT the best variant by ensemble metric on: within_2fold, within_3fold.

- Fu: the final model is NOT the best variant by seed mean on: within_5fold.
- Fu: the final model is NOT the best variant by ensemble metric on: within_3fold, within_5fold.

- pKa_Acidic: the final model is NOT the best variant by seed mean on: R2, RMSE, within_1_pKa_unit, within_2_pKa_unit.
- pKa_Acidic: the final model is NOT the best variant by ensemble metric on: R2, RMSE, within_1_pKa_unit.

- pKa_Basic: the final model is NOT the best variant by seed mean on: R2, MAE, RMSE, within_1_pKa_unit, within_2_pKa_unit.
- pKa_Basic: the final model is NOT the best variant by ensemble metric on: R2, MAE, RMSE, within_1_pKa_unit, within_2_pKa_unit.

## 7. Deviations, failures and warnings

- Fu/mfmn_base_untuned: RUN_WITH_CAVEAT - no candidate (C1 existing script; C2/C3 reconstructed plain-MFMN) reproduces the CSV row within the gate; running C1 because the project's own records attribute the label to that script. It trains MFMN_InteractAux (aux_weight 0.3, pca32 context), not plain MFMN, and does NOT reproduce the CSV row
- Fu/dyngate_unweighted: RUN_WITH_CAVEAT - seed-42 gate FAILED for every candidate (candidate A and B give identical results); running candidate A (primary) anyway at the user's instruction; see the Fu control in the report for environment drift
- pKa_Acidic/no_xtb: RECONSTRUCTED_VERIFIED - reconstructed; seed 42 reproduces the CSV row within the gate
- pKa_Acidic/no_edge_gating: RECONSTRUCTED_UNVERIFIED - reconstructed; seed 42 does NOT reproduce the CSV row within the gate (see candidates_tested)
- pKa_Acidic/no_site_readout: RECONSTRUCTED_UNVERIFIED - reconstructed; seed 42 does NOT reproduce the CSV row within the gate (see candidates_tested)
- pKa_Basic/no_xtb: RECONSTRUCTED_UNVERIFIED - reconstructed; seed 42 does NOT reproduce the CSV row within the gate (see candidates_tested)
- pKa_Basic/no_edge_gating: RECONSTRUCTED_UNVERIFIED - reconstructed; seed 42 does NOT reproduce the CSV row within the gate (see candidates_tested)
- pKa_Basic/no_site_readout: RECONSTRUCTED_UNVERIFIED - reconstructed; seed 42 does NOT reproduce the CSV row within the gate (see candidates_tested)
- `rdkit` is not installed in this environment and is not in `requirements.txt`; the pKa featurization needs it. rdkit 2026.03.6 was installed with `--no-deps` into `ablation_10seed/_deps/` (not the system environment). The rdkit version behind the stored pKa final arrays is unknown.
- Reconstructed variants (see section 8) are inferred from the variant names; no original code for them was available. The seed-42 gate shows how closely each reproduces its CSV row.
- Scripts live inside `ablation_10seed/` (the only place writing was allowed); run them as `python ablation_10seed/<script>.py`.
- Per-seed metrics column order follows the task spec (GMFE before the within-fold columns); the existing CL/VDss CSVs put GMFE last. A `variant` column is appended.
- Determinism: fixed 2 CPU threads per worker regardless of worker count, `torch.use_deterministic_algorithms(True, warn_only=True)`, cudnn deterministic. The original runs did not set these; CL/VDss still reproduce the CSV rows to |dR2| <= 0.0004.
- `run.log` lines appear twice when the runner is started with `nohup ... >> run.log` (the file handler and the redirected stdout both write to it); cosmetic.
- Package `src` and `data` were staged byte-identically in `/dev/shm/pk_ro` (size/hash verified). Candidates `fuA`/`fuC1` use hard-coded absolute paths inside `mfmn/experiments/*.py` and read `artifacts/` from the project directory instead.

## 8. Reconstructed variants: what exactly was changed


### Checks

All assertions in this script passed: **True**

- final CL R2: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=5.55e-17
- final CL MAE: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=5.55e-17
- final CL RMSE: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=5.55e-17
- final VDss R2: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=0.00e+00
- final VDss MAE: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=5.55e-17
- final VDss RMSE: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=5.55e-17
- final Fu R2: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=0.00e+00
- final Fu MAE: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=5.55e-17
- final Fu RMSE: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=5.55e-17
- final pKa_Acidic R2: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=1.11e-16
- final pKa_Acidic MAE: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=5.55e-17
- final pKa_Acidic RMSE: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=0.00e+00
- final pKa_Basic R2: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=1.11e-16
- final pKa_Basic MAE: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=5.55e-17
- final pKa_Basic RMSE: recomputed (this script) vs existing per_seed_metrics.csv max|diff|=0.00e+00
- CL/mfmn_base: y_test identical to final's: True
- CL/mfmn_base: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=5.55e-17
- CL/mfmn_base: seeds present 10/10
- CL/interactaux: y_test identical to final's: True
- CL/interactaux: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=5.55e-17
- CL/interactaux: seeds present 10/10
- VDss/mfmn_base: y_test identical to final's: True
- VDss/mfmn_base: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=5.55e-17
- VDss/mfmn_base: seeds present 10/10
- VDss/dyngate: y_test identical to final's: True
- VDss/dyngate: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=5.55e-17
- VDss/dyngate: seeds present 10/10
- Fu/dyngate_unweighted: y_test identical to final's: True
- Fu/dyngate_unweighted: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=5.55e-17
- Fu/dyngate_unweighted: seeds present 10/10
- Fu/mfmn_base_untuned: y_test identical to final's: True
- Fu/mfmn_base_untuned: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=5.55e-17
- Fu/mfmn_base_untuned: seeds present 10/10
- pKa_Acidic/no_xtb: y_test identical to final's: True
- pKa_Acidic/no_xtb: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=1.11e-16
- pKa_Acidic/no_xtb: seeds present 10/10
- pKa_Acidic/no_edge_gating: y_test identical to final's: True
- pKa_Acidic/no_edge_gating: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=1.11e-16
- pKa_Acidic/no_edge_gating: seeds present 10/10
- pKa_Acidic/no_site_readout: y_test identical to final's: True
- pKa_Acidic/no_site_readout: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=1.11e-16
- pKa_Acidic/no_site_readout: seeds present 10/10
- pKa_Basic/no_xtb: y_test identical to final's: True
- pKa_Basic/no_xtb: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=1.11e-16
- pKa_Basic/no_xtb: seeds present 10/10
- pKa_Basic/no_edge_gating: y_test identical to final's: True
- pKa_Basic/no_edge_gating: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=1.11e-16
- pKa_Basic/no_edge_gating: seeds present 10/10
- pKa_Basic/no_site_readout: y_test identical to final's: True
- pKa_Basic/no_site_readout: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): True; recomputed vs stored per-seed metrics max|diff|=1.11e-16
- pKa_Basic/no_site_readout: seeds present 10/10

### Final models

Existing `seed_analysis_results/<target>/per_seed_preds.npy` arrays were copied (not retrained) to `ablation_10seed/<target>/final/`; per-seed and cumulative metrics recomputed here match the existing CSVs to 1e-6 for all five targets.

### Sources and versions

See `manifest.json` (source file and cell for every variant, SHA-256 of every source file, library/device versions, hyperparameters read from the code).
