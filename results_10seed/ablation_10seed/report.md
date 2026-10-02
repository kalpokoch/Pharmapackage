# Ablation with 10 seeds - report

Generated 2026-10-01T05:11:54 by `analyze_ablation_10seed.py`. Facts only; no recommendations about what to claim.

## 1. Status of all 12 variants

| target | variant | slug | status | detail |
|---|---|---|---|---|
| CL | MFMN (base) | mfmn_base | RUN (10 seeds) |  |
| CL | MFMN_InteractAux | interactaux | RUN (10 seeds) |  |
| VDss | MFMN (base) | mfmn_base | RUN (10 seeds) |  |
| VDss | MFMN_DynGate | dyngate | RUN (10 seeds) |  |
| Fu | MFMN_DynGate (unweighted) | dyngate_unweighted | BLOCKED | seed-42 smoke test failed the gate (|dR2|<=0.002 and |dMAE|<=0.002) for every candidate; see candidates_tested |
| Fu | MFMN base (untuned) | mfmn_base_untuned | BLOCKED | 0 candidates reproduce seed 42 (need exactly 1); candidate seed-42 numbers in candidates_tested |
| pKa_Acidic | GraphMPNN, no xTB features | no_xtb | BLOCKED | the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction |
| pKa_Acidic | GraphMPNN, no edge gating | no_edge_gating | BLOCKED | the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction |
| pKa_Acidic | GraphMPNN, no site readout | no_site_readout | BLOCKED | the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction |
| pKa_Basic | GraphMPNN, no xTB features | no_xtb | BLOCKED | the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction |
| pKa_Basic | GraphMPNN, no edge gating | no_edge_gating | BLOCKED | the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction |
| pKa_Basic | GraphMPNN, no site readout | no_site_readout | BLOCKED | the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction |


## 2. Smoke test: seed 42 vs `proposed_model_ablation.csv`

Gate: |dR2| <= 0.002 and |dMAE| <= 0.002 and y_test identical to the final model's. Run with 8 workers (tag `w8`); candidates for the same variant are listed in priority order.

| target | variant | runner | R2 | csv_R2 | dR2 | MAE | csv_MAE | dMAE | RMSE | GMFE | csv_GMFE | within_2fold | csv_within_2fold | seconds | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CL | MFMN (base) | clvd | 0.3825 | 0.3825 | -0.0000 | 0.3403 | 0.3403 | 0.0000 | 0.4567 | 2.1894 | 2.1893 | 0.5537 | 0.5537 | 10.4900 | True |
| CL | MFMN_InteractAux | clvd | 0.4284 | 0.4284 | 0.0000 | 0.3347 | 0.3346 | 0.0001 | 0.4393 | 2.1610 | 2.1608 | 0.5706 | 0.5706 | 24.5358 | True |
| VDss | MFMN (base) | clvd | 0.5708 | 0.5709 | -0.0001 | 0.2688 | 0.2687 | 0.0001 | 0.3621 | 1.8569 | 1.8566 | 0.6441 | 0.6441 | 8.1122 | True |
| VDss | MFMN_DynGate | clvd | 0.6092 | 0.6096 | -0.0004 | 0.2673 | 0.2672 | 0.0001 | 0.3455 | 1.8506 | 1.8503 | 0.6610 | 0.6667 | 30.0107 | True |
| Fu | MFMN base (untuned) | fuC1 | 0.6351 | 0.6806 | -0.0454 | 0.3560 | 0.3366 | 0.0194 | 0.4899 | 2.2699 | 2.1708 | 0.5577 | 0.5814 | 64.1807 | False |
| Fu | MFMN_DynGate (unweighted) | fuA | 0.6979 | 0.6936 | 0.0043 | 0.3236 | 0.3280 | -0.0044 | 0.4458 | 2.1067 | 2.1284 | 0.6130 | 0.6035 | 43.8989 | False |
| Fu | MFMN_DynGate (unweighted) | fuB | 0.6979 | 0.6936 | 0.0043 | 0.3236 | 0.3280 | -0.0044 | 0.4458 | 2.1067 | 2.1284 | 0.6130 | 0.6035 | 39.1970 | False |


Candidate code paths: `clvd` = `ablation/ablation_official_test_CL_VDss_Fu.ipynb` cells 1,3,5 with `src/mfmn.py`; `fuA` = `MFMN_v2.ipynb` cells 1,13 + `mfmn/experiments/expv2_aux_h7_dyngate.py::fit_predict_dyngate` with `RECIPE['Fu']`; `fuB` = notebook `05_Fu_MFMN_DynGate.ipynb` cells 1-4 with the low-fu multiplier set to 1.0 (only change); `fuC1` = `mfmn/experiments/expv2_aux_targets_final.py::fit_predict` defaults (class `MFMN_InteractAux`, aux_weight 0.3).


**Fu diagnostics (not a recipe change; run to explain the gate failures):** on this machine the existing 1.25x low-fu recipe (notebook 05 path, the FINAL model's own recipe) trained at seed 42 gives R2 0.6856 / MAE 0.3370, whereas the stored final-model seed-42 predictions (`seed_analysis_results/Fu/per_seed_preds.npy`) give R2 0.6890 / MAE 0.3275, with per-compound predictions differing by up to 0.53 log10 units. Repeating a training in this environment is bit-identical (with deterministic settings on or off), so the gap is not run-to-run noise; the stored Fu arrays were produced in a different software environment (`requirements.txt` is unpinned, so the original versions cannot be recovered). CL/VDss reproduce the CSV rows within |dR2| <= 0.0004.


**Worker-count independence (seed 42, same arrays compared bit for bit):**

| comparison | max_abs_diff | bit_identical |
|---|---|---|
| CL/interactaux: workers 8 vs 1 | 0.000000 | True |
| CL/interactaux: workers 8 vs 3 | 0.000000 | True |
| VDss/dyngate: workers 8 vs 1 | 0.000000 | True |
| VDss/dyngate: workers 8 vs 3 | 0.000000 | True |


## 3. Seconds per seed (full run, 10 concurrent workers sharing one MIG slice; smoke-test timings in section 2)

| target | variant | seeds_logged | sec_mean | sec_min | sec_max | failed | device |
|---|---|---|---|---|---|---|---|
| CL | mfmn_base | 10 | 7.1 | 3.9 | 11.1 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| CL | interactaux | 10 | 26.6 | 18.4 | 38.7 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| VDss | mfmn_base | 10 | 4.4 | 3.8 | 4.8 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |
| VDss | dyngate | 10 | 29.5 | 20.6 | 40.5 | 0 | NVIDIA A100-SXM4-40GB MIG 3g.20gb |


## 4. Seeds completed per variant

| target | variant | status | n_seeds | seeds |
|---|---|---|---|---|
| CL | mfmn_base | RUN | 10 | 42 43 44 45 46 47 48 49 50 51 |
| CL | interactaux | RUN | 10 | 42 43 44 45 46 47 48 49 50 51 |
| VDss | mfmn_base | RUN | 10 | 42 43 44 45 46 47 48 49 50 51 |
| VDss | dyngate | RUN | 10 | 42 43 44 45 46 47 48 49 50 51 |
| Fu | dyngate_unweighted | not run (BLOCKED) | 0 |  |
| Fu | mfmn_base_untuned | not run (BLOCKED) | 0 |  |
| pKa_Acidic | no_xtb | not run (BLOCKED) | 0 |  |
| pKa_Acidic | no_edge_gating | not run (BLOCKED) | 0 |  |
| pKa_Acidic | no_site_readout | not run (BLOCKED) | 0 |  |
| pKa_Basic | no_xtb | not run (BLOCKED) | 0 |  |
| pKa_Basic | no_edge_gating | not run (BLOCKED) | 0 |  |
| pKa_Basic | no_site_readout | not run (BLOCKED) | 0 |  |


## 5. Results (only variants that were run)

Seed mean ± SD (ddof=1) over seeds and the metrics of the 10-seed ensemble; `diff` = variant - final, paired by seed index.

| target | variant | name | status | n_seeds | R2_mean_pm_sd | MAE_mean_pm_sd | RMSE_mean_pm_sd | R2_ens | MAE_ens | diff_mean_R2 | diff_in_final_sd_R2 | flag_R2 | diff_mean_MAE | diff_in_final_sd_MAE | flag_MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CL | mfmn_base | MFMN (base) | RUN | 10 | 0.364 ± 0.076 | 0.348 ± 0.022 | 0.463 ± 0.027 | 0.4677 | 0.3133 | -0.0762 | -3.8200 | overlaps_zero | 0.0237 | 3.2100 | overlaps_zero |
| CL | interactaux | MFMN_InteractAux | RUN | 10 | 0.426 ± 0.036 | 0.329 ± 0.013 | 0.440 ± 0.014 | 0.4897 | 0.3088 | -0.0148 | -0.7400 | overlaps_zero | 0.0053 | 0.7200 | overlaps_zero |
| CL | final | FINAL | final | 10 | 0.441 ± 0.020 | 0.324 ± 0.007 | 0.435 ± 0.008 | 0.4967 | 0.3050 |  |  | reference |  |  | reference |
| VDss | mfmn_base | MFMN (base) | RUN | 10 | 0.528 ± 0.029 | 0.287 ± 0.009 | 0.379 ± 0.012 | 0.5713 | 0.2724 | -0.0803 | -2.7800 | excludes_zero | 0.0280 | 2.4100 | excludes_zero |
| VDss | dyngate | MFMN_DynGate | RUN | 10 | 0.609 ± 0.009 | 0.260 ± 0.004 | 0.346 ± 0.004 | 0.6465 | 0.2482 | 0.0002 | 0.0100 | overlaps_zero | 0.0018 | 0.1500 | overlaps_zero |
| VDss | final | FINAL | final | 10 | 0.609 ± 0.029 | 0.259 ± 0.012 | 0.345 ± 0.013 | 0.6478 | 0.2466 |  |  | reference |  |  | reference |
| Fu | final | FINAL | final | 10 | 0.683 ± 0.012 | 0.335 ± 0.007 | 0.457 ± 0.009 | 0.7143 | 0.3185 |  |  | reference |  |  | reference |
| pKa_Acidic | final | FINAL | final | 10 | 0.965 ± 0.002 | 0.473 ± 0.011 | 0.805 ± 0.021 | 0.9695 | 0.4197 |  |  | reference |  |  | reference |
| pKa_Basic | final | FINAL | final | 10 | 0.935 ± 0.004 | 0.476 ± 0.018 | 0.747 ± 0.023 | 0.9463 | 0.4185 |  |  | reference |  |  | reference |


### Interpretation notes

- Seeds share one fixed test set, so the seed-level tests capture training randomness only. The compound-level bootstrap captures test-set sampling. Both are reported and not merged.
- Pairing 'by seed index' is a convenience: the variant and the final model are different architectures, so seed 42 does not mean the same random numbers across them. It is paired by seed index, nothing stronger.
- There are many comparisons (variants x metrics). The text here uses 'CI excludes zero' / 'CI overlaps zero' and reports the Holm-adjusted p-values (`p_holm`, over the variants of the same target and metric, from the exact Wilcoxon p-values; `p_holm_ttest` from the paired t-test).
- Bootstrap: B = 10,000, `np.random.default_rng(0)`, one resampled-index matrix per target shared by every variant and the final model, ensemble metrics recomputed inside each draw; 'ens_p_variant_better' is the share of draws in which the variant's metric is better than the final's.


### Component effects per target: ensemble-level 95% CI of (variant - final)


**CL** (n_test=177)

| variant | metric | n_seeds | diff_mean | diff_in_final_sd | n_seeds_variant_better | p_wilcoxon | p_holm | ens_diff | ens_ci95_lo | ens_ci95_hi | ens_p_variant_better | ens_ci_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mfmn_base | R2 | 10 | -0.0762 | -3.8247 | 2 | 0.0098 | 0.0195 | -0.0289 | -0.0768 | 0.0220 | 0.1274 | overlaps_zero |
| mfmn_base | MAE | 10 | 0.0237 | 3.2131 | 1 | 0.0098 | 0.0195 | 0.0083 | -0.0069 | 0.0236 | 0.1424 | overlaps_zero |
| mfmn_base | RMSE | 10 | 0.0280 | 3.6236 | 2 | 0.0098 | 0.0195 | 0.0117 | -0.0084 | 0.0317 | 0.1274 | overlaps_zero |
| mfmn_base | GMFE | 10 | 0.1205 | 3.3603 | 1 | 0.0098 | 0.0195 | 0.0389 | -0.0322 | 0.1132 | 0.1424 | overlaps_zero |
| mfmn_base | within_2fold | 10 | -0.0181 | -0.9875 | 3 | 0.2324 | 0.4648 | -0.0339 | -0.0847 | 0.0169 | 0.0698 | overlaps_zero |
| mfmn_base | within_3fold | 10 | -0.0350 | -2.1564 | 1 | 0.0059 | 0.0117 | -0.0395 | -0.0791 | 0.0000 | 0.0143 | overlaps_zero |
| mfmn_base | within_5fold | 10 | -0.0203 | -2.0266 | 2 | 0.0371 | 0.0742 | 0.0000 | -0.0282 | 0.0282 | 0.4187 | overlaps_zero |
| interactaux | R2 | 10 | -0.0148 | -0.7443 | 3 | 0.2754 | 0.2754 | -0.0070 | -0.0260 | 0.0127 | 0.2406 | overlaps_zero |
| interactaux | MAE | 10 | 0.0053 | 0.7163 | 2 | 0.4316 | 0.4316 | 0.0038 | -0.0039 | 0.0118 | 0.1712 | overlaps_zero |
| interactaux | RMSE | 10 | 0.0056 | 0.7233 | 3 | 0.2754 | 0.2754 | 0.0029 | -0.0048 | 0.0112 | 0.2406 | overlaps_zero |
| interactaux | GMFE | 10 | 0.0264 | 0.7368 | 2 | 0.4316 | 0.4316 | 0.0177 | -0.0182 | 0.0554 | 0.1712 | overlaps_zero |
| interactaux | within_2fold | 10 | 0.0119 | 0.6481 | 5 | 0.2500 | 0.4648 | 0.0000 | -0.0282 | 0.0339 | 0.4223 | overlaps_zero |
| interactaux | within_3fold | 10 | -0.0186 | -1.1478 | 3 | 0.1055 | 0.1055 | -0.0169 | -0.0395 | 0.0000 | 0.0000 | overlaps_zero |
| interactaux | within_5fold | 10 | 0.0006 | 0.0563 | 4 | 0.9375 | 0.9375 | 0.0056 | -0.0113 | 0.0226 | 0.6093 | overlaps_zero |


- `interactaux`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, GMFE, within_2fold, within_3fold, within_5fold.

- `mfmn_base`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, GMFE, within_2fold, within_3fold, within_5fold.


**VDss** (n_test=177)

| variant | metric | n_seeds | diff_mean | diff_in_final_sd | n_seeds_variant_better | p_wilcoxon | p_holm | ens_diff | ens_ci95_lo | ens_ci95_hi | ens_p_variant_better | ens_ci_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mfmn_base | R2 | 10 | -0.0803 | -2.7821 | 0 | 0.0020 | 0.0039 | -0.0766 | -0.1221 | -0.0367 | 0.0001 | excludes_zero |
| mfmn_base | MAE | 10 | 0.0280 | 2.4124 | 1 | 0.0039 | 0.0078 | 0.0258 | 0.0113 | 0.0408 | 0.0001 | excludes_zero |
| mfmn_base | RMSE | 10 | 0.0339 | 2.6715 | 0 | 0.0020 | 0.0039 | 0.0339 | 0.0164 | 0.0516 | 0.0001 | excludes_zero |
| mfmn_base | GMFE | 10 | 0.1204 | 2.4853 | 1 | 0.0039 | 0.0078 | 0.1081 | 0.0467 | 0.1750 | 0.0001 | excludes_zero |
| mfmn_base | within_2fold | 10 | -0.0384 | -1.3741 | 1 | 0.0078 | 0.0156 | -0.0282 | -0.0791 | 0.0226 | 0.1103 | overlaps_zero |
| mfmn_base | within_3fold | 10 | -0.0520 | -2.9693 | 1 | 0.0039 | 0.0078 | -0.0621 | -0.1017 | -0.0282 | 0.0000 | excludes_zero |
| mfmn_base | within_5fold | 10 | -0.0164 | -1.0218 | 2 | 0.0488 | 0.0977 | -0.0056 | -0.0169 | 0.0000 | 0.0000 | overlaps_zero |
| dyngate | R2 | 10 | 0.0002 | 0.0075 | 5 | 0.9219 | 0.9219 | -0.0013 | -0.0152 | 0.0125 | 0.4260 | overlaps_zero |
| dyngate | MAE | 10 | 0.0018 | 0.1520 | 4 | 0.4922 | 0.4922 | 0.0017 | -0.0036 | 0.0069 | 0.2662 | overlaps_zero |
| dyngate | RMSE | 10 | 0.0001 | 0.0072 | 5 | 0.9219 | 0.9219 | 0.0006 | -0.0057 | 0.0069 | 0.4260 | overlaps_zero |
| dyngate | GMFE | 10 | 0.0069 | 0.1416 | 4 | 0.4922 | 0.4922 | 0.0067 | -0.0147 | 0.0281 | 0.2662 | overlaps_zero |
| dyngate | within_2fold | 10 | 0.0096 | 0.3435 | 6 | 0.5566 | 0.5566 | 0.0113 | -0.0169 | 0.0395 | 0.7314 | overlaps_zero |
| dyngate | within_3fold | 10 | 0.0051 | 0.2905 | 4 | 0.7422 | 0.7422 | 0.0113 | -0.0169 | 0.0395 | 0.7351 | overlaps_zero |
| dyngate | within_5fold | 10 | 0.0023 | 0.1409 | 3 | 0.9375 | 0.9375 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | overlaps_zero |


- `dyngate`: ensemble CI excludes zero for: none; overlaps zero for: R2, MAE, RMSE, GMFE, within_2fold, within_3fold, within_5fold.

- `mfmn_base`: ensemble CI excludes zero for: R2, MAE, RMSE, GMFE, within_3fold; overlaps zero for: within_2fold, within_5fold.


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


- CL: the final model is NOT the best variant by seed mean on: within_2fold, within_5fold.
- CL: the final model is NOT the best variant by ensemble metric on: within_5fold.

- VDss: the final model is NOT the best variant by seed mean on: R2, within_2fold, within_3fold, within_5fold.
- VDss: the final model is NOT the best variant by ensemble metric on: within_2fold, within_3fold.

## 7. Deviations, failures and warnings

- BLOCKED / not run: Fu / MFMN_DynGate (unweighted) - seed-42 smoke test failed the gate (|dR2|<=0.002 and |dMAE|<=0.002) for every candidate; see candidates_tested
- BLOCKED / not run: Fu / MFMN base (untuned) - 0 candidates reproduce seed 42 (need exactly 1); candidate seed-42 numbers in candidates_tested
- BLOCKED / not run: pKa_Acidic / GraphMPNN, no xTB features - the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction
- BLOCKED / not run: pKa_Acidic / GraphMPNN, no edge gating - the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction
- BLOCKED / not run: pKa_Acidic / GraphMPNN, no site readout - the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction
- BLOCKED / not run: pKa_Basic / GraphMPNN, no xTB features - the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction
- BLOCKED / not run: pKa_Basic / GraphMPNN, no edge gating - the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction
- BLOCKED / not run: pKa_Basic / GraphMPNN, no site readout - the code that produced this row is not in the project (graph_mpnn.py has no switch for it; no notebook/script trains it); not reconstructed by instruction
- `Fu/mfmn_base_untuned` and `Fu/dyngate_unweighted`: seed-42 reproduction failed the gate (see section 2), so their 10 seeds were not run. The only plain-`MFMN` path in the project is the CL/VDss notebook function, which is bound to CL/VDss data; using it for Fu would have required editing the recipe, so it was not tried.
- The code behind the CSV row labelled 'MFMN base (untuned)' for Fu is, by name and by `all_models_comparison.csv` provenance (`expv2_aux_targets_final_results.json`), the `MFMN_InteractAux` aux-target script (candidate `fuC1`); that candidate did not reproduce the row (R2 0.6351 vs 0.6806), so the label question is moot here.
- Scripts live inside `ablation_10seed/` (the only place writing was allowed); run them as `python ablation_10seed/<script>.py`.
- Per-seed metrics column order follows the task spec (GMFE before the within-fold columns); the existing CL/VDss CSVs put GMFE last. A `variant` column is appended.
- Determinism: fixed 2 CPU threads per worker regardless of worker count, `torch.use_deterministic_algorithms(True, warn_only=True)`, cudnn deterministic. The original runs did not set these; CL/VDss still reproduce the CSV rows to |dR2| <= 0.0004.
- `run.log` lines appear twice when the runner is started with `nohup ... >> run.log` (the file handler and the redirected stdout both write to it); cosmetic.
- Package `src` and `data` were staged byte-identically in `/dev/shm/pk_ro` (size/hash verified). Candidates `fuA`/`fuC1` use hard-coded absolute paths inside `mfmn/experiments/*.py` and read `artifacts/` from the project directory instead.

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

### Final models

Existing `seed_analysis_results/<target>/per_seed_preds.npy` arrays were copied (not retrained) to `ablation_10seed/<target>/final/`; per-seed and cumulative metrics recomputed here match the existing CSVs to 1e-6 for all five targets.

### Sources and versions

See `manifest.json` (source file and cell for every variant, SHA-256 of every source file, library/device versions, hyperparameters read from the code).
