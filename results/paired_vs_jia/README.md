# Paired comparison with the benchmark's published predictions

Jia et al. publish their own **per-compound** predictions for all five endpoints in sheet `106_cmp_test_set` of the
paper's Supporting Information. On those 106 compounds the two models can therefore be compared with a **paired**
test, instead of against a published aggregate. Produced by `manuscript/paired_vs_jia.py`.

All 106 compounds lie in the official **test** split of every endpoint (asserted), so neither model trained on them,
and both are scored on identical compounds. This is the only subset where the comparison is possible: the SI carries
no per-compound predictions for the rest of the test splits.

## Result

7 of the 23 metric–endpoint pairs are resolved (paired bootstrap 95% CI of the difference excludes zero) and **all 7
favour this work; none favours the benchmark.**

| Endpoint | Metric | Benchmark | This work | Difference | 95% CI | Resolved |
|---|---|---|---|---|---|---|
| pKa_Acidic | R² | 0.963 | 0.982 | +0.019 | [+0.001, +0.039] | yes |
| pKa_Acidic | MAE | 0.514 | 0.354 | −0.160 | [−0.285, −0.039] | yes |
| pKa_Acidic | RMSE | 0.798 | 0.562 | −0.236 | [−0.454, −0.008] | yes |
| pKa_Acidic | within 1 pKa unit | 0.858 | 0.934 | +0.075 | [+0.000, +0.151] | no |
| pKa_Basic | R² | 0.902 | 0.977 | +0.076 | [+0.033, +0.133] | yes |
| pKa_Basic | MAE | 0.634 | 0.314 | −0.320 | [−0.463, −0.194] | yes |
| pKa_Basic | RMSE | 0.976 | 0.468 | −0.508 | [−0.752, −0.264] | yes |
| pKa_Basic | within 1 pKa unit | 0.849 | 0.943 | +0.094 | [+0.028, +0.160] | yes |
| CL | GMFE | 1.969 | 1.953 | −0.016 | [−0.121, +0.093] | no |
| CL | within 2-fold | 0.642 | 0.613 | −0.028 | [−0.104, +0.047] | no |
| VDss | GMFE | 1.889 | 1.759 | −0.130 | [−0.263, +0.004] | no |
| VDss | within 2-fold | 0.604 | 0.698 | +0.094 | [+0.000, +0.189] | no |
| Fu | GMFE | 2.014 | 1.937 | −0.077 | [−0.227, +0.067] | no |
| Fu | within 2-fold | 0.604 | 0.670 | +0.066 | [−0.009, +0.151] | no |

Full metric set in `paired_summary.csv`. Per-compound errors for both models in `per_compound_errors.csv`.

- **pKa**: both endpoints improve on R², MAE and RMSE with the CI excluding zero; pKa_Basic also on within-1-pKa-unit.
  The pKa_Basic MAE reduction is 51%. This work's prediction is closer for 62/106 (acidic) and 76/106 (basic) compounds.
- **CL, VDss, Fu**: nothing resolved. This work holds the better point estimate on every VDss and Fu metric; CL is
  indistinguishable.
- 106 compounds is a smaller sample than the full test splits (776/815/177/177/633), so only larger differences
  resolve. **No resolved difference is not evidence of equivalence.**

## Method

Metrics and bootstrap are those of `manuscript/analysis.py`: B = 10,000 compound-level resamples, same RNG seed, and
for the difference the **same resampled compounds in both arms**, so the pairing is real. Values are compared in the
space each endpoint is modelled in: raw pKa units for pKa, log10 for CL, VDss and Fu. The SI's linear values are
log10-transformed, and our observed values are asserted to agree with the SI's.

Joins: CL and VDss by `row_idx` of `data/cl_vdss/modeling_index.parquet` → `PUBCHEM_CID`; pKa and Fu by the parent
SMILES of sheet `pKas_modeling_set`, which is the `id` of our prediction files. Both are asserted to produce exactly
106 one-to-one matches.

The SI workbook is CC BY-NC-ND and is not redistributed here; `src/downstream.py` downloads it into `data/external/`
and verifies its SHA-256.
