# MFMN vs a parameter-matched MLP on the same descriptors (CL, VDss)

Question: does the seven-factor MFMN design beat ordinary descriptor learning? The MLP gets exactly MFMN's inputs (shared
context of 32 PCA components + MACCS = 199 columns, plus the 19 factor-block columns, concatenated: 218 features), the
same recipe (AdamW, cosine schedule, SmoothL1, batch 128, 15 % validation split drawn with the same `RandomState(seed)`,
early stopping), seeds 42-51, official split, same test rows. Hidden width is chosen so the trainable parameter count
matches the MFMN final model (VDss 12,976; CL 24,328). Produced by `manuscript/mlp_vs_mfmn.py`.

Variants: `mlp_2L` / `mlp_3L` (plain), `mlp_2L_aux` (adds MFMN's seven auxiliary-property heads and weight, which isolates the
factor architecture from the auxiliary supervision). `matched_*` uses MFMN's hyperparameters (lr 3e-3, wd 1e-2, dropout 0.2);
`tuned_*` uses lr / wd / dropout (/ aux weight) chosen by 5-fold CV on the **training split only** (features refit inside each
fold; `tuning_*.csv`, `tuned_hparams.json`). No choice used the test split.

## Results (10-seed ensemble, official test split)

| Target | Model | Params | GMFE | within-2-fold | R² | MAE |
|---|---|---|---|---|---|---|
| VDss | MFMN base | 12,281 | 1.872 | 0.655 | 0.571 | 0.272 |
| VDss | **MFMN final (InteractAux)** | 12,976 | **1.764** | 0.684 | 0.648 | 0.247 |
| VDss | MLP 2L, matched / tuned | 12,913 | 1.808 / 1.794 | 0.689 / 0.678 | 0.614 / 0.626 | 0.257 / 0.254 |
| VDss | MLP 3L, matched | 12,853 | 1.797 | 0.706 | 0.616 | 0.255 |
| VDss | MLP 2L / 3L, untuned (framework defaults) | 12,913 / 12,853 | 1.797 / 1.794 | 0.689 / 0.684 | 0.619 / 0.621 | 0.255 / 0.254 |
| VDss | MLP 2L + aux heads, matched / tuned | 12,933 | 1.755 / 1.751 | 0.723 / 0.695 | 0.653 / 0.650 | 0.244 / 0.243 |
| CL | MFMN base | 12,281 | 2.057 | 0.571 | 0.468 | 0.313 |
| CL | **MFMN final (DynGate)** | 24,328 | **2.018** | 0.605 | 0.497 | 0.305 |
| CL | MLP 2L, matched / tuned | 24,463 | 2.031 / 2.024 | 0.588 / 0.605 | 0.496 / 0.500 | 0.308 / 0.306 |
| CL | MLP 3L, matched | 24,345 | 2.018 | 0.610 | 0.505 | 0.305 |
| CL | MLP 2L / 3L, untuned (framework defaults) | 24,463 / 24,345 | 2.098 / 2.040 | 0.599 / 0.605 | 0.459 / 0.499 | 0.322 / 0.310 |
| CL | MLP 2L + aux heads, matched / tuned | 24,261 | 2.080 / 2.121 | 0.599 / 0.593 | 0.478 / 0.452 | 0.318 / 0.327 |

Paired-bootstrap CIs (10,000 resamples of test compounds, MFMN final minus variant) for every metric are in
`t_mlp_vs_mfmn_VDss.csv` and `t_mlp_vs_mfmn_CL.csv`.

`untuned_*`: AdamW defaults (lr 1e-3, wd 1e-2), no dropout, no tuning of any kind (phase `untuned`).

## What the data support

- **VDss.** The base MFMN (1.872) is *worse* than a plain parameter-matched MLP (1.794-1.808), so the factorised architecture
  alone does not beat ordinary descriptor learning. The final model's gain over base comes with the interaction term and the
  auxiliary supervision; an MLP given the same auxiliary heads, without any factor structure, reaches the same accuracy
  (GMFE 1.751-1.755 vs 1.764). MFMN final vs every MLP variant: no GMFE, MAE or within-2-fold difference is resolved
  (all CIs include 0); one R² difference (vs the matched plain 2L MLP) is resolved in MFMN's favour, the same comparison
  with the tuned MLP is not.
- **CL.** MFMN final (2.018) is indistinguishable from the plain MLPs (2.018-2.031). It is better than the MLP with auxiliary
  heads (GMFE and MAE resolved), i.e. the auxiliary loss does not transfer to a plain MLP on CL, but that is not evidence
  for the factor design over a plain MLP.
- **Untuned MLP.** On VDss the untuned 2-layer MLP (GMFE 1.797) is close to MFMN, but MFMN's R² and RMSE are resolved as better (R² +0.029, CI [0.005, 0.053]); GMFE, MAE and within-2-fold are not. On CL the untuned 2-layer MLP is clearly worse (GMFE 2.098 vs 2.018, CI of the difference [-0.162, -0.006]; MAE also resolved), while the untuned 3-layer MLP (2.040) is not. Tuning (or MFMN's regularisation: dropout 0.2-0.3) closes the gap, so part of MFMN's advantage over a naively configured MLP is regularisation.
- So these experiments do **not** establish that the seven-factor design is better than a conventional MLP on these
  descriptors; they are also not evidence that it is worse (no MLP is resolved as better than MFMN final).

## Limits

- One official split with 177 test compounds: the CIs are wide, and only large differences can resolve.
- MFMN's recipe was developed on this dataset over the project's history; the MLP got a modest CV-tuned grid
  (12 configs, 24 with aux; single seed per fold), so MFMN may still have a tuning advantage. The CV-selected aux weight/lr
  for the CL aux-MLP did worse on test than MFMN's recipe, a sign that the CV selection is noisy.
- Fu uses a different feature pipeline and was not run.
- Ensembles of 10 seeds on identical validation splits per seed for MFMN and MLP.
