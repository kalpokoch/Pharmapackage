# Untuned baselines (library defaults) vs the retrained proposed models

Protocol: `run_baselines_10seed.py`. Features: `mfmn_benchmark_package/ablation_study/baselines/features/<target>__<view>.npz`
(CL/VDss/Fu `model_inputs`, pKa `desc`); StandardScaler fit on the training split; official test split, identical row order to the
proposed models (asserted). No hyperparameter is set (`n_jobs` controls parallelism only).

- **Random Forest** `RandomForestRegressor(random_state=seed)`, seeds 42-51: the seeds give different predictions; 10-seed ensemble reported.
- **XGBoost** `XGBRegressor(random_state=seed)`, seeds 42-51: with library defaults there is no row/column subsampling, so the seed has
  no effect and all 10 fits are identical (verified: max prediction spread across seeds = 0.0 for all five targets). Its "10-seed
  ensemble" therefore equals a single fit.
- **SVM** `SVR()`: deterministic; fit once (n_seeds = 1).

Per model and target: `<target>/<model>/seed_<s>/{predictions.csv,metrics.csv}`, `per_seed_preds.npy`, `per_seed_metrics.csv`,
`cumulative_ensemble_metrics.csv`, `ensemble_predictions.csv`, `parity_ensemble.{png,pdf}`.
Comparison with the retrained final models (`../finals/`): `tables/baselines_and_proposed_summary.csv`,
`tables/baselines_vs_proposed_table.csv`, `tables/baseline_vs_proposed_bootstrap.csv` (compound-level paired bootstrap of the ensemble
difference, B=10,000, same resampled compounds for baseline and proposed; "CI excludes/overlaps zero"), `tables/comparison_<target>.{png,pdf}`.
