"""Evaluation metrics + the published benchmark-paper numbers every notebook compares against.

PAPER: Jia et al., J. Med. Chem. 2025, 68(7), 7737-7750 (CL/VDss/Fu), same project's
established pKa benchmark comparison (pKa_Acidic/pKa_Basic) -- see mfmn/deliverables/
PROJECT_REPORT.md and mfmn/experiments/expv2_aux_targets_final.py for provenance.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PAPER = {
    # CL/VDss R2, MAE, RMSE corrected 2026-09-30 against the primary source (Table 2, p.7740):
    # the paper reports the full metric set for every target -- these two were previously
    # missing R2/MAE/RMSE here (GMFE/within_2fold-only), an incomplete transcription, not a
    # gap in what the paper itself reports.
    "CL": {"R2": 0.48, "MAE": 0.31, "RMSE": 0.42, "GMFE": 2.00, "within_2fold": 0.64},
    "VDss": {"R2": 0.60, "MAE": 0.28, "RMSE": 0.35, "GMFE": 1.88, "within_2fold": 0.62},
    "Fu": {"R2": 0.69, "MAE": 0.30, "RMSE": 0.41, "GMFE": 2.01, "within_2fold": 0.60},
    "pKa_Acidic": {"R2": 0.94, "MAE": 0.61, "RMSE": 1.05, "GMFE": 1.10, "within_2fold": 0.98},
    "pKa_Basic": {"R2": 0.91, "MAE": 0.67, "RMSE": 0.98, "GMFE": 1.28, "within_2fold": 0.91},
}


def eval_preds(y_true, y_pred, is_log10=False):
    """For Fu (is_log10=True, values modeled in log10 space) and pKa_Acidic/pKa_Basic
    (is_log10=False, values can be negative -- evaluated in raw pKa units)."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    out = {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}
    if is_log10:
        obs, pred = 10 ** y_true, 10 ** y_pred
        ratio = np.maximum(obs / pred, pred / obs)
        out["GMFE"] = float(10 ** np.mean(np.abs(np.log10(obs) - np.log10(pred))))
        out.update({f"within_{n}fold": float(np.mean(ratio <= n)) for n in (2, 3, 5)})
    else:
        diff = np.abs(y_true - y_pred)
        out.update({f"within_{n}_pKa_unit": float(np.mean(diff <= n)) for n in (1, 2)})
    return out


def fold_metrics(obs, pred):
    ratio = np.maximum(obs / pred, pred / obs)
    out = {f"within_{n}fold": float(np.mean(ratio <= n)) for n in (2, 3, 5)}
    out["GMFE"] = float(10 ** np.mean(np.abs(np.log10(obs) - np.log10(pred))))
    return out


def eval_log_preds(y_true_log, y_pred_log):
    """For CL/VDss: values are modeled directly in log10 space (lgCL/lgVD)."""
    y_true_log, y_pred_log = np.asarray(y_true_log), np.asarray(y_pred_log)
    r2 = r2_score(y_true_log, y_pred_log)
    mae = mean_absolute_error(y_true_log, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    obs_lin, pred_lin = 10 ** y_true_log, 10 ** y_pred_log
    return {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse), **fold_metrics(obs_lin, pred_lin)}


def bootstrap_ci(obs, pred, n_boot=2000, seed=42):
    """Bootstrap 95% CI for GMFE and within-2-fold, resampling compounds with replacement."""
    rng = np.random.RandomState(seed)
    n = len(obs)
    gmfes, w2s = [], []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        o, p = obs[i], pred[i]
        gmfes.append(10 ** np.mean(np.abs(np.log10(o) - np.log10(p))))
        r = np.maximum(o / p, p / o)
        w2s.append(np.mean(r <= 2))
    return {
        "GMFE_ci": (float(np.percentile(gmfes, 2.5)), float(np.percentile(gmfes, 97.5))),
        "within_2fold_ci": (float(np.percentile(w2s, 2.5)), float(np.percentile(w2s, 97.5))),
    }
