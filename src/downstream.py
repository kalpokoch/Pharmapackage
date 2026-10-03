"""Downstream exposure propagation (AUC / Cmax) under a one-compartment IV model.

Propagates CL and VDss predictions to exposure with a fixed analytic model, so the three
parameter sources (observed, Jia et al. QSAR, this work) can be compared with the
downstream model held constant.

Assumptions, which hold for every number this module produces: one-compartment linear
kinetics, dose linearity, and every profile normalised to a 1 mg/kg IV dose (the
normalisation Jia et al. use). What this measures is how step-1 parameter error
propagates into exposure error. It is NOT a measurement of absolute concentration-profile
accuracy, and its numbers are not comparable with any figure obtained by scoring against
observed concentration-time profiles.

Data source: Supporting Information of Jia et al., J. Med. Chem. 2025, 68(7), 7737-7750
(jm5c00340_si_001.xlsx). The workbook is CC BY-NC-ND and is not redistributed here;
`fetch_si()` downloads it from the Europe PMC open-access mirror and checks its SHA-256.
"""
import hashlib
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import metrics as metrics_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
SI_PATH = os.path.join(ROOT, "data", "external", "jm5c00340_si_001.xlsx")
SI_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11998014/supplementaryFiles"
SI_NAME = "jm5c00340_si_001.xlsx"
SI_SHA256 = "d9e5c6e69fd3c458009738e987da877363959cc6481b0d3cd97c2c9fbe27db3e"

DOSE_MG_PER_KG = 1.0
TEST_SHEET = "106_cmp_test_set"
MODELING_SHEET = "Fu_VDss_CL_modeling_set"
N_TEST = 106

CL_OBS, CL_JIA = "CL_final(L/hour/kg)", "pred_CL_(L/hour/kg)"
VD_OBS, VD_JIA = "VD_final(L/kg)", "pred_VD_(L/kg)"
FU_OBS, FU_JIA = "fu", "pred_fu"
T_MIN = "InfusionTime_min"

# fold-error metrics are reported on log10 exposure; direction: True = higher is better
METRIC_NAMES = ["R2", "MAE", "RMSE", "GMFE", "within_2fold", "within_3fold", "within_5fold",
                "median_FE", "mean_FE"]
HIGHER_IS_BETTER = {"R2": True, "MAE": False, "RMSE": False, "GMFE": False, "within_2fold": True,
                    "within_3fold": True, "within_5fold": True, "median_FE": False, "mean_FE": False}


# --------------------------------------------------------------------------- data

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_si(path=SI_PATH):
    """Return the SI workbook path, downloading it from Europe PMC if absent; verify SHA-256."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(SI_URL, timeout=900) as r:
            bundle = zipfile.ZipFile(io.BytesIO(r.read()))
        with open(path, "wb") as fh:
            fh.write(bundle.read(SI_NAME))
    digest = _sha256(path)
    assert digest == SI_SHA256, f"SI workbook SHA-256 mismatch: {digest} != {SI_SHA256}"
    return path


def load_test_set(si_path=SI_PATH):
    """The 106-compound test set with observed and Jia et al.-predicted parameters.

    Asserts: 106 unique compounds; all parameter columns finite and positive; every
    compound in the official test split for CL, VDss and Fu; and the SI's two sheets
    agreeing on the observed values.
    """
    x = pd.ExcelFile(si_path)
    t = x.parse(TEST_SHEET)
    assert len(t) == N_TEST, f"expected {N_TEST} rows, got {len(t)}"
    assert t.PubChem_CID.notna().all() and t.PubChem_CID.is_unique
    for c in (CL_OBS, CL_JIA, VD_OBS, VD_JIA):
        v = t[c].to_numpy(float)
        assert np.isfinite(v).all() and (v > 0).all(), f"{c}: non-finite or non-positive values"

    m = x.parse(MODELING_SHEET)
    s = m.set_index("PUBCHEM_CID").loc[t.PubChem_CID]
    for c in ("CL_train_test", "VD_train_test", "Fu_train_test"):
        assert (s[c] == "test").all(), f"{c}: not all {N_TEST} compounds are in the official test split"
    assert np.allclose(s[CL_OBS].to_numpy(float), t[CL_OBS].to_numpy(float), rtol=1e-9)
    assert np.allclose(s[VD_OBS].to_numpy(float), t[VD_OBS].to_numpy(float), rtol=1e-9)
    return t.reset_index(drop=True)


def _cid_map():
    """row_idx (the `id` of results/final_models/{CL,VDss}) -> PUBCHEM_CID."""
    mi = pd.read_parquet(os.path.join(ROOT, "data", "cl_vdss", "modeling_index.parquet")).reset_index()
    assert {"row_idx", "PUBCHEM_CID"} <= set(mi.columns)
    return mi[["row_idx", "PUBCHEM_CID"]]


def load_ours(test, results_dir=None):
    """Our 10-seed predictions for the 106 compounds, back-transformed out of log10 space.

    Returns (ens, seeds): ens = {"CL": [106], "VDss": [106]} ensemble predictions in linear
    units; seeds = {"CL": [10, 106], "VDss": [10, 106]} per-seed predictions, same order as
    `test`. The `id` column of ensemble_predictions.csv is row_idx of
    data/cl_vdss/modeling_index.parquet, which carries PUBCHEM_CID; that is the join key.
    Asserts the join is complete (106 rows) and that our back-transformed observed values
    reproduce the SI's observed values.
    """
    results_dir = results_dir or os.path.join(ROOT, "results", "final_models")
    cid = _cid_map()
    ens, seeds = {}, {}
    for target, obs_col in (("CL", CL_OBS), ("VDss", VD_OBS)):
        d = pd.read_csv(os.path.join(results_dir, target, "ensemble_predictions.csv"))
        per_seed = np.load(os.path.join(results_dir, target, "per_seed_preds.npy")).astype(float)
        assert per_seed.shape == (10, len(d)), f"{target}: per-seed array {per_seed.shape} vs {len(d)} rows"
        assert np.allclose(per_seed.mean(0), d.y_pred_ensemble, atol=1e-5), f"{target}: seeds do not average to the ensemble"

        d = d.merge(cid, left_on="id", right_on="row_idx", how="left", validate="one_to_one")
        assert d.PUBCHEM_CID.notna().all(), f"{target}: ids missing from modeling_index"
        d["_pos"] = np.arange(len(d))

        j = test[["PubChem_CID"]].merge(d, left_on="PubChem_CID", right_on="PUBCHEM_CID",
                                        how="left", validate="one_to_one")
        missing = j.loc[j.y_pred_ensemble.isna(), "PubChem_CID"].tolist()
        assert not missing, f"{target}: {len(missing)} of {N_TEST} compounds absent from our predictions: {missing}"
        assert len(j) == N_TEST

        obs_back = 10 ** j.y_true.to_numpy(float)
        assert np.allclose(obs_back, test[obs_col].to_numpy(float), rtol=1e-3), \
            f"{target}: back-transformed y_true disagrees with the SI's observed values -- join is wrong"

        ens[target] = 10 ** j.y_pred_ensemble.to_numpy(float)
        seeds[target] = 10 ** per_seed[:, j._pos.to_numpy(int)]
        assert ens[target].shape == (N_TEST,) and seeds[target].shape == (10, N_TEST)
        assert np.isfinite(ens[target]).all() and (ens[target] > 0).all()
    return ens, seeds


# ------------------------------------------------------------------- exposure model

def auc_cmax(cl, v, t_hours, dose=DOSE_MG_PER_KG):
    """One-compartment IV exposure for a `dose` mg/kg dose.

    cl: L/h/kg, v: L/kg, t_hours: infusion duration in hours (0 = bolus).
    Returns (AUC [mg*h/L = ug*h/mL], Cmax [mg/L = ug/mL]).

    AUC = dose / CL for both bolus and infusion under linear kinetics. For an infusion,
    Cmax = (R0 / CL) * (1 - exp(-k*T)) with R0 = dose / T and k = CL / V; as k*T -> 0 this
    tends to dose / V, so the branch is evaluated with expm1 and falls back to the bolus
    form for k*T below the point where it loses precision.
    """
    cl = np.asarray(cl, float)
    v = np.asarray(v, float)
    t_hours = np.asarray(t_hours, float)
    assert np.isfinite(cl).all() and (cl > 0).all(), "CL must be finite and positive"
    assert np.isfinite(v).all() and (v > 0).all(), "V must be finite and positive"
    assert np.isfinite(t_hours).all() and (t_hours >= 0).all(), "infusion time must be finite and non-negative"

    auc = dose / cl
    k = cl / v
    kt = k * t_hours
    # -expm1(-kt)/kt -> 1 as kt -> 0; guard the 0/0 and the catastrophic-cancellation regime
    small = kt < 1e-8
    ratio = np.where(small, 1.0, -np.expm1(-np.where(small, 1.0, kt)) / np.where(small, 1.0, kt))
    cmax = (dose / v) * ratio          # = (R0/CL)*(1-exp(-kT)) with R0 = dose/T, and = dose/V at T = 0
    assert np.isfinite(cmax).all() and (cmax > 0).all()
    return auc, cmax


def infusion_hours(test):
    """Infusion duration in hours, plus the number of compounds treated as a bolus.

    Blank/NaN InfusionTime_min is treated as a bolus (T = 0) and counted, never imputed.
    """
    raw = test[T_MIN].to_numpy(float)
    n_blank = int(np.isnan(raw).sum())
    n_zero = int((raw == 0).sum())
    t = np.where(np.isnan(raw), 0.0, raw) / 60.0
    assert (t >= 0).all()
    return t, n_blank, n_zero


# ------------------------------------------------------------------------- metrics

def exposure_metrics(obs, pred, names=METRIC_NAMES):
    """Fold-error metrics of `pred` against `obs` (both linear, positive).

    R2/MAE/RMSE/GMFE/within-n-fold come from src/metrics.py (`eval_log_preds` on log10
    values); median/mean fold error are added here.
    """
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    out = dict(metrics_mod.eval_log_preds(np.log10(obs), np.log10(pred)))
    fe = fold_error(obs, pred)
    out["median_FE"] = float(np.median(fe))
    out["mean_FE"] = float(np.mean(fe))
    return {m: out[m] for m in names}


def fold_error(obs, pred):
    """Per-compound fold error max(pred/obs, obs/pred) >= 1."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return np.maximum(pred / obs, obs / pred)


def verdict(lo, hi, higher_is_better, better="ours", worse="jia"):
    """Three-way verdict in the vocabulary the repo already uses, for a CI of (ours - jia)."""
    if lo > 0:
        return f"{better if higher_is_better else worse} better (CI excludes 0)"
    if hi < 0:
        return f"{worse if higher_is_better else better} better (CI excludes 0)"
    return "CI includes 0"


# ------------------------------------------- paired step-1 comparison vs Jia et al.

PAIRED_SPEC = {
    # endpoint: (SI observed column, SI predicted column, values are log10-modelled by us)
    "pKa_Acidic": ("pKa_Acidic", "pred_pKa_Acidic", False),
    "pKa_Basic": ("pKa_Basic", "pred_pKa_Basic", False),
    "CL": (CL_OBS, CL_JIA, True),
    "VDss": (VD_OBS, VD_JIA, True),
    "Fu": (FU_OBS, FU_JIA, True),
}


def load_paired_predictions(si_path=SI_PATH, results_dir=None):
    """Our predictions and Jia et al.'s, for the same 106 compounds, for all five endpoints.

    Jia et al.'s per-compound predictions exist only in sheet `106_cmp_test_set`, so a paired
    comparison is possible on those 106 compounds and nowhere else. Returns
    {endpoint: DataFrame[PubChem_CID, Name, y_obs, y_jia, y_ours, seeds]} with every column in
    the space the endpoint is modelled in (log10 for CL/VDss/Fu, raw pKa units otherwise), plus
    a [10, 106] per-seed array.

    Joins: CL/VDss by `row_idx` -> PUBCHEM_CID (as in `load_ours`); pKa/Fu by the parent SMILES
    of sheet `pKas_modeling_set`, which is the `id` of our pKa and Fu prediction files. Asserts
    106 rows, uniqueness, and that the observed values agree with ours.
    """
    results_dir = results_dir or os.path.join(ROOT, "results", "final_models")
    x = pd.ExcelFile(si_path)
    test = load_test_set(si_path)
    pk = x.parse(MODELING_SHEET.replace("Fu_VDss_CL_modeling_set", "pKas_modeling_set"))
    smiles = pk.set_index("PUBCHEM_CID").reindex(test.PubChem_CID)["SMILES_parent"].to_numpy()
    assert pd.notna(smiles).all(), "parent SMILES missing for some of the 106 compounds"
    cid = _cid_map()

    out = {}
    for endpoint, (obs_col, jia_col, is_log) in PAIRED_SPEC.items():
        ours = pd.read_csv(os.path.join(results_dir, endpoint, "ensemble_predictions.csv"))
        per_seed = np.load(os.path.join(results_dir, endpoint, "per_seed_preds.npy")).astype(float)
        assert per_seed.shape == (10, len(ours))
        ours = ours.assign(_pos=np.arange(len(ours)))

        left = test[["PubChem_CID", "Name_trend", obs_col, jia_col]].copy()
        if endpoint in ("CL", "VDss"):
            ours = ours.merge(cid, left_on="id", right_on="row_idx", how="left", validate="one_to_one")
            j = left.merge(ours, left_on="PubChem_CID", right_on="PUBCHEM_CID", how="left", validate="one_to_one")
        else:
            j = left.assign(_sm=smiles).merge(ours, left_on="_sm", right_on="id", how="left", validate="one_to_one")
        assert len(j) == N_TEST, f"{endpoint}: join produced {len(j)} rows"
        missing = j.loc[j.y_pred_ensemble.isna(), "PubChem_CID"].tolist()
        assert not missing, f"{endpoint}: {len(missing)} compounds absent from our predictions: {missing}"

        obs_si = j[obs_col].to_numpy(float)
        jia = j[jia_col].to_numpy(float)
        assert np.isfinite(obs_si).all() and np.isfinite(jia).all(), endpoint
        if is_log:
            assert (obs_si > 0).all() and (jia > 0).all(), f"{endpoint}: non-positive value cannot be log10-transformed"
            obs_si, jia = np.log10(obs_si), np.log10(jia)
        assert np.allclose(j.y_true.to_numpy(float), obs_si, rtol=1e-2, atol=1e-2), \
            f"{endpoint}: our observed values disagree with the SI's -- join is wrong"

        pos = j._pos.to_numpy(int)
        out[endpoint] = (pd.DataFrame({
            "PubChem_CID": j.PubChem_CID, "Name": j.Name_trend,
            "y_obs": j.y_true.to_numpy(float), "y_jia": jia, "y_ours": j.y_pred_ensemble.to_numpy(float),
        }), per_seed[:, pos])
    return out
