"""
Descriptor-based features for MFMN on Fu (trimmed to the exact H6/H7
"raw_fcfp" recipe used by Fu's final confirmed model -- MFMN_DynGate with
ctx_variant="raw_fcfp"). Every one of the 7 factors is built from purely
deterministic descriptors (no fitted predictor of Fu/pKa in the loop):

  F1 size          MolWt, HeavyAtomCount, NumRotatableBonds
  F2 lipophilic     MolLogP, MolMR
  F3 ionization     acidic/basic functional-group fragment counts
  F4 polarity        TPSA, NumHDonors, NumHAcceptors
  F5 binding         PEOE_VSA / SlogP_VSA charge-weighted surface area bins + LabuteASA
  F6 electronic      Gasteiger partial charges + BalabanJ/BertzCT
  F7 structural      RingCount, NumAromaticRings, FractionCSP3, NumSaturatedRings

Shared context ("raw_fcfp"): variance/correlation-pruned RDKit+Mordred+MACCS
(no PCA compression), plus frequency-pruned FCFP6 circular fingerprint bits.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "fu")
SEED = 42

FACTOR_NAMES = ["size", "lipophilicity", "ionization", "polarity", "binding", "electronic", "structural"]

FACTOR_COLS = {
    "size": ["MolWt", "HeavyAtomCount", "NumRotatableBonds"],
    "lipophilicity": ["MolLogP", "MolMR"],
    "ionization": ["fr_Al_COO", "fr_Ar_COO", "fr_COO", "fr_COO2", "fr_NH0", "fr_NH1", "fr_NH2",
                   "fr_ArN", "fr_Ar_N", "fr_pyridine", "fr_imidazole", "fr_guanido", "fr_amidine",
                   "NHOHCount", "NOCount"],
    "polarity": ["TPSA", "NumHDonors", "NumHAcceptors"],
    "binding": ["PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6",
                "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6",
                "LabuteASA"],
    "electronic": ["MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
                   "BalabanJ", "BertzCT"],
    "structural": ["RingCount", "NumAromaticRings", "FractionCSP3", "NumSaturatedRings"],
}

TARGET_COL = {
    "Fu": ("lgFu", "Fu_train_test", True),
    "pKa_Acidic": ("pKa_Acidic", "pKaAcidic_train_test", False),
    "pKa_Basic": ("pKa_Basic", "pKaBasic_train_test", False),
}

_rdkit = pd.read_parquet(os.path.join(DATA_DIR, "aux_rdkit_desc.parquet"))
_mordred = pd.read_parquet(os.path.join(DATA_DIR, "aux_mordred_desc.parquet"))
_maccs = pd.read_parquet(os.path.join(DATA_DIR, "aux_maccs_desc.parquet"))
_fcfp6 = pd.read_parquet(os.path.join(DATA_DIR, "aux_fcfp6_desc.parquet"))
_aux_index = pd.read_parquet(os.path.join(DATA_DIR, "aux_index.parquet"))


def load_aux_pool(target):
    val_col, split_col, _ = TARGET_COL[target]
    sub = _aux_index[_aux_index[split_col].notna()].copy()
    return sub["canon_smiles"].values, sub[val_col].values.astype(np.float64), sub


def _fit_scale(Xtr, Xte, clip=10):
    sc = StandardScaler().fit(Xtr)
    return np.clip(sc.transform(Xtr), -clip, clip), np.clip(sc.transform(Xte), -clip, clip)


def build_shared_context_raw(train_smi, test_smi, var_thresh=0.2, corr_thresh=0.85):
    """H2: variance/correlation-pruned RDKit+Mordred fed directly (no PCA)."""
    parts_tr, parts_te = [], []
    for name, df in [("rdkit", _rdkit), ("mordred", _mordred)]:
        xtr, xte = df.loc[train_smi].copy(), df.loc[test_smi].copy()
        keep = xtr.columns[xtr.notna().all(axis=0)]
        xtr, xte = xtr[keep], xte[keep]
        parts_tr.append(xtr.add_prefix(f"{name}__"))
        parts_te.append(xte.add_prefix(f"{name}__"))
    Xtr = pd.concat(parts_tr, axis=1)
    Xte = pd.concat(parts_te, axis=1)

    keep = Xtr.columns[Xtr.notna().all(axis=0)]
    Xtr, Xte = Xtr[keep], Xte[keep]
    variances = Xtr.var(axis=0)
    keep = variances[variances >= var_thresh].index
    Xtr, Xte = Xtr[keep], Xte[keep]
    corr = Xtr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > corr_thresh).any()]
    keep = [c for c in Xtr.columns if c not in to_drop]
    Xtr, Xte = Xtr[keep], Xte[keep]

    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    Xte = Xte.replace([np.inf, -np.inf], np.nan)
    med = Xtr.median(axis=0)
    Xtr, Xte = Xtr.fillna(med).values, Xte.fillna(med).values
    cont_tr, cont_te = _fit_scale(Xtr, Xte)

    maccs_tr = _maccs.loc[train_smi].values.astype(np.float32)
    maccs_te = _maccs.loc[test_smi].values.astype(np.float32)
    return (np.hstack([cont_tr, maccs_tr]).astype(np.float32),
            np.hstack([cont_te, maccs_te]).astype(np.float32))


def build_shared_context_raw_fcfp(train_smi, test_smi, var_thresh=0.2, corr_thresh=0.85,
                                   fcfp_min_freq=0.01, fcfp_max_freq=0.99):
    """H6: H2's raw context (RDKit+Mordred+MACCS) plus FCFP6 circular
    fingerprints (1024 bits, variance-pruned) -- Fu's final confirmed context."""
    base_tr, base_te = build_shared_context_raw(train_smi, test_smi, var_thresh, corr_thresh)

    fp_tr = _fcfp6.loc[train_smi].values.astype(np.float32)
    fp_te = _fcfp6.loc[test_smi].values.astype(np.float32)
    freq = fp_tr.mean(axis=0)
    keep = (freq >= fcfp_min_freq) & (freq <= fcfp_max_freq)
    fp_tr, fp_te = fp_tr[:, keep], fp_te[:, keep]

    return (np.hstack([base_tr, fp_tr]).astype(np.float32),
            np.hstack([base_te, fp_te]).astype(np.float32))


def build_factor_blocks(train_smi, test_smi):
    blocks_tr, blocks_te = {}, {}
    for fname, cols in FACTOR_COLS.items():
        raw_tr = _rdkit.loc[train_smi, cols].values.astype(np.float64)
        raw_te = _rdkit.loc[test_smi, cols].values.astype(np.float64)
        med = np.nanmedian(raw_tr, axis=0)
        raw_tr = np.where(np.isnan(raw_tr), med, raw_tr)
        raw_te = np.where(np.isnan(raw_te), med, raw_te)
        Xtr, Xte = _fit_scale(raw_tr, raw_te)
        blocks_tr[fname], blocks_te[fname] = Xtr.astype(np.float32), Xte.astype(np.float32)
    return blocks_tr, blocks_te
