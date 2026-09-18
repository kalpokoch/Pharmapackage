"""
Descriptor-based features for MFMN on CL/VDss (1,501-compound pool).

Two feature groups are built per fold (all fold-dependent fitting -- PCA,
scaling -- uses the training fold only):

  1. SHARED CONTEXT: PCA-compressed RDKit+Mordred(2D) concatenated with raw
     MACCS keys (dense binary, not helped by PCA).
  2. FACTOR-SPECIFIC RAW BLOCKS: for each of the 7 candidate factors, the
     specific deterministic descriptors named after that mechanism, plus
     (for ionization/binding) predicted pKa/fu from a separate, legitimate
     structure-based predictor, and (for electronic) xTB HOMO/LUMO/gap/dipole.

Factor -> raw descriptor mapping:
  F1 size        MolWt, HeavyAtomCount, NumRotatableBonds
  F2 lipophilic  MolLogP, MolMR
  F3 ionization  pred_pKa_Acidic, pred_pKa_Basic
  F4 polarity    TPSA, NumHDonors, NumHAcceptors
  F5 binding     pred_fu
  F6 electronic  homo_ev, lumo_ev, gap_ev, dipole_debye  (xTB)
  F7 structural  RingCount, NumAromaticRings, FractionCSP3, NumSaturatedRings
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cl_vdss")
SEED = 42
N_PCA_CONTEXT = 32

DESC_SETS = {
    "rdkit": pd.read_parquet(os.path.join(DATA_DIR, "rdkit_desc.parquet")),
    "MACCS": pd.read_parquet(os.path.join(DATA_DIR, "maccs_desc.parquet")),
    "mordred": pd.read_parquet(os.path.join(DATA_DIR, "mordred_desc.parquet")),
}

_aux_feat = pd.read_parquet(os.path.join(DATA_DIR, "aux_features_for_cl_vd.parquet"))
_idx = pd.read_parquet(os.path.join(DATA_DIR, "modeling_index.parquet")).reset_index()
_row_to_smiles = dict(zip(_idx["row_idx"], _idx["canon_smiles"]))
_xtb = pd.read_parquet(os.path.join(DATA_DIR, "xtb_full_results.parquet")).set_index("smiles")

FACTOR_COLS = {
    "size": ["MolWt", "HeavyAtomCount", "NumRotatableBonds"],
    "lipophilicity": ["MolLogP", "MolMR"],
    "polarity": ["TPSA", "NumHDonors", "NumHAcceptors"],
    "structural": ["RingCount", "NumAromaticRings", "FractionCSP3", "NumSaturatedRings"],
}
XTB_COLS = ["homo_ev", "lumo_ev", "gap_ev", "dipole_debye"]
FACTOR_NAMES = ["size", "lipophilicity", "ionization", "polarity", "binding", "electronic", "structural"]


def _clean(df, tr, te):
    Xtr = df.loc[tr].replace([np.inf, -np.inf], np.nan)
    Xte = df.loc[te].replace([np.inf, -np.inf], np.nan)
    keep = Xtr.columns[Xtr.notna().all(axis=0)]
    Xtr, Xte = Xtr[keep], Xte[keep]
    keep = Xtr.columns[Xtr.var(axis=0) > 1e-12]
    Xtr, Xte = Xtr[keep], Xte[keep]
    med = Xtr.median(axis=0)
    return Xtr.values, Xte.fillna(med).values


def _fit_scale(Xtr, Xte, clip=10):
    sc = StandardScaler().fit(Xtr)
    return np.clip(sc.transform(Xtr), -clip, clip), np.clip(sc.transform(Xte), -clip, clip)


def _rdkit_cols(names, ids):
    df = DESC_SETS["rdkit"]
    return df.loc[ids, names].values.astype(np.float64)


def _aux_block(ids, cols):
    rows = [[_aux_feat.loc[_row_to_smiles[i]][c] for c in cols] for i in ids]
    return np.array(rows, dtype=np.float64)


def _xtb_block(ids):
    rows = []
    for i in ids:
        smi = _row_to_smiles[i]
        if smi in _xtb.index:
            rows.append([_xtb.loc[smi, c] for c in XTB_COLS])
        else:
            rows.append([np.nan] * len(XTB_COLS))
    return np.array(rows, dtype=np.float64)


def build_shared_context(train_ids, test_ids):
    rd_tr, rd_te = _clean(DESC_SETS["rdkit"], train_ids, test_ids)
    mo_tr, mo_te = _clean(DESC_SETS["mordred"], train_ids, test_ids)
    cont_tr = np.hstack([rd_tr, mo_tr])
    cont_te = np.hstack([rd_te, mo_te])
    cont_tr, cont_te = _fit_scale(cont_tr, cont_te)
    k = min(N_PCA_CONTEXT, cont_tr.shape[0], cont_tr.shape[1])
    pca = PCA(n_components=k, random_state=SEED).fit(cont_tr)
    cont_tr, cont_te = pca.transform(cont_tr), pca.transform(cont_te)
    cont_tr, cont_te = _fit_scale(cont_tr, cont_te)

    maccs_tr = DESC_SETS["MACCS"].loc[train_ids].values.astype(np.float32)
    maccs_te = DESC_SETS["MACCS"].loc[test_ids].values.astype(np.float32)
    return (np.hstack([cont_tr, maccs_tr]).astype(np.float32),
            np.hstack([cont_te, maccs_te]).astype(np.float32))


def build_factor_blocks(train_ids, test_ids):
    """Returns {factor_name: (Xtr, Xte)}, each fold-fit-scaled, median-imputed."""
    blocks_tr, blocks_te = {}, {}

    for fname, cols in FACTOR_COLS.items():
        raw_tr = _rdkit_cols(cols, train_ids)
        raw_te = _rdkit_cols(cols, test_ids)
        med = np.nanmedian(raw_tr, axis=0)
        raw_tr = np.where(np.isnan(raw_tr), med, raw_tr)
        raw_te = np.where(np.isnan(raw_te), med, raw_te)
        Xtr, Xte = _fit_scale(raw_tr, raw_te)
        blocks_tr[fname], blocks_te[fname] = Xtr.astype(np.float32), Xte.astype(np.float32)

    # ionization: predicted pKa (aux)
    raw_tr = _aux_block(train_ids, ["pred_pKa_Acidic", "pred_pKa_Basic"])
    raw_te = _aux_block(test_ids, ["pred_pKa_Acidic", "pred_pKa_Basic"])
    med = np.nanmedian(raw_tr, axis=0)
    raw_tr = np.where(np.isnan(raw_tr), med, raw_tr)
    raw_te = np.where(np.isnan(raw_te), med, raw_te)
    Xtr, Xte = _fit_scale(raw_tr, raw_te)
    blocks_tr["ionization"], blocks_te["ionization"] = Xtr.astype(np.float32), Xte.astype(np.float32)

    # binding: predicted fu (aux)
    raw_tr = _aux_block(train_ids, ["pred_fu"])
    raw_te = _aux_block(test_ids, ["pred_fu"])
    med = np.nanmedian(raw_tr, axis=0)
    raw_tr = np.where(np.isnan(raw_tr), med, raw_tr)
    raw_te = np.where(np.isnan(raw_te), med, raw_te)
    Xtr, Xte = _fit_scale(raw_tr, raw_te)
    blocks_tr["binding"], blocks_te["binding"] = Xtr.astype(np.float32), Xte.astype(np.float32)

    # electronic: xTB
    raw_tr = _xtb_block(train_ids)
    raw_te = _xtb_block(test_ids)
    med = np.nanmedian(raw_tr, axis=0)
    raw_tr = np.where(np.isnan(raw_tr), med, raw_tr)
    raw_te = np.where(np.isnan(raw_te), med, raw_te)
    Xtr, Xte = _fit_scale(raw_tr, raw_te)
    blocks_tr["electronic"], blocks_te["electronic"] = Xtr.astype(np.float32), Xte.astype(np.float32)

    return blocks_tr, blocks_te


def build_aux_targets(ids):
    """Returns {factor_name: standardization-ready raw array (n,)} for the
    auxiliary-supervision targets. NOT scaled here -- caller standardizes
    using train-fold statistics."""
    AUX_TARGET_COL = {
        "size": ("rdkit", "MolWt"), "lipophilicity": ("rdkit", "MolLogP"),
        "ionization": ("aux", "pred_pKa_Acidic"), "polarity": ("rdkit", "TPSA"),
        "binding": ("aux", "pred_fu"), "electronic": ("xtb", "gap_ev"),
        "structural": ("rdkit", "NumAromaticRings"),
    }
    out = {}
    for fname, (src, col) in AUX_TARGET_COL.items():
        if src == "rdkit":
            vals = DESC_SETS["rdkit"].loc[ids, col].values.astype(np.float64)
        elif src == "aux":
            vals = np.array([_aux_feat.loc[_row_to_smiles[i]][col] for i in ids], dtype=np.float64)
        elif src == "xtb":
            vals = np.array([_xtb.loc[_row_to_smiles[i], col] if _row_to_smiles[i] in _xtb.index
                             else np.nan for i in ids], dtype=np.float64)
        out[fname] = vals
    return out
