"""Minimal data loaders for the 5 benchmark-paper-configuration notebooks."""
import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PKA_DIR = os.path.join(_HERE, "..", "data", "pka")
CLVDSS_DIR = os.path.join(_HERE, "..", "data", "cl_vdss")

PKA_TARGET_COL = {
    "pKa_Acidic": ("pKa_Acidic", "pKaAcidic_train_test"),
    "pKa_Basic": ("pKa_Basic", "pKaBasic_train_test"),
}
PKA_GRAPH_PATH = {
    "pKa_Acidic": os.path.join(PKA_DIR, "aux_xtb_graph_pkaacidic.parquet"),
    "pKa_Basic": os.path.join(PKA_DIR, "aux_xtb_graph.parquet"),
}


def load_pka_official_split(target):
    """Returns (graph_df_ok, index_df) for the pKa GNN's official train/test confirmation --
    mirrors mfmn_v3/experiments/e13_final_confirmation.py / e14_final_confirmation.py exactly."""
    val_col, split_col = PKA_TARGET_COL[target]
    graph_df = pd.read_parquet(PKA_GRAPH_PATH[target])
    graph_ok = graph_df[graph_df.status == "ok"].reset_index()
    idx = pd.read_parquet(os.path.join(PKA_DIR, "aux_index.parquet"))[["canon_smiles", split_col, val_col]]
    graph_ok = graph_ok.merge(idx, on="canon_smiles", how="left")
    return graph_ok, val_col, split_col


def load_full_pool(target):
    """Returns (all_row_idx, y_log, sub) for every compound labeled for `target` (CL or VDss),
    i.e. paper's train+test rows combined -- mirrors scripts/cv_harness.py's load_full_pool."""
    idx = pd.read_parquet(os.path.join(CLVDSS_DIR, "modeling_index.parquet")).reset_index()
    log_col, split_col = ("lgCL", "CL_train_test") if target == "CL" else ("lgVD", "VD_train_test")
    sub = idx[idx[split_col].notna()].copy()
    return sub["row_idx"].values, sub[log_col].values, sub
