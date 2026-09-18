"""
MFMN-v3, E13: graph featurization + dense-batching utilities for the GNN.

No torch_geometric/dgl dependency -- this box runs a nightly CUDA build of
torch (2.11.0a0+...nv26.02) that prebuilt graph-library wheels are not
guaranteed to match, and the molecules here are small (a few dozen heavy
atoms), so plain padded dense-adjacency batching is simpler and safer than
managing sparse-graph-library ABI compatibility. This is standard practice
for small-molecule GNNs at this scale.

Per-atom (node) features (concatenated into a fixed-size vector per heavy atom):
  - one-hot atomic number over a fixed vocabulary (C,N,O,S,P,F,Cl,Br,I,other)
  - degree (one-hot, 0-5+)
  - formal charge (one-hot, -2..+2)
  - aromatic flag
  - in-ring flag
  - hybridization (one-hot: SP, SP2, SP3, other)
  - implicit + explicit H count (one-hot, 0-4+)
  - xTB partial charge on this atom (mfmn_v3/features/aux_xtb_graph.py)
  - xTB sum/max charge of this atom's attached H's (0 if none -- summarizes
    "how acidic is my most acidic attached proton" without adding H nodes)
  - is_acidic_site flag (from acidic_site.py, threaded through the same
    aux_xtb_graph.py output)

Per-bond (edge) features:
  - bond type one-hot (single/double/triple/aromatic)
  - conjugated flag
  - in-ring flag
  - xTB Wiberg bond order (continuous, real-valued -- the one genuinely new
    physically-grounded edge feature no prior experiment in this project used)

Graphs are batched as dense, padded tensors: node features
[B, N_max, F_node], an adjacency/edge-feature tensor [B, N_max, N_max, F_edge]
(zero elsewhere), and a boolean mask [B, N_max] for valid atoms. This trades
some memory for simplicity given N_max is small here (see NODE_FEATURE_DIM
usage in mfmn_v3/models/graph_mpnn.py).
"""
import numpy as np
from rdkit import Chem

ATOM_VOCAB = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]  # + "other" bucket
HYBRID_VOCAB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
BOND_VOCAB = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

MAX_DEGREE = 5
MAX_FORMAL_CHARGE = 2
MAX_H_COUNT = 4

NODE_FEATURE_DIM = (
    len(ATOM_VOCAB) + 1        # atomic number one-hot + other
    + (MAX_DEGREE + 1)         # degree one-hot 0..5+
    + (2 * MAX_FORMAL_CHARGE + 1)  # formal charge one-hot -2..+2
    + 1                        # aromatic
    + 1                        # in ring
    + (len(HYBRID_VOCAB) + 1)  # hybridization + other
    + (MAX_H_COUNT + 1)        # total H count 0..4+
    + 1                        # xTB atom charge
    + 1                        # xTB attached-H charge sum
    + 1                        # xTB attached-H charge max
    + 1                        # is_acidic_site flag
)
EDGE_FEATURE_DIM = (len(BOND_VOCAB) + 1) + 1 + 1 + 1  # bond type one-hot(+other) + conjugated + in-ring + WBO


def _onehot(idx, n):
    v = np.zeros(n + 1, dtype=np.float32)  # last slot = "other"/overflow
    if 0 <= idx < n:
        v[idx] = 1.0
    else:
        v[n] = 1.0
    return v


def featurize_atom(atom, xtb_charge, h_charge_sum, h_charge_max, is_site):
    sym = atom.GetSymbol()
    z = ATOM_VOCAB.index(sym) if sym in ATOM_VOCAB else len(ATOM_VOCAB)
    v = [_onehot(z, len(ATOM_VOCAB))]
    v.append(_onehot(min(atom.GetDegree(), MAX_DEGREE), MAX_DEGREE))
    fc = int(np.clip(atom.GetFormalCharge(), -MAX_FORMAL_CHARGE, MAX_FORMAL_CHARGE)) + MAX_FORMAL_CHARGE
    v.append(_onehot(fc, 2 * MAX_FORMAL_CHARGE))
    v.append(np.array([float(atom.GetIsAromatic())], dtype=np.float32))
    v.append(np.array([float(atom.IsInRing())], dtype=np.float32))
    hyb = atom.GetHybridization()
    hyb_idx = HYBRID_VOCAB.index(hyb) if hyb in HYBRID_VOCAB else len(HYBRID_VOCAB)
    v.append(_onehot(hyb_idx, len(HYBRID_VOCAB)))
    n_h = atom.GetTotalNumHs()
    v.append(_onehot(min(n_h, MAX_H_COUNT), MAX_H_COUNT))
    v.append(np.array([xtb_charge, h_charge_sum, h_charge_max, float(is_site)], dtype=np.float32))
    return np.concatenate(v).astype(np.float32)


def featurize_bond(bond, wbo):
    bt = bond.GetBondType()
    bt_idx = BOND_VOCAB.index(bt) if bt in BOND_VOCAB else len(BOND_VOCAB)
    v = [_onehot(bt_idx, len(BOND_VOCAB))]
    v.append(np.array([float(bond.GetIsConjugated()), float(bond.IsInRing()), float(wbo)], dtype=np.float32))
    return np.concatenate(v).astype(np.float32)


def mol_to_graph(smi, row, site_finder=None):
    """row: a pandas Series from aux_xtb_graph.parquet (status must be 'ok').
    `site_finder`, if given, is a function(mol_noH) -> (pattern_name, idx, is_strong)
    used INSTEAD of the row's stored site_idx -- e.g. basic_site.find_basic_site
    for pKa_Basic, since aux_xtb_graph.parquet's own site_idx column is always
    the ACIDIC site (computed once, for pKa_Acidic, when the graph-xtb data was
    generated). Recomputing a different site definition is cheap (pure RDKit
    SMARTS matching, no xTB rerun needed) since the atom charges/bonds already
    computed don't depend on which site definition is used to read them out.
    Returns (node_feats [N,F_node], adj_feats [N,N,F_edge], adj_mask [N,N], site_idx)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    if "." in smi:
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
        m = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    n = m.GetNumAtoms()
    if n != row["n_heavy"]:
        return None  # defensive: featurization must match the xTB run's atom count/order

    charge = row["atom_charge"]
    h_sum = row["atom_h_charge_sum"]
    h_max = row["atom_h_charge_max"]
    if site_finder is not None:
        _, site_idx, _ = site_finder(m)
        site_idx = int(site_idx) if site_idx is not None else -1
    else:
        site_idx = int(row["site_idx"])

    node_feats = np.stack([
        featurize_atom(m.GetAtomWithIdx(i), charge[i], h_sum[i], h_max[i], i == site_idx)
        for i in range(n)
    ])
    adj_feats = np.zeros((n, n, EDGE_FEATURE_DIM), dtype=np.float32)
    adj_mask = np.zeros((n, n), dtype=np.float32)
    bond_i, bond_j, bond_wbo = row["bond_i"], row["bond_j"], row["bond_wbo"]
    for k in range(len(bond_i)):
        i, j = int(bond_i[k]), int(bond_j[k])
        b = m.GetBondBetweenAtoms(i, j)
        if b is None:
            continue
        ef = featurize_bond(b, bond_wbo[k])
        adj_feats[i, j] = ef
        adj_feats[j, i] = ef
        adj_mask[i, j] = 1.0
        adj_mask[j, i] = 1.0
    return node_feats, adj_feats, adj_mask, site_idx


def batch_graphs(graph_list, max_n=None):
    """graph_list: list of (node_feats, adj_feats, adj_mask, site_idx). Returns
    padded batch tensors (numpy) + a valid-atom mask + site index per graph."""
    n_max = max_n or max(g[0].shape[0] for g in graph_list)
    B = len(graph_list)
    node_dim = graph_list[0][0].shape[1]
    edge_dim = graph_list[0][1].shape[2]
    nodes = np.zeros((B, n_max, node_dim), dtype=np.float32)
    adj = np.zeros((B, n_max, n_max, edge_dim), dtype=np.float32)
    adj_mask = np.zeros((B, n_max, n_max), dtype=np.float32)
    node_mask = np.zeros((B, n_max), dtype=np.float32)
    site_idx = np.full(B, -1, dtype=np.int64)
    for b, (nf, ef, am, sidx) in enumerate(graph_list):
        n = nf.shape[0]
        nodes[b, :n] = nf
        adj[b, :n, :n] = ef
        adj_mask[b, :n, :n] = am
        node_mask[b, :n] = 1.0
        site_idx[b] = sidx if 0 <= sidx < n else -1
    return nodes, adj, adj_mask, node_mask, site_idx
