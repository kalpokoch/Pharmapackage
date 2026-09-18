"""
MFMN-v3, E14: basic-site detection for pKa_Basic -- mirrors acidic_site.py.

Same rationale as acidic_site.py (see that module's docstring): a graph
neural network's site-aware readout needs to know WHICH atom is the
ionizable one. For pKa_Basic that is the protonatable heteroatom (usually
an amine or an aromatic N with a lone pair available for protonation), not
an acidic O-H/N-H.

Priority-ordered SMARTS, strongest/most common basic groups first. Each
pattern's `site_pos` indexes the SMARTS match tuple for the atom that
actually gets protonated -- verified explicitly per pattern (the earlier,
now-fixed bug in acidic_site.py was assuming this was always position 0,
which is wrong whenever the basic nitrogen is not the first atom written in
the SMARTS).
"""
from rdkit import Chem

BASIC_SMARTS_PRIORITY = [
    # strong/common bases first
    ("amidine",        "[NX3;!$(NC=O)]=[CX3]", 0),                       # amidine =N- (protonates here)
    ("guanidine",      "[NX3]C(=[NX2])[NX3]", 2),                        # guanidine imine N
    ("primary_amine",  "[NX3H2;!$(NC=O);!$(N=*);!a]", 0),
    ("secondary_amine", "[NX3H1;!$(NC=O);!$(N=*);!a;!$(N[#16](=O)=O)]", 0),
    ("tertiary_amine", "[NX3H0;!$(NC=O);!$(N=*);!a;!$(N[#16](=O)=O)]", 0),
    ("pyridine_n",     "[nX2H0;!$(n(:*):*=O)]", 0),                      # aromatic pyridine-type N (lone pair, sp2, no H)
    ("imidazole_basic_n", "[nX2H0]1[cX3][nX3H1][cX3][cX3]1", 0),          # imidazole's sp2 non-H-bearing N
    ("aniline",        "[NX3H2][c]", 0),                                 # weakly basic, aniline-type
]
_COMPILED = [(name, Chem.MolFromSmarts(s), pos) for name, s, pos in BASIC_SMARTS_PRIORITY]
assert all(p is not None for _, p, _pos in _COMPILED), "invalid SMARTS in BASIC_SMARTS_PRIORITY"

STRONG_PATTERNS = {name for name, _, _ in BASIC_SMARTS_PRIORITY if name not in ("aniline",)}


def find_basic_site(mol):
    """Returns (pattern_name, atom_idx, is_strong) for the first matching
    pattern in priority order, or (None, None, False). `mol` must be the
    plain (no explicit-H) RDKit mol whose atom indices match the xyz file
    written for the xTB run (AddHs appends, never reorders)."""
    for name, patt, pos in _COMPILED:
        match = mol.GetSubstructMatch(patt)
        if match:
            return name, match[pos], name in STRONG_PATTERNS
    return None, None, False


def coverage_report(smiles_list):
    from collections import Counter
    counts = Counter()
    n_strong, n_any = 0, 0
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        name, idx, strong = find_basic_site(m)
        if name is not None:
            counts[name] += 1
            n_any += 1
            if strong:
                n_strong += 1
    return {"n_total": len(smiles_list), "n_any_site": n_any, "n_strong_site": n_strong,
            "by_pattern": dict(counts)}


if __name__ == "__main__":
    import os
    import pandas as pd
    idx = pd.read_parquet(os.path.join(os.path.dirname(__file__), "..", "data", "pka", "aux_index.parquet"))
    sub = idx[idx.pKaBasic_train_test.notna()]
    rep = coverage_report(sub.canon_smiles.tolist())
    print(f"n_total={rep['n_total']}  n_any_site={rep['n_any_site']} "
          f"({rep['n_any_site']/rep['n_total']*100:.1f}%)  "
          f"n_strong_site={rep['n_strong_site']} ({rep['n_strong_site']/rep['n_total']*100:.1f}%)")
    for k, v in sorted(rep["by_pattern"].items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:5d}")
