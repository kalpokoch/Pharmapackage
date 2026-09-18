"""
MFMN-v3, E12: acidic-site detection for pKa_Acidic.

Rationale (why a site detector at all): E11 found that whole-molecule
electronic descriptors (HOMO/LUMO/gap/dipole/solvation terms) do NOT close
pKa_Acidic's fold-accuracy gap -- checked properly, including a
regression-to-the-mean control, in results/v3/e11_xtb_electronic/. A
whole-molecule scalar dilutes exactly the signal that should matter for an
acid/base equilibrium: the electronic environment AT the specific
deprotonating atom. This module identifies that atom via SMARTS matching so
mfmn_v3/features/aux_xtb_site.py can read its (and its neighbors') xTB
partial charge instead of a whole-molecule average.

Design: a priority-ordered list of acidic functional-group patterns, most
specific/strongest acid first. A compound gets the FIRST pattern that
matches (not all matches) so multi-acid molecules get one determinate site
rather than an ambiguous blend -- consistent with how the paper's own
pKa_Acidic label is itself a single value (presumably the most acidic site).
Coverage on the 6,082-compound pKa_Acidic pool: 77.4% by a "strong" pattern
(real ionizable-group heuristics), rising to 91.8% with a weak aliphatic-OH
fallback included. The remaining ~8% (already-drawn anions like
carboxylate/phenolate SMILES, oximes past the added pattern, exotic groups)
get no localized descriptor and fall back to the ordinary median-imputed
NaN handling used throughout this pipeline -- never a guessed site.

Each pattern below carries an explicit `site_pos` -- the index INTO THE
SMARTS MATCH TUPLE of the atom that actually loses H+ (or, for the
already-ionized carboxylate_anion pattern, the atom that carries the
charge). This is NOT always match[0]: RDKit match tuples follow SMARTS
atom-writing order, not chemical role, so e.g. "[SX4](=O)(=O)[OX2H1]" has
the acidic oxygen at position 3, not the sulfur at position 0. (An earlier
version of this file assumed match[0] uniformly and was caught by
sanity-checking the pilot xTB charges: site atoms it flagged were reading
positive/carbonyl-carbon-like charges instead of the expected electronegative
O/N -- fixed here before the production run in mfmn_v3/features/aux_xtb_site.py.)
"""
from rdkit import Chem

# (name, smarts, site_pos) -- site_pos indexes the match tuple, see module docstring.
ACIDIC_SMARTS_PRIORITY = [
    ("sulfonic_acid",     "[SX4](=O)(=O)[OX2H1]",           3),
    ("phosphonic_acid",   "[PX4](=O)([OX2H1])[OX2H1]",      2),
    ("carboxylic_acid",   "[CX3](=O)[OX2H1]",               2),
    ("carboxylate_anion", "[CX3](=O)[O-]",                  2),  # already-ionized SMILES; site for charge lookup only
    ("sulfonamide_NH",    "[SX4](=O)(=O)[NX3;H1,H2]",       3),
    ("hydroxamic_acid",   "[CX3](=O)[NX3H1][OX2H1]",        3),  # ionizes at the N-OH oxygen
    ("phenol",            "[OX2H1][c]",                     0),
    ("oxime_OH",          "[CX3]=[NX2][OX2H1]",             2),
    ("amide_like_NH",     "[NX3H1;!$(NC=[!O])]C(=O)",       0),  # amide/imide/uracil/hydantoin/barbiturate NH
    ("aromatic_NH",       "[nX3H1]",                        0),  # pyrrole/indole/pyrazole/indazole/triazole/tetrazole NH
    ("alkyl_OH_weak",     "[CX4][OX2H1]",                   1),  # weak fallback only
]
_COMPILED = [(name, Chem.MolFromSmarts(s), pos) for name, s, pos in ACIDIC_SMARTS_PRIORITY]
assert all(p is not None for _, p, _pos in _COMPILED), "invalid SMARTS in ACIDIC_SMARTS_PRIORITY"

STRONG_PATTERNS = {name for name, _, _ in ACIDIC_SMARTS_PRIORITY if name != "alkyl_OH_weak"}


def find_acidic_site(mol):
    """Returns (pattern_name, atom_idx, is_strong) for the first matching pattern
    in priority order, or (None, None, False) if nothing matches. `mol` must be
    the plain (no explicit-H) RDKit mol whose atom indices match the xyz file
    written for the xTB run (AddHs appends, never reorders, so this index is
    valid directly against the xtb `charges` file)."""
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
        name, idx, strong = find_acidic_site(m)
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
    sub = idx[idx.pKaAcidic_train_test.notna()]
    rep = coverage_report(sub.canon_smiles.tolist())
    print(f"n_total={rep['n_total']}  n_any_site={rep['n_any_site']} "
          f"({rep['n_any_site']/rep['n_total']*100:.1f}%)  "
          f"n_strong_site={rep['n_strong_site']} ({rep['n_strong_site']/rep['n_total']*100:.1f}%)")
    for k, v in sorted(rep["by_pattern"].items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:5d}")
