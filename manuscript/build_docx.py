"""Build the Results-section draft (.docx) from the computed tables and figures.

The Results, Discussion and Conclusion text lives in `sections_results.py`, shared with
`build_manuscript.py` so the two documents cannot drift apart. This builder emits that part on
its own, numbering tables I.. and figures 1.. from the start.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import docx_common                                          # noqa: E402
import sections_results                                     # noqa: E402

TITLE = {"pKa_Acidic": "pKa (acidic)", "pKa_Basic": "pKa (basic)", "CL": "CL", "VDss": "VDss", "Fu": "Fu"}
MODEL = {"pKa_Acidic": "GraphMPNN", "pKa_Basic": "GraphMPNN (no site readout)", "CL": "MFMN-DynGate",
         "VDss": "MFMN-InteractAux", "Fu": "MFMN-DynGate + LW"}
MLAB = {"R2": "R²", "MAE": "MAE", "RMSE": "RMSE", "GMFE": "GMFE", "within_2fold": "W2F",
        "within_1_pKa_unit": "W1"}
HIGHER = {"R2": True, "MAE": False, "RMSE": False, "GMFE": False, "within_2fold": True,
          "within_1_pKa_unit": True}


def make_context(doc, helpers, numbering, first_section="V", draft_note=False):
    """Assemble the dict `sections_results.emit` expects: data, helpers and the labels its
    tables, figures and section headings will carry."""
    d = {name: pd.read_csv(HERE / f"t_{name}.csv")
         for name in ("main", "paper", "baselines", "ablation", "paired_vs_jia", "downstream")}
    main = d["main"]

    def m(t, metric):
        return main[(main.target == t) & (main.metric == metric)].iloc[0]

    roman = docx_common.ROMAN
    i = roman.index(first_section)
    ctx = dict(helpers)
    ctx.update({
        "D": HERE, "m": m,
        "main": main, "paper": d["paper"], "base": d["baselines"], "abl": d["ablation"],
        "pair": d["paired_vs_jia"], "down": d["downstream"],
        "TITLE": TITLE, "MODEL": MODEL, "MLAB": MLAB, "HIGHER": HIGHER,
        "S_RES": roman[i], "S_DIS": roman[i + 1], "S_CON": roman[i + 2],
        "draft_note": draft_note,
    })
    for n, lab in enumerate(numbering.peek_tables(8), 1):
        ctx[f"T{n}"] = lab
    for n, lab in enumerate(numbering.peek_figures(6), 1):
        ctx[f"F{n}"] = lab
    return ctx


def main():
    doc, helpers = docx_common.make_builder()
    numbering = docx_common.Numbering()
    ctx = make_context(doc, helpers, numbering, first_section="V", draft_note=True)
    sections_results.emit(ctx)
    out = HERE / "Results_section_draft.docx"
    doc.save(out)
    print(out)


if __name__ == "__main__":
    main()
