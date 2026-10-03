"""Build the Results-section draft (.docx) from the computed tables and figures."""
import pandas as pd
from pathlib import Path
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

D = Path(__file__).parent
main = pd.read_csv(D / "t_main.csv")
paper = pd.read_csv(D / "t_paper.csv")
base = pd.read_csv(D / "t_baselines.csv")
abl = pd.read_csv(D / "t_ablation.csv")
pair = pd.read_csv(D / "t_paired_vs_jia.csv")

TITLE = {"pKa_Acidic": "pKa (acidic)", "pKa_Basic": "pKa (basic)", "CL": "CL", "VDss": "VDss", "Fu": "Fu"}
MODEL = {"pKa_Acidic": "GraphMPNN", "pKa_Basic": "GraphMPNN (no site readout)", "CL": "MFMN-DynGate",
         "VDss": "MFMN-InteractAux", "Fu": "MFMN-DynGate + LW"}
MLAB = {"R2": "R²", "MAE": "MAE", "RMSE": "RMSE", "GMFE": "GMFE", "within_2fold": "W2F", "within_1_pKa_unit": "W1"}
HIGHER = {"R2": True, "MAE": False, "RMSE": False, "GMFE": False, "within_2fold": True, "within_1_pKa_unit": True}
NEEDS = RGBColor(0xC0, 0x00, 0x00)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Inches(1))
sec.top_margin = sec.bottom_margin = Inches(1)

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.font.size = Pt(10)
st.paragraph_format.space_after = Pt(4)
st.paragraph_format.line_spacing = 1.1
for name, size in (("Heading 1", 12), ("Heading 2", 10.5)):
    h = doc.styles[name]
    h.font.name = "Times New Roman"
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    h.font.size = Pt(size)
    h.font.bold = name == "Heading 1"
    h.font.italic = name == "Heading 2"
    h.font.color.rgb = RGBColor(0, 0, 0)
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(4)


def para(parts, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, italic=False):
    """parts: str or list of str / (str, 'needs'|'b'|'i'|'sub'|'sup')."""
    p = doc.add_paragraph()
    p.alignment = align
    for part in ([parts] if isinstance(parts, str) else parts):
        txt, kind = (part, None) if isinstance(part, str) else part
        r = p.add_run(txt)
        r.italic = italic or kind == "i"
        r.bold = kind == "b"
        if kind == "needs":
            r.font.color.rgb = NEEDS
            r.bold = True
        if kind == "sub":
            r.font.subscript = True
        if size:
            r.font.size = Pt(size)
    return p


def caption(label, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    r = p.add_run(label)
    r.font.size = Pt(8)
    r.font.small_caps = True
    r2 = p.add_run(text)
    r2.font.size = Pt(8)
    return p


def fig_caption(label, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(label)
    r.font.size = Pt(8)
    r2 = p.add_run(text)
    r2.font.size = Pt(8)


def set_cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, val in kw.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), "6")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)


def table(header, rows, widths, notes=None, bold_cells=None, group_rows=None):
    """IEEE-style three-rule table. rows: list of lists of str. bold_cells: set of (r, c)."""
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    bold_cells = bold_cells or set()
    group_rows = group_rows or set()
    for ri, row in enumerate([header] + rows):
        for ci, val in enumerate(row):
            c = t.cell(ri, ci)
            c.width = Inches(widths[ci])
            c.text = ""
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(val)
            r.font.size = Pt(7.5)
            if ri == 0 or (ri - 1, ci) in bold_cells:
                r.bold = True
            if "PENDING" in val or "NEEDS" in val:
                r.font.color.rgb = NEEDS
            if (ri - 1) in group_rows and ri > 0:
                r.italic = True
            if ri == 0:
                set_cell_border(c, top="single", bottom="single")
            if ri == len(rows):
                set_cell_border(c, bottom="single")
    for ci, w in enumerate(widths):
        for cell in t.columns[ci].cells:
            cell.width = Inches(w)
    if notes:
        for n in notes:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(n)
            r.font.size = Pt(7)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def f(v, d=3):
    return f"{v:.{d}f}"


def m(t, metric):
    return main[(main.target == t) & (main.metric == metric)].iloc[0]


# ======================================================================
h = doc.add_heading("V. Results", level=1)
para([("[Draft – Results section only. Section number, reference numbers and cross-references to the "
       "Methods/Experiments sections are placeholders to be aligned with the full manuscript.]", "needs")], size=8)

para("This section reports the performance of the two proposed model families, GraphMPNN for the acid and base "
     "dissociation constants (pKa) and the multi-factor mechanistic network (MFMN) family for human intravenous "
     "clearance (CL), steady-state volume of distribution (VDss) and fraction unbound in plasma (Fu), on the official "
     "held-out test split of each endpoint. For pKa (basic), the final GraphMPNN omits the ionizable-site readout "
     "(Section D); for pKa (acidic), the full GraphMPNN is used. Results are reported on the official "
     "held-out test split. Every model was trained on the official training split only, with ten random "
     "seeds (42–51); the reported prediction for each compound is the arithmetic mean of the ten seed predictions "
     "(hereafter, the 10-seed ensemble). Variability is reported in two complementary forms: (i) the mean ± sample "
     "standard deviation (SD) of the metric over the ten individually trained seeds, which quantifies training "
     "stochasticity, and (ii) a 95% confidence interval (CI) of the ensemble metric obtained by a nonparametric "
     "bootstrap over test compounds (B = 10 000 resamples with replacement, percentile interval), which quantifies "
     "uncertainty due to the finite test set. Pairwise comparisons between models use a paired bootstrap in which both "
     "models are evaluated on the same resampled compounds; a difference is reported as resolved when the 95% CI of the "
     "paired difference excludes zero.")

para(["pKa endpoints are evaluated in pKa units using the coefficient of determination (R²), mean absolute error (MAE), "
      "root mean squared error (RMSE) and the fraction of compounds predicted within one pKa unit (W1). CL, VDss and Fu are "
      "modeled in log", ("10", "sub"), " space; for these endpoints we report R², MAE and RMSE in log", ("10", "sub"),
      " units, the geometric mean fold error, GMFE = 10^(mean |log", ("10", "sub"), " ŷ − log", ("10", "sub"),
      " y|), and the fraction of compounds predicted within two-fold of the observed value (W2F). Lower is better for "
      "MAE, RMSE and GMFE; higher is better for R², W1 and W2F."])

# ---------------- A. ensemble performance ----------------
doc.add_heading("A. Performance of the Proposed Models", level=2)
pk = m("pKa_Acidic", "R2"), m("pKa_Basic", "R2")
para(f"Tables I and II report the 10-seed ensemble performance of the proposed models, and Fig. 1 shows the "
     f"corresponding parity plots. On pKa, GraphMPNN achieves an ensemble R² of {f(m('pKa_Acidic','R2').ensemble)} "
     f"(95% CI {f(m('pKa_Acidic','R2').ci_lo)}–{f(m('pKa_Acidic','R2').ci_hi)}) and an MAE of "
     f"{f(m('pKa_Acidic','MAE').ensemble)} pKa units on the acidic test set (n = 776), and an R² of "
     f"{f(m('pKa_Basic','R2').ensemble)} ({f(m('pKa_Basic','R2').ci_lo)}–{f(m('pKa_Basic','R2').ci_hi)}) with an MAE of "
     f"{f(m('pKa_Basic','MAE').ensemble)} on the basic test set (n = 815). "
     f"{100*m('pKa_Acidic','within_1_pKa_unit').ensemble:.1f}% and {100*m('pKa_Basic','within_1_pKa_unit').ensemble:.1f}% "
     f"of acidic and basic test compounds, respectively, are predicted within one pKa unit.")
para(f"For the pharmacokinetic endpoints, the MFMN models reach an ensemble GMFE of {f(m('CL','GMFE').ensemble)} "
     f"(95% CI {f(m('CL','GMFE').ci_lo)}–{f(m('CL','GMFE').ci_hi)}) for CL, {f(m('VDss','GMFE').ensemble)} "
     f"({f(m('VDss','GMFE').ci_lo)}–{f(m('VDss','GMFE').ci_hi)}) for VDss and {f(m('Fu','GMFE').ensemble)} "
     f"({f(m('Fu','GMFE').ci_lo)}–{f(m('Fu','GMFE').ci_hi)}) for Fu, with W2F of "
     f"{f(m('CL','within_2fold').ensemble)}, {f(m('VDss','within_2fold').ensemble)} and "
     f"{f(m('Fu','within_2fold').ensemble)}, respectively. The CL and VDss test sets each contain 177 compounds and the "
     f"Fu test set 633 compounds; the CIs for CL and VDss are correspondingly wider (e.g., CL R² "
     f"{f(m('CL','R2').ci_lo)}–{f(m('CL','R2').ci_hi)}).")
para(f"Across all five endpoints, the 10-seed ensemble outperforms the average individual seed on every metric. For "
     f"example, CL R² is {f(m('CL','R2').ensemble)} for the ensemble versus {f(m('CL','R2').seed_mean)} ± "
     f"{f(m('CL','R2').seed_sd)} for individual seeds, and pKa (acidic) MAE is {f(m('pKa_Acidic','MAE').ensemble)} versus "
     f"{f(m('pKa_Acidic','MAE').seed_mean)} ± {f(m('pKa_Acidic','MAE').seed_sd)}. The seed-to-seed SD of R² is at most "
     f"{f(main[main.metric=='R2'].seed_sd.max())} across endpoints, with the largest values for VDss and CL.")

caption("Table I\n", "Test-set performance of GraphMPNN on pKa (10-seed ensemble)")
rows = []
for t in ["pKa_Acidic", "pKa_Basic"]:
    for metric in ["R2", "MAE", "RMSE", "within_1_pKa_unit"]:
        r = m(t, metric)
        rows.append([f"{TITLE[t]} (n = {r.n_test})" if metric == "R2" else "", MLAB[metric],
                     f"{f(r.ensemble)} [{f(r.ci_lo)}, {f(r.ci_hi)}]", f"{f(r.seed_mean)} ± {f(r.seed_sd)}"])
table(["Endpoint", "Metric", "Ensemble [95% CI]", "Seed mean ± SD"], rows, [1.5, 0.7, 1.7, 1.3],
      notes=["Ensemble: mean prediction of 10 seeds; CI: compound-level bootstrap, B = 10 000. Seed mean ± SD over the "
             "10 individually trained seeds. MAE and RMSE in pKa units; W1: fraction within 1 pKa unit. pKa (basic): "
             "GraphMPNN without the ionizable-site readout."])

caption("Table II\n", "Test-set performance of the MFMN models on CL, VDss and Fu (10-seed ensemble)")
rows = []
for t in ["CL", "VDss", "Fu"]:
    for metric in ["R2", "MAE", "RMSE", "GMFE", "within_2fold"]:
        r = m(t, metric)
        rows.append([f"{TITLE[t]} – {MODEL[t]} (n = {r.n_test})" if metric == "R2" else "", MLAB[metric],
                     f"{f(r.ensemble)} [{f(r.ci_lo)}, {f(r.ci_hi)}]", f"{f(r.seed_mean)} ± {f(r.seed_sd)}"])
table(["Endpoint – model", "Metric", "Ensemble [95% CI]", "Seed mean ± SD"], rows, [2.1, 0.6, 1.5, 1.2],
      notes=["MAE and RMSE in log10 units; GMFE: geometric mean fold error; W2F: fraction within two-fold. "
             "LW: 1.25× training-loss weight for compounds with fu < 0.006. CI and seed statistics as in Table I."])

doc.add_picture(str(D / "fig1_parity.png"), width=Inches(6.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
fig_caption("Fig. 1. ", "Predicted versus observed values on the official test sets for the proposed models (10-seed "
            "ensemble). The solid line is the identity; the shaded band marks ±1 pKa unit (pKa) or two-fold error "
            "(CL, VDss, Fu). Inset metrics are ensemble values from Tables I and II.")

# ---------------- B. vs published benchmark ----------------
doc.add_heading("B. Comparison With the Published Benchmark", level=2)
n_better = int(paper.ensemble_better.sum())
n_tot = len(paper)
outside = paper[~paper.paper_inside_ci]
para(["Table III compares the 10-seed ensemble with the published benchmark values on the same official test splits: "
      "Jia et al. ", ("[REF: Jia et al., J. Med. Chem., 2025, 68(7), 7737–7750]", "needs"),
      " for CL, VDss and Fu, and ", ("[NEEDS: citation for the pKa benchmark]", "needs"),
      " for pKa. For most of the test set the benchmark reports aggregate values only, so Table III reports whether the "
      "published value lies inside the bootstrap 95% CI of the ensemble metric, and Fig. 2 shows the relative "
      "differences. The benchmark does, however, publish its own per-compound predictions for a 106-compound subset; "
      "Table IV compares the two models on those compounds with a paired test, which is the stronger comparison."])
pa = paper.set_index(["target", "metric"])
para(f"The ensemble improves on the published value in {n_better} of the {n_tot} metric–endpoint pairs for which a "
     f"benchmark value is available. On both pKa endpoints, all three metrics improve and the published value lies "
     f"outside the ensemble CI: MAE decreases from {f(pa.loc[('pKa_Acidic','MAE'),'paper'],2)} to "
     f"{f(pa.loc[('pKa_Acidic','MAE'),'ensemble'])} (acidic) and from {f(pa.loc[('pKa_Basic','MAE'),'paper'],2)} to "
     f"{f(pa.loc[('pKa_Basic','MAE'),'ensemble'])} (basic). For VDss, all five metrics improve (GMFE "
     f"{f(pa.loc[('VDss','GMFE'),'ensemble'])} vs. {f(pa.loc[('VDss','GMFE'),'paper'],2)}; W2F "
     f"{f(pa.loc[('VDss','within_2fold'),'ensemble'])} vs. {f(pa.loc[('VDss','within_2fold'),'paper'],2)}); of these, "
     f"only the MAE improvement places the published value outside the ensemble CI.")
para(f"For CL, the ensemble improves on R², MAE and RMSE (R² {f(pa.loc[('CL','R2'),'ensemble'])} vs. "
     f"{f(pa.loc[('CL','R2'),'paper'],2)}) but trails on GMFE ({f(pa.loc[('CL','GMFE'),'ensemble'])} vs. "
     f"{f(pa.loc[('CL','GMFE'),'paper'],2)}) and W2F ({f(pa.loc[('CL','within_2fold'),'ensemble'])} vs. "
     f"{f(pa.loc[('CL','within_2fold'),'paper'],2)}). For Fu, the ensemble improves on R² "
     f"({f(pa.loc[('Fu','R2'),'ensemble'])} vs. {f(pa.loc[('Fu','R2'),'paper'],2)}) and W2F "
     f"({f(pa.loc[('Fu','within_2fold'),'ensemble'])} vs. {f(pa.loc[('Fu','within_2fold'),'paper'],2)}) and trails on "
     f"MAE, RMSE and GMFE ({f(pa.loc[('Fu','GMFE'),'ensemble'])} vs. {f(pa.loc[('Fu','GMFE'),'paper'],2)}). For all ten "
     f"CL and Fu metrics, the published value lies inside the ensemble 95% CI.")
cl_seed = paper[(paper.target == "CL")].seed_mean_better.sum()
fu_seed = paper[(paper.target == "Fu")].seed_mean_better.sum()
para(f"The comparison depends on whether the ensemble or an individual model is considered. Using the seed-mean "
     f"metric instead of the ensemble, the proposed models still improve on all pKa and VDss benchmark values, but on "
     f"{int(cl_seed)} of 5 CL metrics and {int(fu_seed)} of 5 Fu metrics (e.g., CL seed-mean R² "
     f"{f(pa.loc[('CL','R2'),'seed_mean'])} vs. {f(pa.loc[('CL','R2'),'paper'],2)}; Fu seed-mean R² "
     f"{f(pa.loc[('Fu','R2'),'seed_mean'])} vs. {f(pa.loc[('Fu','R2'),'paper'],2)}).")

caption("Table III\n", "Comparison of the 10-seed ensemble with the published benchmark (official test splits)")
rows, bolds = [], set()
for t in ["pKa_Acidic", "pKa_Basic", "CL", "VDss", "Fu"]:
    sub = paper[paper.target == t]
    for i, (_, r) in enumerate(sub.iterrows()):
        better = r.ensemble_better
        rows.append([TITLE[t] if i == 0 else "", MLAB[r.metric], f"{r.paper:.2f}",
                     f"{f(r.ensemble)} [{f(r.ci_lo)}, {f(r.ci_hi)}]", f"{r.delta:+.3f}".replace("-", "−"),
                     "yes" if r.paper_inside_ci else "no", f"{f(r.seed_mean)}"])
        bolds.add((len(rows) - 1, 3 if better else 2))
table(["Endpoint", "Metric", "Published", "Ensemble [95% CI]", "Δ (ens. − pub.)", "Pub. in CI", "Seed mean"],
      rows, [0.95, 0.55, 0.7, 1.45, 0.85, 0.65, 0.7], bold_cells=bolds,
      notes=["Bold marks the better value of Published vs. Ensemble. Published values: CL/VDss/Fu from Jia et al. "
             "[REF]; pKa from [NEEDS: citation]. No published W1/GMFE/W2F values are used for pKa. "
             "Published values were transcribed from the source; re-verify before submission."])

doc.add_picture(str(D / "fig2_vs_paper.png"), width=Inches(3.4))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
fig_caption("Fig. 2. ", "Relative difference between the 10-seed ensemble and the published benchmark for each "
            "endpoint and metric, oriented so that positive values favor the proposed model. Dots: ensemble; bars: "
            "bootstrap 95% CI of the ensemble metric expressed relative to the published value. The vertical line "
            "marks parity with the benchmark.")

# ---------------- B2. paired comparison on the published-prediction subset ----------------
doc.add_heading("C. Paired Comparison on the Benchmark's Published Predictions", level=2)


def pv(t, metric):
    return pair[(pair.target == t) & (pair.metric == metric)].iloc[0]


n_res = int((pair.verdict != "CI includes 0").sum())
n_ours = int(pair.verdict.str.startswith("ours").sum())
para(f"The benchmark publishes per-compound predictions for all five endpoints on a {int(pair.n_test.iloc[0])}-compound "
     f"subset of the test data. Every one of these compounds lies in the official test split of every endpoint, so "
     f"neither model was trained on them, and both were evaluated on identical compounds. This permits a paired "
     f"comparison (Table IV, Fig. 3): the same compound-level bootstrap as above, resampling the same compounds for "
     f"both models, which removes the compound-to-compound variation that dominates the marginal CIs of Table III. "
     f"This subset is the only part of the test data for which the benchmark's own predictions are available.")
para(f"Of the {len(pair)} metric–endpoint pairs, {n_res} are resolved, and all {n_ours} favour the proposed models; "
     f"none favours the benchmark. Both pKa endpoints improve on R², MAE and RMSE with paired CIs excluding zero: on "
     f"pKa (acidic), MAE falls from {f(pv('pKa_Acidic','MAE').jia)} to {f(pv('pKa_Acidic','MAE').ours)} "
     f"(difference {f(pv('pKa_Acidic','MAE').diff_ours_minus_jia)}, 95% CI "
     f"[{f(pv('pKa_Acidic','MAE').diff_ci_lo)}, {f(pv('pKa_Acidic','MAE').diff_ci_hi)}]), and on pKa (basic) from "
     f"{f(pv('pKa_Basic','MAE').jia)} to {f(pv('pKa_Basic','MAE').ours)} "
     f"(CI [{f(pv('pKa_Basic','MAE').diff_ci_lo)}, {f(pv('pKa_Basic','MAE').diff_ci_hi)}]), a reduction of "
     f"{100*(1-pv('pKa_Basic','MAE').ours/pv('pKa_Basic','MAE').jia):.0f}%. The W1 improvement is also resolved for "
     f"pKa (basic) ({f(pv('pKa_Basic','within_1_pKa_unit').ours)} vs. "
     f"{f(pv('pKa_Basic','within_1_pKa_unit').jia)}).")
para(f"For CL, VDss and Fu no difference is resolved on this subset. The proposed models hold the better point "
     f"estimate on every VDss and Fu metric (VDss GMFE {f(pv('VDss','GMFE').ours)} vs. {f(pv('VDss','GMFE').jia)}; Fu "
     f"GMFE {f(pv('Fu','GMFE').ours)} vs. {f(pv('Fu','GMFE').jia)}) and the two models are indistinguishable on CL "
     f"(GMFE {f(pv('CL','GMFE').ours)} vs. {f(pv('CL','GMFE').jia)}). With {int(pair.n_test.iloc[0])} compounds this "
     f"subset is smaller than the full test splits of Tables I and II, so it resolves only the larger differences; the "
     f"absence of a resolved difference is not evidence of equivalence.")

caption("Table IV\n", "Paired comparison with the benchmark's published per-compound predictions")
rows, bolds = [], set()
for t in ["pKa_Acidic", "pKa_Basic", "CL", "VDss", "Fu"]:
    sub = pair[pair.target == t]
    for i, (_, r) in enumerate(sub.iterrows()):
        better = (r.ours > r.jia) if HIGHER[r.metric] else (r.ours < r.jia)
        mark = "†" if r.verdict.startswith("ours") else ("‡" if r.verdict.startswith("jia") else "")
        rows.append([f"{TITLE[t]} (n = {int(r.n_test)})" if i == 0 else "", MLAB[r.metric], f(r.jia), f(r.ours),
                     f"{r.diff_ours_minus_jia:+.3f}".replace("-", "\u2212") + mark,
                     f"[{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}]".replace("-", "\u2212")])
        bolds.add((len(rows) - 1, 3 if better else 2))
table(["Endpoint", "Metric", "Benchmark", "This work", "Difference", "95% CI of difference"],
      rows, [1.0, 0.6, 0.85, 0.85, 0.9, 1.3], bold_cells=bolds,
      notes=["Both models evaluated on the same compounds, the subset for which the benchmark publishes per-compound "
             "predictions; all of them lie in the official test split of every endpoint. Difference = this work − "
             "benchmark, oriented by the metric. CI: paired compound-level bootstrap, B = 10 000, the same resampled "
             "compounds for both models. †: CI excludes zero in favour of this work; ‡: in favour of the benchmark "
             "(unused — no metric favours the benchmark); no mark: CI includes zero. Bold: better value. MAE and RMSE "
             "in pKa units (pKa) or log10 units (CL, VDss, Fu)."])

doc.add_picture(str(D / "fig3_paired_vs_jia.png"), width=Inches(6.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
fig_caption("Fig. 3. ", "Per-compound absolute error of the benchmark's published predictions (horizontal) against "
            "this work's (vertical) on the same compounds. Points below the identity line are compounds this work "
            "predicts more accurately; the panel titles give the count. pKa in pKa units, CL, VDss and Fu in log10 "
            "units.")

# ---------------- D. vs untuned baselines ----------------
doc.add_heading("D. Comparison With Untuned Classical Baselines", level=2)
para("Table V compares the proposed models with three classical regressors trained on the same training split and "
     "evaluated on the same test compounds in the same order: Random Forest (RF), XGBoost (XGB) and support vector "
     "regression (SVM), each with library-default hyperparameters (no tuning). RF was trained with seeds 42–51 and is "
     "reported as a 10-seed ensemble. With default settings, XGB uses no row or column subsampling, so all ten seeds "
     "produced identical predictions and its ensemble equals a single fit; SVM is deterministic and was fit once. Fig. 4 "
     "summarizes the headline error metric per endpoint.")


def bv(t, mod, metric):
    return base[(base.target == t) & (base.model == mod) & (base.metric == metric)].iloc[0]


para(f"On both pKa endpoints and on Fu, the proposed model has the better value on every metric against every "
     f"baseline, and every paired-difference CI excludes zero. The strongest pKa baseline by MAE is RF on the acidic set "
     f"({f(bv('pKa_Acidic','Random Forest','MAE').value)} vs. {f(m('pKa_Acidic','MAE').ensemble)} for GraphMPNN) and "
     f"XGB on the basic set ({f(bv('pKa_Basic','XGBoost','MAE').value)} vs. {f(m('pKa_Basic','MAE').ensemble)}). On Fu, "
     f"the strongest baseline is SVM (GMFE {f(bv('Fu','SVM','GMFE').value)} vs. {f(m('Fu','GMFE').ensemble)}; R² "
     f"{f(bv('Fu','SVM','R2').value)} vs. {f(m('Fu','R2').ensemble)}).")
para(f"On VDss, the proposed model has the better point estimate on every metric against all three baselines; the "
     f"paired CIs exclude zero against XGB and SVM on all metrics, but include zero against RF on all metrics (GMFE "
     f"{f(m('VDss','GMFE').ensemble)} vs. {f(bv('VDss','Random Forest','GMFE').value)}, paired-difference CI "
     f"[{f(bv('VDss','Random Forest','GMFE').diff_ci_lo)}, {f(bv('VDss','Random Forest','GMFE').diff_ci_hi)}]). On CL, "
     f"the proposed model is better than XGB on all metrics with CIs excluding zero, but is not resolved from RF or SVM "
     f"on any metric. RF has the better point estimate on CL for MAE ({f(bv('CL','Random Forest','MAE').value)} vs. "
     f"{f(m('CL','MAE').ensemble)}), GMFE ({f(bv('CL','Random Forest','GMFE').value)} vs. {f(m('CL','GMFE').ensemble)}) "
     f"and W2F ({f(bv('CL','Random Forest','within_2fold').value)} vs. {f(m('CL','within_2fold').ensemble)}), and SVM "
     f"has the better point estimate for W2F ({f(bv('CL','SVM','within_2fold').value)}).")

caption("Table V\n", "Proposed models versus untuned classical baselines (official test splits)")
SETS = {"pKa_Acidic": ["R2", "MAE", "RMSE", "within_1_pKa_unit"], "pKa_Basic": ["R2", "MAE", "RMSE", "within_1_pKa_unit"],
        "CL": ["R2", "MAE", "GMFE", "within_2fold"], "VDss": ["R2", "MAE", "GMFE", "within_2fold"],
        "Fu": ["R2", "MAE", "GMFE", "within_2fold"]}
rows, bolds, groups = [], set(), set()
for t in ["pKa_Acidic", "pKa_Basic", "CL", "VDss", "Fu"]:
    mets = SETS[t]
    rows.append([f"{TITLE[t]} (n = {m(t,'R2').n_test})"] + [MLAB[x] for x in mets])
    groups.add(len(rows) - 1)
    vals = {"Proposed": {x: m(t, x).ensemble for x in mets}}
    for mod in ["Random Forest", "XGBoost", "SVM"]:
        vals[mod] = {x: bv(t, mod, x).value for x in mets}
    for mod in ["Proposed", "Random Forest", "XGBoost", "SVM"]:
        cells = []
        for ci_, x in enumerate(mets):
            s = f(vals[mod][x])
            if mod != "Proposed":
                v = bv(t, mod, x).verdict
                s += "†" if v.startswith("proposed") else ("‡" if v.startswith("baseline") else "")
            best = max(vals, key=lambda k: vals[k][x]) if HIGHER[x] else min(vals, key=lambda k: vals[k][x])
            if best == mod:
                bolds.add((len(rows), ci_ + 1))
            cells.append(s)
        rows.append([("  " + MODEL[t]) if mod == "Proposed" else ("  " + mod)] + cells)
hdr = ["Endpoint / model", "Metric values (metric names given in each endpoint row)", "", "", ""]
t4 = table(hdr, rows, [1.75, 1.1, 1.1, 1.1, 1.1], bold_cells=bolds, group_rows=groups,
           notes=["Metric names are given in each italic endpoint row (pKa: R², MAE, RMSE, W1; CL/VDss/Fu: R², MAE, "
                  "GMFE, W2F). Proposed and RF: 10-seed ensembles; XGB: seed-invariant (single fit); SVM: single "
                  "deterministic fit. Bold: best value per endpoint and metric. †: paired-bootstrap 95% CI of the "
                  "difference (proposed − baseline) excludes zero in favor of the proposed model; no mark: CI includes "
                  "zero. No baseline was resolved as better than the proposed model (‡ unused). Full metric set and "
                  "difference CIs: supplementary table [NEEDS: supplementary reference]."])
# merge the header metric cells into one, dropping the empty paragraphs merge() leaves behind
hc = t4.rows[0].cells[1].merge(t4.rows[0].cells[4])
for extra in hc.paragraphs[1:]:
    extra._element.getparent().remove(extra._element)

doc.add_picture(str(D / "fig4_baselines.png"), width=Inches(6.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
fig_caption("Fig. 4. ", "Headline error metric of the proposed models and the untuned baselines on each official "
            "test set (MAE in pKa units for pKa; GMFE for CL, VDss and Fu; lower is better). Error bars: bootstrap "
            "95% CI of each model's metric (B = 10 000). Proposed: GraphMPNN (pKa) or MFMN variant (CL, VDss, Fu).")

# ---------------- E. ablation ----------------
doc.add_heading("E. Ablation Study", level=2)
para("Tables VI and VII report 10-seed ablations retrained under the same protocol, training split and test split as "
     "the final models. For CL and VDss (Table VI), the final model is compared with the base MFMN (no auxiliary "
     "supervision and no dynamic gating) and with the alternative MFMN variant that differs only in the gating "
     "mechanism (MFMN-InteractAux for CL; MFMN-DynGate for VDss); Fig. 5 shows the per-seed GMFE distributions. For "
     "pKa (Table VII), the xTB-derived features, the edge gating and the ionizable-site readout of the full GraphMPNN are "
     "removed one at a time; on pKa (basic), the variant without the site readout performed best and is used as the "
     "final model, so the full GraphMPNN and its other two ablations are compared against it. For Fu, the low-fu training-loss weight is removed. Variant–final differences use the same "
     "paired compound-level bootstrap as Section D.")


def av(t, var, metric):
    return abl[(abl.target == t) & (abl.variant == var) & (abl.metric == metric)].iloc[0]


para(f"On CL, the final MFMN-DynGate model has the best ensemble value on R², MAE, RMSE and GMFE (GMFE "
     f"{f(m('CL','GMFE').ensemble)} vs. {f(av('CL','MFMN-InteractAux','GMFE').ensemble)} for MFMN-InteractAux and "
     f"{f(av('CL','MFMN (base)','GMFE').ensemble)} for the base MFMN) and ties MFMN-InteractAux on W2F "
     f"({f(m('CL','within_2fold').ensemble)}). None of these differences is resolved by the paired bootstrap; all "
     f"paired-difference CIs include zero. Across seeds, the SD of R² decreases from "
     f"{f(av('CL','MFMN (base)','R2').seed_sd)} (base) to {f(av('CL','MFMN-InteractAux','R2').seed_sd)} "
     f"(InteractAux) and {f(m('CL','R2').seed_sd)} (DynGate).")
para(f"On VDss, the final MFMN-InteractAux model improves on the base MFMN on R², MAE, RMSE and GMFE with paired CIs "
     f"excluding zero (GMFE {f(m('VDss','GMFE').ensemble)} vs. {f(av('VDss','MFMN (base)','GMFE').ensemble)}; "
     f"difference CI [{f(av('VDss','MFMN (base)','GMFE').diff_ci_lo)}, {f(av('VDss','MFMN (base)','GMFE').diff_ci_hi)}]); "
     f"the W2F difference is not resolved. MFMN-InteractAux and MFMN-DynGate are not resolved on any metric (GMFE "
     f"{f(m('VDss','GMFE').ensemble)} vs. {f(av('VDss','MFMN-DynGate','GMFE').ensemble)}); MFMN-DynGate has the higher "
     f"ensemble W2F ({f(av('VDss','MFMN-DynGate','within_2fold').ensemble)} vs. {f(m('VDss','within_2fold').ensemble)}) "
     f"and a lower seed-to-seed SD of R² ({f(av('VDss','MFMN-DynGate','R2').seed_sd)} vs. {f(m('VDss','R2').seed_sd)}).")

caption("Table VI\n", "Ablation of the MFMN family on CL and VDss (10 seeds, official test splits)")
rows, bolds = [], set()
for t in ["CL", "VDss"]:
    order = (["MFMN (base)", "MFMN-InteractAux", "final"] if t == "CL" else ["MFMN (base)", "MFMN-DynGate", "final"])
    vals = {}
    for var in order:
        vals[var] = {x: (m(t, x).ensemble if var == "final" else av(t, var, x).ensemble)
                     for x in ["R2", "MAE", "GMFE", "within_2fold"]}
    first = True
    for var in order:
        cells = []
        for ci_, x in enumerate(["R2", "MAE", "GMFE", "within_2fold"]):
            if var == "final":
                sm, sd = m(t, x).seed_mean, m(t, x).seed_sd
                mark = ""
            else:
                r = av(t, var, x)
                sm, sd = r.seed_mean, r.seed_sd
                mark = "†" if r.verdict.startswith("final") else ("‡" if r.verdict.startswith("variant") else "")
            cells.append(f"{f(vals[var][x])}{mark} ({f(sm)} ± {f(sd)})")
            best = max(vals, key=lambda k: vals[k][x]) if HIGHER[x] else min(vals, key=lambda k: vals[k][x])
            if vals[best][x] == vals[var][x]:
                bolds.add((len(rows), ci_ + 2))
        name = f"{MODEL[t]} (final)" if var == "final" else var
        rows.append([t if first else "", name] + cells)
        first = False
table(["Endpoint", "Variant", "R²", "MAE", "GMFE", "W2F"], rows, [0.6, 1.45, 1.1, 1.1, 1.1, 1.1], bold_cells=bolds,
      notes=["Entries: 10-seed ensemble (seed mean ± SD). Bold: best ensemble value per endpoint and metric (ties both "
             "bold). †: paired-bootstrap 95% CI of the difference (final − variant) excludes zero in favor of the final "
             "model; no mark: CI includes zero; no variant was resolved as better than the final model. The final-model "
             "rows are the retrained finals of Table II; the ablation variants were compared against the same test "
             "rows."])

doc.add_picture(str(D / "fig5_ablation.png"), width=Inches(3.4))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
fig_caption("Fig. 5. ", "Per-seed test GMFE (dots, n = 10 per variant) and 10-seed ensemble GMFE (horizontal bar) for "
            "the MFMN ablation on CL and VDss. Lower is better.")

PKA_MET = ["R2", "MAE", "RMSE", "within_1_pKa_unit"]
full, nx, ne = "GraphMPNN (full)", "full, w/o xTB features", "full, w/o edge gating"
para(f"On pKa (acidic), none of the three ablations changes performance measurably: every ensemble metric is within "
     f"{f(abl[abl.target=='pKa_Acidic'].diff_final_minus_variant.abs().max(), 3)} of the final model "
     f"(R² {f(m('pKa_Acidic','R2').ensemble, 4)} for the final model vs. "
     f"{f(abl[(abl.target=='pKa_Acidic')&(abl.metric=='R2')].ensemble.min(), 4)}–"
     f"{f(abl[(abl.target=='pKa_Acidic')&(abl.metric=='R2')].ensemble.max(), 4)} for the variants), and all "
     f"paired-difference CIs include zero. The full GraphMPNN has the lowest MAE ({f(m('pKa_Acidic','MAE').ensemble)}) "
     f"and is retained as the final model; the variants have marginally better point estimates on some other metrics "
     f"(e.g., RMSE {f(av('pKa_Acidic','w/o edge gating','RMSE').ensemble)} without edge gating vs. "
     f"{f(m('pKa_Acidic','RMSE').ensemble)}), by less than one seed-to-seed SD.")
para(f"On pKa (basic), removing the ionizable-site readout gives the best model on every metric: R² "
     f"{f(m('pKa_Basic','R2').ensemble)}, MAE {f(m('pKa_Basic','MAE').ensemble)}, RMSE {f(m('pKa_Basic','RMSE').ensemble)} "
     f"and W1 {f(m('pKa_Basic','within_1_pKa_unit').ensemble)}, compared with {f(av('pKa_Basic',full,'R2').ensemble)}, "
     f"{f(av('pKa_Basic',full,'MAE').ensemble)}, {f(av('pKa_Basic',full,'RMSE').ensemble)} and "
     f"{f(av('pKa_Basic',full,'within_1_pKa_unit').ensemble)} for the full GraphMPNN; it is therefore used as the final "
     f"pKa (basic) model. The improvement over the full model is consistent across seeds (higher R² for 9 of the 10 seeds "
     f"when paired by seed index) but is not resolved by the paired bootstrap (MAE difference CI "
     f"[{f(av('pKa_Basic',full,'MAE').diff_ci_lo, 4)}, {f(av('pKa_Basic',full,'MAE').diff_ci_hi, 4)}]). Relative to the "
     f"final model, removing the xTB-derived features from the full GraphMPNN degrades every metric with all "
     f"paired-difference CIs excluding zero (RMSE {f(av('pKa_Basic',nx,'RMSE').ensemble)} vs. "
     f"{f(m('pKa_Basic','RMSE').ensemble)}), and the full model without edge gating is worse on R² and RMSE with CIs "
     f"excluding zero (RMSE {f(av('pKa_Basic',ne,'RMSE').ensemble)}).")

fu_w = "MFMN-DynGate, no loss weight"
para(f"On Fu, removing the 1.25× low-fu loss weight leaves performance essentially unchanged: GMFE "
     f"{f(av('Fu',fu_w,'GMFE').ensemble)} vs. {f(m('Fu','GMFE').ensemble)}, R² {f(av('Fu',fu_w,'R2').ensemble)} vs. "
     f"{f(m('Fu','R2').ensemble)} and W2F {f(av('Fu',fu_w,'within_2fold').ensemble)} vs. "
     f"{f(m('Fu','within_2fold').ensemble)} for the unweighted and final models, respectively; all paired-difference CIs "
     f"include zero. The final model has the better point estimate on every metric, but the differences are smaller "
     f"than one seed-to-seed SD.")
caption("Table VII\n", "Ablation of GraphMPNN on pKa (10 seeds, official test splits)")
rows, bolds = [], set()
ORDER = {"pKa_Acidic": ["w/o xTB features", "w/o edge gating", "w/o site readout", "final"],
         "pKa_Basic": ["full, w/o xTB features", "full, w/o edge gating", "GraphMPNN (full)", "final"]}
for t in ["pKa_Acidic", "pKa_Basic"]:
    order = ORDER[t]
    vals = {var: {x: (m(t, x).ensemble if var == "final" else av(t, var, x).ensemble) for x in PKA_MET}
            for var in order}
    for i, var in enumerate(order):
        cells = []
        for ci_, x in enumerate(PKA_MET):
            if var == "final":
                sm, sd, mark = m(t, x).seed_mean, m(t, x).seed_sd, ""
            else:
                r = av(t, var, x)
                sm, sd = r.seed_mean, r.seed_sd
                mark = "†" if r.verdict.startswith("final") else ("‡" if r.verdict.startswith("variant") else "")
            cells.append(f"{f(vals[var][x])}{mark} ({f(sm)} ± {f(sd)})")
            best = max(vals, key=lambda k: vals[k][x]) if HIGHER[x] else min(vals, key=lambda k: vals[k][x])
            if vals[best][x] == vals[var][x]:
                bolds.add((len(rows), ci_ + 2))
        name = f"{MODEL[t]} (final)" if var == "final" else var
        rows.append([TITLE[t] if i == 0 else "", name] + cells)
table(["Endpoint", "Variant", "R²", "MAE", "RMSE", "W1"], rows, [0.85, 1.35, 1.08, 1.08, 1.08, 1.08],
      bold_cells=bolds,
      notes=["Entries: 10-seed ensemble (seed mean ± SD); MAE and RMSE in pKa units. Bold: best ensemble value per "
             "endpoint and metric. †: paired-bootstrap 95% CI of the difference (final − variant) excludes zero in "
             "favor of the final model; no mark: CI includes zero; no variant was resolved as better than the final "
             "model. pKa (acidic): final = full GraphMPNN. pKa (basic): final = GraphMPNN without the ionizable-site "
             "readout; 'full' rows are the complete GraphMPNN and its ablations. Final-model rows are those of Table I."])

out = D / "Results_section_draft.docx"
doc.save(out)
print(out)
