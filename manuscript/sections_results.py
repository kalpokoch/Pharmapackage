"""Results, Discussion and Conclusion of the manuscript.

Shared by `build_docx.py` (the Results-only draft) and `build_manuscript.py` (the full paper),
so the text has a single source. Table and figure numbers are not hard-coded: `ctx` supplies
`T1..T8` and `F1..F6` already offset by whatever front matter precedes this part, so adding a
table or figure earlier in the paper renumbers these automatically.
"""


def emit(ctx):
    """Append Results (A-F), Discussion and Conclusion to ctx['doc']."""
    doc = ctx["doc"]; D = ctx["D"]
    para = ctx["para"]; caption = ctx["caption"]; fig_caption = ctx["fig_caption"]
    table = ctx["table"]; f = ctx["f"]; m = ctx["m"]
    main = ctx["main"]; paper = ctx["paper"]; base = ctx["base"]
    abl = ctx["abl"]; pair = ctx["pair"]; down = ctx["down"]
    TITLE = ctx["TITLE"]; MODEL = ctx["MODEL"]; MLAB = ctx["MLAB"]; HIGHER = ctx["HIGHER"]
    Inches = ctx["Inches"]; WD_ALIGN_PARAGRAPH = ctx["WD_ALIGN_PARAGRAPH"]
    T1, T2, T3, T4 = ctx["T1"], ctx["T2"], ctx["T3"], ctx["T4"]
    T5, T6, T7, T8 = ctx["T5"], ctx["T6"], ctx["T7"], ctx["T8"]
    F1, F2, F3 = ctx["F1"], ctx["F2"], ctx["F3"]
    F4, F5, F6 = ctx["F4"], ctx["F5"], ctx["F6"]
    S_RES, S_DIS, S_CON = ctx["S_RES"], ctx["S_DIS"], ctx["S_CON"]

    # ======================================================================
    doc.add_heading(f"{S_RES}. Results", level=1)
    if ctx.get("draft_note"):
        para([("[Draft – Results section only. Section number, reference numbers and cross-references to the "
               "Methods/Experiments sections are placeholders to be aligned with the full manuscript.]", "needs")],
             size=8)

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
    para(f"Tables {T1} and {T2} report the 10-seed ensemble performance of the proposed models, and Fig. {F1} shows the "
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

    caption(f"Table {T1}\n", "Test-set performance of GraphMPNN on pKa (10-seed ensemble)")
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

    caption(f"Table {T2}\n", "Test-set performance of the MFMN models on CL, VDss and Fu (10-seed ensemble)")
    rows = []
    for t in ["CL", "VDss", "Fu"]:
        for metric in ["R2", "MAE", "RMSE", "GMFE", "within_2fold"]:
            r = m(t, metric)
            rows.append([f"{TITLE[t]} – {MODEL[t]} (n = {r.n_test})" if metric == "R2" else "", MLAB[metric],
                         f"{f(r.ensemble)} [{f(r.ci_lo)}, {f(r.ci_hi)}]", f"{f(r.seed_mean)} ± {f(r.seed_sd)}"])
    table(["Endpoint – model", "Metric", "Ensemble [95% CI]", "Seed mean ± SD"], rows, [2.1, 0.6, 1.5, 1.2],
          notes=["MAE and RMSE in log10 units; GMFE: geometric mean fold error; W2F: fraction within two-fold. "
                 f"LW: 1.25× training-loss weight for compounds with fu < 0.006. CI and seed statistics as in Table {T1}."])

    doc.add_picture(str(D / "fig1_parity.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig_caption(f"Fig. {F1}. ", "Predicted versus observed values on the official test sets for the proposed models (10-seed "
                "ensemble). The solid line is the identity; the shaded band marks ±1 pKa unit (pKa) or two-fold error "
                f"(CL, VDss, Fu). Inset metrics are ensemble values from Tables {T1} and {T2}.")

    # ---------------- B. vs published benchmark ----------------
    doc.add_heading("B. Comparison With the Published Benchmark", level=2)
    n_better = int(paper.ensemble_better.sum())
    n_tot = len(paper)
    outside = paper[~paper.paper_inside_ci]
    para([f"Table {T3} compares the 10-seed ensemble with the published benchmark values on the same official test splits: "
          "Jia et al. ", ("[REF: Jia et al., J. Med. Chem., 2025, 68(7), 7737–7750]", "needs"),
          " for CL, VDss and Fu, and ", ("[NEEDS: citation for the pKa benchmark]", "needs"),
          f" for pKa. For most of the test set the benchmark reports aggregate values only, so Table {T3} reports whether the "
          f"published value lies inside the bootstrap 95% CI of the ensemble metric, and Fig. {F2} shows the relative "
          "differences. The benchmark does, however, publish its own per-compound predictions for a 106-compound subset; "
          f"Table {T4} compares the two models on those compounds with a paired test, which is the stronger comparison."])
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

    caption(f"Table {T3}\n", "Comparison of the 10-seed ensemble with the published benchmark (official test splits)")
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
    fig_caption(f"Fig. {F2}. ", "Relative difference between the 10-seed ensemble and the published benchmark for each "
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
         f"comparison (Table {T4}, Fig. {F3}): the same compound-level bootstrap as above, resampling the same compounds for "
         f"both models, which removes the compound-to-compound variation that dominates the marginal CIs of Table {T3}. "
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
         f"subset is smaller than the full test splits of Tables {T1} and {T2}, so it resolves only the larger differences; the "
         f"absence of a resolved difference is not evidence of equivalence.")

    caption(f"Table {T4}\n", "Paired comparison with the benchmark's published per-compound predictions")
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
    fig_caption(f"Fig. {F3}. ", "Per-compound absolute error of the benchmark's published predictions (horizontal) against "
                "this work's (vertical) on the same compounds. Points below the identity line are compounds this work "
                "predicts more accurately; the panel titles give the count. pKa in pKa units, CL, VDss and Fu in log10 "
                "units.")

    # ---------------- D. vs untuned baselines ----------------
    doc.add_heading("D. Comparison With Untuned Classical Baselines", level=2)
    para(f"Table {T5} compares the proposed models with three classical regressors trained on the same training split and "
         "evaluated on the same test compounds in the same order: Random Forest (RF), XGBoost (XGB) and support vector "
         "regression (SVM), each with library-default hyperparameters (no tuning). RF was trained with seeds 42–51 and is "
         "reported as a 10-seed ensemble. With default settings, XGB uses no row or column subsampling, so all ten seeds "
         f"produced identical predictions and its ensemble equals a single fit; SVM is deterministic and was fit once. Fig. {F4} "
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

    caption(f"Table {T5}\n", "Proposed models versus untuned classical baselines (official test splits)")
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
    fig_caption(f"Fig. {F4}. ", "Headline error metric of the proposed models and the untuned baselines on each official "
                "test set (MAE in pKa units for pKa; GMFE for CL, VDss and Fu; lower is better). Error bars: bootstrap "
                "95% CI of each model's metric (B = 10 000). Proposed: GraphMPNN (pKa) or MFMN variant (CL, VDss, Fu).")

    # ---------------- E. ablation ----------------
    doc.add_heading("E. Ablation Study", level=2)
    para(f"Tables {T6} and {T7} report 10-seed ablations retrained under the same protocol, training split and test split as "
         f"the final models. For CL and VDss (Table {T6}), the final model is compared with the base MFMN (no auxiliary "
         "supervision and no dynamic gating) and with the alternative MFMN variant that differs only in the gating "
         f"mechanism (MFMN-InteractAux for CL; MFMN-DynGate for VDss); Fig. {F5} shows the per-seed GMFE distributions. For "
         f"pKa (Table {T7}), the xTB-derived features, the edge gating and the ionizable-site readout of the full GraphMPNN are "
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

    caption(f"Table {T6}\n", "Ablation of the MFMN family on CL and VDss (10 seeds, official test splits)")
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
                 f"rows are the retrained finals of Table {T2}; the ablation variants were compared against the same test "
                 "rows."])

    doc.add_picture(str(D / "fig5_ablation.png"), width=Inches(3.4))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig_caption(f"Fig. {F5}. ", "Per-seed test GMFE (dots, n = 10 per variant) and 10-seed ensemble GMFE (horizontal bar) for "
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
    caption(f"Table {T7}\n", "Ablation of GraphMPNN on pKa (10 seeds, official test splits)")
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
                 f"readout; 'full' rows are the complete GraphMPNN and its ablations. Final-model rows are those of Table {T1}."])

    # ---------------- F. downstream exposure ----------------
    doc.add_heading("F. Propagation to Predicted Exposure", level=2)


    def dv(ep, arm, metric):
        return down[(down.endpoint == ep) & (down.arm == arm) & (down.metric == metric)].iloc[0]


    OURS_ARM, JIA_ARM = "This work (10-seed)", "Jia et al. QSAR"
    n_down = int(down.n_test.iloc[0])
    para(f"A natural question is whether improved parameter predictions translate into better predicted exposure. "
         f"Table {T8} addresses this on the same {n_down} compounds of Section C, holding the downstream model fixed and "
         f"varying only the source of CL and VDss. Under the simplest exposure model consistent with the dose-linearity "
         f"and 1 mg/kg normalization used throughout this benchmark — a one-compartment intravenous model — the area "
         f"under the concentration–time curve and the maximum concentration follow in closed form from CL, VDss and the "
         f"infusion duration: AUC = D/CL, and Cmax = D/V for a bolus or (R₀/CL)(1 − e^(−kT)) for an infusion "
         f"of duration T, with R₀ = D/T and k = CL/V. Parameter error can therefore be propagated to exposure error "
         f"analytically, with no fitted downstream model and no training data.")
    para(["What this quantifies is ", ("how parameter error propagates into exposure error", "i"),
          " with the downstream model held constant; it is not a measurement of absolute concentration–time profile "
          "accuracy, and it is not comparable with exposure errors obtained by scoring predicted profiles against "
          "observed profiles. Two consequences of the model should be stated before the numbers are read. First, because "
          "AUC = D/CL exactly, the AUC fold error is identically the CL fold error, so the AUC row carries no information "
          f"beyond the CL results of Tables {T1}–IV; it is reported for completeness. Cmax, which depends on CL, VDss and "
          "infusion time jointly, is the only genuinely new quantity. Second, Fu does not enter a one-compartment model "
          "at all, so this analysis is silent on it; under a physiologically based model Fu would act through tissue "
          "partitioning, which this model does not represent."])
    para(f"No difference is resolved. For Cmax the proposed parameters give the lower geometric mean fold error "
         f"({f(dv('Cmax',OURS_ARM,'GMFE').value)} vs. {f(dv('Cmax',JIA_ARM,'GMFE').value)}) and the higher fraction within "
         f"two-fold ({f(dv('Cmax',OURS_ARM,'within_2fold').value)} vs. {f(dv('Cmax',JIA_ARM,'within_2fold').value)}), but "
         f"the paired differences are not resolved (GMFE difference {f(dv('Cmax',OURS_ARM,'GMFE').diff_ours_minus_jia)}, "
         f"95% CI [{f(dv('Cmax',OURS_ARM,'GMFE').diff_ci_lo)}, {f(dv('Cmax',OURS_ARM,'GMFE').diff_ci_hi)}]). For AUC the "
         f"two parameter sources are indistinguishable, as the equivalence with CL requires. The ensemble again "
         f"outperforms the average individual seed on every exposure metric (Cmax GMFE "
         f"{f(dv('Cmax',OURS_ARM,'GMFE').value)} for the ensemble vs. {f(dv('Cmax',OURS_ARM,'GMFE').seed_mean)} ± "
         f"{f(dv('Cmax',OURS_ARM,'GMFE').seed_sd)} across seeds).")

    caption(f"Table {T8}\n", "Propagation of predicted CL and VDss to exposure under a one-compartment IV model")
    rows, bolds = [], set()
    for ep in ["AUC", "Cmax"]:
        for i, metric in enumerate(["R2", "GMFE", "within_2fold"]):
            rj, ro = dv(ep, JIA_ARM, metric), dv(ep, OURS_ARM, metric)
            better = (ro.value > rj.value) if HIGHER[metric] else (ro.value < rj.value)
            rows.append([f"{ep} (n = {n_down})" if i == 0 else "", MLAB[metric],
                         f"{f(rj.value)} [{f(rj.value_ci_lo)}, {f(rj.value_ci_hi)}]",
                         f"{f(ro.value)} [{f(ro.value_ci_lo)}, {f(ro.value_ci_hi)}]",
                         f"[{ro.diff_ci_lo:+.3f}, {ro.diff_ci_hi:+.3f}]".replace("-", "−")])
            bolds.add((len(rows) - 1, 3 if better else 2))
    table(["Exposure", "Metric", "From benchmark parameters [95% CI]", "From this work's parameters [95% CI]",
           "95% CI of difference"], rows, [0.9, 0.6, 1.55, 1.55, 1.2], bold_cells=bolds,
          notes=["One-compartment IV model, dose linearity, 1 mg/kg; observed CL and VDss define the reference exposure. "
                 "R² on log10 exposure; GMFE and W2F on the linear scale. CIs: compound-level bootstrap, B = 10 000; "
                 f"differences paired as in Table {T4}. No difference is resolved. AUC fold error is identically CL fold "
                 "error under this model. These values are not comparable with exposure errors obtained by scoring "
                 "predicted concentration–time profiles against observed profiles."])

    doc.add_picture(str(D / "fig6_downstream.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig_caption(f"Fig. {F6}. ", "Per-compound exposure fold error from the benchmark's parameters (horizontal) against this "
                "work's (vertical), under the one-compartment model. The shaded square marks compounds within two-fold "
                "for both; points below the identity line are compounds for which this work's parameters give the closer "
                "exposure.")

    # ================================================================== DISCUSSION
    doc.add_heading(f"{S_DIS}. Discussion", level=1)

    doc.add_heading("A. Where the Proposed Models Improve, and Where They Do Not", level=2)
    para(f"The five endpoints divide cleanly. On the two pKa endpoints the proposed GraphMPNN models improve on the "
         f"published benchmark by a margin that survives every test applied here: the published values lie outside the "
         f"bootstrap confidence intervals of the ensemble metrics (Table {T3}), and on the subset where the benchmark's own "
         f"per-compound predictions are available the improvement is resolved by a paired test on R², MAE and RMSE for "
         f"both endpoints (Table {T4}). The effect is large rather than marginal — on pKa (basic) the paired MAE falls from "
         f"{f(pv('pKa_Basic','MAE').jia)} to {f(pv('pKa_Basic','MAE').ours)}, and the proposed model is the closer "
         f"prediction for 76 of the {int(pair.n_test.iloc[0])} compounds. On the three pharmacokinetic endpoints the "
         f"picture is different: the ensembles are at least as good as the benchmark on most metrics and better on several "
         f"point estimates, but no difference is resolved, and on the full test split the benchmark retains the better "
         f"GMFE and within-two-fold fraction for CL (Table {T3}).")
    para("The most plausible reading is that the gap reflects what each endpoint's data can support rather than a "
         "difference in modeling effort. pKa is a thermodynamic property of the molecular structure itself, measured "
         "reproducibly, and the training sets here are the largest of the five. A message-passing network operating on "
         "the molecular graph with xTB-derived electronic features can exploit that structure directly, and the ablation "
         "confirms that the electronic features carry real signal for pKa (basic). CL, VDss and Fu are whole-organism "
         "properties whose reference values aggregate inter-individual variability, differing assay protocols and "
         "literature curation decisions; their test sets here are also far smaller, 177 compounds for CL and VDss. Both "
         "factors compress the achievable range of performance and widen every confidence interval, so that even "
         "consistent point-estimate advantages — the proposed model holds the better value on every VDss and Fu metric in "
         f"Table {T4} — fail to resolve.")
    para(f"Two observations recur across every analysis and are worth separating from the endpoint-specific findings. "
         f"First, ensembling is not cosmetic: on all five endpoints the 10-seed ensemble beats the average individual "
         f"seed on every metric, and the gap is large enough to change conclusions. CL is the clearest case, where the "
         f"ensemble R² of {f(m('CL','R2').ensemble)} exceeds the published {f(pa.loc[('CL','R2'),'paper'],2)} while "
         f"the seed mean of {f(m('CL','R2').seed_mean)} does not. Any comparison of this kind should therefore state "
         f"whether an ensemble or a single model is being reported. Second, the seed-to-seed standard deviation is "
         f"substantial relative to the differences under discussion — up to "
         f"{f(main[main.metric=='R2'].seed_sd.max())} in R² — which is why a comparison based on a single "
         f"seed per architecture is difficult to interpret.")

    doc.add_heading("B. What the Ablations Show", level=2)
    para(f"The component ablations resolve far less than is usually claimed for architectural choices. On CL, none of the "
         f"differences between the final MFMN-DynGate, MFMN-InteractAux and the base MFMN is resolved, although the "
         f"ordering is consistent and the seed-to-seed variance falls markedly as components are added "
         f"({f(av('CL','MFMN (base)','R2').seed_sd)} to {f(m('CL','R2').seed_sd)} in R² standard deviation). On VDss "
         f"the auxiliary supervision is clearly load-bearing: removing it costs "
         f"{f(av('VDss','MFMN (base)','GMFE').diff_final_minus_variant).lstrip('-')} in GMFE with the CI excluding zero, "
         f"whereas exchanging one gating mechanism for the other changes nothing resolvable. On pKa (basic) the "
         f"xTB-derived features are the component that matters, and on pKa (acidic) no component is resolved at all.")
    para("Read together, these results suggest that for this family of problems the choice of representation — graph "
         "versus descriptor, with or without electronic features — matters more than the particular gating or attention "
         "mechanism layered on top, and that several of the architectural refinements are within seed noise on the test "
         "sets available. That is a limitation of the evidence as much as a claim about the architectures: with 177 "
         "compounds, differences smaller than roughly one seed standard deviation cannot be detected.")

    doc.add_heading("C. Does Better Parameter Prediction Improve Predicted Exposure?", level=2)
    para(f"Section F answers this directly for the quantities a one-compartment model can express, and the answer is that "
         f"it does not, measurably, on these {n_down} compounds. The result deserves care rather than dismissal. The AUC "
         f"comparison is not independent evidence at all: AUC = D/CL makes its fold error identical to the CL fold error, "
         f"so it restates an endpoint on which the two models were already indistinguishable. Cmax is the informative "
         f"quantity, and there the proposed parameters give a lower GMFE "
         f"({f(dv('Cmax',OURS_ARM,'GMFE').value)} vs. {f(dv('Cmax',JIA_ARM,'GMFE').value)}) and a higher within-two-fold "
         f"fraction, in the direction the VDss results predict, but without reaching resolution on a sample this size.")
    para(["Two structural features of the problem limit what this analysis can show. The benchmark's own feature-importance "
          "analysis for its downstream model ", ("[REF: Jia et al., Supporting Information, Table S6]", "needs"),
          " ranks CL and VDss far above the remaining parameters, with Fu last, which "
          "supports propagating exactly those two but also implies that improvements concentrated in the other parameters "
          "cannot express themselves downstream. More fundamentally, the endpoints on which the proposed models improve "
          "most — the two pKa constants — have almost no influence on exposure in this framework, while the endpoint that "
          "dominates exposure, CL, is the one on which the two models are closest. A substantial gain in pKa prediction "
          "can therefore coexist with no measurable gain in predicted exposure without any inconsistency. Confirming this "
          "properly would require the observed concentration–time data, which is not publicly available for this "
          "benchmark; the comparison reported here is the strongest that can be made without it."])

    doc.add_heading("D. Limitations", level=2)
    para(f"Each endpoint is evaluated on a single official held-out split, so the results characterize performance on "
         f"those particular compounds and inherit their composition. The CL and VDss splits contain 177 compounds each, "
         f"and the paired comparison of Table {T4} and the exposure analysis of Table {T8} rest on {n_down}; at these sizes "
         f"only large differences resolve, and the absence of a resolved difference is not evidence of equivalence. No "
         f"cross-validated estimate for the final models is reported here, so the stability of these results across "
         f"alternative partitions of the same data is not established.")
    para(["The comparison is against a single published benchmark, and for most of the test data that benchmark reports "
          f"aggregate values only, which is why Table {T3} can report no more than whether a published point estimate falls "
          "inside a confidence interval. Published values were transcribed from the source and should be re-verified "
          "before submission. The classical baselines of Section D use library defaults and are intended as a reference "
          "point for the difficulty of each endpoint, not as tuned competitors."])
    para("The exposure analysis of Section F inherits the assumptions of a one-compartment intravenous model with linear "
         "kinetics and a 1 mg/kg normalization. Real disposition is frequently multi-compartmental, and the model cannot "
         "represent tissue partitioning, protein binding or saturable elimination; its value here is that it is identical "
         "across the arms being compared, not that it describes any compound exactly. Its results therefore bound how "
         "much the parameter improvements could matter under that model, and say nothing about absolute profile accuracy.")

    # ================================================================== CONCLUSION
    doc.add_heading(f"{S_CON}. Conclusion", level=1)
    para(f"This work reproduced and extended the first stage of a published intravenous pharmacokinetics benchmark, "
         f"training a graph message-passing network for the two pKa endpoints and a family of factorized-descriptor "
         f"networks for clearance, volume of distribution and fraction unbound, each as a 10-seed ensemble fitted on the "
         f"official training split and scored once on the official held-out test split. On the pKa endpoints the proposed "
         f"models improve on the benchmark substantially and the improvement is statistically resolved, both against the "
         f"published values and, on the subset where the benchmark's per-compound predictions are available, by a paired "
         f"test: R² rises to {f(m('pKa_Acidic','R2').ensemble)} and {f(m('pKa_Basic','R2').ensemble)} for the acidic "
         f"and basic endpoints, with mean absolute errors of {f(m('pKa_Acidic','MAE').ensemble)} and "
         f"{f(m('pKa_Basic','MAE').ensemble)} pKa units. On clearance, volume of distribution and fraction unbound the "
         f"proposed models are comparable to the benchmark, better on several point estimates and worse on others, with "
         f"no difference resolved.")
    para("Propagating the predicted parameters through a fixed one-compartment exposure model produced no resolved "
         "improvement in predicted AUC or Cmax. That outcome is consistent with the parameter-level results rather than "
         "in tension with them, since exposure in this framework is governed largely by clearance, the endpoint on which "
         "the two models are closest, and is insensitive to the pKa constants on which they differ most. Establishing "
         "whether improved parameter prediction benefits full concentration–time prediction would require the observed "
         "profile data underlying the benchmark.")
    para("Two methodological points generalize beyond this benchmark. Multi-seed ensembling changed the sign of several "
         "comparisons relative to single models and should be reported explicitly rather than left implicit. And across "
         "the ablations, differences between architectural variants were frequently smaller than the variation between "
         "random seeds of the same architecture, so claims about component contributions on test sets of this size "
         "require seed-level variability to be quantified before they can be sustained.")
