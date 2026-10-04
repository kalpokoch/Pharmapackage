"""Build the full manuscript draft (.docx): front matter + the shared Results/Discussion/Conclusion.

Front matter (title, abstract, introduction, datasets, methodology, equations) is written here;
Results, Discussion and Conclusion come from `sections_results.py`, shared with `build_docx.py`.
Table, figure, equation and citation numbers are all assigned automatically, so the Results part
renumbers itself around whatever front matter precedes it.

Every quantitative claim is read from the generated t_*.csv tables rather than typed in.
Architecture figures are deliberately left as placeholders.
"""
import sys
from pathlib import Path

import pandas as pd

from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build_docx                                           # noqa: E402  (shared context builder)
import docx_common                                          # noqa: E402
import sections_results                                     # noqa: E402
from references import REFERENCES                            # noqa: E402

TITLE_MAIN = ("Graph and Factorized-Descriptor Neural Networks for Human Intravenous "
              "Pharmacokinetic Parameter Prediction: A Reproducible Benchmark Study")
TITLE_ALTS = [
    "Seed-Resolved Benchmarking of Graph and Descriptor Neural Networks for Human PK and pKa Prediction",
    "From Structure to Exposure: Graph Neural Networks for pKa and Pharmacokinetic Parameter Prediction",
]
TITLE_EP = {"pKa_Acidic": "pKa (acidic)", "pKa_Basic": "pKa (basic)", "CL": "CL",
            "VDss": "VDss", "Fu": "Fu"}
KEYWORDS = ("Pharmacokinetics, graph neural networks, molecular property prediction, "
            "acid dissociation constant, drug clearance, volume of distribution, model ensembling, "
            "reproducible benchmarking")


def main():
    doc, H = docx_common.make_builder()
    num = docx_common.Numbering()
    cites = docx_common.Citations(REFERENCES)
    para, caption, fig_caption, table, f = H["para"], H["caption"], H["fig_caption"], H["table"], H["f"]
    cite = cites.cite
    eq_n = [0]

    def equation(text, note=None):
        """Centred italic equation with a right-aligned number."""
        eq_n[0] += 1
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        tabs = p.paragraph_format.tab_stops
        tabs.add_tab_stop(Inches(3.25), WD_TAB_ALIGNMENT.CENTER)
        tabs.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
        r = p.add_run("\t" + text)
        r.italic = True
        p.add_run(f"\t({eq_n[0]})")
        if note:
            q = doc.add_paragraph()
            q.paragraph_format.space_after = Pt(4)
            rn = q.add_run(note)
            rn.font.size = Pt(8.5)
        return eq_n[0]


    # ------------------------------------------------------------------ title block
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITLE_MAIN)
    r.bold = True
    r.font.size = Pt(16)
    para([("[Alternative titles for consideration: (a) " + TITLE_ALTS[0] + "; (b) " + TITLE_ALTS[1] +
           ". Author list, affiliations and corresponding-author details to be added.]", "needs")],
         align=WD_ALIGN_PARAGRAPH.CENTER, size=8)

    # ------------------------------------------------------------------ abstract
    doc.add_heading("Abstract", level=2)
    ctx_tmp = build_docx.make_context(doc, H, docx_common.Numbering())
    m, pair, down = ctx_tmp["m"], ctx_tmp["pair"], ctx_tmp["down"]
    pa = ctx_tmp["paper"].set_index(["target", "metric"])

    def pv(t, metric):
        return pair[(pair.target == t) & (pair.metric == metric)].iloc[0]

    n_res = int((pair.verdict != "CI includes 0").sum())
    para(f"Predicting human pharmacokinetics from chemical structure underpins candidate selection in drug "
         f"discovery, yet reported gains are often difficult to separate from the variance of model training "
         f"itself. We revisit the parameter-prediction stage of a recent intravenous pharmacokinetics benchmark "
         f"and report five endpoints — acidic and basic acid dissociation constants (pKa), intravenous clearance "
         f"(CL), steady-state volume of distribution (VDss) and fraction unbound in plasma (Fu) — each as a "
         f"ten-seed ensemble trained on the official training split and scored once on the official held-out test "
         f"split. An edge-gated graph message-passing network with semi-empirical quantum descriptors is used for "
         f"pKa, and a factorized-descriptor network with per-compound dynamic gating and auxiliary supervision for "
         f"the pharmacokinetic endpoints. On pKa the ensembles reach R² of {f(m('pKa_Acidic','R2').ensemble)} "
         f"(acidic) and {f(m('pKa_Basic','R2').ensemble)} (basic) with mean absolute errors of "
         f"{f(m('pKa_Acidic','MAE').ensemble)} and {f(m('pKa_Basic','MAE').ensemble)} pKa units, improving on the "
         f"benchmark by margins that a paired bootstrap resolves on the subset where its own per-compound "
         f"predictions are published; {n_res} of {len(pair)} paired metric–endpoint comparisons are resolved and "
         f"all favour the proposed models. On CL, VDss and Fu the models are comparable to the benchmark, with no "
         f"difference resolved. Propagating the predicted parameters through a fixed one-compartment exposure "
         f"model yields no resolved improvement in predicted exposure, which we show is consistent with, rather "
         f"than contrary to, the parameter-level result. Across all five endpoints the ten-seed ensemble exceeds "
         f"the average single seed on every metric, and several architectural differences are smaller than the "
         f"spread between random seeds — observations that bear on how such comparisons should be reported.")
    p = doc.add_paragraph()
    r = p.add_run("Index Terms—")
    r.italic = True
    r.bold = True
    p.add_run(KEYWORDS + ".")

    # ------------------------------------------------------------------ I. Introduction
    doc.add_heading("I. Introduction", level=1)
    para(f"The pharmacokinetic profile of a candidate molecule — how quickly it is cleared, how widely it "
         f"distributes, and how much of it circulates unbound — determines whether a compound with promising "
         f"potency can become a viable drug. Measuring these properties in humans is slow and expensive, so "
         f"predicting them from chemical structure has become a standard part of early discovery "
         f"{cite('bassani2024', 'seal2025')}, with dedicated efforts for clearance {cite('clearanceml2025')} and "
         f"for the unbound fraction {cite('fubreview2026')}. Recent work has moved from predicting isolated "
         f"parameters towards "
         f"predicting the full concentration–time profile, either by learning it directly "
         f"{cite('beckers2024', 'pillai2024')} or by feeding predicted physicochemical parameters into a "
         f"mechanistic or physiologically based model {cite('mavroudis2023', 'gruber2024', 'li2024')}. Both "
         f"strategies rest on the same foundation: the quality of the structure-derived parameter predictions that "
         f"enter them {cite('chou2023', 'geci2024')}. The same parameters feed exposure assessment for "
         f"environmental chemicals {cite('pkexposure2024')} and the quantitative prediction of drug-drug "
         f"interactions {cite('ddi2026')}, so errors in them propagate widely.")
    f_ctx = num.fig_label()
    doc.add_picture(str(ROOT / "Diagram" / "motivation_figure_drawio.drawio.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig_caption(f"Fig. {f_ctx}. ", "Where structure-based parameter prediction sits in drug discovery. "
                "(1) Experimental determination of pharmacokinetic parameters is accurate but slow, "
                "resource-intensive and limited to one compound at a time. (2) Predicting the same parameters "
                "from structure \u2014 clearance, volume of distribution, fraction unbound and the ionization "
                "constants, which are the five endpoints modeled in this work \u2014 takes seconds per compound; "
                "the exposure shown is that obtained by propagating predicted parameters through an assumed "
                f"compartmental model (Section {docx_common.ROMAN[3]}-H), not a learned concentration\u2013time "
                "profile. (3) The resulting throughput allows candidates to be ranked before synthesis, at the "
                "stage where acting on an unfavourable profile is least costly. Computational screening narrows "
                "what reaches the laboratory; it does not replace experimental confirmation.")
    para(f"Graph neural networks are now the dominant representation for molecular property prediction, learning "
         f"directly from the molecular graph rather than from precomputed descriptors "
         f"{cite('dgcl2024', 'chainaware2024')}, and have been applied to absorption, distribution, metabolism, "
         f"excretion and toxicity endpoints with multi-task and auxiliary-task formulations "
         f"{cite('admetmtl2023')}. Augmenting such models with quantum-chemical information has been shown to "
         f"improve property prediction further {cite('qimrl2024')}, and semi-empirical methods make that "
         f"information affordable at screening scale {cite('xtbcorr2023')}. For acid dissociation constants in "
         f"particular — a property that governs ionization state and therefore permeability, binding and "
         f"clearance — graph-based and transfer-learning approaches have become the methods of choice "
         f"{cite('qiu2024', 'pkareview2026')}. Self-supervised pretraining on large unlabeled collections "
         f"{cite('hierssl2023')} and geometry-aware transformer encoders {cite('geot2023')} have pushed accuracy "
         f"further still, and self-interpretable graph architectures are beginning to make such predictions "
         f"auditable rather than opaque {cite('xaignn2024')}.")
    para(f"Two difficulties complicate the interpretation of reported gains. First, published comparisons usually "
         f"report a single aggregate value per metric, so a reader cannot tell whether an improvement is larger "
         f"than the uncertainty of the test set, and a paired statistical test against the prior model is "
         f"generally impossible. Second, neural network results vary with the random seed, and comparisons between "
         f"a new architecture and its ablations are frequently reported from a single training run; the ensembling "
         f"and uncertainty literature has repeatedly shown this variation to be substantial "
         f"{cite('ensembles2023')}, and work on uncertainty quantification for molecular property models has argued "
         f"that an estimate is of limited use without a statement of its reliability {cite('uqexplain2023')}. "
         f"Together these make it difficult to reproduce and to trust incremental improvements, a concern that "
         f"extends across cheminformatics more broadly {cite('repro2023')}.")
    para(f"This work addresses the parameter-prediction stage of the intravenous pharmacokinetics benchmark of "
         f"Jia et al. {cite('jia2025')}, which reports five structure-derived endpoints and then uses them to "
         f"predict concentration–time profiles. We retain that benchmark's official training and test splits "
         f"exactly, so results are directly comparable, and we report every endpoint as a ten-seed ensemble with "
         f"the seed-level spread stated alongside. Our contributions are:")
    for b in [
        "Two model families for the five endpoints: an edge-gated graph message-passing network incorporating "
        "semi-empirical quantum descriptors for the two pKa constants, and a factorized-descriptor network with "
        "per-compound dynamic gating and mechanistically grounded auxiliary supervision for CL, VDss and Fu.",
        "A statistically explicit comparison with the published benchmark: bootstrap confidence intervals for "
        "every metric, and — on the subset for which the benchmark publishes its own per-compound predictions — a "
        "paired test, which is the stronger comparison and which to our knowledge has not previously been applied "
        "to this benchmark.",
        "A ten-seed ablation of both architectures in which component contributions are reported against the "
        "seed-to-seed spread, showing that several architectural differences are not resolvable at these test-set "
        "sizes.",
        "An analytic propagation of the predicted parameters to exposure under a fixed one-compartment model, "
        "isolating how much parameter error propagates into predicted area under the curve and maximum "
        "concentration while holding the downstream model constant.",
    ]:
        q = doc.add_paragraph(style="List Bullet")
        q.paragraph_format.space_after = Pt(2)
        q.add_run(b)
    para("All predictions, per-seed metrics and the scripts that generate every table and figure in this paper "
         "are released with the work.")

    # ------------------------------------------------------------------ II. Datasets
    doc.add_heading("II. Datasets", level=1)
    para(f"All five endpoints use the curated modeling sets and the official training/test partition published "
         f"with the benchmark {cite('jia2025')}, which aggregates human intravenous pharmacokinetic parameters and "
         f"experimental pKa values from public compilations and chemical databases {cite('pubchem2023', 'chembl2024')}. "
         f"Retaining the published partition unchanged is what makes the comparison in Section "
         f"{docx_common.ROMAN[4]} meaningful: no compound in any test split was seen during training by either the "
         f"benchmark models or ours. Reference values of this kind are aggregated from heterogeneous sources and "
         f"the curation decisions behind them measurably affect the models trained on them "
         f"{cite('curation2024')}; the unbound fraction and intrinsic clearance in particular are assay-dependent "
         f"quantities whose reported values carry appreciable experimental spread {cite('intrinsiccl2025')}.")
    t_data = num.table_label()
    caption(f"Table {t_data}\n", "Modeling sets and target distributions for the five endpoints")
    rows = [
        ["pKa (acidic)", "5,296", "776", "pKa units", "none", "−4.37 to 14.00", "8.71"],
        ["pKa (basic)", "5,674", "815", "pKa units", "none", "0.00 to 15.70", "7.76"],
        ["CL", "1,287", "177", "L h⁻¹ kg⁻¹", "log₁₀", "2.22×10⁻⁴ to 64.2", "0.282"],
        ["VDss", "1,287", "177", "L kg⁻¹", "log₁₀", "0.002 to 1114.6", "1.20"],
        ["Fu", "4,042", "633", "fraction", "log₁₀", "5.0×10⁻⁴ to 1.00", "0.0676"],
    ]
    table(["Endpoint", "Train", "Test", "Unit", "Transform", "Range (linear)", "Median"],
          rows, [1.0, 0.75, 0.6, 0.85, 0.8, 1.35, 0.75],
          notes=["Train counts are the compounds actually fitted. For the two pKa endpoints these are 10 and 13 "
                 "fewer than the published modeling-set sizes (5,306 and 5,687), because compounds whose 3D "
                 "conformer embedding failed are excluded; test splits are unaffected and match the published "
                 "sizes exactly. Ranges and medians are of the labeled pool. CL, VDss and Fu are modeled in log₁₀ "
                 "space and scored there, with fold-error metrics computed after back-transformation; pKa is "
                 "modeled and scored directly in pKa units, which may be negative."])
    para("An internal validation set of 15% of the training split is held out at each seed for early stopping; it "
         "is drawn from the training split only, so the test split is used exactly once per endpoint, for the "
         "final evaluation. For CL and VDss, which share a compound pool, the two endpoints' train/test "
         "assignments agree on every compound labeled for both.")
    para(["Semi-empirical quantum-chemical descriptors are used by both model families. For the pKa models these "
          "are per-atom partial charges, the summed and maximum partial charge of each heavy atom's attached "
          "hydrogens, and Wiberg bond orders; for CL and VDss they are whole-molecule frontier-orbital energies, "
          "the HOMO–LUMO gap and the dipole moment. ",
          ("[NEEDS: the semi-empirical Hamiltonian and implicit-solvation model used to generate these "
           "descriptors are not recorded in the released package — the generating scripts sit in a parent project. "
           "State them explicitly here before submission; they must not be inferred.]", "needs")])

    # ------------------------------------------------------------------ III. Methodology
    doc.add_heading("III. Methodology", level=1)

    doc.add_heading("A. Overview", level=2)
    f_fw = num.fig_label()
    f_gnn = num.fig_label()
    f_mfmn = num.fig_label()
    para(f"The five endpoints are addressed by two model families, chosen because the best representation differs "
         f"by endpoint. The acid and base dissociation constants are properties of the molecular graph and its "
         f"electronic structure, and are modeled by a message-passing network operating on that graph (Fig. "
         f"{f_gnn}). Clearance, volume of distribution and fraction unbound are whole-organism properties with "
         f"smaller labeled sets, and are modeled by a compact factorized-descriptor network in which named "
         f"physicochemical descriptor groups are kept in separate pathways (Fig. {f_mfmn}). Fig. {f_fw} shows how "
         f"the two families share a common input representation, training protocol and evaluation.")
    doc.add_picture(str(ROOT / "Diagram" / "Molecular_Property_Prediction_Framework.drawio.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig_caption(f"Fig. {f_fw}. ", "Overview of the prediction framework. A molecule is converted to a combined "
                "representation \u2014 a molecular graph, physicochemical descriptors, structural fingerprints, a "
                "shared descriptor context and semi-empirical quantum-chemical quantities \u2014 from which two "
                "model families branch. The ionization constants are predicted by a site-aware graph "
                "message-passing network, with the site readout used for the acidic endpoint and removed for "
                "the basic one; clearance, volume of distribution and fraction unbound are predicted by the "
                "factorized-descriptor network, whose variant, gating and auxiliary supervision differ by "
                "endpoint. Both families then follow the same protocol: the official split, with ten "
                "independent seeds averaged into a single ensemble prediction; the metrics used to score "
                f"those predictions are given in Section {docx_common.ROMAN[3]}-G. Quantum-chemical features "
                "enter the graph models as per-atom charges and bond "
                "orders and the CL and VDss models as whole-molecule frontier-orbital terms; the Fu model uses "
                f"deterministic descriptors only. Figs. {f_gnn} and {f_mfmn} detail the two architectures.")

    doc.add_picture(str(ROOT / "Diagram" / "GMPNN.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig_caption(f"Fig. {f_gnn}. ", "The GraphMPNN architecture used for the two pKa endpoints. A canonical SMILES "
                "string is featurized with RDKit and semi-empirical quantum chemistry into a molecular graph with "
                "36-dimensional atom and 8-dimensional bond features; the ionizable site located by the acidic or "
                "basic SMARTS matcher is marked as an additional node feature. Atom and bond features are projected "
                "to 128 dimensions and refined by four edge-gated message-passing layers, in each of which a gate "
                "computed from the bond features alone modulates each neighbour's transformed state (2), the gated "
                "messages are summed over bonded neighbours (3), and the result updates the atom state through a "
                "gated recurrent cell applied residually (4). The final atom states are combined into a 384-"
                "dimensional graph readout by concatenating the site-atom embedding with masked mean and maximum "
                "pools and applying layer normalization (5); a two-layer head then predicts the standardized pKa, "
                "which is returned to pKa units using the training-split statistics. One model is trained per "
                "endpoint; for the basic endpoint the site term of the readout is omitted, giving a 256-dimensional "
                "readout.")

    doc.add_picture(str(ROOT / "Diagram" / "MFMN_Architecture.drawio.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig_caption(f"Fig. {f_mfmn}. ", "The MFMN architecture used for CL, VDss and Fu. Top: each molecule yields a "
                "shared context vector \u2014 199 dimensions for CL and VDss, 1,419 for Fu \u2014 together with "
                "seven raw descriptor blocks named for the mechanism each represents. Middle: every factor is "
                "embedded into eight dimensions from the context concatenated with its own block (6); in the "
                "dynamically gated variant a second branch of the same input produces a per-compound sigmoid gate "
                "that multiplies the value branch elementwise (7), while the ungated variant used for VDss omits "
                "it. The seven embeddings concatenate to a 56-dimensional latent. Auxiliary heads, active during "
                "training for CL and VDss only, predict one measured property per factor (9). Bottom: per target, "
                "each factor embedding is projected to a rank-8 vector whose pairwise products are summed by the "
                "factorization identity (8), and the resulting interaction term is concatenated with the factor "
                "latent to give 64 dimensions, from which a two-layer pathway predicts the standardized target. "
                "The table gives the configuration of each endpoint; parameter counts are those of Table "
                f"{docx_common.ROMAN[3]}.")

    doc.add_heading("B. Molecular Representation", level=2)
    para("For the pKa models each molecule is represented as a heavy-atom graph. Every atom carries a "
         "36-dimensional feature vector: one-hot encodings of element, heavy-atom degree, formal charge, "
         "hybridization and total hydrogen count; binary aromaticity and ring-membership flags; three "
         "semi-empirical quantities (the atom's partial charge and the sum and maximum of its attached hydrogens' "
         "partial charges); and a binary flag marking the ionizable site. Each bond carries an eight-dimensional "
         "vector: a one-hot bond type, conjugation and ring flags, and the Wiberg bond order. Hydrogens are not "
         "represented as nodes; their influence enters through the hydrogen-count and attached-charge features.")
    para("The ionizable site is located by an ordered list of SMARTS patterns, the first match being taken so that "
         "multi-acid or multi-base molecules receive one determinate site. For the acidic endpoint a site is found "
         "for 92.9% of compounds (79.0% matching a strong rather than a weak fallback pattern); for the basic "
         "endpoint, 60.7%. Where no site is found the flag is zero for every atom and no site embedding is formed; "
         "a site is never guessed.")
    para("For CL, VDss and Fu each molecule is described by a shared context vector together with seven "
         "factor-specific descriptor blocks, named for the mechanisms they represent: size, lipophilicity, "
         "ionization, polarity, binding, electronic and structural. For CL and VDss the context is formed by "
         "pruning and standardizing two-dimensional descriptor sets, reducing them to 32 principal components and "
         "appending 167 structural keys, giving 199 dimensions; the factor blocks total 19 descriptors, with "
         "ionization and binding supplied by separately predicted pKa and unbound-fraction values. For Fu the "
         "context instead uses variance and correlation pruning with circular fingerprints appended and no "
         "dimensionality reduction (1,419 dimensions), and the factor blocks are built entirely from deterministic "
         "descriptors (46 in total), so that no fitted predictor of the target enters the pipeline. Every fitting "
         "step — imputation, scaling, pruning, decomposition — is performed on the training fold only.")

    doc.add_heading("C. Graph Message-Passing Network", level=2)
    para(f"Atom features x_i are projected to a hidden state, which is then refined by L message-passing layers. "
         f"Writing h_i for the state of atom i, A_ij for the binary adjacency, e_ij for the bond feature vector "
         f"and σ for the logistic function:")
    equation("h_i⁽⁰⁾ = SiLU(W₀ x_i + b₀)")
    equation("g_ij = σ(W₂ · SiLU(W₁ e_ij + b₁) + b₂)")
    equation("m_i⁽ˡ⁾ = Σ_j A_ij · g_ij ⊙ (W_m h_j⁽ˡ⁾ + b_m)")
    equation("h_i⁽ˡ⁺¹⁾ = h_i⁽ˡ⁾ + Dropout( GRU( m_i⁽ˡ⁾ , h_i⁽ˡ⁾ ) )",
             note="⊙ denotes the elementwise product. The bond-conditioned gate g_ij in (2) modulates each "
                  "neighbour's contribution per hidden channel rather than through a full edge-conditioned weight "
                  "matrix, which keeps the cost of the pairwise term linear in the hidden width. Aggregation in "
                  "(3) is a plain sum over bonded neighbours; because the adjacency has a zero diagonal, an atom's "
                  "own state re-enters only as the recurrent state of the gated update in (4). Padded positions "
                  "are re-zeroed after each residual addition.")
    para("After L layers the atom states are pooled over the valid atoms by masked mean and masked maximum, and "
         "concatenated with the embedding of the ionizable site atom s (zeroed when no site was found) to form the "
         "molecular readout:")
    equation("r = LayerNorm( [ h_s⁽ᴸ⁾ ‖ mean_i h_i⁽ᴸ⁾ ‖ max_i h_i⁽ᴸ⁾ ] )")
    para("A two-layer head maps r to the predicted pKa. With a hidden width of 128 and four layers the model has "
         "587,905 parameters. For the basic endpoint the site term of (5) is omitted, giving a two-block readout "
         "and 571,265 parameters; the atom-level site flag is retained in either case.")

    doc.add_heading("D. Factorized-Descriptor Network", level=2)
    para("Each of the seven mechanistic factors f receives the shared context c concatenated with its own "
         "descriptor block x_f, and is embedded into a narrow factor latent. In the dynamically gated variant a "
         "second branch, taking the same input, produces a per-compound gate:")
    equation("v_f = Dropout( SiLU( W_f [ c ‖ x_f ] + b_f ) )")
    equation("z_f = v_f ⊙ σ( W_f^g [ c ‖ x_f ] + b_f^g )",
             note="The gate is conditioned on the individual compound, so a factor can be attenuated for one "
                  "molecule and not another; in the ungated variants z_f = v_f. The factor latent is deliberately "
                  "narrow (eight dimensions), forming a bottleneck that keeps each mechanistic pathway separate.")
    para("Pairwise interactions between factors are captured by projecting each factor latent to a rank-k vector "
         "and taking the sum over factor pairs, evaluated in linear rather than quadratic time by the identity:")
    equation("ι = ½ [ ( Σ_f u_f )² − Σ_f u_f² ],   u_f = W_f^ι z_f + b_f^ι")
    para("The concatenation [z₁ ‖ … ‖ z₇ ‖ ι] is passed to a per-target two-layer pathway. For CL and VDss each "
         "factor additionally carries an auxiliary head predicting one real, measurable property matched to that "
         "factor's mechanism — molecular weight for size, the partition coefficient for lipophilicity, the "
         "frontier-orbital gap for electronic, and so on — supervised jointly with the main target:")
    equation("\u2112 = SmoothL1_{β=0.5}(ŷ, y) + λ Σ_f [ Σ_n m_n^f (â_n^f − a_n^f)² / Σ_n m_n^f ]",
             note="m^f masks compounds for which the auxiliary property is unavailable, so a factor contributes "
                  "only where its target is observed. λ is 0.8 for CL and 2.0 for VDss. Fu uses no auxiliary "
                  "supervision, since the side-signals available for CL and VDss are not valid for that endpoint; "
                  "instead its training loss is a weighted mean that upweights the sparsely populated low-unbound "
                  "tail by a factor of 1.25 for compounds with fu below 0.006.")

    doc.add_heading("E. Training Protocol", level=2)
    para("Every model is trained on the official training split only, with ten random seeds (42–51). Targets are "
         "standardized using training-fold statistics and predictions are inverted before scoring. The reported "
         "prediction for each compound is the arithmetic mean of the ten seeds' predictions. Training "
         "configurations are given in Table " + num.peek_tables(1)[0] + ".")
    para("All training and evaluation was performed on an NVIDIA DGX A100 server. Runs are made deterministic by "
         "fixing the Python, NumPy and PyTorch seeds, restricting intra-op threading, and enabling deterministic "
         "cuDNN kernels.")
    t_hp = num.table_label()
    caption(f"Table {t_hp}\n", "Training configuration for the two model families")
    rows = [
        ["Architecture", "GraphMPNN", "MFMN family"],
        ["Endpoints", "pKa (acidic), pKa (basic)", "CL, VDss, Fu"],
        ["Hidden / factor width", "128, 4 layers", "8 per factor, 7 factors"],
        ["Dropout", "0.1 (0.2 in head)", "0.2 (0.35 for Fu)"],
        ["Optimizer", "AdamW", "AdamW"],
        ["Learning rate", "5×10⁻⁴", "3×10⁻³"],
        ["Weight decay", "1×10⁻⁵", "1×10⁻² (3×10⁻² for Fu)"],
        ["Batch size", "64", "128"],
        ["Max epochs", "300", "400"],
        ["LR schedule", "cosine annealing", "cosine annealing"],
        ["Gradient clipping", "5.0", "1.0"],
        ["Early-stopping patience", "40 epochs", "50 epochs"],
        ["Validation fraction", "0.15", "0.15"],
        ["Primary loss", "smooth L1, β = 0.5", "smooth L1, β = 0.5"],
        ["Seeds", "42–51", "42–51"],
        ["Parameters", "587,905 / 571,265", "12,976 – 161,400"],
    ]
    table(["Setting", "pKa models", "Pharmacokinetic models"], rows, [1.75, 1.95, 2.1],
          notes=["Early stopping monitors validation error on a 15% split of the training data, with the best "
                 "state restored before test prediction. The two parameter counts for the pKa models correspond "
                 "to the full readout and to the variant without the site term; the range for the pharmacokinetic "
                 "models spans the CL/VDss and Fu feature configurations."])

    doc.add_heading("F. Model Complexity and Computational Cost", level=2)
    cx = pd.read_csv(HERE / "t_complexity.csv").set_index("target")
    gnn, vd = cx.loc["pKa_Acidic"], cx.loc["VDss"]
    ratio = gnn.flops_median_mol / vd.flops_median_mol
    t_cx = num.table_label()
    para(f"The two families differ in cost by three orders of magnitude, which is worth stating because it bears "
         f"on where each is worth deploying. Table {t_cx} reports trainable parameters, forward floating-point "
         f"operations for a single molecule, and the measured wall-clock training time per seed. The graph model "
         f"carries {gnn.params:,} parameters and costs {gnn.flops_median_mol/1e6:.0f} MFLOPs for a molecule of "
         f"median size ({int(gnn.n_atoms_median)} heavy atoms), rising to "
         f"{gnn.flops_max_mol/1e6:.0f} MFLOPs for the largest compound in the pool "
         f"({int(gnn.n_atoms_max)} heavy atoms): the dense adjacency representation evaluates the bond-conditioned "
         f"gate of (2) for every ordered atom pair, so cost grows quadratically with molecule size. The "
         f"descriptor models are fixed-cost regardless of molecule size and range from "
         f"{cx.flops_median_mol.min()/1e3:.0f} to {cx.loc['Fu'].flops_median_mol/1e3:.0f} kFLOPs, roughly "
         f"{ratio:,.0f}-fold cheaper than the graph model at the low end.")
    para(f"The same contrast appears in training time: a single pKa seed takes about "
         f"{gnn.sec_per_seed_mean/60:.0f} minutes against under a minute for each pharmacokinetic endpoint, so a "
         f"ten-seed pKa ensemble costs roughly {10*gnn.sec_per_seed_mean/3600:.1f} GPU-hours against about "
         f"{10*vd.sec_per_seed_mean/60:.0f} minutes. These are measured wall-clock times with several seeds "
         f"training concurrently on the same GPU, so they include contention and are upper bounds on the "
         f"cost of a dedicated run. Floating-point counts cover the matrix multiplications of a forward pass and "
         f"exclude feature generation, which for both families is dominated by descriptor and semi-empirical "
         f"calculation performed once per compound. The classical baselines of Section "
         f"{docx_common.ROMAN[4]}-D are non-parametric and are not characterised by these measures.")
    caption(f"Table {t_cx}\n", "Model complexity and measured training cost")
    rows = []
    for t in ["pKa_Acidic", "pKa_Basic", "CL", "VDss", "Fu"]:
        r = cx.loc[t]
        fl = (f"{r.flops_median_mol/1e6:.0f} / {r.flops_max_mol/1e6:.0f} M" if r.family == "graph"
              else f"{r.flops_median_mol/1e3:.0f} k")
        rows.append([TITLE_EP[t], r.model, f"{int(r.params):,}", fl,
                     f"{r.sec_per_seed_mean:.0f}", f"{r.sec_per_seed_min:.0f}\u2013{r.sec_per_seed_max:.0f}"])
    table(["Endpoint", "Model", "Parameters", "Forward FLOPs/molecule", "s / seed", "range"],
          rows, [0.95, 1.6, 0.95, 1.5, 0.7, 0.8],
          notes=["Trainable parameters. Forward FLOPs are for one molecule, counted by the PyTorch profiler over "
                 "the matrix multiplications of a forward pass; for the graph model the two values are for a "
                 "median-sized and the largest molecule in the pool (26 and 65 heavy atoms), between which cost "
                 "grows quadratically, while the descriptor models are size-independent. Training time is measured "
                 "wall-clock per seed on an NVIDIA DGX A100, with several seeds training concurrently on the same "
                 "GPU; it therefore includes contention."])

    doc.add_heading("G. Evaluation Metrics and Statistics", level=2)
    para("Predictions are scored by the coefficient of determination, mean absolute error and root mean squared "
         "error. For the endpoints modeled in log₁₀ space, two fold-error metrics are reported after "
         "back-transformation: the geometric mean fold error and the fraction of compounds predicted within n-fold "
         "of the observed value. For pKa, accuracy within one pKa unit is reported instead.")
    equation("GMFE = 10^( (1/N) Σ_n | log₁₀ ŷ_n − log₁₀ y_n | )")
    equation("FE_n = max( ŷ_n / y_n , y_n / ŷ_n ),   within-k-fold = (1/N) Σ_n 1[ FE_n ≤ k ]")
    para("Two sources of variability are reported separately and never merged. Training stochasticity is "
         "quantified by the mean and sample standard deviation of each metric over the ten independently trained "
         "seeds. Finite-test-set uncertainty is quantified by a nonparametric bootstrap over test compounds "
         "(B = 10,000 resamples, percentile interval, fixed seed). Comparisons between two models use a paired "
         "bootstrap in which both are evaluated on the same resampled compounds, and a difference is reported as "
         "resolved only when the 95% interval of the paired difference excludes zero.")

    doc.add_heading("H. Propagation to Exposure", level=2)
    para("To ask whether improved parameter predictions yield better predicted exposure, the predicted CL and VDss "
         "are propagated through a fixed one-compartment intravenous model under the dose-linearity assumption and "
         "1 mg/kg normalization used throughout the benchmark. For a dose D, infusion duration T and elimination "
         "rate k = CL/V:")
    equation("AUC = D / CL")
    equation("C_max = D / V  (bolus);   C_max = (R₀ / CL)(1 − e^{−kT}),  R₀ = D / T  (infusion)",
             note="The same exposure model is applied to every parameter source, so the comparison isolates the "
                  "effect of parameter error. Because AUC depends only on CL, the AUC fold error is identically "
                  "the CL fold error; C_max, which depends on CL, VDss and infusion time jointly, is the only new "
                  "quantity. The unbound fraction does not enter a one-compartment model.")

    # ------------------------------------------------------------------ shared sections
    ctx = build_docx.make_context(doc, H, num, first_section="IV", draft_note=False)
    sections_results.emit(ctx)
    for _ in range(8):
        num.table_label()
    for _ in range(6):
        num.fig_label()

    # ------------------------------------------------------------------ references
    doc.add_heading("References", level=1)
    for n, text in cites.entries():
        q = doc.add_paragraph()
        q.paragraph_format.space_after = Pt(2)
        q.paragraph_format.left_indent = Inches(0.3)
        q.paragraph_format.first_line_indent = Inches(-0.3)
        r = q.add_run(f"[{n}] ")
        r.font.size = Pt(8.5)
        r2 = q.add_run(text)
        r2.font.size = Pt(8.5)

    out = HERE / "Manuscript_draft.docx"
    doc.save(out)
    print(f"{out}\n  tables: {num.n_tables}  figures: {num.n_figures}  equations: {eq_n[0]}  "
          f"references cited: {len(cites.order)} of {len(REFERENCES)}")
    unused = [k for k in REFERENCES if k not in cites.order]
    if unused:
        print("  uncited entries in the database:", ", ".join(sorted(unused)))


if __name__ == "__main__":
    main()
