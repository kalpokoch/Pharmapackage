"""Verified reference database for the manuscript (IEEE numeric style).

Every entry was confirmed against its CrossRef record; `doi` resolves. All entries are dated
2023-2026. Keys are cited from the builders via `cite(key)`, which assigns numbers in order of
first appearance, so this file carries no numbering of its own.
"""

REFERENCES = {
    "admetmtl2023": (
        "B. X. Du, Y. Xu, S. M. Yiu, H. Yu, and J. Y. Shi, "
        "\u201cADMET property prediction via multi-task graph learning under adaptive auxiliary task selection,\u201d "
        "iScience, vol. 26, no. 11, pp. 108285, 2023, doi: 10.1016/j.isci.2023.108285."),
    "chou2023": (
        "W. C. Chou, and Z. Lin, "
        "\u201cMachine learning and artificial intelligence in physiologically based pharmacokinetic modeling,\u201d "
        "Toxicological Sciences, vol. 191, no. 1, pp. 1-14, 2023, doi: 10.1093/toxsci/kfac101."),
    "ensembles2023": (
        "J. Carrete, H. Montes-Campos, R. Wanzenböck, E. Heid, and G. K. H. Madsen, "
        "\u201cDeep ensembles vs committees for uncertainty estimation in neural-network force fields: Comparison and application to active learning,\u201d "
        "The Journal of Chemical Physics, vol. 158, no. 20, 2023, doi: 10.1063/5.0146905."),
    "mavroudis2023": (
        "P. D. Mavroudis, D. Teutonico, A. Abos, and N. Pillai, "
        "\u201cApplication of machine learning in combination with mechanistic modeling to predict plasma exposure of small molecules,\u201d "
        "Frontiers in Systems Biology, vol. 3, 2023, doi: 10.3389/fsysb.2023.1180948."),
    "pubchem2023": (
        "S. Kim, J. Chen, T. Cheng, A. Gindulyte, J. He, S. He, et al., "
        "\u201cPubChem 2023 update,\u201d "
        "Nucleic Acids Research, vol. 51, no. D1, pp. D1373-D1380, 2023, doi: 10.1093/nar/gkac956."),
    "repro2023": (
        "C. T. Hoyt, B. Zdrazil, R. Guha, N. Jeliazkova, K. Martinez-Mayorga, and E. Nittinger, "
        "\u201cImproving reproducibility and reusability in the Journal of Cheminformatics,\u201d "
        "Journal of Cheminformatics, vol. 15, no. 1, 2023, doi: 10.1186/s13321-023-00730-y."),
    "xtbcorr2023": (
        "E. M. Cabaleiro-Lago, B. Fernández, R. Rodríguez-Fernández, J. Rodríguez-Otero, and S. A. Vázquez, "
        "\u201cFunctional group corrections to the GFN2-xTB and PM6 semiempirical methods for noncovalent interactions in alkanes and alkenes,\u201d "
        "The Journal of Chemical Physics, vol. 158, no. 12, 2023, doi: 10.1063/5.0140668."),
    "bassani2024": (
        "D. Bassani, N. J. Parrott, N. Manevski, and J. D. Zhang, "
        "\u201cAnother string to your bow: machine learning prediction of the pharmacokinetic properties of small molecules,\u201d "
        "Expert Opinion on Drug Discovery, vol. 19, no. 6, pp. 683-698, 2024, doi: 10.1080/17460441.2024.2348157."),
    "beckers2024": (
        "M. Beckers, D. Yonchev, S. Desrayaud, G. Gerebtzoff, and R. Rodríguez-Pérez, "
        "\u201cDeepCt: Predicting Pharmacokinetic Concentration–Time Curves and Compartmental Models from Chemical Structure Using Deep Learning,\u201d "
        "Molecular Pharmaceutics, vol. 21, no. 12, pp. 6220-6233, 2024, doi: 10.1021/acs.molpharmaceut.4c00562."),
    "chainaware2024": (
        "H. Wang, A. Zhang, Y. Zhong, J. Tang, K. Zhang, and P. Li, "
        "\u201cChain-aware graph neural networks for molecular property prediction,\u201d "
        "Bioinformatics, vol. 40, no. 10, 2024, doi: 10.1093/bioinformatics/btae574."),
    "chembl2024": (
        "B. Zdrazil, E. Felix, F. Hunter, E. J. Manners, J. Blackshaw, S. Corbett, et al., "
        "\u201cThe ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods,\u201d "
        "Nucleic Acids Research, vol. 52, no. D1, pp. D1180-D1192, 2024, doi: 10.1093/nar/gkad1004."),
    "dgcl2024": (
        "X. Jiang, L. Tan, and Q. Zou, "
        "\u201cDGCL: dual-graph neural networks contrastive learning for molecular property prediction,\u201d "
        "Briefings in Bioinformatics, vol. 25, no. 6, 2024, doi: 10.1093/bib/bbae474."),
    "geci2024": (
        "R. Geci, D. Gadaleta, M. G. de Lomana, R. Ortega-Vallbona, E. Colombo, E. Serrano-Candelas, et al., "
        "\u201cSystematic evaluation of high-throughput PBK modelling strategies for the prediction of intravenous and oral pharmacokinetics in humans,\u201d "
        "Archives of Toxicology, vol. 98, no. 8, pp. 2659-2676, 2024, doi: 10.1007/s00204-024-03764-9."),
    "gruber2024": (
        "A. Gruber, F. Führer, S. Menz, H. Diedam, A. H. Göller, and S. Schneckener, "
        "\u201cPrediction of Human Pharmacokinetics From Chemical Structure: Combining Mechanistic Modeling with Machine Learning,\u201d "
        "Journal of Pharmaceutical Sciences, vol. 113, no. 1, pp. 55-63, 2024, doi: 10.1016/j.xphs.2023.10.035."),
    "li2024": (
        "Y. Li, Z. Wang, Y. Li, J. Du, X. Gao, Y. Li, et al., "
        "\u201cA Combination of Machine Learning and PBPK Modeling Approach for Pharmacokinetics Prediction of Small Molecules in Humans,\u201d "
        "Pharmaceutical Research, vol. 41, no. 7, pp. 1369-1379, 2024, doi: 10.1007/s11095-024-03725-y."),
    "pillai2024": (
        "N. Pillai, A. Abos, D. Teutonico, and P. D. Mavroudis, "
        "\u201cMachine learning framework to predict pharmacokinetic profile of small molecule drugs based on chemical structure,\u201d "
        "Clinical and Translational Science, vol. 17, no. 5, 2024, doi: 10.1111/cts.13824."),
    "qimrl2024": (
        "J. Kim, W. Chang, H. Ji, and I. Joung, "
        "\u201cQuantum-Informed Molecular Representation Learning Enhancing ADMET Property Prediction,\u201d "
        "Journal of Chemical Information and Modeling, vol. 64, no. 13, pp. 5028-5040, 2024, doi: 10.1021/acs.jcim.4c00772."),
    "qiu2024": (
        "Y. Qiu, J. Chen, K. Xie, R. Gu, Z. Qi, and Z. Song, "
        "\u201cGraph transformer based transfer learning for aqueous pK prediction of organic small molecules,\u201d "
        "Chemical Engineering Science, vol. 300, pp. 120559, 2024, doi: 10.1016/j.ces.2024.120559."),
    "jia2025": (
        "X. Jia, D. Teutonico, S. Dhakal, Y. M. Psarellis, A. Abos, H. Zhu, et al., "
        "\u201cApplication of Machine Learning and Mechanistic Modeling to Predict Intravenous Pharmacokinetic Profiles in Humans,\u201d "
        "Journal of Medicinal Chemistry, vol. 68, no. 7, pp. 7737-7750, 2025, doi: 10.1021/acs.jmedchem.5c00340."),
    "seal2025": (
        "S. Seal, M. A. Trapotsi, M. Mahale, V. Subramanian, N. Greene, O. Spjuth, et al., "
        "\u201cPKSmart: an open-source computational model to predict intravenous pharmacokinetics of small molecules,\u201d "
        "Journal of Cheminformatics, vol. 17, no. 1, 2025, doi: 10.1186/s13321-025-01066-5."),
    "pkareview2026": (
        "J. Baikété, A. Malloum, and J. Conradie, "
        "\u201cpKa prediction for small molecules: an overview of experimental, quantum, and machine learning-based approaches,\u201d "
        "Journal of Computer-Aided Molecular Design, vol. 40, no. 1, 2026, doi: 10.1007/s10822-025-00719-9."),
}
