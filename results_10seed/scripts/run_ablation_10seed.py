#!/usr/bin/env python
"""
run_ablation_10seed.py -- re-train the ablation variants with seeds 42..51 and save every seed's raw
predictions.  Nothing here defines or changes a recipe: the training code of every variant is EXECUTED
VERBATIM from the original notebook / script cells (see SOURCES below and manifest.json).  The only thing
that varies between runs is the seed.

Writes only inside  <project>/ablation_10seed/  (this file lives there).

Modes
  --smoke      seed 42 for every candidate code path, compared with proposed_model_ablation.csv
               (gate: |dR2|<=0.002 and |dMAE|<=0.002) -> smoke/<tag>/ + candidate_selection.json
  (default)    full run of every variant that passed the gate; resumable; completed seeds never rerun
  --dry-run    print the plan, train nothing

Examples
  python ablation_10seed/run_ablation_10seed.py --smoke --workers 8
  nohup python ablation_10seed/run_ablation_10seed.py --workers 12 >> ablation_10seed/run.log 2>&1 &
  python ablation_10seed/run_ablation_10seed.py --targets CL --variants interactaux --seeds 42-44
"""
import os

# --- thread settings are FIXED (independent of the number of workers) so results cannot depend on it.
THREADS = "2"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = THREADS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # required by torch deterministic mode
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation; no numerical effect

import sys
from pathlib import Path as _P
_deps = _P(__file__).resolve().parent / "_deps"          # rdkit wheel installed here with --no-deps (not in the system env)
if _deps.exists():
    sys.path.insert(0, str(_deps))
import argparse, csv, datetime, hashlib, json, logging, platform, random, shutil, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PK_ROOT", HERE.parent))      # /workspace/Pharmacokinetics
OUT = Path(os.environ.get("ABL_OUT", HERE))              # output dir (default: ablation_10seed/)
PKG = ROOT / "mfmn_benchmark_package"
STAGE = Path("/dev/shm/pk_ro")          # read-only copy of package src+data (identical bytes, checked by hash)
ABL_CSV = ROOT / "ablation" / "proposed_model_ablation.csv"
SEED_ROOT = Path(os.environ.get("FINAL_ROOT", ROOT / "seed_analysis_results"))   # where the final-model arrays live

NB_ABL = ROOT / "ablation" / "ablation_official_test_CL_VDss_Fu.ipynb"
NB_FU = PKG / "notebooks" / "05_Fu_MFMN_DynGate.ipynb"
NB_V2 = ROOT / "MFMN_v2.ipynb"
NB_10S = ROOT / "seed_analysis_results" / "10seed_training_and_analysis.ipynb"
PY_UNTUNED = ROOT / "mfmn" / "experiments" / "expv2_aux_targets_final.py"
PY_H7 = ROOT / "mfmn" / "experiments" / "expv2_aux_h7_dyngate.py"

SEEDS_ALL = list(range(42, 52))
TARGETS = ["CL", "VDss", "Fu", "pKa_Acidic", "pKa_Basic"]          # cheapest first
FINAL_FILES = {"CL": "y_test_log.npy", "VDss": "y_test_log.npy", "Fu": "y_test.npy",
               "pKa_Acidic": "y_test.npy", "pKa_Basic": "y_test.npy"}
LOG_TARGETS = {"CL", "VDss", "Fu"}
GATE_DR2, GATE_DMAE = 0.002, 0.002

# ----------------------------------------------------------------------------------------------------------
# Variant registry.  (target, CSV model name) -> slug + ordered candidate code paths ("runners").
# Runnable candidates execute original code verbatim; BLOCKED ones have no code in the project.
# ----------------------------------------------------------------------------------------------------------
_UNUSED_BLOCK_PKA = ("the code that produced this row is not in the project (graph_mpnn.py has no switch for it; "
             "no notebook/script trains it); not reconstructed by instruction")
VARIANTS = {
    ("CL", "MFMN (base)"):               dict(slug="mfmn_base",      runners=["clvd"], cls="MFMN",             use_aux=False),
    ("CL", "MFMN_InteractAux"):          dict(slug="interactaux",    runners=["clvd"], cls="MFMN_InteractAux", use_aux=True),
    ("VDss", "MFMN (base)"):             dict(slug="mfmn_base",      runners=["clvd"], cls="MFMN",             use_aux=False),
    ("VDss", "MFMN_DynGate"):            dict(slug="dyngate",        runners=["clvd"], cls="MFMN_DynGate",     use_aux=True),
    # Fu: original code exists for the unweighted variant (candidate A, then B); both failed the seed-42 gate in this
    # environment (see report).  The 'untuned base' row has no known code: C1 = the only existing script, C2/C3 = RECONSTRUCTED
    # plain-MFMN candidates defined a priori (see _setup / SOURCES).
    ("Fu", "MFMN_DynGate (unweighted)"): dict(slug="dyngate_unweighted", runners=["fuA", "fuB"]),
    ("Fu", "MFMN base (untuned)"):       dict(slug="mfmn_base_untuned",  runners=["fuC1", "fuC2", "fuC3"], reconstructed=True, test_all=True),
}
PKA_MOD = {"no_xtb": "pka_no_xtb", "no_edge_gating": "pka_no_edge_gating", "no_site_readout": "pka_no_site_readout"}
for _t in ("pKa_Acidic", "pKa_Basic"):
    for _n, _s in (("GraphMPNN, no xTB features", "no_xtb"), ("GraphMPNN, no edge gating", "no_edge_gating"),
                   ("GraphMPNN, no site readout", "no_site_readout")):
        VARIANTS[(_t, _n)] = dict(slug=_s, runners=[PKA_MOD[_s]], reconstructed=True)
# controls (not variants): unmodified pKa model and the final Fu recipe through the same harness, to measure drift
CONTROLS = [("pKa_Acidic", "pka_none"), ("pKa_Basic", "pka_none"), ("Fu", "fuB125")]

RUNNER_GROUP = {"final_clvd": "pkg", "final_fu": "pkg", "clvd": "pkg", "fuB": "pkg", "fuB125": "pkg", "fuA": "orig", "fuC1": "orig", "fuC2": "orig", "fuC3": "orig",
                "pka_none": "pka_none", "pka_no_xtb": "pka_no_xtb", "pka_no_edge_gating": "pka_no_edge_gating",
                "pka_no_site_readout": "pka_no_site_readout"}
SOURCES = {
    "clvd": dict(file=str(NB_ABL.relative_to(ROOT)), cells="cell 1 (imports/constants), cell 3 (fit_predict), "
                 "cell 5 (VARIANTS list + loop logic reproduced line for line)", code="src/mfmn.py"),
    "fuA": dict(file="MFMN_v2.ipynb cell 1 + cell 13 (data prep) and mfmn/experiments/expv2_aux_h7_dyngate.py::"
                     "fit_predict_dyngate with RECIPE['Fu'] (no sample weights)", cells="MFMN_v2.ipynb cells 1, 13, 14"),
    "fuB": dict(file=str(NB_FU.relative_to(ROOT)), cells="cells 1,2,3,4; ONLY change: LOWFU_WEIGHT_FN = "
                "make_binary_weight(1.0) (existing multiplier set to 1.0)"),
    "fuC1": dict(file=str(PY_UNTUNED.relative_to(ROOT)), cells="fit_predict(...) defaults (aux_weight=0.3, "
                 "ctx pca32, MFMN_InteractAux) + data prep lines of its __main__ block"),
    "fuC2": dict(file=str(PY_H7.relative_to(ROOT)) + "::fit_predict_dyngate (RECONSTRUCTED)", cells="source taken programmatically with 2 textual "
                 "substitutions (MFMN_DynGate->MFMN, interact_rank kwarg dropped); called with ctx_variant='pca32' + function defaults"),
    "fuC3": dict(file=str(PY_H7.relative_to(ROOT)) + "::fit_predict_dyngate (RECONSTRUCTED)", cells="same substituted source, called with RECIPE['Fu']"),
    "pka_*": dict(file=str(NB_10S.relative_to(ROOT)), cells="cell 1 (imports), cell 9 (fit_predict_gnn, verbatim) + data prep lines of cell 10; "
                  "RECONSTRUCTED modification applied in-process (no file edited): no_edge_gating / no_site_readout derive from the original "
                  "graph_mpnn.py source by textual substitution; no_xtb drops feature columns in to_device_batch"),
}
HASH_FILES = [NB_10S, NB_ABL, NB_FU, NB_V2, PY_UNTUNED, PY_H7, ABL_CSV, Path(__file__).resolve()] + \
    sorted((PKG / "src").glob("*.py")) + sorted((ROOT / "mfmn" / "features").glob("*.py")) + \
    sorted((ROOT / "mfmn" / "models").glob("*.py"))

log = logging.getLogger("ablation")


# ----------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def parse_seeds(s):
    if not s:
        return list(SEEDS_ALL)
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def vdir(target, slug):
    return OUT / target / slug


def stage_shm():
    """Copy package src + data into /dev/shm once (read-only shared data); verify identical bytes."""
    (STAGE / "ablation").mkdir(parents=True, exist_ok=True)
    (STAGE / "mfmn_benchmark_package" / "notebooks").mkdir(parents=True, exist_ok=True)
    for sub in ("src", "data"):
        dst = STAGE / "mfmn_benchmark_package" / sub
        if not dst.exists():
            shutil.copytree(PKG / sub, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    n = 0
    for f in (PKG / "src").glob("*.py"):
        assert sha256(f) == sha256(STAGE / "mfmn_benchmark_package" / "src" / f.name), f"shm copy differs: {f}"
        n += 1
    for f in (PKG / "data").rglob("*"):
        if f.is_file():
            g = STAGE / "mfmn_benchmark_package" / "data" / f.relative_to(PKG / "data")
            assert g.exists() and g.stat().st_size == f.stat().st_size, f"shm copy differs: {f}"
    os.system(f"chmod -R a-w {STAGE}/mfmn_benchmark_package/src {STAGE}/mfmn_benchmark_package/data")
    return n


# ----------------------------------------------------------------------------------------------------------
# worker side (runs in spawned processes; each process only ever loads ONE group of modules)
# ----------------------------------------------------------------------------------------------------------
_NS = {}


def _nb_src(nb_path, idx):
    nb = json.load(open(nb_path))
    c = nb["cells"][idx]
    assert c["cell_type"] == "code", (nb_path, idx)
    return "".join(c["source"])


def _signature_defaults(fn):
    import inspect
    sig = inspect.signature(fn)
    return {k: (v.default if v.default is not inspect.Parameter.empty else "<required>")
            for k, v in sig.parameters.items()}


def _setup(runner):
    """Build (once per process) the namespace for a runner by exec-ing the ORIGINAL cells."""
    if runner in _NS:
        return _NS[runner]
    import numpy as np
    ns = {"__name__": f"orig_{runner}"}
    if runner == "clvd":
        os.chdir(STAGE / "ablation")                      # cell 1 does sys.path.insert('../mfmn_benchmark_package/src')
        for i in (1, 3):
            exec(compile(_nb_src(NB_ABL, i), f"{NB_ABL.name}:cell{i}", "exec"), ns)
        head = _nb_src(NB_ABL, 5).split("rows = []")[0]   # only the VARIANTS = [...] statement
        exec(compile(head, f"{NB_ABL.name}:cell5(head)", "exec"), ns)
        ns["_hp"] = dict(EPOCHS=ns["EPOCHS"], LR=ns["LR"], WD=ns["WD"], PATIENCE=ns["PATIENCE"], BATCH=ns["BATCH"],
                         VAL_FRAC=ns["VAL_FRAC"], AUX_WEIGHT=ns["AUX_WEIGHT"],
                         model_defaults={n: _signature_defaults(c.__init__) for n, c, _ in ns["VARIANTS"]})
    elif runner in ("fuB", "fuB125"):
        os.chdir(STAGE / "mfmn_benchmark_package" / "notebooks")   # cell 1 does sys.path.insert('../src')
        for i in (1, 2, 3, 4):
            exec(compile(_nb_src(NB_FU, i), f"{NB_FU.name}:cell{i}", "exec"), ns)
        mult = 1.0 if runner == "fuB" else 1.25                     # fuB: the one permitted change; fuB125 = control, unchanged
        ns["LOWFU_WEIGHT_FN"] = ns["make_binary_weight"](mult)
        ns["_hp"] = dict(DROPOUT=ns["DROPOUT"], WD=ns["WD"], FACTOR_DIM=ns["FACTOR_DIM"],
                         INTERACT_RANK=ns["INTERACT_RANK"], EPOCHS=ns["EPOCHS"], LR=ns["LR"], PATIENCE=ns["PATIENCE"],
                         BATCH=ns["BATCH"], VAL_FRAC=ns["VAL_FRAC"], lowfu_multiplier=mult, lowfu_thresh=0.006,
                         model="MFMN_DynGate", model_defaults=_signature_defaults(ns["MFMN_DynGate"].__init__))
    elif runner == "fuA":
        os.chdir(ROOT)                                    # notebook cell 1: PROJECT_ROOT = Path.cwd()
        for i in (1, 13):
            exec(compile(_nb_src(NB_V2, i), f"{NB_V2.name}:cell{i}", "exec"), ns)
        ns["_hp"] = dict(RECIPE_Fu=ns["RECIPE"]["Fu"], fit_predict_dyngate_defaults=_signature_defaults(ns["fit_predict_dyngate"]),
                         model="MFMN_DynGate", sample_weights="none")
    elif runner == "fuC1":
        os.chdir(ROOT)
        for p in ("scripts", "mfmn/features", "mfmn/models", "mfmn/experiments"):
            sys.path.insert(0, str(ROOT / p))
        import expv2_aux_targets_final as m
        from aux_descriptor_features import load_aux_pool, TARGET_COL
        ns.update(m=m, load_aux_pool=load_aux_pool, TARGET_COL=TARGET_COL)
        # data prep: lines of the module's __main__ block, target="Fu"
        t = "Fu"
        smi_all, y_all, sub = load_aux_pool(t)
        split_col = TARGET_COL[t][1]
        ns["train_smi"] = sub[sub[split_col] == "train"]["canon_smiles"].values
        ns["test_smi"] = sub[sub[split_col] == "test"]["canon_smiles"].values
        ns["y_train_full"] = sub.set_index("canon_smiles").loc[ns["train_smi"], TARGET_COL[t][0]].values.astype(np.float64)
        ns["y_test"] = sub.set_index("canon_smiles").loc[ns["test_smi"], TARGET_COL[t][0]].values.astype(np.float64)
        ns["_hp"] = dict(fit_predict_defaults={k: (str(v) if callable(v) else v) for k, v in
                                               _signature_defaults(m.fit_predict).items()},
                         model="MFMN_InteractAux (aux-target loss)", AUX_WEIGHT=m.AUX_WEIGHT)
    elif runner == "final_clvd":
        # FINAL CL (MFMN_DynGate, aux 0.8) / VDss (MFMN_InteractAux, aux 2.0): the cells that produced the stored 10-seed arrays
        os.chdir(STAGE / "ablation")
        for i in (1, 3):
            exec(compile(_nb_src(NB_10S, i), f"{NB_10S.name}:cell{i}", "exec"), ns)
        ns["_hp"] = dict(CONFIGS={k: dict(model=v["model_cls"].__name__, aux_weight=v["aux_weight"]) for k, v in ns["CONFIGS"].items()},
                         EPOCHS=ns["EPOCHS"], LR=ns["LR"], WD=ns["WD"], PATIENCE=ns["PATIENCE"], BATCH=ns["BATCH"], VAL_FRAC=ns["VAL_FRAC"])
    elif runner == "final_fu":
        # FINAL Fu (MFMN_DynGate + 1.25x low-fu weight): 10seed notebook cells 1,3,6 + data prep lines of cell 7
        os.chdir(STAGE / "ablation")
        for i in (1, 3, 6):
            exec(compile(_nb_src(NB_10S, i), f"{NB_10S.name}:cell{i}", "exec"), ns)
        t = ns["TARGET_FU"]
        smi_all, y_all, sub = ns["load_aux_pool"](t)
        split_col = ns["TARGET_COL"][t][1]
        ns["train_smi"] = sub[sub[split_col] == "train"]["canon_smiles"].values
        ns["test_smi"] = sub[sub[split_col] == "test"]["canon_smiles"].values
        ns["y_train_full"] = sub.set_index("canon_smiles").loc[ns["train_smi"], ns["TARGET_COL"][t][0]].values.astype(np.float64)
        ns["y_test_fu"] = sub.set_index("canon_smiles").loc[ns["test_smi"], ns["TARGET_COL"][t][0]].values.astype(np.float64)
        ns["_hp"] = dict(DROPOUT=ns["DROPOUT"], WD=ns["WD_FU"], FACTOR_DIM=ns["FACTOR_DIM"], INTERACT_RANK=ns["INTERACT_RANK"],
                         EPOCHS=ns["EPOCHS_FU"], LR=ns["LR_FU"], PATIENCE=ns["PATIENCE_FU"], BATCH=ns["BATCH_FU"], VAL_FRAC=ns["VAL_FRAC_FU"],
                         lowfu_multiplier=1.25, lowfu_thresh=0.006)
    elif runner in ("fuC2", "fuC3"):
        # RECONSTRUCTED plain-MFMN candidates for 'MFMN base (untuned)'.  The training loop is the ORIGINAL
        # fit_predict_dyngate source (expv2_aux_h7_dyngate.py) with exactly two textual substitutions: the model class
        # MFMN_DynGate -> MFMN (plain base, no gating/interaction/aux) and dropping the `interact_rank` kwarg MFMN lacks.
        import inspect, textwrap, importlib
        ns = dict(_setup("fuC1"))
        h7 = importlib.import_module("expv2_aux_h7_dyngate")
        mfmn_mod = importlib.import_module("mfmn")
        src = textwrap.dedent(inspect.getsource(h7.fit_predict_dyngate))
        subs = [("def fit_predict_dyngate(", "def fit_predict_plain_mfmn("),
                ("model = MFMN_DynGate(ctx_tr.shape[1]", "model = MFMN(ctx_tr.shape[1]"),
                ("factor_dim=factor_dim, interact_rank=interact_rank, dropout=dropout", "factor_dim=factor_dim, dropout=dropout")]
        for a, b in subs:
            assert src.count(a) == 1, f"substitution target not unique: {a!r}"
            src = src.replace(a, b)
        h7.__dict__["MFMN"] = mfmn_mod.MFMN
        exec(compile(src, "reconstructed:fit_predict_plain_mfmn", "exec"), h7.__dict__)
        ns["h7"] = h7
        ns["fit_plain"] = h7.fit_predict_plain_mfmn
        ns["_recipe"] = dict(ctx_variant="pca32") if runner == "fuC2" else dict(h7.RECIPE["Fu"])
        ns["_hp"] = dict(RECONSTRUCTED=True, model="MFMN (plain base class)", loop="verbatim fit_predict_dyngate source with 2 textual "
                         "substitutions (MFMN_DynGate->MFMN; interact_rank kwarg dropped)", recipe=ns["_recipe"],
                         fit_predict_defaults=_signature_defaults(h7.fit_predict_plain_mfmn),
                         candidate=("C2: 'untuned' defaults: pca32 context, dropout 0.2, wd 1e-2 (the settings of expv2_aux_targets_final)"
                                    if runner == "fuC2" else "C3: Fu tuned recipe RECIPE['Fu'] (raw_fcfp context, dropout 0.35, wd 3e-2) with plain MFMN"))
    elif runner.startswith("pka_"):
        # pKa component ablations: original fit_predict_gnn (10seed_training_and_analysis.ipynb cell 9) + RECONSTRUCTED switch.
        import inspect, textwrap, importlib
        mod = runner[len("pka_"):]
        os.chdir(STAGE / "ablation")                      # cell 1: sys.path.insert('../mfmn_benchmark_package/src')
        exec(compile(_nb_src(NB_10S, 1), f"{NB_10S.name}:cell1", "exec"), ns)
        exec("import data_loader", ns)
        exec(compile(_nb_src(NB_10S, 9), f"{NB_10S.name}:cell9", "exec"), ns)
        gm = importlib.import_module("graph_mpnn")
        desc = {"none": "control: unmodified model"}
        if mod == "no_edge_gating":
            # message = h_j * adj_mask (gate fixed at 1); module construction unchanged so the random init is identical
            src = textwrap.dedent(inspect.getsource(gm.MPNNLayer.forward))
            subs = [("gate = self.edge_gate(adj_feats)", "gate = None"),
                    ("msg = gate * h_j * adj_mask.unsqueeze(-1)", "msg = h_j * adj_mask.unsqueeze(-1)")]
            for a, b in subs:
                i = src.index(a)                                   # first token of the line is enough (comments follow)
                assert src.count(a) == 1, a
                src = src.replace(a, b)
            loc = {}
            exec(compile(src, "reconstructed:MPNNLayer.forward(no gate)", "exec"), gm.__dict__, loc)
            gm.MPNNLayer.forward = loc["forward"]
            desc = {"no_edge_gating": "MPNNLayer message = msg_linear(h_j) * adj_mask; edge_gate output replaced by 1 "
                                      "(edge_gate module still constructed, unused)"}
        elif mod == "no_site_readout":
            nn_ = ns["nn"]
            src = textwrap.dedent(inspect.getsource(gm.GraphMPNN.encode))
            a, b = "readout = torch.cat([site_embed, mean_pool, max_pool], dim=-1)", "readout = torch.cat([mean_pool, max_pool], dim=-1)"
            assert src.count(a) == 1
            loc = {}
            exec(compile(src.replace(a, b), "reconstructed:GraphMPNN.encode(no site)", "exec"), gm.__dict__, loc)

            class GraphMPNNNoSite(gm.GraphMPNN):
                def __init__(self, *a, **k):
                    super().__init__(*a, **k)
                    self.readout_norm = nn_.LayerNorm(self.hidden_dim * 2)

                def add_finetune_head(self, out_dim=1, hidden=None):
                    hidden = hidden or self.hidden_dim
                    self.finetune_head = nn_.Sequential(nn_.Linear(self.hidden_dim * 2, hidden), nn_.SiLU(), nn_.Dropout(0.2),
                                                        nn_.Linear(hidden, out_dim))
                    return self
            GraphMPNNNoSite.encode = loc["encode"]
            ns["GraphMPNN"] = GraphMPNNNoSite
            desc = {"no_site_readout": "readout = LayerNorm([mean_pool, max_pool]) (2H) instead of [site, mean, max] (3H); finetune head "
                                       "input 2H; the is_acidic_site node-feature flag is kept"}
        elif mod == "no_xtb":
            D, E = ns["NODE_FEATURE_DIM"], ns["EDGE_FEATURE_DIM"]
            keep_n = list(range(D - 4)) + [D - 1]            # drop xTB charge, H-charge sum, H-charge max (cols D-4..D-2)
            orig = ns["to_device_batch"]

            def to_device_batch(graph_list):
                nodes, adj, adj_mask, node_mask, site_idx = orig(graph_list)
                return nodes[..., keep_n].contiguous(), adj[..., :E - 1].contiguous(), adj_mask, node_mask, site_idx
            ns["to_device_batch"] = to_device_batch
            ns["NODE_FEATURE_DIM"], ns["EDGE_FEATURE_DIM"] = D - 3, E - 1   # drop Wiberg bond order (last edge col)
            desc = {"no_xtb": f"node dim {D}->{D - 3} (xTB atom charge, attached-H charge sum/max removed; SMARTS site flag kept), "
                              f"edge dim {E}->{E - 1} (xTB Wiberg bond order removed)"}
        ns["_prep"] = {}
        ns["_hp"] = dict(RECONSTRUCTED=(mod != "none"), modification=desc, HIDDEN_DIM=ns["HIDDEN_DIM"], N_LAYERS=ns["N_LAYERS"],
                         DROPOUT_GNN=ns["DROPOUT_GNN"], EPOCHS_GNN=ns["EPOCHS_GNN"], LR_GNN=ns["LR_GNN"], BATCH_GNN=ns["BATCH_GNN"],
                         PATIENCE_GNN=ns["PATIENCE_GNN"], VAL_FRAC_GNN=ns["VAL_FRAC_GNN"], weight_decay=1e-5, clip_grad_norm=5.0)
    else:
        raise KeyError(runner)
    _NS[runner] = ns
    return ns


def run_task(runner, target, vname, seed, device, determ=True):
    """Train ONE (variant, seed) in this process.  Seeds are set here, inside the call, in addition to the
    seeding the original recipe performs itself (RandomState(seed) for the validation split, torch.manual_seed)."""
    import numpy as np
    import torch
    t0 = time.time()
    if determ:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(int(THREADS))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ns = _setup(runner)
    dev = torch.device(device)
    if runner in ("clvd", "fuB", "fuB125", "final_clvd", "final_fu") or runner.startswith("pka_"):
        ns["DEVICE"] = dev
    elif runner in ("fuA", "fuC2", "fuC3"):
        sys.modules["expv2_aux_h7_dyngate"].DEVICE = dev
    if runner in ("fuC1", "fuC2", "fuC3"):
        ns["m"].DEVICE = dev

    spec = VARIANTS.get((target, vname), {})
    if runner == "clvd":
        SPLIT, YCOL = ns["SPLIT_COL"][target], ns["Y_COL"][target]
        _, _, sub = ns["data_loader"].load_full_pool(target)
        train_ids = sub[sub[SPLIT] == "train"]["row_idx"].values
        test_ids = sub[sub[SPLIT] == "test"]["row_idx"].values
        y_test = sub[sub[SPLIT] == "test"][YCOL].values
        match = [(n, c, a) for n, c, a in ns["VARIANTS"] if n == vname]
        assert len(match) == 1, f"variant {vname!r} not in cell-5 VARIANTS"
        _, model_cls, use_aux = match[0]
        assert model_cls.__name__ == spec["cls"] and use_aux == spec["use_aux"]
        pred = ns["fit_predict"](model_cls, target, train_ids, test_ids, seed,
                                 aux_weight=ns["AUX_WEIGHT"][target] if use_aux else None)
    elif runner == "final_clvd":
        cfg = ns["CONFIGS"][target]
        _, _, sub = ns["data_loader"].load_full_pool(target)
        train_ids = sub[sub[cfg["split_col"]] == "train"]["row_idx"].values
        test_ids = sub[sub[cfg["split_col"]] == "test"]["row_idx"].values
        y_test = sub[sub[cfg["split_col"]] == "test"][cfg["y_col"]].values
        ids = test_ids
        pred = ns["fit_predict_cl_vdss"](cfg["model_cls"], cfg["aux_weight"], train_ids, test_ids, target, seed)
    elif runner == "final_fu":
        pred = ns["fit_predict_fu"](ns["train_smi"], ns["test_smi"], ns["y_train_full"], seed)
        y_test = ns["y_test_fu"]
        ids = ns["test_smi"]
    elif runner in ("fuB", "fuB125"):
        pred = ns["fit_predict_dyngate"](ns["train_smi"], ns["test_smi"], ns["y_train_full"], seed=seed, verbose=False)
        y_test = ns["y_test"]
    elif runner in ("fuC2", "fuC3"):
        pred = ns["fit_plain"](ns["train_smi"], ns["test_smi"], ns["y_train_full"], seed=seed, **ns["_recipe"])
        y_test = ns["y_test"]
    elif runner.startswith("pka_"):
        if target not in ns["_prep"]:                     # data prep: lines of cell 10 of 10seed_training_and_analysis.ipynb
            graph_ok, val_col, split_col = ns["data_loader"].load_pka_official_split(target)
            graphs, keep_rows = [], []
            for i, row in graph_ok.iterrows():
                g = ns["mol_to_graph"](row["canon_smiles"], row, site_finder=ns["SITE_FINDER"][target])
                if g is not None:
                    graphs.append(g)
                    keep_rows.append(i)
            graph_ok = graph_ok.iloc[keep_rows].reset_index(drop=True)
            y_all = graph_ok[val_col].values.astype(np.float64)
            train_pos = np.where(graph_ok[split_col].values == "train")[0]
            test_pos = np.where(graph_ok[split_col].values == "test")[0]
            ns["_prep"][target] = (graphs, y_all, train_pos, test_pos)
            ns.setdefault("_ids", {})[target] = graph_ok["canon_smiles"].values[test_pos]
        graphs, y_all, train_pos, test_pos = ns["_prep"][target]
        pred = ns["fit_predict_gnn"](graphs, y_all, train_pos, test_pos, seed)
        y_test = y_all[test_pos]
        ids = ns["_ids"][target]
    elif runner == "fuA":
        pred = ns["fit_predict_dyngate"](ns["train_smi"], ns["test_smi"], ns["y_train"], seed=seed, **ns["RECIPE"]["Fu"])
        y_test = ns["y_test_Fu"]
    elif runner == "fuC1":
        pred = ns["m"].fit_predict(ns["train_smi"], ns["test_smi"], ns["y_train_full"], seed=seed)
        y_test = ns["y_test"]
    pred = np.asarray(pred, dtype=np.float64)
    ids = locals().get("ids", None)
    peak = torch.cuda.max_memory_reserved() / 1e9 if dev.type == "cuda" else 0.0
    if dev.type == "cuda":
        torch.cuda.empty_cache()                      # an idle worker must not hold GPU memory other workers need
    return dict(pred=pred, y_test=np.asarray(y_test, dtype=np.float64), seconds=time.time() - t0,
                device=(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"), pid=os.getpid(), ids=(None if ids is None else np.asarray(ids).astype(str)),
                hparams=ns["_hp"], seed=seed, runner=runner, target=target, vname=vname,
                peak_gpu_gb=peak)


# ----------------------------------------------------------------------------------------------------------
# main-process side
# ----------------------------------------------------------------------------------------------------------
def metric_fn(target):
    sys.path.insert(0, str(PKG / "src"))
    from metrics import eval_log_preds, eval_preds
    if target in ("CL", "VDss"):
        return eval_log_preds
    if target == "Fu":
        return lambda y, p: eval_preds(y, p, True)
    return lambda y, p: eval_preds(y, p, False)


COLS = {"log": ["R2", "MAE", "RMSE", "GMFE", "within_2fold", "within_3fold", "within_5fold"],
        "pka": ["R2", "MAE", "RMSE", "within_1_pKa_unit", "within_2_pKa_unit"]}


def metric_cols(target):
    return COLS["log"] if target in LOG_TARGETS else COLS["pka"]


def write_metrics(d, target, label, seeds, preds, y):
    """per_seed_metrics.csv + cumulative_ensemble_metrics.csv from the stored predictions (rows ordered by seed)."""
    import numpy as np
    import pandas as pd
    f = metric_fn(target)
    cols = metric_cols(target)
    rows = []
    for i, (s, p) in enumerate(zip(seeds, preds)):
        m = f(y, p)
        rows.append(dict(target=target, seed_idx=s - 42, seed=s, **{c: m[c] for c in cols}, variant=label))
    pd.DataFrame(rows).to_csv(d / "per_seed_metrics.csv", index=False)
    rows = []
    for k in range(1, len(preds) + 1):
        m = f(y, np.mean(preds[:k], axis=0))
        rows.append(dict(target=target, k_seeds=k, **{c: m[c] for c in cols}, variant=label))
    pd.DataFrame(rows).to_csv(d / "cumulative_ensemble_metrics.csv", index=False)


def prepare_finals(only=None):
    """Copy the existing final-model arrays and recompute their metrics; assert agreement to 1e-6."""
    import numpy as np
    import pandas as pd
    info = {}
    for t in TARGETS:
        if only and t not in only:
            continue
        src = SEED_ROOT / t
        d = OUT / t / "final"
        d.mkdir(parents=True, exist_ok=True)
        p = np.load(src / "per_seed_preds.npy")
        y = np.load(src / FINAL_FILES[t])
        shutil.copyfile(src / "per_seed_preds.npy", d / "per_seed_preds.npy")
        shutil.copyfile(src / FINAL_FILES[t], d / FINAL_FILES[t])
        write_metrics(d, t, "final", SEEDS_ALL[:len(p)], p, y)
        (d / "seeds_done.json").write_text(json.dumps(SEEDS_ALL[:len(p)]))
        cols = metric_cols(t)
        for name, mine, theirs, key in (("per_seed_metrics.csv", d / "per_seed_metrics.csv", src / "per_seed_metrics.csv", "seed"),
                                        ("cumulative_ensemble_metrics.csv", d / "cumulative_ensemble_metrics.csv",
                                         src / "cumulative_ensemble_metrics.csv", "k_seeds")):
            a, b = pd.read_csv(mine), pd.read_csv(theirs)
            assert len(a) == len(b) == len(p), (t, name, len(a), len(b))
            assert (a[key].values == b[key].values).all(), (t, name, "key mismatch")
            for c in cols:
                dmax = float(np.max(np.abs(a[c].values - b[c].values)))
                assert dmax <= 1e-6, f"final {t} {name} {c} differs by {dmax}"
        info[t] = dict(n_seeds=int(p.shape[0]), n_test=int(p.shape[1]), checked="matches existing CSVs to 1e-6")
    return info


def load_csv_rows():
    import pandas as pd
    df = pd.read_csv(ABL_CSV)
    return df


def plan_variants(df, targets=None, slugs=None):
    """Every non-final row of proposed_model_ablation.csv -> registry entry (read from the CSV, not hard-coded)."""
    out = []
    for t in TARGETS:
        sub = df[(df.target == t) & (~df.is_final_model.astype(bool))]
        for _, r in sub.iterrows():
            key = (t, r["model"])
            if key not in VARIANTS:
                raise SystemExit(f"CSV variant {key} is not in the registry -- stop and report")
            spec = dict(VARIANTS[key]); spec.update(target=t, name=r["model"], csv_row=r.to_dict())
            if targets and t not in targets:
                continue
            if slugs and spec["slug"] not in slugs:
                continue
            out.append(spec)
    return out


def make_pools(workers, groups):
    ctx = get_context("spawn")
    return {g: ProcessPoolExecutor(max_workers=workers, mp_context=ctx) for g in groups}


def save_seed(d, target, label, seed, res, y_final, final_name):
    """Persist one finished seed (never overwrites), then rebuild the consolidated files from seed files."""
    import numpy as np
    d.mkdir(parents=True, exist_ok=True)
    (d / "seeds").mkdir(exist_ok=True)
    sf = d / "seeds" / f"seed_{seed}.npy"
    if sf.exists():
        raise RuntimeError(f"refusing to overwrite completed seed {sf}")
    assert np.array_equal(res["y_test"], y_final), f"{target}/{label}: y_test differs from the final model's y_test"
    np.save(sf, res["pred"])
    np.save(d / final_name, res["y_test"])
    done = sorted(int(p.stem.split("_")[1]) for p in (d / "seeds").glob("seed_*.npy"))
    preds = np.vstack([np.load(d / "seeds" / f"seed_{s}.npy") for s in done])
    np.save(d / "per_seed_preds.npy", preds)
    (d / "seeds_done.json").write_text(json.dumps(done))
    write_metrics(d, target, label, done, preds, res["y_test"])


def append_run_log(d, seed, seconds, device, status):
    d.mkdir(parents=True, exist_ok=True)
    f = d / "run_log.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["seed", "seconds", "device", "status", "timestamp"])
        w.writerow([seed, f"{seconds:.1f}", device, status, now()])


def write_manifest(extra=None):
    import numpy as np, pandas as pd, scipy, sklearn, torch
    man_p = OUT / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    man.update(
        generated=now(),
        note="Training code is exec'd verbatim from the listed notebook cells / scripts; only the seed varies.",
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                         device=(torch.cuda.get_device_name(0) + " / " + str(torch.cuda.get_device_properties(0).name)
                                 if torch.cuda.is_available() else "cpu"),
                         platform=platform.platform()),
        determinism=dict(thread_env=THREADS, CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                         cudnn_deterministic=True, cudnn_benchmark=False,
                         use_deterministic_algorithms="True (warn_only=True)",
                         seeding="random.seed/np.random.seed/torch.manual_seed/cuda.manual_seed_all inside every task, "
                                 "plus the recipe's own RandomState(seed) validation split and torch.manual_seed(seed)",
                         data_staging="mfmn_benchmark_package/src + data copied byte-identically to /dev/shm/pk_ro (hash/size verified)"),
        sha256={str(p.relative_to(ROOT)): sha256(p) for p in HASH_FILES if p.exists()},
        sources=SOURCES,
        slug_to_name={f"{t}/{v['slug']}": n for (t, n), v in VARIANTS.items()},
    )
    if extra:
        man.update(extra)
    man_p.write_text(json.dumps(man, indent=2, default=str))


def read_selection():
    p = OUT / "candidate_selection.json"
    return json.loads(p.read_text()) if p.exists() else {}


def run_bounded(pools, tasks, pka_cap, handle):
    """Submit tasks in priority order; at most `pka_cap` pKa trainings in flight at once (each holds ~6-10 GB of the
    ~20 GB MIG slice).  tasks: [(tag, runner, args_tuple_for_run_task_after_runner)].  handle(tag, result|Exception)."""
    from concurrent.futures import wait, FIRST_COMPLETED
    pending, inflight = list(tasks), {}
    npka = lambda: sum(1 for _, r in inflight.values() if r.startswith("pka_"))

    def fill():
        i = 0
        while i < len(pending):
            tag, r, a = pending[i]
            if r.startswith("pka_") and npka() >= pka_cap:
                i += 1
                continue
            inflight[pools[RUNNER_GROUP[r]].submit(run_task, r, *a)] = (tag, r)
            pending.pop(i)
    fill()
    while inflight:
        done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        for f in done:
            tag, r = inflight.pop(f)
            try:
                res = f.result()
            except Exception as e:
                res = e
                res.tb = traceback.format_exc()
            handle(tag, res)
        fill()


def smoke(args, specs):
    import numpy as np
    import pandas as pd
    tag = args.tag
    sd = OUT / "smoke" / tag
    sd.mkdir(parents=True, exist_ok=True)
    runnable = [s for s in specs if s["runners"]]
    groups = {RUNNER_GROUP[r] for s in runnable for r in s["runners"]}
    if args.control:
        groups |= {RUNNER_GROUP[r] for t, r in CONTROLS if not args.targets or t in args.targets}
    pools = make_pools(args.workers, groups)
    results = {}

    ctl_results = {}

    def launch(batch, controls=()):
        tasks = [(("v", s["target"], s["slug"], r), r, (s["target"], s["name"], 42, args.device)) for s, r in batch]
        tasks += [(("c", t, r), r, (t, "control", 42, args.device)) for t, r in controls]

        def handle(tag, res):
            if tag[0] == "c":
                ctl_results[(tag[1], tag[2])] = res
                log.info("control done %s via %s%s", tag[1], tag[2], "" if not isinstance(res, Exception) else f" FAILED {res!r}")
                return
            _, t, slug, r = tag
            if isinstance(res, Exception):
                log.error("smoke FAILED %s/%s via %s: %r", t, slug, r, res)
                results[(t, slug, r)] = dict(error=repr(res))
                return
            results[(t, slug, r)] = res
            log.info("smoke done %s/%s via %s: %.0fs (peak GPU %.1f GB)", t, slug, r, res["seconds"], res.get("peak_gpu_gb", 0))
        run_bounded(pools, tasks, args.pka_concurrency, handle)

    # stage 1: first candidate of every variant; stage 2: fall-backs only for those whose first candidate failed
    first = lambda s: s["runners"] if s.get("test_all") else s["runners"][:1]
    ctl = [(t, r) for t, r in CONTROLS if args.control and (not args.targets or t in args.targets)]
    launch([(s, r) for s in runnable for r in first(s)], ctl)
    rows, selection = [], {}
    f_by_target = {}

    def judge(s, r):
        res = results[(s["target"], s["slug"], r)]
        row = dict(target=s["target"], variant=s["name"], slug=s["slug"], runner=r)
        if "error" in res:
            return dict(row, status="ERROR", error=res["error"]), res
        t = s["target"]
        f = f_by_target.setdefault(t, metric_fn(t))
        y_final = np.load(OUT / t / "final" / FINAL_FILES[t])
        m = f(res["y_test"], res["pred"])
        c = s["csv_row"]
        d2, dm = m["R2"] - float(c["R2"]), m["MAE"] - float(c["MAE"])
        row.update(R2=m["R2"], MAE=m["MAE"], RMSE=m["RMSE"], csv_R2=float(c["R2"]), csv_MAE=float(c["MAE"]),
                   dR2=d2, dMAE=dm, seconds=res["seconds"], y_test_identical_to_final=bool(np.array_equal(res["y_test"], y_final)))
        for k in ("GMFE", "within_2fold"):
            if k in m and k in c and pd.notna(c[k]):
                row[k], row["csv_" + k] = m[k], float(c[k])
        row["passes"] = bool(abs(d2) <= GATE_DR2 and abs(dm) <= GATE_DMAE and row["y_test_identical_to_final"])
        return row, res

    stage1 = {}
    for s in runnable:
        for r in first(s):
            row, res = judge(s, r)
            stage1[(s["target"], s["slug"], r)] = (row, res)
            rows.append(row)
    fallback = [(s, s["runners"][1]) for s in runnable
                if len(s["runners"]) > 1 and not s.get("test_all") and not stage1[(s["target"], s["slug"], s["runners"][0])][0].get("passes")]
    if fallback:
        launch(fallback)
        for s, r in fallback:
            row, res = judge(s, r)
            rows.append(row)
    if args.control:
        ctl_rows = []
        for (t, r), res in ctl_results.items():
            if isinstance(res, Exception):
                continue
            stored = np.load(SEED_ROOT / t / "per_seed_preds.npy")[0]
            fm = metric_fn(t)
            m, m0 = fm(res["y_test"], res["pred"]), fm(res["y_test"], stored)
            ctl_rows.append(dict(target=t, runner=r, R2=m["R2"], MAE=m["MAE"], stored_final_seed42_R2=m0["R2"],
                                 stored_final_seed42_MAE=m0["MAE"], dR2=m["R2"] - m0["R2"], dMAE=m["MAE"] - m0["MAE"],
                                 max_abs_pred_diff=float(np.abs(res["pred"] - stored).max()), seconds=res["seconds"]))
            log.info("control %s via %s: R2 %.4f vs stored %.4f (max|dpred| %.2e)", t, r, m["R2"], m0["R2"], ctl_rows[-1]["max_abs_pred_diff"])
        pd.DataFrame(ctl_rows).to_csv(sd / "controls.csv", index=False)
    for p in pools.values():
        p.shutdown()
    df = pd.DataFrame(rows)
    df.to_csv(sd / "smoke_results.csv", index=False)
    for (t, slug, r), res in results.items():
        if "pred" in res:
            vd = sd / t / slug
            vd.mkdir(parents=True, exist_ok=True)
            np.save(vd / f"seed42_{r}.npy", res["pred"])

    for s in specs:
        key = f"{s['target']}/{s['slug']}"
        mine = df[(df.target == s["target"]) & (df.slug == s["slug"])]
        passed = mine[mine["passes"] == True] if "passes" in mine else mine.iloc[0:0]
        hp = {r: results[(s["target"], s["slug"], r)].get("hparams") for r in s["runners"]
              if (s["target"], s["slug"], r) in results}
        cand_cols = [c for c in ("runner", "R2", "MAE", "dR2", "dMAE", "passes") if c in mine.columns]
        entry = dict(name=s["name"], candidates_tested=mine[cand_cols].to_dict("records"), hparams=hp)
        ran = [r for r in mine["runner"]] if len(mine) else []
        errored = [r for r in ran if "error" in results.get((s["target"], s["slug"], r), {})]
        if s["slug"] == "mfmn_base_untuned":
            # rule 3: identify by reproduction among C1 (existing script), C2/C3 (reconstructed plain-MFMN, defined a priori)
            if len(passed) == 1:
                r = passed.iloc[0]["runner"]
                entry.update(runner=r)
                if r == "fuC1":
                    entry.update(status="RUN_WITH_CAVEAT", reason="exactly one candidate reproduces seed 42: the existing script; it trains "
                                 "MFMN_InteractAux (aux 0.3), not plain MFMN -> the CSV label is inaccurate")
                else:
                    entry.update(status="RECONSTRUCTED_VERIFIED", reason=f"exactly one candidate reproduces seed 42: reconstructed {r} (plain MFMN)")
            elif len(passed) == 0:
                # none reproduces: choose by the project's own provenance (all_models_comparison.csv attributes this label to
                # expv2_aux_targets_final.py with aux_weight=0.3), NOT by closeness of the smoke-test numbers
                entry.update(runner="fuC1", status="RUN_WITH_CAVEAT",
                             reason="no candidate (C1 existing script; C2/C3 reconstructed plain-MFMN) reproduces the CSV row within the gate; running "
                                    "C1 because the project's own records attribute the label to that script. It trains MFMN_InteractAux "
                                    "(aux_weight 0.3, pca32 context), not plain MFMN, and does NOT reproduce the CSV row")
            else:
                entry.update(status="BLOCKED", reason=f"{len(passed)} candidates reproduce seed 42 (need exactly 1)")
        elif s.get("reconstructed"):
            r = s["runners"][0]
            if r in errored or r not in ran:
                entry.update(status="BLOCKED", reason="reconstructed candidate raised an error in the smoke test: " + str(results.get((s["target"], s["slug"], r), {}).get("error")))
            elif len(passed):
                entry.update(runner=r, status="RECONSTRUCTED_VERIFIED", reason="reconstructed; seed 42 reproduces the CSV row within the gate")
            else:
                entry.update(runner=r, status="RECONSTRUCTED_UNVERIFIED", reason="reconstructed; seed 42 does NOT reproduce the CSV row within the gate "
                             "(see candidates_tested)")
        elif len(passed):
            r = passed.iloc[0]["runner"]       # first passing candidate in priority order (rows are in that order)
            note = None
            if r == "fuB":
                note = "candidate A failed the gate; ran notebook-05 recipe with low-fu multiplier 1.0 (only change)"
            entry.update(status="RUN_WITH_CAVEAT" if note else "RUN", runner=r, **({"reason": note} if note else {}))
        else:
            entry.update(runner=s["runners"][0], status="RUN_WITH_CAVEAT",
                         reason="seed-42 gate FAILED for every candidate (candidate A and B give identical results); running candidate A (primary) "
                                "anyway at the user's instruction; see the Fu control in the report for environment drift")
        selection[key] = entry
    merged = read_selection()            # a partial smoke run (--targets/--variants) must not drop other variants
    merged.update(selection)
    (OUT / "candidate_selection.json").write_text(json.dumps(merged, indent=2, default=str))
    show = [c for c in ("target", "variant", "runner", "R2", "csv_R2", "dR2", "MAE", "csv_MAE", "dMAE", "seconds", "passes", "status") if c in df.columns]
    log.info("\n%s", df[show].round(4).to_string(index=False))
    for k, v in selection.items():
        log.info("selection %-34s %-16s %s", k, v["status"], v.get("runner", ""))
    man_p = OUT / "manifest.json"
    prev = json.loads(man_p.read_text()).get("hyperparameters_by_variant", {}) if man_p.exists() else {}
    prev.update({k: v.get("hparams") for k, v in selection.items()})
    write_manifest(dict(smoke_workers=args.workers, smoke_tag=tag, hyperparameters_by_variant=prev))


def full_run(args, specs):
    import numpy as np
    sel = read_selection()
    seeds = parse_seeds(args.seeds)
    todo, report = [], []
    for s in specs:
        key = f"{s['target']}/{s['slug']}"
        st = sel.get(key, {})
        if not s["runners"] or not st.get("status", "").startswith(("RUN", "RECONSTRUCTED")):
            report.append((key, "SKIPPED (" + st.get("status", "no smoke result -- run --smoke first") + ")", []))
            continue
        d = vdir(s["target"], s["slug"])
        done = set(json.loads((d / "seeds_done.json").read_text())) if (d / "seeds_done.json").exists() else set()
        pend = [x for x in seeds if x not in done]
        report.append((key, f"{st['status']} runner={st['runner']}", pend))
        todo.append((s, st["runner"], pend))
    for key, status, pend in report:
        log.info("plan  %-34s %-40s pending seeds: %s", key, status, pend if pend else "-")
    if args.dry_run:
        return
    # seed-42 results of the smoke test (same code, same environment, bit-identical on re-run) are imported instead of retrained
    smoke_dir = OUT / "smoke"
    imported = 0
    for s, r, pend in todo:
        if 42 not in pend:
            continue
        for tagdir in sorted(smoke_dir.glob("*")):
            f = tagdir / s["target"] / s["slug"] / f"seed42_{r}.npy"
            sr = tagdir / "smoke_results.csv"
            if f.exists() and sr.exists():
                import pandas as pd
                row = pd.read_csv(sr)
                row = row[(row.target == s["target"]) & (row.slug == s["slug"]) & (row.runner == r)]
                if not len(row) or not bool(row.iloc[0].get("y_test_identical_to_final", False)):
                    continue
                pred = np.load(f)
                y_final = np.load(OUT / s["target"] / "final" / FINAL_FILES[s["target"]])
                res = dict(pred=pred, y_test=y_final, seconds=float(row.iloc[0]["seconds"]), device="imported from smoke test " + tagdir.name)
                d = vdir(s["target"], s["slug"])
                save_seed(d, s["target"], s["slug"], 42, res, y_final, FINAL_FILES[s["target"]])
                append_run_log(d, 42, res["seconds"], res["device"], "ok (imported from smoke)")
                pend.remove(42)
                imported += 1
                log.info("IMPORTED seed 42 for %s/%s from %s", s["target"], s["slug"], tagdir.name)
                break
    stage = [(s, r, x) for s, r, pend in todo for x in pend]
    if not stage:
        log.info("nothing to do")
        return
    # cheapest first by target; within a target seed-major (so seed 42 of every variant comes first), interleaving variants
    order = {t: i for i, t in enumerate(TARGETS)}
    stage.sort(key=lambda a: (order[a[0]["target"]] if not a[1].startswith("pka_") else 3, a[2], a[0]["target"], a[0]["slug"]))
    pools = make_pools(args.workers, {RUNNER_GROUP[r] for _, r, _ in stage})
    stats = dict(ok=0, fail=0)
    total = len(stage)

    def handle(tag, res):
        s, r, x = tag
        t = s["target"]
        d = vdir(t, s["slug"])
        if isinstance(res, Exception):
            stats["fail"] += 1
            append_run_log(d, x, 0.0, args.device, "failed: " + repr(res)[:200])
            log.error("FAIL  %s/%s seed %d: %r\n%s", t, s["slug"], x, res, getattr(res, "tb", ""))
            return
        try:
            y_final = np.load(OUT / t / "final" / FINAL_FILES[t])
            save_seed(d, t, s["slug"], x, res, y_final, FINAL_FILES[t])
            append_run_log(d, x, res["seconds"], res["device"], "ok")
            stats["ok"] += 1
            log.info("DONE  %s/%s seed %d  %.0fs peakGPU %.1fGB  (%d ok, %d failed, %d left)", t, s["slug"], x, res["seconds"],
                     res.get("peak_gpu_gb", 0), stats["ok"], stats["fail"], total - stats["ok"] - stats["fail"])
        except Exception as e:
            stats["fail"] += 1
            append_run_log(d, x, 0.0, args.device, "save failed: " + repr(e)[:200])
            log.error("SAVE FAIL %s/%s seed %d: %r", t, s["slug"], x, e)

    tasks = [((s, r, x), r, (s["target"], s["name"], x, args.device)) for s, r, x in stage]
    log.info("submitting %d trainings (%d seed-42 results imported); workers per group=%d, pKa concurrency=%d",
             total, imported, args.workers, args.pka_concurrency)
    run_bounded(pools, tasks, args.pka_concurrency, handle)
    for p in pools.values():
        p.shutdown()
    log.info("finished: %d ok, %d failed", stats["ok"], stats["fail"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="*", help="subset of CL VDss Fu pKa_Acidic pKa_Basic (default all)")
    ap.add_argument("--variants", nargs="*", help="variant slugs (default all)")
    ap.add_argument("--seeds", default=None, help="e.g. 42-51 or 42,43 (default 42-51)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="concurrent worker processes per code-path group")
    ap.add_argument("--tag", default="default", help="smoke output tag")
    ap.add_argument("--pka-concurrency", type=int, default=2, help="max pKa trainings in flight at once (GPU memory bound)")
    ap.add_argument("--control", action="store_true", help="smoke: also run the unmodified pKa model and the final Fu recipe at seed 42 "
                                                           "and compare with the stored final arrays (environment-drift control)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "run.log"), logging.StreamHandler(sys.stdout)])
    log.info("start %s args=%s", now(), vars(args))
    df = load_csv_rows()
    specs = plan_variants(df, args.targets, args.variants)
    if not args.dry_run:
        stage_shm()
    info = prepare_finals(args.targets)
    log.info("final models: arrays copied, metrics recomputed and match existing CSVs to 1e-6: %s", info)
    if not args.dry_run:
        write_manifest(dict(final_check=info))
    if args.smoke:
        smoke(args, specs)
    else:
        full_run(args, specs)
    log.info("end %s", now())


if __name__ == "__main__":
    main()
