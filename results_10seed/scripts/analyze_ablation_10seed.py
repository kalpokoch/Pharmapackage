#!/usr/bin/env python
"""
analyze_ablation_10seed.py -- summary tables, paired statistics and report.md from whatever seeds exist in
ablation_10seed/.  Re-runnable; works on partial results; blocked variants are listed as "not run" and get no
statistics rows.  Writes only inside ablation_10seed/ (tables/, report.md, tables/checks.json).

Metric definitions (modelling scale: log10 for CL/VDss/Fu, pKa units for pKa):
  R2 = 1 - SSE/SST ; MAE, RMSE ; GMFE = 10**mean|pred-obs| ; within_k_fold = mean(|pred-obs| <= log10 k)
  pKa: within_k_pKa_unit = mean(|pred-obs| <= k).   Higher is better: R2, within_*.  Lower: MAE, RMSE, GMFE.

Run:  python ablation_10seed/analyze_ablation_10seed.py [--boot 10000]
"""
import argparse, datetime, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
import os
OUT = Path(os.environ.get("ABL_OUT", HERE))
FINAL_ROOT = Path(os.environ.get("FINAL_ROOT", HERE.parent / "seed_analysis_results"))
TAB = OUT / "tables"
ROOT = HERE.parent
TARGETS = ["CL", "VDss", "Fu", "pKa_Acidic", "pKa_Basic"]
LOG_TARGETS = {"CL", "VDss", "Fu"}
FINAL_Y = {"CL": "y_test_log.npy", "VDss": "y_test_log.npy", "Fu": "y_test.npy",
           "pKa_Acidic": "y_test.npy", "pKa_Basic": "y_test.npy"}
LOGM = ["R2", "MAE", "RMSE", "GMFE", "within_2fold", "within_3fold", "within_5fold"]
PKAM = ["R2", "MAE", "RMSE", "within_1_pKa_unit", "within_2_pKa_unit"]
HIGHER = lambda m: m == "R2" or m.startswith("within_")
B_DEFAULT = 10_000


def metrics_of(t):
    return LOGM if t in LOG_TARGETS else PKAM


# ------------------------------------------------------------------ metric implementation (spec definitions)
def metric_values(y, p, t):
    """y,p: (..., n) arrays broadcast on the last axis -> dict of arrays (...,)."""
    e = p - y
    sse = (e ** 2).sum(-1)
    sst = ((y - y.mean(-1, keepdims=True)) ** 2).sum(-1)
    out = {"R2": 1 - sse / sst, "MAE": np.abs(e).mean(-1), "RMSE": np.sqrt((e ** 2).mean(-1))}
    if t in LOG_TARGETS:
        out["GMFE"] = 10 ** np.abs(e).mean(-1)
        for k in (2, 3, 5):
            out[f"within_{k}fold"] = (np.abs(e) <= np.log10(k)).mean(-1)
    else:
        for k in (1, 2):
            out[f"within_{k}_pKa_unit"] = (np.abs(e) <= k).mean(-1)
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------ loading
def load_variant(d):
    seeds = json.loads((d / "seeds_done.json").read_text())
    p = np.load(d / "per_seed_preds.npy")
    assert p.shape[0] == len(seeds), (d, p.shape, seeds)
    ys = sorted(d.glob("y_test*.npy"))
    assert len(ys) == 1, (d, ys)
    return dict(seeds=seeds, preds=p, y=np.load(ys[0]), dir=d)


def discover():
    sel = json.loads((OUT / "candidate_selection.json").read_text()) if (OUT / "candidate_selection.json").exists() else {}
    man = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else {}
    names = man.get("slug_to_name", {})
    finals, variants, notrun = {}, {}, []
    for t in TARGETS:
        fd = OUT / t / "final"
        if (fd / "per_seed_preds.npy").exists():
            finals[t] = load_variant(fd)
        for key, nm in names.items():
            tt, slug = key.split("/", 1)
            if tt != t:
                continue
            d = OUT / t / slug
            st = sel.get(key, {})
            if (d / "seeds_done.json").exists() and (d / "per_seed_preds.npy").exists():
                v = load_variant(d)
                v.update(name=nm, slug=slug, status=st.get("status", "RUN"), reason=st.get("reason", ""))
                variants[(t, slug)] = v
            else:
                notrun.append(dict(target=t, slug=slug, name=nm, status=st.get("status", "NOT RUN"),
                                   reason=st.get("reason", "no results found")))
    return sel, man, finals, variants, notrun


# ------------------------------------------------------------------ checks
def run_checks(finals, variants):
    msgs, ok = [], True
    for t, f in finals.items():
        ex = FINAL_ROOT / t
        a = pd.read_csv(ex / "per_seed_metrics.csv")
        mine = metric_values(f["y"][None, :], f["preds"], t)
        for m in ["R2", "MAE", "RMSE"]:
            dmax = float(np.max(np.abs(mine[m] - a[m].values)))
            if dmax > 1e-6:
                ok = False
            msgs.append(f"final {t} {m}: recomputed (this script) vs existing per_seed_metrics.csv max|diff|={dmax:.2e}")
    for (t, slug), v in variants.items():
        f = finals[t]
        same = bool(np.array_equal(v["y"], f["y"]))
        ok &= same
        msgs.append(f"{t}/{slug}: y_test identical to final's: {same}")
        csvp = v["dir"] / "per_seed_metrics.csv"
        if csvp.exists():
            c = pd.read_csv(csvp)
            seeds_ok = list(c["seed"]) == v["seeds"] and list(c["seed_idx"]) == [s - 42 for s in v["seeds"]]
            mine = metric_values(v["y"][None, :], v["preds"], t)
            dmax = max(float(np.max(np.abs(mine[m] - c[m].values))) for m in ["R2", "MAE", "RMSE"])
            ok &= seeds_ok and dmax <= 1e-6
            msgs.append(f"{t}/{slug}: row-to-seed mapping consistent (preds/seeds_done.json/per_seed_metrics.csv): "
                        f"{seeds_ok}; recomputed vs stored per-seed metrics max|diff|={dmax:.2e}")
        missing = sorted(set(range(42, 52)) - set(v["seeds"]))
        msgs.append(f"{t}/{slug}: seeds present {len(v['seeds'])}/10" + (f"; MISSING {missing}" if missing else ""))
    return ok, msgs


# ------------------------------------------------------------------ tables
def seed_metric_table(v, t):
    return metric_values(v["y"][None, :], v["preds"], t)          # dict metric -> (n_seeds,)


def build_summary(finals, variants, notrun):
    rows = []
    for t in TARGETS:
        entries = []
        if t in finals:
            entries.append((("final"), "FINAL", finals[t], "final"))
        for (tt, slug), v in variants.items():
            if tt == t:
                entries.append((slug, v["name"], v, v["status"]))
        for slug, nm, v, st in entries:
            sm = seed_metric_table(v, t)
            ens = metric_values(v["y"], v["preds"].mean(0), t)
            r = dict(target=t, variant=slug, name=nm, status=st, n_seeds=len(v["seeds"]),
                     seeds=" ".join(map(str, v["seeds"])), n_test=len(v["y"]))
            for m in metrics_of(t):
                r[f"{m}_mean"] = float(sm[m].mean())
                r[f"{m}_sd"] = float(sm[m].std(ddof=1)) if len(v["seeds"]) > 1 else np.nan
                r[f"{m}_ens"] = float(ens[m])
            rows.append(r)
        for n in notrun:
            if n["target"] == t:
                rows.append(dict(target=t, variant=n["slug"], name=n["name"], status="not run (" + n["status"] + ")",
                                 n_seeds=0, seeds="", n_test=np.nan))
    return pd.DataFrame(rows)


def build_vs_final(finals, variants, B):
    rows = []
    for t in TARGETS:
        if t not in finals:
            continue
        vs = [(slug, v) for (tt, slug), v in variants.items() if tt == t]
        if not vs:
            continue
        f = finals[t]
        n_test = len(f["y"])
        rng = np.random.default_rng(0)                  # one index matrix per target, shared by all variants
        idx = rng.integers(0, n_test, size=(B, n_test))  # same resampled compounds for variant and final in a draw
        per_metric = {}
        for slug, v in vs:
            common = [s for s in v["seeds"] if s in f["seeds"]]
            vi = [v["seeds"].index(s) for s in common]
            fi = [f["seeds"].index(s) for s in common]
            vp, fp = v["preds"][vi], f["preds"][fi]
            vm = metric_values(v["y"][None, :], vp, t)
            fm = metric_values(f["y"][None, :], fp, t)
            # ensemble-level paired bootstrap (ensemble over the common seeds; =10 seeds when complete)
            ev, ef = vp.mean(0), fp.mean(0)
            y = f["y"]
            boot_v = {m: [] for m in metrics_of(t)}
            boot_f = {m: [] for m in metrics_of(t)}
            for c0 in range(0, B, 2000):
                ii = idx[c0:c0 + 2000]
                bv = metric_values(y[ii], ev[ii], t)
                bf = metric_values(y[ii], ef[ii], t)
                for m in metrics_of(t):
                    boot_v[m].append(bv[m]); boot_f[m].append(bf[m])
            pt_v, pt_f = metric_values(y, ev, t), metric_values(y, ef, t)
            for m in metrics_of(t):
                d = vm[m] - fm[m]
                n = len(d)
                r = dict(target=t, variant=slug, name=v["name"], metric=m, higher_is_better=HIGHER(m), n_seeds=n)
                r["diff_mean"] = float(d.mean())
                r["diff_sd"] = float(d.std(ddof=1)) if n > 1 else np.nan
                if n > 1:
                    se = r["diff_sd"] / np.sqrt(n)
                    tcrit = stats.t.ppf(0.975, n - 1)
                    r["ci95_lo"], r["ci95_hi"] = r["diff_mean"] - tcrit * se, r["diff_mean"] + tcrit * se
                    r["p_ttest"] = float(stats.ttest_1samp(d, 0.0).pvalue) if r["diff_sd"] > 0 else np.nan
                else:
                    r["ci95_lo"] = r["ci95_hi"] = r["p_ttest"] = np.nan
                better = (d > 0) if HIGHER(m) else (d < 0)
                r["n_seeds_variant_better"] = int(better.sum())
                nz = d[d != 0]
                r["n_nonzero_diffs"] = int(len(nz))
                r["p_wilcoxon"] = float(stats.wilcoxon(nz, method="exact").pvalue) if len(nz) > 0 else np.nan
                fsd = float(fm[m].std(ddof=1)) if n > 1 else np.nan
                r["final_seed_sd"] = fsd
                r["diff_in_final_sd"] = r["diff_mean"] / fsd if fsd and fsd > 0 else np.nan
                bd = np.concatenate(boot_v[m]) - np.concatenate(boot_f[m])
                r["ens_diff"] = float(pt_v[m] - pt_f[m])
                r["ens_ci95_lo"], r["ens_ci95_hi"] = (float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)))
                r["ens_p_variant_better"] = float(((bd > 0) if HIGHER(m) else (bd < 0)).mean())
                r["ens_ci_flag"] = "excludes_zero" if (r["ens_ci95_lo"] > 0 or r["ens_ci95_hi"] < 0) else "overlaps_zero"
                r["boot_B"] = B
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        for col, new in (("p_wilcoxon", "p_holm"), ("p_ttest", "p_holm_ttest")):
            df[new] = np.nan
            for (t, m), g in df.groupby(["target", "metric"]):
                ok = g[col].notna()
                if ok.sum():
                    df.loc[g.index[ok], new] = holm(g.loc[ok, col].values)
    return df


def fmt_pm(mean, sd, nd=3):
    return f"{mean:.{nd}f} ± {sd:.{nd}f}" if pd.notna(sd) else f"{mean:.{nd}f} ± n/a"


def build_table(summary, vs):
    rows = []
    for t in TARGETS:
        g = summary[summary.target == t]
        if g.empty:
            continue
        order = list(g[g.variant != "final"].index) + list(g[g.variant == "final"].index)   # final last
        for i in order:
            s = g.loc[i]
            r = dict(target=t, variant=s["variant"], name=s["name"], status=s["status"], n_seeds=s["n_seeds"])
            if s["n_seeds"] == 0:
                rows.append(r); continue
            for m in ("R2", "MAE", "RMSE"):
                r[m + "_mean_pm_sd"] = fmt_pm(s[f"{m}_mean"], s[f"{m}_sd"])
            r["R2_ens"], r["MAE_ens"] = round(s["R2_ens"], 4), round(s["MAE_ens"], 4)
            if s["variant"] == "final":
                r.update(flag_R2="reference", flag_MAE="reference")
            else:
                for m in ("R2", "MAE"):
                    q = vs[(vs.target == t) & (vs.variant == s["variant"]) & (vs.metric == m)]
                    if len(q):
                        q = q.iloc[0]
                        r[f"diff_mean_{m}"] = round(q["diff_mean"], 4)
                        r[f"diff_in_final_sd_{m}"] = round(q["diff_in_final_sd"], 2) if pd.notna(q["diff_in_final_sd"]) else np.nan
                        r[f"flag_{m}"] = q["ens_ci_flag"]
            rows.append(r)
    return pd.DataFrame(rows)


def build_ranking(summary):
    rows = []
    for t in TARGETS:
        g = summary[(summary.target == t) & (summary.n_seeds > 0)]
        if g.empty:
            continue
        for m in metrics_of(t):
            gg = g.sort_values(f"{m}_mean", ascending=not HIGHER(m))
            for rank, (_, s) in enumerate(gg.iterrows(), 1):
                rows.append(dict(target=t, metric=m, rank=rank, variant=s["variant"], name=s["name"],
                                 mean_over_seeds=s[f"{m}_mean"], sd_over_seeds=s[f"{m}_sd"], ens=s[f"{m}_ens"],
                                 n_seeds=s["n_seeds"], is_final=(s["variant"] == "final")))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def md_table(df, cols=None, nd=4):
    if df.empty:
        return "_(none)_"
    df = df[cols] if cols else df
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cells.append("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def build_report(sel, man, summary, vs, ranking, notrun, finals, variants, check_ok, check_msgs):
    L = []
    A = L.append
    A("# Ablation with 10 seeds - report\n")
    A(f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} by `analyze_ablation_10seed.py`. "
      "Facts only; no recommendations about what to claim.\n")
    # --- 1. status table of all 12 variants
    A("## 1. Status of all 12 variants\n")
    srows = []
    for key, nm in man.get("slug_to_name", {}).items():
        t, slug = key.split("/", 1)
        st = sel.get(key, {})
        have = variants.get((t, slug))
        status = st.get("status", "not run")
        n = len(have["seeds"]) if have else 0
        if status == "RUN" and n >= 10: label = "RUN (10 seeds)"
        elif status == "RUN" and have: label = f"RUN (partial: {n}/10 seeds)"
        elif status == "RUN_WITH_CAVEAT": label = f"RUN WITH CAVEAT ({n}/10 seeds)"
        elif status == "RECONSTRUCTED_VERIFIED": label = f"RECONSTRUCTED, seed-42 gate passed ({n}/10 seeds)"
        elif status == "RECONSTRUCTED_UNVERIFIED": label = f"RECONSTRUCTED, seed-42 gate FAILED ({n}/10 seeds)"
        else: label = "BLOCKED" if status == "BLOCKED" else status
        why = st.get("reason", "")
        srows.append(dict(target=t, variant=nm, slug=slug, status=label, detail=why))
    A(md_table(pd.DataFrame(srows)))
    # --- 2. smoke test
    A("\n## 2. Smoke test: seed 42 vs `proposed_model_ablation.csv`\n")
    A("Gate: |dR2| <= 0.002 and |dMAE| <= 0.002 and y_test identical to the final model's. The latest smoke run of each "
      "(variant, code path) is shown (tag in the last column). Candidates for the same variant are listed in priority order; "
      "`fuC2`/`fuC3` and every pKa variant are RECONSTRUCTED code paths.\n")
    frames = []
    for sp in sorted((OUT / "smoke").glob("*/smoke_results.csv"), key=lambda q: q.stat().st_mtime):
        f_ = pd.read_csv(sp); f_["smoke_tag"] = sp.parent.name; frames.append(f_)
    if frames:
        sm = pd.concat(frames, ignore_index=True).drop_duplicates(["target", "slug", "runner"], keep="last")
        sm["_o"] = sm.target.map({t: i for i, t in enumerate(TARGETS)})
        sm = sm.sort_values(["_o", "slug", "runner"])
        cols = [c for c in ["target", "variant", "runner", "R2", "csv_R2", "dR2", "MAE", "csv_MAE", "dMAE", "RMSE", "GMFE",
                            "csv_GMFE", "within_2fold", "csv_within_2fold", "seconds", "passes", "smoke_tag"] if c in sm.columns]
        A(md_table(sm, cols))
    A("\nCandidate code paths: `clvd` = `ablation/ablation_official_test_CL_VDss_Fu.ipynb` cells 1,3,5 with `src/mfmn.py`; "
      "`fuA` = `MFMN_v2.ipynb` cells 1,13 + `mfmn/experiments/expv2_aux_h7_dyngate.py::fit_predict_dyngate` with `RECIPE['Fu']`; "
      "`fuB` = notebook `05_Fu_MFMN_DynGate.ipynb` cells 1-4 with the low-fu multiplier set to 1.0 (only change); "
      "`fuC1` = `mfmn/experiments/expv2_aux_targets_final.py::fit_predict` defaults (class `MFMN_InteractAux`, aux_weight 0.3); "
      "`fuC2`/`fuC3` = RECONSTRUCTED plain-`MFMN` versions of the original `fit_predict_dyngate` loop (C2: pca32 context, dropout 0.2, "
      "wd 1e-2; C3: `RECIPE['Fu']`); `pka_*` = original `fit_predict_gnn` (10seed notebook cell 9) with a RECONSTRUCTED modification "
      "(descriptions in section 8).\n")
    ctl_frames = [pd.read_csv(q) for q in sorted((OUT / "smoke").glob("*/controls.csv"), key=lambda q: q.stat().st_mtime)]
    if ctl_frames:
        ctl = pd.concat(ctl_frames, ignore_index=True).drop_duplicates(["target", "runner"], keep="last")
        A("\n**Environment-drift controls (seed 42, unmodified final recipes run through the same harness, compared with the stored "
          "final-model arrays `seed_analysis_results/<target>/per_seed_preds.npy[0]`):**\n")
        A(md_table(ctl))
        A("\nIf the control does not match the stored final-model seed-42 predictions, the stored final arrays were produced in a different "
          "software environment (`requirements.txt` is unpinned; original versions cannot be recovered), and every variant-vs-final "
          "difference below includes that environment difference as well as the component effect.\n")
    dc = OUT / "determinism_check.json"
    if dc.exists():
        A("\n**Worker-count independence (seed 42, same arrays compared bit for bit):**\n")
        d = json.loads(dc.read_text())
        A(md_table(pd.DataFrame([dict(comparison=k, max_abs_diff=v["max_abs_diff"], bit_identical=v["bit_identical"])
                                 for k, v in d.items()]), nd=6))
    # --- 3. seconds
    A("\n## 3. Seconds per seed (full run, 10 concurrent workers sharing one MIG slice; smoke-test timings in section 2)\n")
    rl = []
    for (t, slug), v in variants.items():
        p = v["dir"] / "run_log.csv"
        if p.exists():
            r = pd.read_csv(p)
            ok = r[r.status == "ok"]
            rl.append(dict(target=t, variant=slug, seeds_logged=len(ok), sec_mean=ok.seconds.mean(), sec_min=ok.seconds.min(),
                           sec_max=ok.seconds.max(), failed=int((r.status != "ok").sum()),
                           device=ok.device.iloc[0] if len(ok) else ""))
    A(md_table(pd.DataFrame(rl), nd=1))
    # --- 4. seeds
    A("\n## 4. Seeds completed per variant\n")
    A(md_table(summary[summary.variant != "final"][["target", "variant", "status", "n_seeds", "seeds"]]))
    if any(len(v["seeds"]) < 10 for v in variants.values()):
        A("\n**WARNING: at least one run variant has fewer than 10 seeds.**\n")
    # --- 5. results
    A("\n## 5. Results (only variants that were run)\n")
    A("Seed mean ± SD (ddof=1) over seeds and the metrics of the 10-seed ensemble; `diff` = variant - final, paired by seed index.\n")
    tab = build_table(summary, vs)
    A(md_table(tab[tab.n_seeds > 0] if "n_seeds" in tab else tab))
    A("\n### Interpretation notes\n")
    A("- Seeds share one fixed test set, so the seed-level tests capture training randomness only. The compound-level bootstrap "
      "captures test-set sampling. Both are reported and not merged.\n"
      "- Pairing 'by seed index' is a convenience: the variant and the final model are different architectures, so seed 42 does not "
      "mean the same random numbers across them. It is paired by seed index, nothing stronger.\n"
      "- There are many comparisons (variants x metrics). The text here uses 'CI excludes zero' / 'CI overlaps zero' and reports the "
      "Holm-adjusted p-values (`p_holm`, over the variants of the same target and metric, from the exact Wilcoxon p-values; "
      "`p_holm_ttest` from the paired t-test).\n"
      "- Bootstrap: B = 10,000, `np.random.default_rng(0)`, one resampled-index matrix per target shared by every variant and the final "
      "model, ensemble metrics recomputed inside each draw; 'ens_p_variant_better' is the share of draws in which the variant's metric is "
      "better than the final's.\n")
    A("\n### Component effects per target: ensemble-level 95% CI of (variant - final)\n")
    for t in TARGETS:
        g = vs[vs.target == t] if len(vs) else vs
        if g is None or len(g) == 0:
            continue
        A(f"\n**{t}** (n_test={len(finals[t]['y'])})\n")
        keep = g[["variant", "metric", "n_seeds", "diff_mean", "diff_in_final_sd", "n_seeds_variant_better", "p_wilcoxon", "p_holm",
                  "ens_diff", "ens_ci95_lo", "ens_ci95_hi", "ens_p_variant_better", "ens_ci_flag"]]
        A(md_table(keep))
        for slug, gg in g.groupby("variant"):
            ex = list(gg[gg.ens_ci_flag == "excludes_zero"].metric)
            ov = list(gg[gg.ens_ci_flag == "overlaps_zero"].metric)
            A(f"\n- `{slug}`: ensemble CI excludes zero for: {', '.join(ex) if ex else 'none'}; overlaps zero for: {', '.join(ov) if ov else 'none'}.")
        A("")
    # --- 6. ranking / is final best
    A("\n## 6. Is the final model the best variant on each metric?\n")
    A("By seed mean, and by the 10-seed ensemble metric, among the variants that were run for that target.\n")
    brow = []
    for t in TARGETS:
        g = summary[(summary.target == t) & (summary.n_seeds > 0)]
        if len(g) < 2:
            continue
        for m in metrics_of(t):
            asc = not HIGHER(m)
            best_mean = g.sort_values(f"{m}_mean", ascending=asc).iloc[0]["variant"]
            best_ens = g.sort_values(f"{m}_ens", ascending=asc).iloc[0]["variant"]
            brow.append(dict(target=t, metric=m, best_by_seed_mean=best_mean, final_is_best_by_seed_mean=(best_mean == "final"),
                             best_by_ensemble=best_ens, final_is_best_by_ensemble=(best_ens == "final")))
    bdf = pd.DataFrame(brow)
    A(md_table(bdf))
    for t in TARGETS:
        b = bdf[bdf.target == t] if len(bdf) else bdf
        if len(b):
            no = b[(~b.final_is_best_by_seed_mean)]
            if len(no):
                A(f"\n- {t}: the final model is NOT the best variant by seed mean on: {', '.join(no.metric)}.")
            no = b[(~b.final_is_best_by_ensemble)]
            if len(no):
                A(f"- {t}: the final model is NOT the best variant by ensemble metric on: {', '.join(no.metric)}.")
    # --- 7. deviations / warnings
    A("\n## 7. Deviations, failures and warnings\n")
    for n in notrun:
        A(f"- BLOCKED / not run: {n['target']} / {n['name']} - {n['reason']}")
    for key, st in sel.items():
        if st.get("status") in ("RUN_WITH_CAVEAT", "RECONSTRUCTED_UNVERIFIED", "RECONSTRUCTED_VERIFIED"):
            A(f"- {key}: {st['status']} - {st.get('reason', '')}")
    A("- `rdkit` is not installed in this environment and is not in `requirements.txt`; the pKa featurization needs it. rdkit 2026.03.6 "
      "was installed with `--no-deps` into `ablation_10seed/_deps/` (not the system environment). The rdkit version behind the stored pKa "
      "final arrays is unknown.")
    A("- Reconstructed variants (see section 8) are inferred from the variant names; no original code for them was available. The seed-42 gate "
      "shows how closely each reproduces its CSV row.")
    A("- Scripts live inside `ablation_10seed/` (the only place writing was allowed); run them as `python ablation_10seed/<script>.py`.")
    A("- Per-seed metrics column order follows the task spec (GMFE before the within-fold columns); the existing CL/VDss CSVs put GMFE last. "
      "A `variant` column is appended.")
    A("- Determinism: fixed 2 CPU threads per worker regardless of worker count, `torch.use_deterministic_algorithms(True, warn_only=True)`, "
      "cudnn deterministic. The original runs did not set these; CL/VDss still reproduce the CSV rows to |dR2| <= 0.0004.")
    A("- `run.log` lines appear twice when the runner is started with `nohup ... >> run.log` (the file handler and the redirected stdout both write to it); cosmetic.")
    A("- Package `src` and `data` were staged byte-identically in `/dev/shm/pk_ro` (size/hash verified). Candidates `fuA`/`fuC1` use "
      "hard-coded absolute paths inside `mfmn/experiments/*.py` and read `artifacts/` from the project directory instead.")
    A("\n## 8. Reconstructed variants: what exactly was changed\n")
    hp = man.get("hyperparameters_by_variant", {})
    for key, h in hp.items():
        if not h:
            continue
        for r, v in h.items():
            if isinstance(v, dict) and (v.get("RECONSTRUCTED") or v.get("modification")):
                A(f"- `{key}` via `{r}`: " + json.dumps({k: v[k] for k in ("modification", "candidate", "loop", "recipe", "model") if k in v}, default=str))
    A("\n### Checks\n")
    A(f"All assertions in this script passed: **{check_ok}**\n")
    A("\n".join(f"- {m}" for m in check_msgs))
    A("\n### Final models\n")
    A("Existing `seed_analysis_results/<target>/per_seed_preds.npy` arrays were copied (not retrained) to "
      "`ablation_10seed/<target>/final/`; per-seed and cumulative metrics recomputed here match the existing CSVs to 1e-6 for all five targets.")
    A("\n### Sources and versions\n")
    A("See `manifest.json` (source file and cell for every variant, SHA-256 of every source file, library/device versions, hyperparameters "
      "read from the code).")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=B_DEFAULT)
    args = ap.parse_args()
    TAB.mkdir(exist_ok=True)
    sel, man, finals, variants, notrun = discover()
    if not finals:
        sys.exit("no final-model arrays found: run run_ablation_10seed.py first")
    for (t, slug), v in variants.items():
        n = len(v["seeds"])
        msg = f"{t}/{slug}: n_seeds={n}" + ("  WARNING: fewer than 10 seeds" if n < 10 else "")
        print(msg)
    for n in notrun:
        print(f"{n['target']}/{n['slug']}: not run ({n['status']})")
    ok, msgs = run_checks(finals, variants)
    (TAB / "checks.json").write_text(json.dumps(dict(all_passed=bool(ok), messages=msgs), indent=2))
    summary = build_summary(finals, variants, notrun)
    summary.to_csv(TAB / "ablation_10seed_summary.csv", index=False)
    vs = build_vs_final(finals, variants, args.boot)
    vs.to_csv(TAB / "ablation_10seed_vs_final.csv", index=False)
    build_table(summary, vs).to_csv(TAB / "ablation_10seed_table.csv", index=False)
    ranking = build_ranking(summary)
    ranking.to_csv(TAB / "ablation_10seed_ranking.csv", index=False)
    (OUT / "report.md").write_text(build_report(sel, man, summary, vs, ranking, notrun, finals, variants, ok, msgs))
    print("wrote", [p.name for p in sorted(TAB.glob("*.csv"))], "report.md; checks passed:", ok)
    if not ok:
        sys.exit("CHECK FAILURE - see tables/checks.json")


if __name__ == "__main__":
    main()
