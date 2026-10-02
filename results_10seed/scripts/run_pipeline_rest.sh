#!/bin/bash
# Runs after the pKa finals: (1) wait for run_finals.py, (2) pKa ablation vs the retrained finals, (3) ablation analysis,
# (4) baselines-vs-proposed comparison for all targets. Safe to re-run: every stage is resumable.
cd /workspace/Pharmacokinetics
LOG=retrain_10seed/pipeline.log
echo "$(date -Is) pipeline waiting for finals" >> $LOG
python - <<'PY' >> $LOG 2>&1
import json
for t in ("pKa_Acidic","pKa_Basic"):
    n=len(json.load(open(f"retrain_10seed/finals/{t}/seeds_done.json"))); print(t,"final seeds:",n)
    assert n==10, "pKa finals incomplete -- rerun: python retrain_10seed/run_finals.py"
PY
[ $? -eq 0 ] || { echo "$(date -Is) finals incomplete, stopping" >> $LOG; exit 1; }
echo "$(date -Is) starting pKa ablation" >> $LOG
ABL_OUT=retrain_10seed/ablation FINAL_ROOT=retrain_10seed/finals python ablation_10seed/run_ablation_10seed.py \
    --targets pKa_Acidic pKa_Basic --workers 2 --pka-concurrency 2 >> retrain_10seed/ablation/run_ablation.out 2>&1
echo "$(date -Is) ablation stage ended; analysing" >> $LOG
ABL_OUT=retrain_10seed/ablation FINAL_ROOT=retrain_10seed/finals python ablation_10seed/analyze_ablation_10seed.py >> $LOG 2>&1
python retrain_10seed/run_baselines_10seed.py --stage compare >> $LOG 2>&1
python retrain_10seed/run_finals.py --finalize-only >> $LOG 2>&1
echo "$(date -Is) PIPELINE COMPLETE" >> $LOG
