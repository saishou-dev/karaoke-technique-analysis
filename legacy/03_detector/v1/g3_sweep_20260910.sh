#!/usr/bin/env bash
cd "/g/研究アーカイブ/成果物/動画解析"
export PYTHONIOENCODING=utf-8
for conf in 0.0; do for md in 0.08 0.12; do for cw in balanced none; do
  tag="c${conf}_d${md}_${cw}"
  G3_CONF=$conf G3_MINDUR=$md G3_CW=$cw G3_TAG=$tag python scripts/g3_train_v1.py > "scripts/g3_sweep_${tag}.log" 2>&1
  echo "done $tag $(grep -c 'done' scripts/g3_sweep_${tag}.log)"
done; done; done
echo SWEEP_FINISHED
