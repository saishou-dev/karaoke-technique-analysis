#!/usr/bin/env python3
"""score_gran_z.py = 条件Z（全画面のフレーム）を含めて採点する（2026-10-30）

score_gran.py を書き換えずにそのまま使い、条件の並びに Z を足し、手法に Tesseract psm 11 を足すだけ。
採点の尺度・後処理（帯消失ガード→6コマ安定→減らない制約）・「1が11」型の数え方は score_gran.py と同一。
使い方: $PY score_gran_z.py --pred pred --out result --sub 4 --tag sub4z
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_gran as S   # noqa: E402

S.CONDS = ["A", "B", "C", "C2", "Z"]
if "tess_psm11" not in S.CAND_ORDER:
    S.CAND_ORDER.insert(S.CAND_ORDER.index("tess_psm8") + 1, "tess_psm11")
S.LABEL["tess_psm11"] = "Tesseract 5.5.0（psm 11・文字の検出あり・前処理あり）"

if __name__ == "__main__":
    sys.exit(S.main())
