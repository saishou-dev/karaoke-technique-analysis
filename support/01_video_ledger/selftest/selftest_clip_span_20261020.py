# -*- coding: utf-8 -*-
"""clip_span の自己テスト（2026-10-20・評価の既定を exclude → exclude_band に変えたときに作成）。

確かめること
  1. 既定が exclude_band であること（引数を省いたときの挙動が exclude_band と一致）。
  2. 帯消失列（band_last_pts）が**空**の曲は exclude_band でも exclude と**同じ結果**（＝旧挙動へフォールバック）。
     実データ＝8/07 の 18 曲（帯動画が残っておらず band_* 列が空のまま）。
  3. 帯消失列が**埋まっている**曲は exclude_band のほうが終端が早い（または同じ）＝決して広がらない。
  4. 表に行が無い曲は span をそのまま返す。
  5. keep は何もしない。
使い方: python scripts/selftest_clip_span_20261020.py
"""
import os, sys, csv, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from song_valid_range_20261015 import clip_span, load, OUT  # noqa: E402

rows = list(csv.DictReader(open(OUT, encoding="utf-8-sig")))
by_video = {r["video"]: r for r in rows}
empty = [r for r in rows if r["first_frame_pts"] != "" and not r["band_last_pts"]]
filled = [r for r in rows if r["first_frame_pts"] != "" and r["band_last_pts"]]
ng = 0


def check(name, cond, extra=""):
    global ng
    print(("PASS " if cond else "FAIL ") + name + (("  " + extra) if extra else ""))
    if not cond:
        ng += 1


SPAN = [0.0, 10_000.0]   # 十分広い span＝必ず表の側で切られる

# 1. 既定が exclude_band
v = filled[0]["video"]
check("1 既定が exclude_band",
      clip_span(v, SPAN, 0.0) == clip_span(v, SPAN, 0.0, "exclude_band"),
      f"{v} 既定={clip_span(v, SPAN, 0.0)[0]}")

# 2. 帯列が空の曲は exclude と同じ（8/07 など）
same = [r["video"] for r in empty
        if clip_span(r["video"], SPAN, 0.0, "exclude_band")[0] == clip_span(r["video"], SPAN, 0.0, "exclude")[0]]
ses = collections.Counter(r["session"] for r in empty)
check("2 帯列が空の曲は exclude と同じ結果（フォールバック）",
      len(same) == len(empty) and len(empty) > 0,
      f"{len(same)}/{len(empty)} 曲一致  セッション内訳={dict(ses)}")

# 3. 帯列が埋まっている曲は終端が早い or 同じ（広がらない）
bad = [r["video"] for r in filled
       if clip_span(r["video"], SPAN, 0.0, "exclude_band")[0][1] > clip_span(r["video"], SPAN, 0.0, "exclude")[0][1] + 1e-9]
shorter = [r["video"] for r in filled
           if clip_span(r["video"], SPAN, 0.0, "exclude_band")[0][1] < clip_span(r["video"], SPAN, 0.0, "exclude")[0][1] - 1e-9]
check("3 帯列が埋まった曲で終端が広がらない", not bad,
      f"狭まる {len(shorter)}/{len(filled)} 曲・広がる {len(bad)} 曲")

# 4. 表に無い曲は素通し
check("4 表に無い曲は span をそのまま返す",
      clip_span("__NO_SUCH_VIDEO__", SPAN, 0.0, "exclude_band") == (SPAN, False))

# 5. keep は何もしない
check("5 keep は何もしない", clip_span(v, SPAN, 0.0, "keep") == (SPAN, False))

print(f"\n表 {os.path.basename(OUT)}: {len(rows)} 曲（帯列あり {len(filled)}・帯列なし {len(empty)}）")
print("RESULT " + ("ALL PASS" if ng == 0 else f"{ng} FAIL"))
sys.exit(1 if ng else 0)
