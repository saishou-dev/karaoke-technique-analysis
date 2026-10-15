# -*- coding: utf-8 -*-
"""目視で増加を確認した欠落イベントをセッションの G1（台帳）に追記する・全セッション版（2026-10-15）。

append_found_20260915.py の一般化。追記する行は `dataset/found_20261015/<key>/append_list.csv`
（列 video,counter,value,event_frame,why）＝ found_exact_20261015.py の EXACT 行を **タイル画像で目視確認した分だけ** 残したもの。

- 退避＝<G1>_pre_found_20261015.bak.csv（初回のみ）。
- 冪等＝(video, counter, count_value) が既にあればその行は足さない。何も足さなければ G1 は書き換えない。
- 時刻＝`dataset/<pts>/<video>.csv` の pts_sec[event_frame]（既存行と同じ基準＝video_time_sec = pts[event_frame_est] を実測で確認済み）。
- delta＝そのカウンターの count_value を昇順に並べ直した連鎖（先頭行の delta は読取器の初期値を含むのでそのまま残す）。
- note 列＝found_visual_20261015（既存の found_visual_20260915 は残す）。

使い方: python append_found_20261015.py <key> [--dry]
"""
import os, sys, csv, shutil, pandas as pd

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
OUTROOT = os.path.join(DS, "found_20261015")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import SESS  # noqa: E402

NOTE = "found_visual_20261015"


def pts_of(cfg, video):
    p = os.path.join(DS, cfg["pts"], video + ".csv")
    return [float(r["pts_sec"]) for r in csv.DictReader(open(p, encoding="utf-8"))]


def run(key, dry=False):
    cfg = SESS[key]
    g1p = os.path.join(DS, cfg["g1"])
    bak = g1p[:-4] + "_pre_found_20261015.bak.csv"
    lst = os.path.join(OUTROOT, key, "append_list.csv")
    add = list(csv.DictReader(open(lst, encoding="utf-8-sig")))
    G = pd.read_csv(g1p, encoding="utf-8-sig")
    have = set(zip(G["video"], G["counter"], G["count_value"].astype(int)))
    rows = []
    for a in add:
        v, c, val, f = a["video"], a["counter"], int(a["value"]), int(a["event_frame"])
        if (v, c, val) in have:
            print(f"  skip(既存) {v[:24]} {c} {val}")
            continue
        pts = pts_of(cfg, v)
        t = round(pts[f], 3)
        r = dict(video=v, counter=c, count_value=val, delta=0, frame=f, event_frame_est=f,
                 video_time_sec=t, stable_read_time_sec=t, wav=v + ".wav", audio_time_sec=t,
                 multi_increment=0)
        if "pts_source" in G.columns:
            r["pts_source"] = cfg["pts_source"]
        if "audio_source" in G.columns:
            src = G.loc[G["video"] == v, "audio_source"].dropna()
            r["audio_source"] = src.iloc[0] if len(src) else ""
        r["note"] = NOTE
        rows.append(r)
        print(f"  ADD {v[:26]:28s} {c:10s} {val:3d} f={f} t={t:.3f}  {a.get('why','')}")
    if not rows:
        print(f"{key}: 追記なし（すべて既存）"); return 0
    if dry:
        print(f"{key}: dry-run {len(rows)} 行"); return len(rows)
    if not os.path.exists(bak):
        shutil.copy(g1p, bak); print("  backup", os.path.basename(bak))
    if "note" not in G.columns:
        G["note"] = ""
    A = pd.DataFrame(rows)
    out = pd.concat([G, A], ignore_index=True)
    # delta の連鎖を、新規行が入ったカウンターだけ組み直す
    for (v, c) in set((r["video"], r["counter"]) for r in rows):
        m = (out["video"] == v) & (out["counter"] == c)
        sub = out.loc[m].sort_values("count_value")
        idx = sub.index.tolist(); vals = sub["count_value"].astype(int).tolist()
        base = vals[0] - int(sub["delta"].iloc[0])     # 読取器の初期値（先頭行の delta はそのまま残す）
        prev = base
        for i, ix in enumerate(idx):
            d = vals[i] - prev
            out.at[ix, "delta"] = d
            out.at[ix, "multi_increment"] = int(d > 1)
            prev = vals[i]
    out = out.sort_values(["video", "video_time_sec", "counter"], kind="stable").reset_index(drop=True)
    out.to_csv(g1p, index=False, encoding="utf-8-sig")
    print(f"WROTE {g1p}  {len(G)} -> {len(out)}  (+{len(rows)})")
    return len(rows)


if __name__ == "__main__":
    key = sys.argv[1]
    run(key, "--dry" in sys.argv)
