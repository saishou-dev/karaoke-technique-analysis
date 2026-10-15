# -*- coding: utf-8 -*-
"""目視追記（found_visual_20261015）を統合台帳 dataset/G1_events_all.csv に反映する（2026-10-15・第2版）。

merge_g1_all_*.py は「新しいセッションを足す」道具で、既に入っているセッションには使えない。
本スクリプトは **新規行だけを足し、追記が入ったカウンターの delta 連鎖だけを組み直す**。

第1版は「追記が入った video の行をセッション G1 から作り直して差し替える」方式だったが、それだと
G1_events_all 固有の列（video_time_idx_sec・stable_read_time_idx_sec・audio_time_sec）が壊れる
（10/15 実測＝9/11 の 47 行で idx 列が別基準＝旧フレーム/fps 時刻だった）。そこで新規行の
これらの列は、**同じ video の既存行から線形補間**して作る（idx と video_time_sec が一致する
セッションでは補間しても一致する）。take も同じ video の既存行から引く（1 video に複数 take が
紐づく 9/11 Midnight のような例があるため、take ごとに 1 行ずつ足す）。

検算＝(1) 既存行の md5 が 1 行も欠けないこと（delta 連鎖を組み直した行だけは「変更」として列挙）
      (2) 行数・技法別件数・セッション別件数の前後 (3) 重複の前後
20261014 には触らない（本人指示 10/15・別件調査中）。

使い方: python merge_found_all_20261015.py [--dry]
"""
import os, sys, hashlib, shutil, numpy as np, pandas as pd

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
ALL = os.path.join(DS, "G1_events_all.csv")
BAK = os.path.join(DS, "G1_events_all_pre_found_20261015.bak.csv")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import SESS, ORDER  # noqa: E402

NOTE = "found_visual_20261015"
FORBIDDEN_SESSIONS = {20261014}


def rowhash(df):
    s = df.astype(str).apply(lambda r: "\x1f".join(map(str, r.tolist())), axis=1)
    return s.map(lambda x: hashlib.md5(x.encode()).hexdigest()) if len(s) else s


def interp(xs, ys, x):
    """既存行の (video_time_sec -> 目的の列) を線形補間。範囲外は端の差分を平行移動。"""
    o = np.argsort(xs); xs = np.asarray(xs)[o]; ys = np.asarray(ys)[o]
    if x <= xs[0]:
        return float(ys[0] + (x - xs[0]))
    if x >= xs[-1]:
        return float(ys[-1] + (x - xs[-1]))
    return float(np.interp(x, xs, ys))


def main(dry=False):
    A = pd.read_csv(ALL, encoding="utf-8-sig")
    cols = list(A.columns)
    n0 = len(A)
    print("before rows =", n0)

    new = []
    for key in ORDER:
        cfg = SESS[key]
        s = cfg["session"]
        assert s not in FORBIDDEN_SESSIONS, s
        g = pd.read_csv(os.path.join(DS, cfg["g1"]), encoding="utf-8-sig")
        if "note" not in g.columns:
            continue
        add = g[g["note"] == NOTE]
        for _, r in add.iterrows():
            v = r["video"]
            ex = A[(A["session"] == s) & (A["video"] == v)]
            assert len(ex), f"統合台帳に {s} {v} が無い"
            takes = sorted(ex["take"].unique())
            vt = float(r["video_time_sec"]); st = float(r["stable_read_time_sec"])
            for tk in takes:
                e = ex[ex["take"] == tk]
                new.append(dict(session=s, video=v, counter=r["counter"],
                                count_value=int(r["count_value"]), delta=int(r["delta"]),
                                video_time_sec=vt, stable_read_time_sec=st, take=tk,
                                audio_time_sec=round(interp(e["video_time_sec"], e["audio_time_sec"], vt), 3),
                                video_time_idx_sec=round(interp(e["video_time_sec"], e["video_time_idx_sec"], vt), 3),
                                stable_read_time_idx_sec=round(interp(e["stable_read_time_sec"], e["stable_read_time_idx_sec"], st), 3),
                                multi_increment=int(r["multi_increment"])))
    if not new:
        print("反映するものがありません"); return
    N = pd.DataFrame(new)[cols]
    assert N.isna().sum().sum() == 0, "新規行に欠損"
    print(f"新規行 = {len(N)}")
    print(N.groupby(["session", "counter"]).size().to_string())

    out = pd.concat([A, N], ignore_index=True)
    # delta 連鎖の組み直し（新規行が入った (session, video, take, counter) だけ）
    changed = []
    for (s, v, tk, c) in set(zip(N["session"], N["video"], N["take"], N["counter"])):
        m = (out["session"] == s) & (out["video"] == v) & (out["take"] == tk) & (out["counter"] == c)
        sub = out.loc[m].sort_values("count_value")
        idx = sub.index.tolist(); vals = sub["count_value"].astype(int).tolist()
        prev = vals[0] - int(sub["delta"].iloc[0])      # 読取器の初期値（先頭行の delta はそのまま）
        for i, ix in enumerate(idx):
            d = vals[i] - prev
            if int(out.at[ix, "delta"]) != d:
                changed.append((s, v, c, vals[i], int(out.at[ix, "delta"]), d))
            out.at[ix, "delta"] = d
            out.at[ix, "multi_increment"] = int(d > 1)
            prev = vals[i]
    out["_s"] = out["session"]
    out = out.sort_values("_s", kind="stable").drop(columns="_s").reset_index(drop=True)

    h_new = set(rowhash(out))
    h_old = set(rowhash(A))
    missing = h_old - h_new
    print("検算1 既存行の md5 欠落 =", len(missing), f"（delta を組み直した {len(changed)} 行と一致するはず）")
    for r in changed:
        print("   delta 変更:", r)
    assert len(missing) == len(set(changed)) or len(missing) <= len(changed), "想定外の既存行の変化"
    print(f"検算2 rows {n0} -> {len(out)} (+{len(out)-n0})")
    cmp = pd.concat([A.groupby("counter").size().rename("before"), out.groupby("counter").size().rename("after")], axis=1)
    cmp["diff"] = cmp["after"] - cmp["before"]; print(cmp.to_string())
    cs = pd.concat([A.groupby("session").size().rename("before"), out.groupby("session").size().rename("after")], axis=1)
    cs["diff"] = cs["after"] - cs["before"]; print(cs.to_string())
    for k in (["session", "take", "counter", "count_value"], ["session", "video", "take", "counter", "count_value"]):
        print("検算3 dup", k, A.duplicated(k).sum(), "->", out.duplicated(k).sum())
    if dry:
        print("dry-run（書き込みなし）"); return
    if not os.path.exists(BAK):
        shutil.copy(ALL, BAK); print("backup", os.path.basename(BAK), os.path.getsize(BAK))
    out.to_csv(ALL, index=False, encoding="utf-8-sig")
    print("WROTE", ALL, len(out))


if __name__ == "__main__":
    main("--dry" in sys.argv)
