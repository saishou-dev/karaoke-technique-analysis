# -*- coding: utf-8 -*-
"""10/14 版 G1（dataset/G1_events_20261014_wao.csv・読取器 v0.8e ＋ 目視追記 35 件）を統合台帳 dataset/G1_events_all.csv へ結合する（2026-10-15）。
   merge_g1_all_20261006.py のコピー。列対応は同じ（session=20261014／take＝wav の stem／idx列＝同値／audio_time_sec＝同期済み）。
   10/14 は全曲 WAO（Hearts）の1機種なので入力は1本。目視追記の note 列は統合台帳には持たせない（セッション台帳に残る＝過去セッションと同じ扱い）。
   退避＝G1_events_all_pre20261014.bak.csv（初回のみ）。冪等（すでに 20261014 があれば何もしない）。

   検算（標準出力）＝行数の前後・セッション別件数・技法別件数・**既存行の md5 一致**・重複・欠損・take/動画の数。
"""
import os, shutil, hashlib, pandas as pd

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
ALL = os.path.join(ROOT, "dataset", "G1_events_all.csv")
BAK = ALL[:-4] + "_pre20261014.bak.csv"
SRCS = [os.path.join(ROOT, "dataset", "G1_events_20261014_wao.csv")]
SESSION = 20261014

A = pd.read_csv(ALL, encoding="utf-8-sig"); n0 = len(A)
if (A["session"] == SESSION).any():
    print("already merged:", int((A["session"] == SESSION).sum())); raise SystemExit(0)
G = pd.concat([pd.read_csv(s, encoding="utf-8-sig") for s in SRCS], ignore_index=True)
assert G["pts_source"].isin(["pts_bands_20261014"]).all(), "pts_source が揃っていない: " + str(G["pts_source"].unique())
assert G["wav"].notna().all() and G["audio_time_sec"].notna().all(), "wav/audio_time_sec に欠損"
N = pd.DataFrame(dict(session=SESSION, video=G["video"], counter=G["counter"], count_value=G["count_value"].astype(int), delta=G["delta"].astype(int),
                      video_time_sec=G["video_time_sec"], stable_read_time_sec=G["stable_read_time_sec"], take=G["wav"].str[:-4],
                      audio_time_sec=G["audio_time_sec"], video_time_idx_sec=G["video_time_sec"], stable_read_time_idx_sec=G["stable_read_time_sec"],
                      multi_increment=G["multi_increment"].astype(int)))[list(A.columns)]
assert N.isna().sum().sum() == 0, "新規行に欠損"


def rowmd5(df):
    s = df.astype(object).apply(lambda r: hashlib.md5("\x1f".join(map(str, r.tolist())).encode("utf-8")).hexdigest(), axis=1)
    return sorted(s) if len(s) else []


h0 = rowmd5(A)
if not os.path.exists(BAK): shutil.copy(ALL, BAK)
out = pd.concat([A, N], ignore_index=True); out.to_csv(ALL, index=False, encoding="utf-8-sig")
chk = pd.read_csv(ALL, encoding="utf-8-sig")
h1 = rowmd5(chk[chk.session != SESSION])
print(f"rows {n0} -> {len(chk)} (+{len(N)})")
print(chk.groupby("session").size().to_string())
print("既存行 md5 一致 =", h0 == h1, f"（既存 {len(h0)} 行 / 結合後の非 {SESSION} 行 {len(h1)} 行）")
new = chk[chk.session == SESSION]
print("dup(session,take,counter,count_value) new =", int(new.duplicated(["session", "take", "counter", "count_value"]).sum()),
      " all =", int(chk.duplicated(["session", "take", "counter", "count_value"]).sum()))
print("dup(video,counter,count_value) all =", int(chk.duplicated(["video", "counter", "count_value"]).sum()))
print("na new =", int(new.isna().sum().sum()))
print(f"{SESSION} videos =", new["video"].nunique(), " takes =", new["take"].nunique(), " counters =", new["counter"].value_counts().to_dict())
print("count_value 単調・重複・時刻逆行の異常 =", [
    (v, c) for (v, c), g in new.groupby(["video", "counter"])
    if g.sort_values("count_value")["count_value"].duplicated().any()
    or not g.sort_values("count_value")["video_time_sec"].is_monotonic_increasing])
print("backup", BAK, os.path.getsize(BAK))
