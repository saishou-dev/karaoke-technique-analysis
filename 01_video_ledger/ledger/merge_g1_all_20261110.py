# -*- coding: utf-8 -*-
"""【2026-11-10 版＝merge_g1_all_20261104.py の複製・セッションとパスだけ変更（Opus）】11/10 版 G1（dataset/G1_events_20261110_wao.csv・v0.8d＋目視追記 2026-11-10）を統合台帳 dataset/G1_events_all.csv へ結合する（2026-11-09・kaishu20261110-merge）。
   merge_g1_all_20261021.py のコピー（セッションとパスを変更）。列対応は同じ（session=20261110／take＝wav の stem／idx列＝同値／audio_time_sec＝同期済み）。
   11/4 は全曲 WAO（Hearts）の1機種なので入力は1本。目視追記の note 列は統合台帳には持たせない（セッション台帳に残る＝過去セッションと同じ扱い）。
   sync の旗（sync_flag）も統合台帳には持たせない＝列を足すと既存行のバイトが変わるため。旗は dataset/sync/sync_flags_20261110.csv（sync_flags_20261110.py）。

   20261021 版からの変更＝**書き方**。既存の版は pandas で全体を書き直していた（行単位の md5 で検算）。
   今回は「既存ファイルのバイト列そのまま＋新しい行を末尾に足す」で書く＝既存行は 1 バイトも変わらない。
   検算＝結合後ファイルの先頭 len(旧) バイトの md5＝旧ファイル全体の md5／行単位の md5（20261021 版と同じ）／行数の前後。

   退避＝G1_events_all_pre20261110.bak.csv（初回のみ）。冪等（すでに 20261110 があれば何もしない）。
"""
import os, io, shutil, hashlib, pandas as pd

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
ALL = os.path.join(ROOT, "dataset", "G1_events_all.csv")
BAK = ALL[:-4] + "_pre20261110.bak.csv"
SRCS = [os.path.join(ROOT, "dataset", "G1_events_20261110_wao.csv")]
SESSION = 20261110


def md5b(b):
    return hashlib.md5(b).hexdigest()


def rowmd5(df):
    s = df.astype(object).apply(lambda r: hashlib.md5("\x1f".join(map(str, r.tolist())).encode("utf-8")).hexdigest(), axis=1)
    return sorted(s) if len(s) else []


raw0 = open(ALL, "rb").read()
A = pd.read_csv(ALL, encoding="utf-8-sig"); n0 = len(A)
if (A["session"] == SESSION).any():
    print("already merged:", int((A["session"] == SESSION).sum())); raise SystemExit(0)
assert raw0.endswith(b"\r\n"), "末尾が CRLF でない＝足し方を決められない"
n_lf = raw0.count(b"\n"); n_crlf = raw0.count(b"\r\n")
assert n_lf == n_crlf, f"改行が混在（LF {n_lf}・CRLF {n_crlf}）"
G = pd.concat([pd.read_csv(s, encoding="utf-8-sig") for s in SRCS], ignore_index=True)
assert G["pts_source"].isin(["pts_bands_20261110"]).all(), "pts_source が揃っていない: " + str(G["pts_source"].unique())
assert G["wav"].notna().all() and G["audio_time_sec"].notna().all(), "wav/audio_time_sec に欠損"
N = pd.DataFrame(dict(session=SESSION, video=G["video"], counter=G["counter"], count_value=G["count_value"].astype(int), delta=G["delta"].astype(int),
                      video_time_sec=G["video_time_sec"], stable_read_time_sec=G["stable_read_time_sec"], take=G["wav"].str[:-4],
                      audio_time_sec=G["audio_time_sec"], video_time_idx_sec=G["video_time_sec"], stable_read_time_idx_sec=G["stable_read_time_sec"],
                      multi_increment=G["multi_increment"].astype(int)))[list(A.columns)]
assert N.isna().sum().sum() == 0, "新規行に欠損"

h0 = rowmd5(A)
if not os.path.exists(BAK): shutil.copy(ALL, BAK)
assert md5b(open(BAK, "rb").read()) == md5b(raw0), "退避の md5 が元と違う"
buf = io.StringIO(); N.to_csv(buf, index=False, header=False, lineterminator="\r\n")
add = buf.getvalue().encode("utf-8")
open(ALL, "wb").write(raw0 + add)

raw1 = open(ALL, "rb").read()
chk = pd.read_csv(ALL, encoding="utf-8-sig")
h1 = rowmd5(chk[chk.session != SESSION])
print(f"rows {n0} -> {len(chk)} (+{len(N)})")
print(chk.groupby("session").size().to_string())
print("先頭バイト md5 一致 =", md5b(raw1[:len(raw0)]) == md5b(raw0), f"（旧 {md5b(raw0)}・{len(raw0)} バイト／新 {md5b(raw1)}・{len(raw1)} バイト）")
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
print("新規行が元の G1 と値で一致 =", all((new[c].astype(str).values == N[c].astype(str).values).all() for c in N.columns))
print("backup", BAK, os.path.getsize(BAK), md5b(open(BAK, "rb").read()))
