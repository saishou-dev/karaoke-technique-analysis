# -*- coding: utf-8 -*-
"""9/15 再走査版 G1（dataset/G1_events_20260915_rescan.csv・目視追記＋annotate 済み）を統合台帳 dataset/G1_events_all.csv に結合する（2026-09-15 23:1x）。
   列対応（9/12 の 9/11 結合と同じ）：session=20260915／take＝wav の stem（9/15 はスマホ音声なので動画名と同じ）／
   video_time_idx_sec・stable_read_time_idx_sec＝video_time_sec・stable_read_time_sec と同値（帯pts時刻・index 換算の旧値は無い）／
   audio_time_sec＝G1 の audio_time_sec（同期オフセット 0）。退避＝G1_events_all_pre20260915.bak.csv（初回のみ）。冪等（session 20260915 が既にあれば何もしない）。
   検証＝行数・session 別件数・重複ゼロ（session+take+counter+count_value・session+take+counter+video_time_sec）。"""
import os, shutil, pandas as pd
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; ALL = os.path.join(ROOT, "dataset", "G1_events_all.csv"); BAK = ALL[:-4] + "_pre20260915.bak.csv"
SRC = os.path.join(ROOT, "dataset", "G1_events_20260915_rescan.csv")
A = pd.read_csv(ALL, encoding="utf-8-sig"); n0 = len(A)
if (A["session"] == 20260915).any(): print("already merged:", (A["session"] == 20260915).sum()); raise SystemExit(0)
G = pd.read_csv(SRC, encoding="utf-8-sig")
assert (G["pts_source"] == "pts_bands_20260915").all() and (G["audio_source"] == "phone").all() and G["wav"].notna().all(), "前提列が揃っていない"
N = pd.DataFrame(dict(session=20260915, video=G["video"], counter=G["counter"], count_value=G["count_value"].astype(int), delta=G["delta"].astype(int),
                      video_time_sec=G["video_time_sec"], stable_read_time_sec=G["stable_read_time_sec"], take=G["wav"].str[:-4],
                      audio_time_sec=G["audio_time_sec"], video_time_idx_sec=G["video_time_sec"], stable_read_time_idx_sec=G["stable_read_time_sec"],
                      multi_increment=G["multi_increment"].astype(int)))[list(A.columns)]
if not os.path.exists(BAK): shutil.copy(ALL, BAK)
out = pd.concat([A, N], ignore_index=True); out.to_csv(ALL, index=False, encoding="utf-8-sig")
chk = pd.read_csv(ALL, encoding="utf-8-sig")
print(f"rows {n0} -> {len(chk)} (+{len(N)})"); print(chk.groupby("session").size().to_string())
print("dup(session,take,counter,count_value) =", chk.duplicated(["session", "take", "counter", "count_value"]).sum())
print("dup(session,take,counter,video_time_sec) =", chk.duplicated(["session", "take", "counter", "video_time_sec"]).sum())
print("9/15 videos =", N["video"].nunique(), " multi_increment =", int(N["multi_increment"].sum()), " counters =", N["counter"].value_counts().to_dict())
print("backup", BAK)
