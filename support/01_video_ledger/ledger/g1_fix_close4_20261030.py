# -*- coding: utf-8 -*-
"""close4（2026-10-30）＝g1_edits_v2 に無い2つの操作だけを当てる補助（既存スクリプトは無変更）。
   op=drop     読み過ぎの偽イベント（採点CSVが0なのに読取器が1を出した等）の行を消す。消す前に画像で確かめたことを why に書く。
   op=multifix relabel で delta が 1 になった行の multi_increment を delta>1 の真偽に合わせ直す。
   op=mark     video の目視追記行（note が found_visual_ で始まる行・counter=* で全技法）の note 末尾に why の文字列を足す（採否の印）。
   編集リスト（utf-8-sig）列＝op,video,counter,value,why。冪等（対象行が無ければ skip）。退避＝<G1>.pre_<tag>_20261030.bak（初回のみ）。
   検算＝消した行／変えた行以外の md5 が変わらないこと。使い方: python g1_fix_close4_20261030.py <G1.csv> <edits.csv> --tag=<退避名>
"""
import os, sys, shutil, hashlib
import pandas as pd
g1p, ep = sys.argv[1], sys.argv[2]; tag = next((a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("--tag=")), "fix")
G0 = pd.read_csv(g1p, encoding="utf-8-sig"); G = G0.copy(); E = pd.read_csv(ep, encoding="utf-8-sig")
def h(df): return set(df.fillna("").astype(str).apply(lambda r: hashlib.md5("\x1f".join(r.tolist()).encode()).hexdigest(), axis=1))
drop_idx = []; chg = 0
for _, e in E.iterrows():
    m = (G.video == e.video) & (G.counter == e.counter) & (G.count_value.astype(int) == (int(float(e.value)) if pd.notna(e.value) and str(e.value).strip() != '' else -1))
    if e.op == "drop":
        if not m.any(): print("skip(済) drop", e.video[:24], e.counter, e.value); continue
        drop_idx += list(G[m].index); print("DROP", e.video[:24], e.counter, e.value, "frame", G.loc[m, "frame"].tolist(), "|", e.why)
    elif e.op == "multifix":
        if not m.any(): print("skip(対象なし) multifix", e.video[:24], e.counter, e.value); continue
        nv = (G.loc[m, "delta"].astype(int) > 1).astype(int)
        if (G.loc[m, "multi_increment"].astype(int) == nv).all(): print("skip(済) multifix", e.video[:24], e.counter, e.value); continue
        G.loc[m, "multi_increment"] = nv; chg += int(m.sum())
        G.loc[m, "note"] = G.loc[m, "note"].fillna("").astype(str).apply(lambda s: (s + "; " if s else "") + "multifix_20261030")
        print("MULTIFIX", e.video[:24], e.counter, e.value, "->", nv.tolist())
    elif e.op == "mark":
        mm = (G.video == e.video) & G.note.fillna("").astype(str).str.startswith("found_visual_") & ((G.counter == e.counter) | (e.counter == "*"))
        mm &= ~G.note.fillna("").astype(str).str.contains(str(e.why), regex=False)
        if not mm.any(): print("skip(済) mark", e.video[:24]); continue
        G.loc[mm, "note"] = G.loc[mm, "note"].astype(str) + "; " + str(e.why); chg += int(mm.sum()); print("MARK", e.video[:24], int(mm.sum()), "行", e.why)
    else: raise SystemExit("op は drop／multifix／mark: " + str(e.op))
G = G.drop(index=drop_idx).reset_index(drop=True)
bak = g1p + f".pre_{tag}_20261030.bak"
if not os.path.exists(bak): shutil.copy(g1p, bak)
G.to_csv(g1p, index=False, encoding="utf-8-sig")
print(f"rows {len(G0)} -> {len(G)}（drop {len(drop_idx)}・変更 {chg}）／ 消えた既存行 {len(h(G0) - h(G))}・新しい行 {len(h(G) - h(G0))}（＝drop＋変更のみなら正常）")
