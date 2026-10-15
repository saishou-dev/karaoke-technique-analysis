# -*- coding: utf-8 -*-
"""台帳ベースの突合（2026-10-15）。

これまでの verify_*.csv は「読取器の最終値 vs DAM の採点 CSV」で、目視追記（found_visual_*）を反映しない。
本スクリプトは **セッションの G1（台帳）の max(count_value) vs DAM の採点 CSV** を突合する＝台帳の完全性そのもの。
正解値は既存 verify_*.csv の csv_<counter> 列をそのまま使う（DAM CSV の再取得をしない＝正本1か所）。

使い方: python verify_g1_20261015.py [<sess-key> ...]   （既定＝全キー）
出力: dataset/found_20261015/verify_g1_<tag>.csv（tag＝実行時に --tag で指定・既定 now）＋標準出力にセッション別一致率
"""
import os, sys, csv, pandas as pd

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import SESS, ORDER, WAO_COUNTERS, AI_COUNTERS  # noqa: E402


def verify(key):
    cfg = SESS[key]
    counters = WAO_COUNTERS if cfg["reader"] == "wao" else AI_COUNTERS
    ver = list(csv.DictReader(open(os.path.join(DS, cfg["verify"]), encoding="utf-8-sig")))
    g = pd.read_csv(os.path.join(DS, cfg["g1"]), encoding="utf-8-sig")
    mx = g.groupby(["video", "counter"])["count_value"].max()
    rows = []
    for r in ver:
        name = r["video"]; d = []
        rec = dict(session=key, video=name)
        for c in counters:
            tgt = int(r["csv_" + c])
            got = int(mx.get((name, c), 0))
            rec["g1_" + c] = got; rec["csv_" + c] = tgt
            if got != tgt:
                d.append(f"{c}{got-tgt:+d}")
        rec["status"] = "PERFECT" if not d else "DIFF"
        rec["diff"] = ";".join(d)
        rec["over"] = int(any(rec["g1_" + c] > rec["csv_" + c] for c in counters))
        rec["missing"] = sum(max(0, rec["csv_" + c] - rec["g1_" + c]) for c in counters)
        rows.append(rec)
    df = pd.DataFrame(rows)
    ok = (df.status == "PERFECT").sum()
    print(f"{key:12s} PERFECT {ok}/{len(df)}  missing_events={df.missing.sum()}  over_songs={df.over.sum()}")
    for _, r in df[df.status == "DIFF"].iterrows():
        print(f"    {r['video'][:40]:42s} {r['diff']}")
    return df


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = "now"
    for a in sys.argv[1:]:
        if a.startswith("--tag="):
            tag = a.split("=", 1)[1]
    keys = args or ORDER
    out = pd.concat([verify(k) for k in keys], ignore_index=True)
    os.makedirs(os.path.join(DS, "found_20261015"), exist_ok=True)
    p = os.path.join(DS, "found_20261015", f"verify_g1_{tag}.csv")
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print("WROTE", p)
    s = out.groupby("session").agg(n=("video", "size"), perfect=("status", lambda x: (x == "PERFECT").sum()), missing=("missing", "sum"))
    print(s.to_string())
