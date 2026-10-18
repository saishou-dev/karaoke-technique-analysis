#!/usr/bin/env python3
"""merge_shards.py ＝ 分割実行した候補の予測と処理時間を1つにまとめる（2026-10-18）

分割（--shard）は「どのコマをどのプロセスが読むか」を変えるだけで、読む画像も設定も同じ。
処理時間は「1プロセス＝専用4コア」のまま測っているので、
  1,000コマあたり秒 ＝ （各分割の秒数の合計）÷（各分割のコマ数の合計）×1000
でそのまま比較できる（壁時計の時間だけが分割数ぶん短くなる）。

使い方: python merge_shards.py --pred <pred dir> --cand paddle_rec --nshards 4
"""
import argparse
import csv
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--nshards", type=int, required=True)
    a = ap.parse_args()
    rows = []
    sec = 0.0
    cells = 0
    for k in range(a.nshards):
        p = os.path.join(a.pred, f"pred_{a.cand}.shard{k}of{a.nshards}.csv")
        t = os.path.join(a.pred, f"time_{a.cand}.shard{k}of{a.nshards}.json")
        if not (os.path.exists(p) and os.path.exists(t)):
            raise SystemExit(f"missing shard {k}: {p}")
        with open(p, encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            next(rd)
            rows.extend(rd)
        j = json.load(open(t))
        sec += j["seconds"]
        cells += j["cells"]
    rows.sort(key=lambda r: (r[0], r[2], int(r[1])))
    with open(os.path.join(a.pred, f"pred_{a.cand}.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "sample_idx", "counter", "pred", "conf"])
        w.writerows(rows)
    json.dump({"cand": a.cand, "cells": cells, "seconds": round(sec, 3),
               "sec_per_1000_cells": round(1000.0 * sec / cells, 3),
               "cores": 4, "merged_from_shards": a.nshards},
              open(os.path.join(a.pred, f"time_{a.cand}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"merged {a.cand}: rows={len(rows)} cells={cells} "
          f"{1000.0*sec/cells:.3f} s/1000cells")


if __name__ == "__main__":
    main()
