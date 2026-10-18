#!/usr/bin/env python3
"""tess_timing.py ＝ Tesseract の処理時間を2つの使い方で測り直す（2026-10-18・補足測定）

なぜ必要か＝本実験で Tesseract 5.5.0 に次の挙動が出た。
  ファイル一覧を渡す一括実行（`tesseract list.txt out --psm N`）が、入力の組み合わせによって
  SIGFPE（signal 8）で異常終了する。1枚ずつ渡す実行では同じ画像が問題なく通る。
  このため run_candidates.py は「一括で落ちたまとまりだけ1枚ずつ再実行する」作りにしてあり、
  その実測秒数には落ちた一括実行の無駄時間が混ざる。
  Tesseract に不利にならないよう、ここで (a) 落ちない一括実行 と (b) 1枚ずつ実行 を別に測る。

使い方: python tess_timing.py --work <work> --out <out.json> [--n 2000] [--psm 7 8]
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np

COUNTERS = ["shakuri", "kobushi", "fall", "vibrato", "accent", "hammering"]


def prep(crop, scale=3):
    g = crop.max(axis=2)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - b


def tess(files, base, psm):
    lst = base + ".list"
    with open(lst, "w") as f:
        f.write("\n".join(files) + "\n")
    t0 = time.perf_counter()
    r = subprocess.run(["tesseract", lst, base, "--psm", str(psm), "-l", "eng",
                        "-c", "tessedit_char_whitelist=0123456789",
                        "-c", "classify_bln_numeric_mode=1"],
                       capture_output=True, text=True)
    return time.perf_counter() - t0, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=500)
    a = ap.parse_args()
    songs = [r["video"] for r in csv.DictReader(
        open(os.path.join(a.work, "input_summary.csv"), encoding="utf-8"))]
    # 4曲×6カウンターから均等に拾う
    per = max(1, a.n // (len(songs) * len(COUNTERS)))
    tmp = tempfile.mkdtemp(prefix="tesst_")
    files = []
    try:
        for s in songs:
            for c in COUNTERS:
                arr = np.load(os.path.join(a.work, s, f"roi_{c}.npy"))
                idx = np.linspace(0, arr.shape[0] - 1, per).astype(int)
                for i in idx:
                    p = os.path.join(tmp, f"{len(files):06d}.png")
                    cv2.imwrite(p, prep(arr[i]))
                    files.append(p)
        base = os.path.join(tmp, "o")
        res = {"cells": len(files), "chunk": a.chunk}
        for psm in (7, 8):
            # (a) 一括（まとまりごと・落ちたまとまりは秒数に数えない）
            ok_t = 0.0
            ok_n = 0
            ng = 0
            for b0 in range(0, len(files), a.chunk):
                ch = files[b0:b0 + a.chunk]
                dt, rc = tess(ch, base, psm)
                if rc == 0:
                    ok_t += dt
                    ok_n += len(ch)
                else:
                    ng += 1
            # (b) 1枚ずつ
            one_t = 0.0
            one_n = min(300, len(files))
            for p in files[:one_n]:
                dt, rc = tess([p], base, psm)
                one_t += dt
            res[f"psm{psm}"] = {
                "batch_ok_cells": ok_n, "batch_failed_chunks": ng,
                "batch_sec_per_1000": round(1000.0 * ok_t / ok_n, 3) if ok_n else None,
                "single_cells": one_n,
                "single_sec_per_1000": round(1000.0 * one_t / one_n, 3),
            }
            print(psm, res[f"psm{psm}"], flush=True)
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("wrote", a.out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
