#!/usr/bin/env python3
"""extract_bands.py ＝ 条件A（帯まるごと）の入力を作る（2026-10-29）

2026-10-18 の比較実験と**同じコマ列**（stride 8・逐次デコード）で、帯動画のコマを丸ごと保存する。
保存したコマ番号が ocr_compare_20261018/work/<曲>/frames.npy と1件も違わないことを確認して止める。

出力 ＝ --out/<曲>.npy  (n, 120, 2480, 3) uint8
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

PREV = "/mnt/g/研究アーカイブ/成果物/動画解析/ocr_compare_20261018"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/g/研究アーカイブ/成果物/動画解析")
    ap.add_argument("--prev-work", default=os.path.join(PREV, "work"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    songs = [d for d in sorted(os.listdir(a.prev_work))
             if os.path.isdir(os.path.join(a.prev_work, d))]
    for s in songs:
        dst = os.path.join(a.out, s + ".npy")
        ref = np.load(os.path.join(a.prev_work, s, "frames.npy"))
        if os.path.exists(dst):
            n = np.load(dst, mmap_mode="r").shape[0]
            if n == len(ref):
                print(f"skip {s} (already {n})", flush=True)
                continue
        t0 = time.time()
        cap = cv2.VideoCapture(os.path.join(a.root, "bands_wao", s + ".mp4"))
        buf = []
        frames = []
        fidx = -1
        while True:
            ok, img = cap.read()
            if not ok:
                break
            fidx += 1
            if fidx % a.stride:
                continue
            frames.append(fidx)
            buf.append(img)
        cap.release()
        frames = np.asarray(frames, dtype=np.int32)
        if len(frames) != len(ref) or not np.array_equal(frames, ref):
            print(f"FATAL {s}: コマ列が 2026-10-18 と一致しない "
                  f"({len(frames)} vs {len(ref)})", flush=True)
            return 2
        arr = np.stack(buf)
        np.save(dst, arr)
        print(f"{s}: {arr.shape} {arr.nbytes/1e9:.2f}GB {time.time()-t0:.0f}s", flush=True)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
