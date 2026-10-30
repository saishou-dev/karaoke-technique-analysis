#!/usr/bin/env python3
"""identity_check.py ＝ 参照実装 glyph_core.py が正本 v0.8e と同じ答えを出すことの実測（2026-10-18）

やること＝指定の帯動画を全コマ走査し、glyph_core の関数だけで counter_events.csv 相当を作って、
正式結果 results_wao_20261014/<video>/counter_events.csv と1行ずつ突き合わせる。
一致すれば「比較実験の基準（現行手法）は正本と同じものである」と言える。

使い方: python identity_check.py <video_name> [--root <動画解析>]
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glyph_core as gc


def run_reader(band_mp4, roi, tpl):
    cap = cv2.VideoCapture(band_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    values = {k: [] for k in roi}
    fidx = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        fidx += 1
        g = img.max(axis=2)
        for name, (x1, y1, x2, y2) in roi.items():
            rg = g[y1:y2, x1:x2]
            boxes = gc.glyph_boxes(rg)
            if boxes is None:
                continue
            digits = []
            okread = True
            for b in boxes:
                d = tpl.classify_ncc(gc.norm_glyph(rg, b))
                if d is None:
                    okread = False
                    break
                digits.append(d)
            if not okread:
                continue
            values[name].append((fidx, digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]))
    cap.release()
    nframes = fidx + 1
    # 帯消失ガード（元コマ単位なので半幅はそのまま150）
    values, _ = gc.band_trim(values, nframes, gc.BAND_W_FRAMES)
    results = {}
    for name in roi:
        accepted, _ = gc.detect_increments(values[name])
        results[name] = accepted
    rows = []
    for name, acc in results.items():
        seq = values[name]
        for i, (fr, v, d) in enumerate(acc):
            ev_fr = fr
            if i > 0:
                pv = acc[i - 1][1]
                last_old = None
                for f2, v2 in seq:
                    if f2 >= fr:
                        break
                    if v2 == pv:
                        last_old = f2
                if last_old is not None:
                    ev_fr = last_old + 1
            rows.append((fr, name, v, d, ev_fr))
    out = []
    for fr, name, v, d, ev_fr in sorted(rows):
        out.append([name, str(fr), f"{fr/fps:.3f}", str(v), str(d), str(ev_fr), f"{ev_fr/fps:.3f}"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--root", default="/mnt/g/研究アーカイブ/成果物/動画解析")
    ap.add_argument("--results", default="results_wao_20261014")
    a = ap.parse_args()
    root = a.root
    band = os.path.join(root, "bands_wao", a.video + ".mp4")
    roi_path = os.path.join(root, a.results, a.video, "roi.json")
    import json
    roi = {k: tuple(v) for k, v in json.load(open(roi_path)).items()}
    tpl = gc.Templates(os.path.join(root, "templates", "wao_20260816", "cluster_reps_wao_ext.npy"))
    got = run_reader(band, roi, tpl)
    ref_path = os.path.join(root, a.results, a.video, "counter_events.csv")
    with open(ref_path, newline="") as f:
        rd = list(csv.reader(f))
    ref = [r for r in rd[1:]]
    print(f"video={a.video}")
    print(f"rows: reference={len(ref)} reimplementation={len(got)}")
    same = (len(ref) == len(got)) and all(a_ == b_ for ra, rb in zip(ref, got) for a_, b_ in zip(ra, rb))
    if same:
        print("RESULT: IDENTICAL")
    else:
        print("RESULT: DIFFERENT")
        for i, (ra, rb) in enumerate(zip(ref, got)):
            if ra != rb:
                print("  ref", ra)
                print("  got", rb)
                if i > 20:
                    break
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
