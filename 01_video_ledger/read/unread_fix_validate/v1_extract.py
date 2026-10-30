#!/usr/bin/env python3
"""v1_extract.py ＝ 検証1の入力＝別の日の収録（9/24・10/6）から 10/18 と同じ採り方でコマと正解値を作る（2026-10-30）

ocr_compare_20261018/extract_rois.py の写し。変えたのは (a) 曲＝verify_<S>.csv の全曲（--verify で指定）
(b) 正解データのファイル名を --g1 で指定（9/24 は G1_events_20260924.csv・10/6 は G1_events_20261006_wao.csv）
(c) ROI は --results の正式結果の roi.json の3点だけ。採り方（逐次デコード・--stride コマごと）と正解値の式は同じ。

--- 以下は写し元の説明 ---
extract_rois.py ＝ 比較実験の入力（同一のコマ）と正解値を1回だけ作る（2026-10-18）

やること
  1. 対象曲の帯動画（画面下部の加点表示領域だけを切り出した mp4）を**逐次デコード**する
     （帯動画は可変フレームレートで cv2 のシークがずれるため・2026-10-15 の既知の癖）。
  2. --stride コマごとに1枚を採り、6つの加点回数表示それぞれの関心領域（ROI）を切って uint8 で保存する。
     ROI は正式結果 results_wao_<S>/<video>/roi.json（曲ごとの校正値）をそのまま使う＝現行手法と同じ入力。
  3. 各コマ・各カウンターの正解値を技法イベント正解データ（dataset/G1_events_<S>_wao.csv）から復元する。
     正解値(f) ＝ frame <= f の最後のイベントの count_value（イベントが無ければ 0）。
     ただし [event_frame_est, frame) の区間は「表示が旧値から新値へ変わる途中（読めていない区間）」で
     真の表示値が一意に決まらないため certain=0 とし、主たる集計から外す。

出力（--out 配下）
  <video>/roi_<counter>.npy   … (n, h, w, 3) uint8 ＝切り出した関心領域そのもの（全候補が同じものを読む）
  <video>/frames.npy          … (n,) int32 ＝元の帯動画のコマ番号
  gt.csv                      … video,sample_idx,frame,counter,gt_value,certain
  input_summary.csv           … 曲ごとの内訳

使い方:
  python extract_rois.py --out <dir> [--stride 8] [--root /mnt/g/研究アーカイブ/成果物/動画解析]
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

COUNTERS = ["shakuri", "kobushi", "fall", "vibrato", "accent", "hammering"]

# 10/14 収録（WAO 30曲）のうち曲単位の完全一致 27曲から、加点回数の大きさが散るように4曲を選んだ。
# （2桁が多い曲・ビブラートが多い曲・フォールが0でない曲・全部1桁の曲）
SONGS = [
    "サイレントマジョリティー_20261014_105318",
    "スターダスト_20261014_120507",
    "共犯_20261014_111506",
    "恋人ごっこ(生音)_20261014_124950",
]


def load_ledger(ds, g1name):
    led = {}
    path = os.path.join(ds, g1name)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            led.setdefault(r["video"], {}).setdefault(r["counter"], []).append(
                (int(r["frame"]), int(r["count_value"]), int(r["event_frame_est"])))
    for v in led:
        for c in led[v]:
            led[v][c].sort()
    return led


def gt_for(events, frame):
    """正解値と確定可否。events＝[(frame, value, event_frame_est)] 昇順。"""
    val = 0
    certain = True
    for fr, v, est in events:
        if frame >= fr:
            val = v
        elif est <= frame < fr:
            certain = False
    return val, certain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/mnt/g/研究アーカイブ/成果物/動画解析")
    ap.add_argument("--session", default="20261014")
    ap.add_argument("--results", default=None)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--g1", required=True)
    ap.add_argument("--verify", required=True)
    a = ap.parse_args()
    results = a.results or f"results_wao_{a.session}"
    ds = os.path.join(a.root, "dataset")
    os.makedirs(a.out, exist_ok=True)
    led = load_ledger(ds, a.g1)
    with open(os.path.join(ds, a.verify), encoding="utf-8-sig", newline="") as f:
        SONGS = [r["video"] for r in csv.DictReader(f)]

    gt_rows = []
    summ = []
    for song in SONGS:
        t0 = time.time()
        band = os.path.join(a.root, "bands_wao", song + ".mp4")
        roi = {k: tuple(v) for k, v in json.load(
            open(os.path.join(a.root, results, song, "roi.json"))).items()}
        odir = os.path.join(a.out, song)
        os.makedirs(odir, exist_ok=True)
        cap = cv2.VideoCapture(band)
        fps = cap.get(cv2.CAP_PROP_FPS)
        buf = {c: [] for c in COUNTERS}
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
            for c in COUNTERS:
                x1, y1, x2, y2 = roi[c]
                buf[c].append(img[y1:y2, x1:x2].copy())
        cap.release()
        nframes = fidx + 1
        np.save(os.path.join(odir, "frames.npy"), np.asarray(frames, dtype=np.int32))
        for c in COUNTERS:
            np.save(os.path.join(odir, f"roi_{c}.npy"), np.stack(buf[c]))
        for i, f in enumerate(frames):
            for c in COUNTERS:
                v, cer = gt_for(led.get(song, {}).get(c, []), f)
                gt_rows.append([song, i, f, c, v, 1 if cer else 0])
        ncer = sum(1 for r in gt_rows if r[0] == song and r[5] == 1)
        summ.append([song, nframes, len(frames), f"{fps:.3f}", len(frames) * 6, ncer,
                     f"{100.0*ncer/max(1,len(frames)*6):.2f}",
                     " ".join(f"{c}:{max([x[1] for x in led.get(song, {}).get(c, [(0,0,0)])])}" for c in COUNTERS)])
        print(f"{song}: frames={nframes} sampled={len(frames)} "
              f"certain={ncer}/{len(frames)*6} ({100.0*ncer/max(1,len(frames)*6):.1f}%) "
              f"{time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(a.out, "gt.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "sample_idx", "frame", "counter", "gt_value", "certain"])
        w.writerows(gt_rows)
    with open(os.path.join(a.out, "input_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frames_total", "frames_sampled", "fps", "cells", "cells_certain",
                    "certain_pct", "final_values"])
        w.writerows(summ)
    print("total cells:", len(gt_rows), "stride:", a.stride)
    return 0


if __name__ == "__main__":
    sys.exit(main())
