#!/usr/bin/env python3
"""v2_fullscan.py ＝ 検証2の走査＝10/14 の30曲を全フレーム走査し、箱ごとの類似度を1回で書き出す（2026-10-30）

1曲ずつ逐次デコード（帯動画は可変フレームレート＝シークしない）。各フレーム×6技法で
  v0.8e と同じ glyph_boxes → norm_glyph → 見本照合（NCC）を行い、箱ごとに
    現行の見本100個での最良（類似度・値／数字でない＝-2）と、追加見本30個での最良
  を保存する（<out>/<曲>.npz）。2条件（見本追加なし・k＝3）の値はこの保存から決まる。
  見本追加なしの値で counter_events を作り、正式結果 results_wao_20261014/<曲>/counter_events.csv と
  1行ずつ突き合わせる（参照実装と正本の同一性の確認・1曲目で不一致なら止まる）。
既に .npz がある曲は飛ばす（再開できる）。全曲が終わると <out>/SCAN_DONE.txt を書く。
入力は読み取り専用（bands_wao・results_wao_20261014・templates・追加見本）。
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glyph_core as gc

ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"
COUNTERS = gc.COUNTERS
FLAT = -2.0
NONE_LAB = -9    # 箱なし／NCC が定義できない
NONDIG = -2      # 最も似た見本が「数字でない」見本


def lab_code(l):
    return NONDIG if l is None else int(l)


def scan(band, roi, tpl, ext_flat, ext_lab):
    cap = cv2.VideoCapture(band)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nb_l, bs_l, bl_l, es_l, el_l = [], [], [], [], []
    nbase = tpl.reps_flat.shape[0]
    allflat = np.concatenate([tpl.reps_flat, ext_flat], axis=0).astype(np.float32)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        nb = np.zeros(6, np.int8)
        bs = np.full((6, 2), FLAT, np.float32)
        bl = np.full((6, 2), NONE_LAB, np.int8)
        es = np.full((6, 2), FLAT, np.float32)
        el = np.full((6, 2), NONE_LAB, np.int8)
        for ci, name in enumerate(COUNTERS):
            x1, y1, x2, y2 = roi[name]
            rg = img[y1:y2, x1:x2].max(axis=2)
            boxes = gc.glyph_boxes(rg)
            if boxes is None:
                continue
            nb[ci] = len(boxes)
            for bi_, b in enumerate(boxes):
                gl = gc.norm_glyph(rg, b)
                v = gl.ravel() - gl.mean()
                n = np.linalg.norm(v)
                if n < 1e-6:
                    continue
                sc_ = allflat @ (v / n)
                i0 = int(np.argmax(sc_[:nbase]))
                i1 = int(np.argmax(sc_[nbase:]))
                bs[ci, bi_] = sc_[i0]
                bl[ci, bi_] = lab_code(tpl.labels.get(i0))
                es[ci, bi_] = sc_[nbase + i1]
                el[ci, bi_] = int(ext_lab[i1])
        nb_l.append(nb)
        bs_l.append(bs)
        bl_l.append(bl)
        es_l.append(es)
        el_l.append(el)
    cap.release()
    return fps, np.stack(nb_l), np.stack(bs_l), np.stack(bl_l), np.stack(es_l), np.stack(el_l)


def values_from(nb, bs, bl, es, el, use_ext, t=gc.NCC_MIN):
    """各フレーム×技法の値（読まない＝-1）。use_ext＝追加見本も使う（130個の中の最良＝追加側が真に上回る時だけ置き換え）。"""
    n = nb.shape[0]
    out = np.full((n, 6), -1, np.int32)
    s = bs.copy()
    l = bl.copy()
    if use_ext:
        rep = (es > bs) & (bs != FLAT)
        s[rep] = es[rep]
        l[rep] = el[rep]
    for ci in range(6):
        for k in (1, 2):
            m = nb[:, ci] == k
            if not m.any():
                continue
            ok = m.copy()
            for j in range(k):
                ok &= (s[:, ci, j] != FLAT) & (s[:, ci, j] >= t) & (l[:, ci, j] >= 0)
            if k == 1:
                out[ok, ci] = l[ok, ci, 0]
            else:
                out[ok, ci] = l[ok, ci, 0].astype(np.int32) * 10 + l[ok, ci, 1]
    return out


def events_from(vals, fps):
    """identity_check.run_reader と同じ後処理（帯消失ガード→安定判定＋減らない制約→イベント行）。"""
    nframes = vals.shape[0]
    values = {}
    for ci, name in enumerate(COUNTERS):
        idx = np.nonzero(vals[:, ci] >= 0)[0]
        values[name] = [(int(i), int(vals[i, ci])) for i in idx]
    values, band = gc.band_trim(values, nframes, gc.BAND_W_FRAMES)
    rows = []
    accs = {}
    for name in COUNTERS:
        acc, _ = gc.detect_increments(values[name])
        accs[name] = acc
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
    return out, accs, band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default="results_wao_20261014")
    ap.add_argument("--verify", default="dataset/verify_20261014_wao.csv")
    ap.add_argument("--ext", default=ROOT + "/unread_fix_20261030/templates_k3_all4.npy")
    ap.add_argument("--songs", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tpl = gc.Templates(ROOT + "/templates/wao_20260816/cluster_reps_wao_ext.npy")
    ext = np.load(a.ext)
    ext_lab_d = json.load(open(os.path.splitext(a.ext)[0] + "_labels.json"))
    ext_lab = [ext_lab_d[str(i)] for i in range(len(ext))]
    ext_flat = []
    for r in ext:
        v = r.ravel() - r.mean()
        ext_flat.append(v / max(1e-6, np.linalg.norm(v)))
    ext_flat = np.stack(ext_flat)
    with open(os.path.join(ROOT, a.verify), encoding="utf-8-sig", newline="") as f:
        songs = [r["video"] for r in csv.DictReader(f)]
    if a.songs:
        songs = [s for s in songs if s in a.songs.split("|")]
    T0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] start songs={len(songs)} ext={len(ext)}", flush=True)
    for n_, s in enumerate(songs):
        npz = os.path.join(a.out, s + ".npz")
        if os.path.exists(npz):
            print(f"skip {s}", flush=True)
            continue
        t0 = time.time()
        roi = {k: tuple(v) for k, v in json.load(open(os.path.join(ROOT, a.results, s, "roi.json"))).items()}
        fps, nb, bs, bl, es, el = scan(os.path.join(ROOT, "bands_wao", s + ".mp4"), roi, tpl, ext_flat, ext_lab)
        el_ = time.time() - t0
        base_vals = values_from(nb, bs, bl, es, el, False)
        got, _accs, _band = events_from(base_vals, fps)
        with open(os.path.join(ROOT, a.results, s, "counter_events.csv"), newline="") as f:
            ref = list(csv.reader(f))[1:]
        same = len(ref) == len(got) and all(x == y for ra, rb in zip(ref, got) for x, y in zip(ra, rb))
        tmp = npz + ".tmp.npz"
        np.savez_compressed(tmp, fps=fps, nb=nb, bs=bs, bl=bl, es=es, el=el)
        os.replace(tmp, npz)
        with open(os.path.join(a.out, "scan_log.csv"), "a", encoding="utf-8") as f:
            f.write(f"{s},{nb.shape[0]},{fps:.3f},{el_:.1f},{'IDENTICAL' if same else 'DIFFERENT'},{len(ref)},{len(got)}\n")
        print(f"[{time.strftime('%H:%M:%S')}] {n_+1}/{len(songs)} {s} frames={nb.shape[0]} {el_:.1f}s "
              f"identity={'IDENTICAL' if same else 'DIFFERENT'} rows ref={len(ref)} got={len(got)}", flush=True)
        if not same and n_ == 0:
            print("STOP: 参照実装と正本の同一性が崩れている（1曲目）", flush=True)
            open(os.path.join(a.out, "SCAN_STOPPED.txt"), "w").write(f"identity DIFFERENT {s}\n")
            return 3
    open(os.path.join(a.out, "SCAN_DONE.txt"), "w").write(f"done {time.strftime('%F %T')} {time.time()-T0:.1f}s\n")
    print(f"[{time.strftime('%H:%M:%S')}] SCAN DONE {time.time()-T0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
