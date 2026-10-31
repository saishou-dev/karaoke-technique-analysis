#!/usr/bin/env python3
"""v3_fullscan_all.py ＝ 読取器 v0.8f（見本 k＝3・計30個＋歯止め規則(c)）で、過去の全収録の帯を全フレーム読み直す（2026-10-31）

v0.8f の中身（材料50・追記 15:11）
  読み＝v0.8e と同じ glyph_boxes → norm_glyph → 見本照合（NCC）に、見本 30個（templates_k3_all4.npy）を足す。
  規則(c)＝1桁→2桁に上がる最初の確定は、2桁の値が12コマ続いた時だけ（後処理＝a4_rules.py の detect(rule_c=True)）。
  本スクリプトは「読み」だけを行い、v2_fullscan.py と同じ形の保存（<曲>.npz＝箱ごとの最良の類似度と値・
  現行の見本100個と追加見本30個の両方）を収録日ごとに書く。(c) を当てた曲末の値・採点・正解表は後続（a4_rules.py 系）で作る。
  読みの関数は v2_fullscan.py から import する（写さない＝10/14 の検証2と同一の読み）。

対象＝bands_wao の WAO 機の収録（既定＝10/14・9/24・10/6・9/11・9/15・10/21・9/2・8/18・8/12）と、
  Ai 機の収録（10/6 Ai＝bands_ai・6/28＝bands）。Ai は 10/31 の試し（--ai-test・CHEERS 10/6）で曲末 4/4 一致を確かめて加えた。
  8/7 は帯の切り出し映像が無い（対象外）。
  10/14 は 10/30 の検証2で同じ読み（v2_fullscan.py・同じ見本）が済んでいるので、その npz を写す（--reuse）。
関心領域（ROI）＝正式結果の曲別 roi.json（<結果>/<曲>/roi.json）。無い収録（8/12〜9/15）は既定 roi_wao.json
  （run_read_20260915.sh と同じ選び方）。
参照＝見本追加なしの読みから counter_events を作り、正式結果の counter_events.csv と1行ずつ突き合わせて記録する
  （v0.8e で読んだ収録は IDENTICAL になるはず・v0.8d で読んだ収録は版の差で DIFFERENT になり得る＝記録のみで止めない）。

並列＝--worker i --workers n で n 本を同時に走らせる。曲の取り合いは /tmp の mkdir ロック（--run で走行ごとに分ける）。
  既に npz がある曲は飛ばす（再開できる）。最後に終わった1本が progress.log に「ALL DONE v0.8f」と SCAN_DONE.txt を書く。
進捗＝<出力の親>/progress.log に曲ごとに1行（時刻つき）。入力（bands_wao・results_*・templates・追加見本）は読み取り専用。

Ai 機の試し＝--ai-test <曲>（bands_ai の1曲を Ai 読取器の ROI（と --gain で輝度ゲイン）で切り、WAO の見本＋追加見本で読んで
  曲末の値を公式採点データ（verify_20261006_ai.csv）と突き合わせる）。
"""
import argparse
import csv
import fcntl
import glob
import json
import os
import shutil
import sys
import time
import traceback

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glyph_core as gc
from v2_fullscan import FLAT, NONE_LAB, lab_code, scan, values_from, events_from

ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"
OUT_ROOT = ROOT + "/reread_v08f_20261031"
COUNTERS = gc.COUNTERS
DEFAULT_ROI = ROOT + "/templates/wao_20260816/roi_wao.json"
TPL = ROOT + "/templates/wao_20260816/cluster_reps_wao_ext.npy"
EXT = ROOT + "/unread_fix_20261030/templates_k3_all4.npy"
REUSE_1014 = ROOT + "/unread_fix_validate_20261030/result/fullscan"

# 収録キー → 帯のフォルダ・正式結果のフォルダ（roi.json と counter_events.csv の出所）
SESS = {
    "20260924": "results_wao_20260924",
    "20261006": "results_wao_20261006",
    "20260911": "results_wao_20260911",
    "20260915": "results_wao_20260915_rescan",
    "20261021": "results_wao_20261021",
    "20260902": "results_wao_20260902",
    "20260818": "results_wao_20260818",
    "20260812": "results_wao_20260812",
    "20261014": "results_wao_20261014",
    "20261006_ai": "results_ai_20261006",
    "20260628_ai": "results",
}
BANDS = {"20261006_ai": "bands_ai", "20260628_ai": "bands"}   # 既定＝bands_wao
# Ai 機＝Ai 読取器の ROI（4技法）と輝度ゲインの有無を、その収録を読んだ正式の読取器に合わせる
#   10/6 Ai＝read_counters_ai_20261006ai.py（vibrato 1305〜1395・ゲインあり）
#   6/28  ＝read_counters.py v0.8（vibrato 1230〜1460・ゲインなし）
AI_BASE = {"shakuri": (245, 85, 350, 148), "kobushi": (565, 85, 690, 148), "fall": (925, 85, 1020, 148)}
AI_CFG = {
    "20261006_ai": dict(rois=dict(AI_BASE, vibrato=(1305, 85, 1395, 148)), gain=True),
    "20260628_ai": dict(rois=dict(AI_BASE, vibrato=(1230, 85, 1460, 148)), gain=False),
}
ORDER = ["20261014", "20260924", "20261006", "20260911", "20260915", "20261021", "20260902", "20260818", "20260812",
         "20261006_ai", "20260628_ai"]
LOCK = "/tmp/reread_v08f_log.lock"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_line(path, msg):
    """複数の worker が同じ progress.log に書くので、/tmp のロックで1行ずつ書く。"""
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{now()}] {msg}\n")
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(f"[{now()}] {msg}", flush=True)


def load_templates():
    tpl = gc.Templates(TPL)
    ext = np.load(EXT)
    ext_lab_d = json.load(open(os.path.splitext(EXT)[0] + "_labels.json"))
    ext_lab = [ext_lab_d[str(i)] for i in range(len(ext))]
    flat = []
    for r in ext:
        v = r.ravel() - r.mean()
        flat.append(v / max(1e-6, np.linalg.norm(v)))
    return tpl, np.stack(flat), ext_lab


def band_dir(key):
    return BANDS.get(key, "bands_wao")


def songs_of(key):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, band_dir(key), f"*_{key[:8]}_*.mp4"))):
        if p.endswith(".part.mp4") or not os.path.isfile(p):
            continue
        out.append(os.path.splitext(os.path.basename(p))[0])
    return out


def task_list(dates):
    return [(d, s) for d in dates for s in songs_of(d)]


def roi_of(date, song):
    p = os.path.join(ROOT, SESS[date], song, "roi.json")
    src = "roi.json"
    if not os.path.exists(p):
        p, src = DEFAULT_ROI, "roi_wao.json"
    return {k: tuple(v) for k, v in json.load(open(p)).items()}, src


def identity(date, song, base_vals, fps):
    ref_p = os.path.join(ROOT, SESS[date], song, "counter_events.csv")
    got, _a, _b = events_from(base_vals, fps)
    if not os.path.exists(ref_p):
        return "NO_REF", 0, len(got)
    with open(ref_p, newline="", encoding="utf-8-sig") as f:
        ref = list(csv.reader(f))[1:]
    same = len(ref) == len(got) and all(x == y for ra, rb in zip(ref, got) for x, y in zip(ra, rb))
    return ("IDENTICAL" if same else "DIFFERENT"), len(ref), len(got)


def scan_one(date, song, out_dir, tpl, ext_flat, ext_lab):
    npz = os.path.join(out_dir, date, song + ".npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    band = os.path.join(ROOT, band_dir(date), song + ".mp4")
    t0 = time.time()
    if date in AI_CFG:
        cfg = AI_CFG[date]
        roi_src = "ai_rois" + ("+gain" if cfg["gain"] else "")
        fps, nb, bs, bl, es, el = scan_ai(band, tpl, ext_flat, ext_lab, cfg["gain"], cfg["rois"])
    else:
        roi, roi_src = roi_of(date, song)
        fps, nb, bs, bl, es, el = scan(band, roi, tpl, ext_flat, ext_lab)
    el_ = time.time() - t0
    if nb.shape[0] == 0:
        raise RuntimeError("0 フレーム（帯が読めない）")
    base_vals = values_from(nb, bs, bl, es, el, False)
    idt, nref, ngot = identity(date, song, base_vals, fps)
    tmp = npz + ".tmp.npz"
    np.savez_compressed(tmp, fps=fps, nb=nb, bs=bs, bl=bl, es=es, el=el)
    os.replace(tmp, npz)
    ext_vals = values_from(nb, bs, bl, es, el, True)
    rd_b = 100.0 * (base_vals >= 0).mean()
    rd_e = 100.0 * (ext_vals >= 0).mean()
    st = os.stat(band)
    with open(os.path.join(out_dir, date, "scan_log.csv"), "a", encoding="utf-8") as f:
        f.write(f"{song},{nb.shape[0]},{fps:.3f},{el_:.1f},{idt},{nref},{ngot},{roi_src},{st.st_size},"
                f"{rd_b:.2f},{rd_e:.2f},{SESS[date]}\n")
    return nb.shape[0], el_, idt, nref, ngot, roi_src, rd_b, rd_e


def run_worker(a):
    out_dir = os.path.join(OUT_ROOT, "fullscan")
    plog = os.path.join(OUT_ROOT, "progress.log")
    os.makedirs(out_dir, exist_ok=True)
    claims = f"/tmp/reread_v08f_claims_{a.run}"
    os.makedirs(claims, exist_ok=True)
    dates = a.dates.split(",")
    tasks = task_list(dates)
    tpl, ext_flat, ext_lab = load_templates()
    log_line(plog, f"w{a.worker} start run={a.run} workers={a.workers} songs={len(tasks)} dates={a.dates} pid={os.getpid()}")
    done_here = 0
    for k, (date, song) in enumerate(tasks):
        npz = os.path.join(out_dir, date, song + ".npz")
        if os.path.exists(npz):
            continue
        try:
            os.mkdir(os.path.join(claims, f"{date}__{song}"))
        except FileExistsError:
            continue
        try:
            src = os.path.join(a.reuse, song + ".npz") if a.reuse else ""
            if date == "20261014" and src and os.path.exists(src):
                os.makedirs(os.path.dirname(npz), exist_ok=True)
                shutil.copy2(src, npz + ".tmp.npz")
                os.replace(npz + ".tmp.npz", npz)
                with open(os.path.join(out_dir, date, "scan_log.csv"), "a", encoding="utf-8") as f:
                    f.write(f"{song},,,,REUSED_v2_fullscan,,,,,,,{SESS[date]}\n")
                log_line(plog, f"w{a.worker} {k+1}/{len(tasks)} {date} {song} reuse=v2_fullscan(10/30)")
                continue
            n, el_, idt, nref, ngot, roi_src, rd_b, rd_e = scan_one(date, song, out_dir, tpl, ext_flat, ext_lab)
            done_here += 1
            log_line(plog, f"w{a.worker} {k+1}/{len(tasks)} {date} {song} frames={n} {el_:.1f}s "
                           f"ms/frame={1000*el_/max(1,n):.1f} identity={idt} rows ref={nref} got={ngot} roi={roi_src} "
                           f"readable base={rd_b:.1f}% k3={rd_e:.1f}%")
        except Exception as e:
            log_line(plog, f"w{a.worker} {k+1}/{len(tasks)} {date} {song} ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
    open(os.path.join(claims, f"worker_{a.worker}.done"), "w").close()
    log_line(plog, f"w{a.worker} finished scanned={done_here}")
    # 最後に終わった worker が全体の完了を書く
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        fin = len(glob.glob(os.path.join(claims, "worker_*.done")))
        if fin >= a.workers and not os.path.exists(os.path.join(out_dir, "SCAN_DONE.txt")):
            miss = [f"{d}/{s}" for d, s in tasks if not os.path.exists(os.path.join(out_dir, d, s + ".npz"))]
            have = len(tasks) - len(miss)
            with open(os.path.join(out_dir, "SCAN_DONE.txt"), "w", encoding="utf-8") as f:
                f.write(f"done {now()} npz={have}/{len(tasks)} missing={len(miss)}\n" + "\n".join(miss) + "\n")
            with open(plog, "a", encoding="utf-8") as f:
                f.write(f"[{now()}] ALL DONE v0.8f npz={have}/{len(tasks)} missing={len(miss)}\n")
        fcntl.flock(lk, fcntl.LOCK_UN)
    return 0


# ---------------- Ai 機の試し（1曲） ----------------
AI_ROIS = {  # read_counters_ai_20261006ai.py と同じ（Ai は4技法）
    "shakuri": (245, 85, 350, 148),
    "kobushi": (565, 85, 690, 148),
    "fall": (925, 85, 1020, 148),
    "vibrato": (1305, 85, 1395, 148),
}


def roi_gain(rg, target=235.0, gmax=3.0):
    """read_counters_ai_20261006ai.py と同じ輝度ゲイン（99.5パーセンタイルを target へ・上限 gmax）。"""
    p = float(np.percentile(rg, 99.5))
    k = min(gmax, target / max(p, 40.0))
    return rg if k <= 1.0 else np.clip(rg.astype(np.float32) * k, 0, 255).astype(np.uint8)


def scan_ai(band, tpl, ext_flat, ext_lab, use_gain, rois=None):
    rois = rois or AI_ROIS
    cap = cv2.VideoCapture(band)
    fps = cap.get(cv2.CAP_PROP_FPS)
    keys = ("nb", "bs", "bl", "es", "el")
    L = {k: [] for k in keys}
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
            if name not in rois:
                continue
            x1, y1, x2, y2 = rois[name]
            rg = img[y1:y2, x1:x2].max(axis=2)
            if use_gain:
                rg = roi_gain(rg)
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
        for k_, v_ in zip(keys, (nb, bs, bl, es, el)):
            L[k_].append(v_)
    cap.release()
    return (fps,) + tuple(np.stack(L[k]) for k in keys)


def run_ai_test(a):
    from a4_rules import accs_from
    out_dir = os.path.join(OUT_ROOT, "ai_test")
    os.makedirs(out_dir, exist_ok=True)
    tpl, ext_flat, ext_lab = load_templates()
    ver = {}
    with open(os.path.join(ROOT, "dataset/verify_20261006_ai.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ver[r["video"]] = r
    s = a.ai_test
    t0 = time.time()
    fps, nb, bs, bl, es, el = scan_ai(os.path.join(ROOT, "bands_ai", s + ".mp4"), tpl, ext_flat, ext_lab, a.gain)
    el_ = time.time() - t0
    tag = "gain" if a.gain else "nogain"
    np.savez_compressed(os.path.join(out_dir, f"{s}_{tag}.npz"), fps=fps, nb=nb, bs=bs, bl=bl, es=es, el=el)
    rows = []
    for cond, ext, rc in [("base_WAOtpl", False, False), ("k3", True, False), ("v0.8f_k3+c", True, True)]:
        vals = values_from(nb, bs, bl, es, el, ext)
        accs, _sk = accs_from(vals, False, rc, False)
        for ci, c in enumerate(COUNTERS[:4]):
            fin = accs[c][-1][1] if accs[c] else 0
            dam = int(ver[s]["csv_" + c])
            ai_read = int(ver[s]["read_" + c])
            rows.append([s, tag, cond, c, fin, dam, ai_read, int(fin == dam), f"{100.0*(vals[:, ci] >= 0).mean():.2f}"])
    with open(os.path.join(out_dir, "ai_test.csv"), "a", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"[{now()}] ai_test {s} {tag} frames={nb.shape[0]} {el_:.1f}s", flush=True)
    print("  cond counter final dam ai_reader hit readable%", flush=True)
    for r in rows:
        print("  " + " ".join(map(str, r[2:])), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--run", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--dates", default=",".join(ORDER))
    ap.add_argument("--reuse", default=REUSE_1014)
    ap.add_argument("--ai-test", default="")
    ap.add_argument("--gain", action="store_true")
    a = ap.parse_args()
    if a.ai_test:
        return run_ai_test(a)
    return run_worker(a)


if __name__ == "__main__":
    sys.exit(main())
