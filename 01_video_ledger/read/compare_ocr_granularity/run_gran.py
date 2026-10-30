#!/usr/bin/env python3
"""run_gran.py = 切り出し粒度を変えて同じ手法群を走らせる（2026-10-29）

条件（切り出し粒度）
  A  = 帯まるごと（2480x120・1コマに6つの数字が同時に写る）
  B  = 技法別の数字窓（2026-10-18 の比較実験と同一の入力）
  C  = 固定ピッチの桁窓（桁数に関わらず cx±P/2 に2枠）
  C2 = 桁ごとの窓（インク幅で桁数を決めてから1枠または2枠）

出力
  pred_<cond>_<cand>[.shardIofN].csv  : video,sample_idx,counter,pred,conf
  time_<cond>_<cand>[.shardIofN].json : 秒数・件数・1,000件あたり秒・場所外の読みの件数
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gran_core as G                      # noqa: E402
import glyph_core as gc                    # noqa: E402

COUNTERS = G.COUNTERS
TESS_CHUNK = 200


def load_inputs(a):
    songs = [d for d in sorted(os.listdir(a.prev_work))
             if os.path.isdir(os.path.join(a.prev_work, d))]
    sel = {}
    for s in songs:
        n = np.load(os.path.join(a.prev_work, s, "frames.npy")).shape[0]
        idx = [i for i in range(n) if i % a.sub == 0]
        if a.nshards > 1:
            idx = idx[a.shard::a.nshards]
        sel[s] = idx
    roi = {s: {k: tuple(v) for k, v in json.load(open(
        os.path.join(a.root, a.results, s, "roi.json"))).items()} for s in songs}
    return songs, sel, roi


def band_arr(a, s):
    return np.load(os.path.join(a.band, s + ".npy"), mmap_mode="r")


def win_arr(a, s, c):
    return np.load(os.path.join(a.prev_work, s, "roi_" + c + ".npy"), mmap_mode="r")


def slots_for(slots, s, c, ndig):
    p = slots["per"][s][c]
    return G.slot_boxes(p["cx"], p["cy"], slots["slot_w"], slots["slot_h"],
                        slots["pitch"], ndig)


def iter_units(a, songs, sel, slots):
    """読みの単位を1つずつ返す。
       条件A  = (key=(song,idx), [band])
       条件B  = (key=(song,idx,counter), [window])
       条件C/C2 = (key=(song,idx,counter), [slot, ...])"""
    if a.cond == "A":
        for s in songs:
            arr = band_arr(a, s)
            for i in sel[s]:
                yield (s, i), [np.asarray(arr[i])]
    else:
        for s in songs:
            for c in COUNTERS:
                arr = win_arr(a, s, c)
                for i in sel[s]:
                    w = np.asarray(arr[i])
                    if a.cond == "B":
                        yield (s, i, c), [w]
                    else:
                        nd = 2 if a.cond == "C" else G.ndigits_of(
                            w.max(axis=2), slots["ndigit_thr"])
                        yield (s, i, c), [G.crop_pad(w, b)
                                          for b in slots_for(slots, s, c, nd)]


def run_tpl(a, songs, sel, roi, slots, how):
    tpl = gc.Templates(os.path.join(a.root, "templates", "wao_20260816",
                                    "cluster_reps_wao_ext.npy"))
    rows = []
    stray = 0
    nimg = 0
    t0 = time.perf_counter()
    if a.cond == "A":
        for s in songs:
            arr = band_arr(a, s)
            for i in sel[s]:
                vals, st = G.read_band_tpl(np.asarray(arr[i]), roi[s], tpl, how)
                stray += st
                nimg += 1
                for c in COUNTERS:
                    v = vals[c]
                    rows.append([s, i, c, "" if v is None else v, ""])
    elif a.cond == "B":
        for s in songs:
            for c in COUNTERS:
                arr = win_arr(a, s, c)
                for i in sel[s]:
                    w = np.asarray(arr[i])
                    v = gc.read_value_ncc(w, tpl) if how == "ncc" else gc.read_value_nn(w, tpl)
                    nimg += 1
                    rows.append([s, i, c, "" if v is None else v, ""])
    else:
        for s in songs:
            for c in COUNTERS:
                arr = win_arr(a, s, c)
                for i in sel[s]:
                    w = np.asarray(arr[i])
                    nd = 2 if a.cond == "C" else G.ndigits_of(w.max(axis=2),
                                                              slots["ndigit_thr"])
                    ds = [G.classify_slot_tpl(G.crop_pad(w, b), tpl, how)
                          for b in slots_for(slots, s, c, nd)]
                    nimg += len(ds)
                    v = G.combine_slots(ds)
                    rows.append([s, i, c, "" if v is None else v, ""])
    return rows, time.perf_counter() - t0, {"stray": stray, "images": nimg}


def _tess(lst, base, psm, tsv):
    cmd = ["tesseract", lst, base, "--psm", str(psm), "-l", "eng",
           "-c", "tessedit_char_whitelist=0123456789",
           "-c", "classify_bln_numeric_mode=1"]
    if tsv:
        cmd.append("tsv")
    return subprocess.run(cmd, capture_output=True, text=True)


def _tsv_pages(path, npages):
    pages = [[] for _ in range(npages)]
    if not os.path.exists(path):
        return pages
    with open(path, encoding="utf-8", errors="replace") as f:
        rd = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for r in rd:
            try:
                if int(r["level"]) != 5:
                    continue
                p = int(r["page_num"]) - 1
                t = (r.get("text") or "").strip()
                if not t or p >= npages or p < 0:
                    continue
                pages[p].append((float(r["left"]) + float(r["width"]) / 2.0, t))
            except (ValueError, TypeError, KeyError):
                continue
    return pages


def run_tess(a, songs, sel, roi, slots, psm):
    """Tesseract。条件A は tsv で位置を取り x で割り当てる。B/C/C2 は文字列だけ。
    処理時間は 2026-10-18 と同じく tesseract 本体の実行だけを測る。"""
    tsv = (a.cond == "A")
    tmp = tempfile.mkdtemp(prefix="tessg_")
    rows = []
    state = {"dt": 0.0, "crash": 0, "stray": 0, "nimg": 0}
    buf = []
    try:
        def flush():
            if not buf:
                return
            files = []
            for k, (key, imgs) in enumerate(buf):
                for j, im in enumerate(imgs):
                    p = os.path.join(tmp, "%04d_%d.png" % (k, j))
                    cv2.imwrite(p, G.prep(im))
                    files.append(p)
                    state["nimg"] += 1
            lst = os.path.join(tmp, "list.txt")
            with open(lst, "w") as f:
                f.write("\n".join(files) + "\n")
            base = os.path.join(tmp, "out")
            t0 = time.perf_counter()
            r = _tess(lst, base, psm, tsv)
            state["dt"] += time.perf_counter() - t0
            ok = r.returncode == 0
            if tsv:
                pages = _tsv_pages(base + ".tsv", len(files)) if ok else None
                if pages is None:
                    pages = []
                    for p in files:
                        with open(lst, "w") as f:
                            f.write(p + "\n")
                        t0 = time.perf_counter()
                        r1 = _tess(lst, base, psm, True)
                        state["dt"] += time.perf_counter() - t0
                        if r1.returncode == 0:
                            pages += _tsv_pages(base + ".tsv", 1)
                        else:
                            pages.append([])
                            state["crash"] += 1
                for (key, _im), pg in zip(buf, pages):
                    s, i = key
                    hit, st = G.assign_by_x(pg, roi[s], scale=3.0)
                    state["stray"] += st
                    for c in COUNTERS:
                        v = G.parse_digits(hit.get(c))
                        rows.append([s, i, c, "" if v is None else v, ""])
            else:
                txts = None
                if ok and os.path.exists(base + ".txt"):
                    pg = open(base + ".txt", encoding="utf-8",
                              errors="replace").read().split("\f")
                    if pg and pg[-1].strip() == "":
                        pg = pg[:-1]
                    if len(pg) == len(files):
                        txts = pg
                if txts is None:
                    txts = []
                    for p in files:
                        with open(lst, "w") as f:
                            f.write(p + "\n")
                        t0 = time.perf_counter()
                        r1 = _tess(lst, base, psm, False)
                        state["dt"] += time.perf_counter() - t0
                        if r1.returncode == 0 and os.path.exists(base + ".txt"):
                            txts.append(open(base + ".txt", encoding="utf-8",
                                             errors="replace").read())
                        else:
                            txts.append("")
                            state["crash"] += 1
                pos = 0
                for key, imgs in buf:
                    s, i, c = key
                    parts = txts[pos:pos + len(imgs)]
                    pos += len(imgs)
                    if a.cond == "B":
                        v = G.parse_digits(parts[0] if parts else "")
                    else:
                        v = G.combine_slots([G.parse_digits(x) for x in parts])
                    rows.append([s, i, c, "" if v is None else v, ""])
            for p in files:
                try:
                    os.remove(p)
                except OSError:
                    pass
            del buf[:]

        for key, imgs in iter_units(a, songs, sel, slots):
            buf.append((key, imgs))
            if len(buf) >= TESS_CHUNK:
                flush()
                if len(rows) % 6000 < 7:
                    print("  tess %d rows %.0fs" % (len(rows), state["dt"]), flush=True)
        flush()
        return rows, state["dt"], {"stray": state["stray"],
                                   "crash_cells": state["crash"],
                                   "images": state["nimg"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_easyocr(a, songs, sel, roi, slots, mode):
    import easyocr
    rd = easyocr.Reader(["en"], gpu=False, verbose=False)
    rows = []
    stray = 0
    nimg = 0
    t0 = time.perf_counter()
    for key, imgs in iter_units(a, songs, sel, slots):
        if a.cond == "A":
            s, i = key
            im = imgs[0].max(axis=2)
            nimg += 1
            if mode == "rec":
                res = rd.recognize(im, allowlist="0123456789")
                txt = res[0][1] if res else ""
                cf = res[0][2] if res else ""
                hit, st = G.assign_in_order(txt)
                conf = dict((c, cf) for c in hit)
            else:
                res = rd.readtext(im, allowlist="0123456789")
                items = [(float(np.mean([p[0] for p in b[0]])), b[1]) for b in res]
                hit, st = G.assign_by_x(items, roi[s])
                conf = {}
            stray += st
            for c in COUNTERS:
                v = G.parse_digits(hit.get(c))
                cf2 = conf.get(c, "")
                rows.append([s, i, c, "" if v is None else v,
                             "" if cf2 == "" else "%.4f" % float(cf2)])
        else:
            s, i, c = key
            vals = []
            cfs = []
            for im0 in imgs:
                im = im0.max(axis=2)
                nimg += 1
                if mode == "rec":
                    res = rd.recognize(im, allowlist="0123456789")
                else:
                    res = rd.readtext(im, allowlist="0123456789")
                if res:
                    vals.append(G.parse_digits(res[0][1]))
                    cfs.append(float(res[0][2]))
                else:
                    vals.append(None)
            v = vals[0] if a.cond == "B" else G.combine_slots(vals)
            cf = min(cfs) if cfs else ""
            rows.append([s, i, c, "" if v is None else v,
                         "" if cf == "" else "%.4f" % cf])
        if len(rows) % 12000 < 7:
            print("  easyocr %d rows %.0fs" % (len(rows), time.perf_counter() - t0),
                  flush=True)
    return rows, time.perf_counter() - t0, {"stray": stray, "images": nimg}


def run_paddle_rec(a, songs, sel, roi, slots, batch=64):
    from paddleocr import TextRecognition
    m = TextRecognition()
    rows = []
    state = {"dt": 0.0, "stray": 0, "nimg": 0}
    pend = []

    def flush():
        if not pend:
            return
        imgs = [im for _k, ims in pend for im in ims]
        t0 = time.perf_counter()
        res = list(m.predict(imgs, batch_size=len(imgs)))
        state["dt"] += time.perf_counter() - t0
        state["nimg"] += len(imgs)
        pos = 0
        for key, ims in pend:
            out = res[pos:pos + len(ims)]
            pos += len(ims)
            if a.cond == "A":
                s, i = key
                txt = out[0].get("rec_text", "") if out else ""
                sc = out[0].get("rec_score", "") if out else ""
                hit, st = G.assign_in_order(txt)
                state["stray"] += st
                for c in COUNTERS:
                    v = G.parse_digits(hit.get(c))
                    rows.append([s, i, c, "" if v is None else v,
                                 "" if sc == "" else "%.4f" % float(sc)])
            else:
                s, i, c = key
                vs = [G.parse_digits(o.get("rec_text", "")) for o in out]
                sc = min([float(o.get("rec_score", 0)) for o in out]) if out else ""
                v = vs[0] if a.cond == "B" else G.combine_slots(vs)
                rows.append([s, i, c, "" if v is None else v,
                             "" if sc == "" else "%.4f" % float(sc)])
        del pend[:]

    nb = 0
    for key, imgs in iter_units(a, songs, sel, slots):
        pend.append((key, imgs))
        nb += len(imgs)
        if nb >= batch:
            flush()
            nb = 0
            if len(rows) % 6000 < 7:
                print("  paddle %d rows %.0fs" % (len(rows), state["dt"]), flush=True)
    flush()
    return rows, state["dt"], {"stray": state["stray"], "images": state["nimg"]}


def run_paddle_full(a, songs, sel, roi, slots):
    from paddleocr import PaddleOCR
    m = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False, enable_mkldnn=False)
    rows = []
    stray = 0
    nimg = 0
    dt = 0.0
    for key, imgs in iter_units(a, songs, sel, slots):
        t0 = time.perf_counter()
        res = m.predict(imgs[0])
        dt += time.perf_counter() - t0
        nimg += 1
        items = []
        for o in res:
            ts = o.get("rec_texts", [])
            bx = o.get("rec_boxes", o.get("dt_polys", []))
            for j, t in enumerate(ts):
                try:
                    cx = float(np.asarray(bx[j]).reshape(-1, 2)[:, 0].mean())
                except Exception:
                    cx = 0.0
                items.append((cx, t))
            break
        if a.cond == "A":
            s, i = key
            hit, st = G.assign_by_x(items, roi[s])
            stray += st
            for c in COUNTERS:
                v = G.parse_digits(hit.get(c))
                rows.append([s, i, c, "" if v is None else v, ""])
        else:
            s, i, c = key
            v = G.parse_digits("".join(t for _x, t in sorted(items)))
            rows.append([s, i, c, "" if v is None else v, ""])
        if len(rows) % 3000 < 7:
            print("  paddle_full %d rows %.0fs" % (len(rows), dt), flush=True)
    return rows, dt, {"stray": stray, "images": nimg}


CANDS = {
    "ncc": lambda a, S, sel, roi, sl: run_tpl(a, S, sel, roi, sl, "ncc"),
    "nn_l2": lambda a, S, sel, roi, sl: run_tpl(a, S, sel, roi, sl, "nn"),
    "tess_psm7": lambda a, S, sel, roi, sl: run_tess(a, S, sel, roi, sl, 7),
    "tess_psm8": lambda a, S, sel, roi, sl: run_tess(a, S, sel, roi, sl, 8),
    "easyocr_rec": lambda a, S, sel, roi, sl: run_easyocr(a, S, sel, roi, sl, "rec"),
    "easyocr_readtext": lambda a, S, sel, roi, sl: run_easyocr(a, S, sel, roi, sl, "read"),
    "paddle_rec": lambda a, S, sel, roi, sl: run_paddle_rec(a, S, sel, roi, sl),
    "paddle_full": lambda a, S, sel, roi, sl: run_paddle_full(a, S, sel, roi, sl),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["A", "B", "C", "C2"])
    ap.add_argument("--cand", required=True, choices=sorted(CANDS))
    ap.add_argument("--root", default="/mnt/g/研究アーカイブ/成果物/動画解析")
    ap.add_argument("--results", default="results_wao_20261014")
    ap.add_argument("--prev-work", default="/home/toshiaki/ocr_gran_20261029/work_b")
    ap.add_argument("--band", default="/home/toshiaki/ocr_gran_20261029/band")
    ap.add_argument("--slots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slots = json.load(open(a.slots))
    songs, sel, roi = load_inputs(a)
    ncells = sum(len(sel[s]) for s in songs) * len(COUNTERS)
    sfx = "" if a.nshards == 1 else ".shard%dof%d" % (a.shard, a.nshards)
    print("cond=%s cand=%s sub=%d shard=%d/%d cells=%d"
          % (a.cond, a.cand, a.sub, a.shard, a.nshards, ncells), flush=True)
    rows, dt, extra = CANDS[a.cand](a, songs, sel, roi, slots)
    assert len(rows) == ncells, "rows %d != cells %d" % (len(rows), ncells)
    name = "%s_%s" % (a.cond, a.cand)
    with open(os.path.join(a.out, "pred_%s%s.csv" % (name, sfx)), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "sample_idx", "counter", "pred", "conf"])
        w.writerows(rows)
    meta = {"cond": a.cond, "cand": a.cand, "sub": a.sub, "cells": ncells,
            "seconds": round(dt, 3),
            "sec_per_1000_cells": round(1000.0 * dt / max(1, ncells), 3),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "shard": a.shard, "nshards": a.nshards,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta.update(extra)
    json.dump(meta, open(os.path.join(a.out, "time_%s%s.json" % (name, sfx)), "w"),
              ensure_ascii=False, indent=1)
    print("done %s: %.1fs %.3f s/1000cells stray=%s"
          % (name, dt, 1000.0 * dt / max(1, ncells), extra.get("stray")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
