#!/usr/bin/env python3
"""run_candidates.py ＝ 同一の入力（extract_rois.py が保存した関心領域）を各手法に読ませる（2026-10-18）

候補（--cand で1つ指定・1プロセス1候補＝処理時間を混ぜないため）
  ncc              現行＝自作の見本照合（正規化相互相関 NCC・読取器 v0.8e と同一の判定式）  ＝基準
  nn_l2            対照＝同じ箱の切り出しのまま、分類だけ「0〜9の見本への生ピクセル最近傍」（正規化なし・棄権なし）
  tess_psm7        Tesseract 5.5.0（psm 7＝1行として読む・数字のみ許可・前処理あり）
  tess_psm8        Tesseract 5.5.0（psm 8＝1語として読む・数字のみ許可・前処理あり）
  tess_psm8_raw    Tesseract 5.5.0（psm 8・前処理なし＝V画像そのまま）
  easyocr_rec      EasyOCR 1.7.2（認識のみ recognize・数字許可リスト・前処理なし）
  easyocr_rec_prep EasyOCR 1.7.2（認識のみ・前処理あり＝Tesseract と同じ前処理）
  easyocr_readtext EasyOCR 1.7.2（検出＋認識 readtext・数字許可リスト・前処理なし）
  paddle_rec       PaddleOCR 3.7.0 / PP-OCRv6_medium_rec（認識のみ TextRecognition・前処理なし）

前処理（「前処理あり」の中身）＝V=max(B,G,R) → 3倍に拡大（双三次）→ 大津の二値化 → 白黒反転。
  既製OCRは「白地に黒字」を前提にしているため、暗い帯に明るい数字が乗る画面はそのまま渡すと不利になる。
  これを渡すことで「前処理をしていないから読めない」という反論を潰す。

出力（--out 配下）
  pred_<cand>.csv  … video,sample_idx,counter,pred,conf（pred 空欄＝「読まない」＝棄権）
  time_<cand>.json … 処理時間（モデル読み込みを除いた本体の秒数・1,000コマあたり秒・コア数）

使い方:
  python run_candidates.py --work <work> --out <out> --cand ncc
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glyph_core as gc

COUNTERS = gc.COUNTERS
DIGITS = re.compile(r"\d+")


def prep(crop, scale=3):
    """既製OCR向けの前処理＝V → 3倍拡大 → 大津の二値化 → 反転（白地に黒字）。"""
    g = crop.max(axis=2)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - b


def parse_digits(s, maxlen=2):
    """OCRの出力文字列から数字だけ拾う。空なら None（棄権）。長すぎる並びはそのまま整数化（＝誤読として数える）。"""
    if s is None:
        return None
    m = DIGITS.findall(str(s))
    if not m:
        return None
    t = "".join(m)
    if len(t) > 6:
        t = t[:6]
    return int(t)


def load_work(work, shard=0, nshards=1):
    """nshards>1 のときは、(曲×カウンター) ごとに添字を飛び飛びに取り分ける
    （同じ大きさの画像がまとまるので、まとめ読みの効率は落ちない）。
    どの分割でも同じ画像を読むので結果は変わらない。処理時間も1プロセス4コアのままで比較できる。"""
    songs = [d for d in sorted(os.listdir(work))
             if os.path.isdir(os.path.join(work, d)) and os.path.exists(os.path.join(work, d, "frames.npy"))]
    data = {}
    idxmap = {}
    for s in songs:
        data[s] = {}
        idxmap[s] = {}
        for c in COUNTERS:
            arr = np.load(os.path.join(work, s, f"roi_{c}.npy"))
            idx = list(range(shard, arr.shape[0], nshards)) if nshards > 1 else list(range(arr.shape[0]))
            data[s][c] = arr[idx] if nshards > 1 else arr
            idxmap[s][c] = idx
    return songs, data, idxmap


# ---------------- 各候補 ----------------

def cand_ncc(songs, data, root):
    tpl = gc.Templates(os.path.join(root, "templates", "wao_20260816", "cluster_reps_wao_ext.npy"))
    out = []
    t0 = time.perf_counter()
    for s in songs:
        for c in COUNTERS:
            arr = data[s][c]
            for i in range(arr.shape[0]):
                v = gc.read_value_ncc(arr[i], tpl)
                out.append([s, i, c, "" if v is None else v, ""])
    return out, time.perf_counter() - t0


def cand_nn(songs, data, root):
    tpl = gc.Templates(os.path.join(root, "templates", "wao_20260816", "cluster_reps_wao_ext.npy"))
    out = []
    t0 = time.perf_counter()
    for s in songs:
        for c in COUNTERS:
            arr = data[s][c]
            for i in range(arr.shape[0]):
                v = gc.read_value_nn(arr[i], tpl)
                out.append([s, i, c, "" if v is None else v, ""])
    return out, time.perf_counter() - t0


TESS_CHUNK = 500


def _tess_call(lst, base, psm):
    return subprocess.run(["tesseract", lst, base, "--psm", str(psm), "-l", "eng",
                           "-c", "tessedit_char_whitelist=0123456789",
                           "-c", "classify_bln_numeric_mode=1"],
                          capture_output=True, text=True)


def cand_tess(songs, data, root, psm, do_prep):
    """Tesseract は「ファイル一覧を1回で渡す」一括実行で測る（1枚ごとにプロセスを起こす
    オーバーヘッドを処理時間に混ぜないため）。
    ただし tesseract 5.5.0 は一部のコマ（二値化後に一様な画像など）で SIGFPE で落ちるため、
    500枚ずつのまとまりで実行し、落ちたまとまりだけ1枚ずつ再実行する。
    1枚でも落ちたコマは「読まない」として扱い、件数を crash_cells として記録する。"""
    tmp = tempfile.mkdtemp(prefix="tess_")
    index = []
    files = []
    try:
        for s in songs:
            for c in COUNTERS:
                arr = data[s][c]
                for i in range(arr.shape[0]):
                    img = prep(arr[i]) if do_prep else arr[i].max(axis=2)
                    p = os.path.join(tmp, f"{len(files):06d}.png")
                    cv2.imwrite(p, img)
                    files.append(p)
                    index.append((s, i, c))
        texts = [None] * len(files)
        crash = 0
        dt = 0.0
        for b0 in range(0, len(files), TESS_CHUNK):
            chunk = files[b0:b0 + TESS_CHUNK]
            lst = os.path.join(tmp, "list.txt")
            with open(lst, "w") as f:
                f.write("\n".join(chunk) + "\n")
            base = os.path.join(tmp, "out")
            t0 = time.perf_counter()
            r = _tess_call(lst, base, psm)
            pages = None
            if r.returncode == 0:
                txt = open(base + ".txt", encoding="utf-8", errors="replace").read()
                pg = txt.split("\f")
                if pg and pg[-1].strip() == "":
                    pg = pg[:-1]
                if len(pg) == len(chunk):
                    pages = pg
            dt += time.perf_counter() - t0
            if pages is None:
                # 落ちたまとまりだけ1枚ずつ
                pages = []
                for p in chunk:
                    with open(lst, "w") as f:
                        f.write(p + "\n")
                    t0 = time.perf_counter()
                    r1 = _tess_call(lst, base, psm)
                    dt += time.perf_counter() - t0
                    if r1.returncode == 0 and os.path.exists(base + ".txt"):
                        pages.append(open(base + ".txt", encoding="utf-8", errors="replace").read())
                    else:
                        pages.append("")
                        crash += 1
            texts[b0:b0 + len(chunk)] = pages
            if (b0 // TESS_CHUNK) % 10 == 0:
                print(f"  tess {b0+len(chunk)}/{len(files)} crash={crash} {dt:.0f}s", flush=True)
        out = []
        for (s, i, c), pg in zip(index, texts):
            v = parse_digits(pg)
            out.append([s, i, c, "" if v is None else v, ""])
        print(f"  tess crash_cells={crash}", flush=True)
        return out, dt, {"crash_cells": crash}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cand_easyocr(songs, data, root, mode, do_prep, join_lr=False):
    import easyocr
    rd = easyocr.Reader(["en"], gpu=False, verbose=False)
    out = []
    t0 = time.perf_counter()
    for s in songs:
        for c in COUNTERS:
            arr = data[s][c]
            print(f"  easyocr {s[:12]} {c} n={arr.shape[0]} "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
            for i in range(arr.shape[0]):
                img = prep(arr[i]) if do_prep else arr[i].max(axis=2)
                if mode == "rec":
                    res = rd.recognize(img, allowlist="0123456789")
                else:
                    res = rd.readtext(img, allowlist="0123456789")
                if res:
                    if join_lr and len(res) > 1:
                        # 検出された枠が複数あるときは左から右へつなぐ（2桁が2枠に分かれる場合の対策）
                        srt = sorted(res, key=lambda r: min(p[0] for p in r[0]))
                        txt = "".join(str(r[1]) for r in srt)
                        conf = min(float(r[2]) for r in srt)
                    else:
                        txt = res[0][1]
                        conf = res[0][2]
                else:
                    txt, conf = "", ""
                v = parse_digits(txt)
                out.append([s, i, c, "" if v is None else v,
                            "" if conf == "" else f"{float(conf):.4f}"])
    return out, time.perf_counter() - t0


def cand_paddle(songs, data, root, batch=64):
    from paddleocr import TextRecognition
    m = TextRecognition()
    out = []
    dt = 0.0
    for s in songs:
        for c in COUNTERS:
            arr = data[s][c]
            n = arr.shape[0]
            print(f"  paddle {s[:12]} {c} n={n} {dt:.0f}s", flush=True)
            for b0 in range(0, n, batch):
                imgs = [arr[i] for i in range(b0, min(n, b0 + batch))]
                t0 = time.perf_counter()
                res = list(m.predict(imgs, batch_size=len(imgs)))
                dt += time.perf_counter() - t0
                for k, o in enumerate(res):
                    txt = o.get("rec_text", "")
                    sc = o.get("rec_score", "")
                    v = parse_digits(txt)
                    out.append([s, b0 + k, c, "" if v is None else v,
                                "" if sc == "" else f"{float(sc):.4f}"])
    return out, dt


def cand_paddle_full(songs, data, root):
    """PaddleOCR の本来の使い方（文字の位置検出＋認識の一式）。認識のみより重いので、
    実行は --nshards で間引いた部分集合に対して行い、採点側でその部分集合だけを集計する。"""
    from paddleocr import PaddleOCR
    # enable_mkldnn=False が要る。既定（oneDNN 有効）だと文字検出モデルの推論が
    #   NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
    #     [pir::ArrayAttribute<pir::DoubleAttribute>] (onednn_instruction.cc:116)
    # で落ちる（paddlepaddle 3.3.1 + paddleocr 3.7.0・2026-10-18 実測）。
    m = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False, enable_mkldnn=False)
    out = []
    dt = 0.0
    for s in songs:
        for c in COUNTERS:
            arr = data[s][c]
            print(f"  paddle_full {s[:12]} {c} n={arr.shape[0]} {dt:.0f}s", flush=True)
            for i in range(arr.shape[0]):
                t0 = time.perf_counter()
                res = m.predict(arr[i])
                dt += time.perf_counter() - t0
                txt, sc = "", ""
                for o in res:
                    ts = o.get("rec_texts", [])
                    ss = o.get("rec_scores", [])
                    bx = o.get("rec_boxes", o.get("dt_polys", []))
                    if len(ts):
                        try:
                            orderx = sorted(range(len(ts)),
                                            key=lambda j: float(np.asarray(bx[j]).reshape(-1, 2)[:, 0].min()))
                        except Exception:
                            orderx = list(range(len(ts)))
                        txt = "".join(str(ts[j]) for j in orderx)
                        sc = min(float(x) for x in ss) if len(ss) else ""
                    break
                v = parse_digits(txt)
                out.append([s, i, c, "" if v is None else v,
                            "" if sc == "" else f"{float(sc):.4f}"])
    return out, dt


CANDS = {
    "ncc": lambda S, D, R: cand_ncc(S, D, R),
    "nn_l2": lambda S, D, R: cand_nn(S, D, R),
    "tess_psm7": lambda S, D, R: cand_tess(S, D, R, 7, True),
    "tess_psm8": lambda S, D, R: cand_tess(S, D, R, 8, True),
    "tess_psm8_raw": lambda S, D, R: cand_tess(S, D, R, 8, False),
    "easyocr_rec": lambda S, D, R: cand_easyocr(S, D, R, "rec", False),
    "easyocr_rec_prep": lambda S, D, R: cand_easyocr(S, D, R, "rec", True),
    "easyocr_readtext": lambda S, D, R: cand_easyocr(S, D, R, "read", False),
    "easyocr_readtext_lr": lambda S, D, R: cand_easyocr(S, D, R, "read", False, join_lr=True),
    "paddle_rec": lambda S, D, R: cand_paddle(S, D, R),
    "paddle_full": lambda S, D, R: cand_paddle_full(S, D, R),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cand", required=True, choices=sorted(CANDS))
    ap.add_argument("--root", default="/mnt/g/研究アーカイブ/成果物/動画解析")
    ap.add_argument("--cores", type=int, default=int(os.environ.get("KV_CORES", "4")))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    songs, data, idxmap = load_work(a.work, a.shard, a.nshards)
    ncells = sum(data[s][c].shape[0] for s in songs for c in COUNTERS)
    suffix = "" if a.nshards == 1 else f".shard{a.shard}of{a.nshards}"
    print(f"cand={a.cand} songs={len(songs)} cells={ncells} shard={a.shard}/{a.nshards}", flush=True)
    res = CANDS[a.cand](songs, data, a.root)
    extra = {}
    if len(res) == 3:
        rows, dt, extra = res
    else:
        rows, dt = res
    assert len(rows) == ncells, f"rows {len(rows)} != cells {ncells}"
    if a.nshards > 1:
        # 行の sample_idx は分割後の連番なので、元の添字へ戻す
        pos = {}
        for r in rows:
            k = (r[0], r[2])
            r[1] = idxmap[r[0]][r[2]][r[1]]
    with open(os.path.join(a.out, f"pred_{a.cand}{suffix}.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "sample_idx", "counter", "pred", "conf"])
        w.writerows(rows)
    meta = {"cand": a.cand, "cells": ncells, "seconds": round(dt, 3),
            "sec_per_1000_cells": round(1000.0 * dt / ncells, 3),
            "cores": a.cores,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta.update(extra)
    meta["shard"] = a.shard
    meta["nshards"] = a.nshards
    json.dump(meta, open(os.path.join(a.out, f"time_{a.cand}{suffix}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"done {a.cand}: {dt:.1f}s  {1000.0*dt/ncells:.3f} s/1000cells", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
