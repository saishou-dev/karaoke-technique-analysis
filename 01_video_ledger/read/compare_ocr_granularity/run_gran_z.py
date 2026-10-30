#!/usr/bin/env python3
"""run_gran_z.py ＝ 条件Z（切り出し前＝全画面のフレーム）を A〜C2 と同じ土俵で測る（2026-10-30）

条件Z
  入力 ＝ 元動画（3840x2160）の全画面フレームそのもの。関心領域の切り出しをしない。
  フレーム列 ＝ sub4 と同じ（work_b/<曲>/frames.npy の標本番号 i のうち i % 4 == 0）。
  手法 ＝ 文字の検出を持つ3つだけ（EasyOCR readtext・PaddleOCR 一式・Tesseract psm 11）。
  割り当て ＝ 検出した文字箱の中心を帯の座標へ写し、帯の矩形（0≦x<2480・0≦y<120）の中にある箱だけを
             条件A と同じ規則（gran_core.assign_by_x＝中心 x がカウンターの数字窓の x 範囲に入るものを採る）で
             6カウンターへ割り当てる。帯の外の箱は「帯外の読み」として件数だけ数える。
  後処理と採点 ＝ A〜C2 と同一（score_gran_z.py が score_gran.py をそのまま使う）。

帯の矩形 ＝ dataset/band_probe_v2_20261014.csv（xL,yL,w,h,angle_deg）と、帯動画を作ったときの変換
  crop(余白 20/48) → rotate(-angle) → crop(w,h) → scale 2480x120（scripts/wincrop_20260915.py と同じ式）。
  cv2 で同じ変換を作り、帯動画のフレーム（band/<曲>.npy）との画素差で対応を確かめてから進む。

可変フレームレートのため、元動画はシークせず逐次デコードで読む。

使い方
  $PY run_gran_z.py extract --out ~/ocr_gran_20261029/full_z     # フレーム採取＋対応確認（1曲目で確認できなければ止まる）
  $PY run_gran_z.py run --cand paddle_full --frames ~/ocr_gran_20261029/full_z --out pred
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

COUNTERS = G.COUNTERS
ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"
VIDEO = "/mnt/g/研究アーカイブ/ビデオ素材/20261014"
WORK = "/home/toshiaki/ocr_gran_20261029/work_b"
BAND = "/home/toshiaki/ocr_gran_20261029/band"
HB = "/home/toshiaki/ocr_gran_20261029/heartbeat_z.txt"
MX, MY = 20, 48          # 帯動画を作ったときの余白（wincrop と同じ）
BW, BH = 2480, 120
SUB = 4
CANDS = ["easyocr_readtext", "paddle_full", "tess_psm11"]
TESS_CHUNK = 20          # 4K の画像を一度に渡す枚数


def hb(msg):
    with open(HB, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (msg, time.strftime("%Y-%m-%d %H:%M:%S")))


def songs_of(work):
    return [d for d in sorted(os.listdir(work)) if os.path.isdir(os.path.join(work, d))]


def probe_params(root):
    p = {}
    with open(os.path.join(root, "dataset", "band_probe_v2_20261014.csv"),
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p[r["video"]] = (int(r["xL"]), int(r["yL"]), int(r["w"]), int(r["h"]),
                             float(r["angle_deg"]))
    return p


class BandMap:
    """全画面 ↔ 帯の座標の変換。ffmpeg の rotate=-radians(angle) は時計回り＝cv2 の角度 +angle。
    （符号は 2026-10-30 の試作で実測＝+angle で画素差の中央値 3.2・-angle で 11.8）"""

    def __init__(self, prm):
        self.xL, self.yL, self.w, self.h, self.ang = prm
        self.x0, self.y0 = self.xL - MX, self.yL - MY
        self.W1, self.H1 = self.w + 2 * MX, self.h + 2 * MY
        self.M = cv2.getRotationMatrix2D(((self.W1 - 1) / 2.0, (self.H1 - 1) / 2.0),
                                         self.ang, 1.0)
        self.sx, self.sy = BW / float(self.w), BH / float(self.h)

    def band_image(self, img):
        c = img[self.y0:self.y0 + self.H1, self.x0:self.x0 + self.W1]
        r = cv2.warpAffine(c, self.M, (self.W1, self.H1), flags=cv2.INTER_LINEAR,
                           borderValue=0)
        r = r[MY:MY + self.h, MX:MX + self.w]
        return cv2.resize(r, (BW, BH), interpolation=cv2.INTER_CUBIC)

    def to_band(self, X, Y):
        px, py = X - self.x0, Y - self.y0
        qx = self.M[0, 0] * px + self.M[0, 1] * py + self.M[0, 2] - MX
        qy = self.M[1, 0] * px + self.M[1, 1] * py + self.M[1, 2] - MY
        return qx * self.sx, qy * self.sy

    def inside(self, bx, by):
        return 0 <= bx < BW and 0 <= by < BH


# ---------------- フレーム採取と対応確認 ----------------

def cmd_extract(a):
    os.makedirs(a.out, exist_ok=True)
    prm = probe_params(a.root)
    totals = {}
    with open(os.path.join(a.work, "input_summary.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            totals[r["video"]] = int(r["frames_total"])
    report = {}
    cj = os.path.join(a.out, "correspondence.json")
    if os.path.exists(cj):
        report.update(json.load(open(cj, encoding="utf-8")))
    for k, s in enumerate(songs_of(a.work)):
        if a.songs and s not in a.songs.split(","):
            continue
        if s in report and not report[s].get("ok_pixel", True):
            continue
        fr = np.load(os.path.join(a.work, s, "frames.npy"))
        band = np.load(os.path.join(a.band, s + ".npy"), mmap_mode="r")
        sel = [i for i in range(len(fr)) if i % SUB == 0]
        want = {int(fr[i]): j for j, i in enumerate(sel)}
        bm = BandMap(prm[s])
        dst = os.path.join(a.out, s + ".npy")
        part = dst + ".part.npy"
        if os.path.exists(dst) and np.load(dst, mmap_mode="r").shape[0] == len(sel):
            print("skip %s (already extracted)" % s, flush=True)
            report.setdefault(s, {"video": s, "skipped": "already extracted"})
            continue
        arr = None
        diffs = []
        t0 = time.time()
        cap = cv2.VideoCapture(os.path.join(a.video, s + ".mp4"))
        fidx = -1
        got = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            fidx += 1
            j = want.get(fidx)
            if j is None:
                continue
            if arr is None:
                arr = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8,
                                                shape=(len(sel),) + img.shape)
            arr[j] = img
            i = sel[j]
            b = bm.band_image(img).astype(np.int16)
            d = []
            for di in (-1, 0, 1):
                ii = i + di
                if 0 <= ii < len(band):
                    d.append(float(np.abs(b - np.asarray(band[ii]).astype(np.int16)).mean()))
                else:
                    d.append(float("nan"))
            diffs.append([i] + d)
            got += 1
            if got % 100 == 0:
                hb("Z extract %s %d/%d" % (s, got, len(sel)))
        cap.release()
        n_dec = fidx + 1
        D = np.array(diffs, dtype=float)
        med = [float(np.nanmedian(D[:, c])) for c in (1, 2, 3)]
        p95 = float(np.nanpercentile(D[:, 2], 95))
        not_min = int(np.sum((D[:, 2] > np.fmin(D[:, 1], D[:, 3])) &
                             ~np.isnan(D[:, 1]) & ~np.isnan(D[:, 3])))
        ok_n = (got == len(sel)) and (n_dec == totals.get(s, -1))
        # 対応の判定（2026-10-30 改め）＝同じ番号の差の中央値が前後の番号の半分未満・95%点が 10 以下・
        # 同じ番号が最小にならないフレームは「画面が止まっていて前後と差が付かない」場合だけ（超過 2.0 以下）。
        # 当初の「中央値 5.0 以下」はスターダストで 5.45（前後 12.97／12.65・95%点 7.58・超過の最大 1.21）となり、
        # 回転と拡大の補間・再圧縮の残差の大きさが曲ごとに違うだけで対応のずれではないため、絶対値の条件を外した。
        exc = D[:, 2] - np.fmin(D[:, 1], D[:, 3])
        exc_max = float(np.nanmax(np.where(exc > 0, exc, 0.0)))
        ok_px = (med[1] < 0.5 * min(med[0], med[2])) and p95 <= 10.0 and exc_max <= 2.0
        rep = {"video": s, "decoded_frames": n_dec, "frames_total_ref": totals.get(s),
               "selected": len(sel), "got": got,
               "median_absdiff_prev_same_next": [round(x, 3) for x in med],
               "p95_absdiff_same": round(p95, 3),
               "same_not_minimum": not_min, "same_not_minimum_max_excess": round(exc_max, 3),
               "ok_count": bool(ok_n), "ok_pixel": bool(ok_px),
               "seconds": round(time.time() - t0, 1)}
        report[s] = rep
        np.savetxt(os.path.join(a.out, s + "_diff.csv"), D, delimiter=",",
                   header="sample_idx,diff_prev,diff_same,diff_next", comments="", fmt="%.4f")
        print(json.dumps(rep, ensure_ascii=False), flush=True)
        hb("Z extract %s done same=%.2f prev=%.2f next=%.2f ok=%s" %
           (s, med[1], med[0], med[2], ok_n and ok_px))
        del arr
        json.dump(report, open(os.path.join(a.out, "correspondence.json"), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=1)
        if not (ok_n and ok_px):
            # 対応が確かめられない曲は条件Z から外す（元動画は読み直さない・2026-10-30 Fable 指示「読み込みは1回だけ」）。
            # 1曲目で確かめられなければ Z そのものを中止する。
            if os.path.exists(part):
                os.remove(part)
            if k == 0:
                hb("Z_ABORT correspondence %s" % s)
                return 2
            hb("Z_EXCLUDE correspondence %s" % s)
            continue
        os.replace(part, dst)
    json.dump(report, open(os.path.join(a.out, "correspondence.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    hb("Z_EXTRACT_DONE")
    return 0


# ---------------- 読み取り ----------------

def split_items(items_full, bm):
    """[(X, Y, text)] 全画面の座標 → (帯の中の [(bx, text)], 帯外の件数, 帯外で数字を含む件数)"""
    inside, out, out_dig = [], 0, 0
    for X, Y, t in items_full:
        bx, by = bm.to_band(X, Y)
        if bm.inside(bx, by):
            inside.append((bx, t))
        else:
            out += 1
            if G.parse_runs(t):
                out_dig += 1
    return inside, out, out_dig


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    prm = probe_params(a.root)
    songs = [s for s in songs_of(a.work)
             if os.path.exists(os.path.join(a.frames, s + ".npy"))]   # 対応が確かめられた曲だけ
    roi = {s: {k: tuple(v) for k, v in json.load(open(
        os.path.join(a.root, a.results, s, "roi.json"))).items()} for s in songs}
    sel = {}
    for s in songs:
        n = np.load(os.path.join(a.work, s, "frames.npy")).shape[0]
        sel[s] = [i for i in range(n) if i % SUB == 0]
    nfr = sum(len(sel[s]) for s in songs)
    # 部分集合（4K の全画面は検出の重い手法で 1フレーム 20〜75秒かかるため・2026-10-30 試走の実測）
    #   --every N ＝ sub4 のフレーム列のうち標本番号 i が N の倍数のものだけ（N は 4 の倍数）
    #   --shard/--nshards ＝ 同じ列を k 本おきに分け、別プロセス（各専用4コア）で並べて走らせる
    if a.every % SUB:
        raise SystemExit("--every は 4 の倍数")
    jmap = {s: {i: j for j, i in enumerate(sel[s])} for s in songs}
    for s in songs:
        sel[s] = [i for i in sel[s] if i % a.every == 0]
    if a.nshards > 1:
        allk = [(s, i) for s in songs for i in sel[s]][a.shard::a.nshards]
        for s in songs:
            sel[s] = [i for (t, i) in allk if t == s]
    if a.limit:
        for s in songs:
            sel[s] = sel[s][:a.limit]
    ncells = sum(len(sel[s]) for s in songs) * len(COUNTERS)
    print("cond=Z cand=%s frames=%d cells=%d" % (a.cand, ncells // 6, ncells), flush=True)

    st = {"dt": 0.0, "stray": 0, "out": 0, "out_digit": 0, "inside": 0, "nimg": 0,
          "crash": 0}
    rows = []
    done = [0]

    def emit(s, i, items_full, bm):
        inside, o, od = split_items(items_full, bm)
        st["out"] += o
        st["out_digit"] += od
        st["inside"] += len(inside)
        hit, sy = G.assign_by_x(inside, roi[s])
        st["stray"] += sy
        for c in COUNTERS:
            v = G.parse_digits(hit.get(c))
            rows.append([s, i, c, "" if v is None else v, ""])
        done[0] += 1
        if done[0] % a.hb_every == 0:
            hb("Z %s %d/%d dt=%.0fs" % (a.cand, done[0], ncells // 6, st["dt"]))

    if a.cand == "easyocr_readtext":
        import easyocr
        rd = easyocr.Reader(["en"], gpu=False, verbose=False)
        for s in songs:
            bm = BandMap(prm[s])
            arr = np.load(os.path.join(a.frames, s + ".npy"), mmap_mode="r")
            for i in sel[s]:
                im = np.asarray(arr[jmap[s][i]]).max(axis=2)
                t0 = time.perf_counter()
                res = rd.readtext(im, allowlist="0123456789")
                st["dt"] += time.perf_counter() - t0
                st["nimg"] += 1
                items = [(float(np.mean([p[0] for p in b[0]])),
                          float(np.mean([p[1] for p in b[0]])), b[1]) for b in res]
                emit(s, i, items, bm)
    elif a.cand == "paddle_full":
        from paddleocr import PaddleOCR
        m = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                      use_textline_orientation=False, enable_mkldnn=False)
        for s in songs:
            bm = BandMap(prm[s])
            arr = np.load(os.path.join(a.frames, s + ".npy"), mmap_mode="r")
            for i in sel[s]:
                img = np.ascontiguousarray(arr[jmap[s][i]])
                t0 = time.perf_counter()
                res = m.predict(img)
                st["dt"] += time.perf_counter() - t0
                st["nimg"] += 1
                items = []
                for o in res:
                    ts = o.get("rec_texts", [])
                    bx = o.get("rec_boxes", o.get("dt_polys", []))
                    for k, t in enumerate(ts):
                        try:
                            q = np.asarray(bx[k]).reshape(-1, 2)
                            X, Y = float(q[:, 0].mean()), float(q[:, 1].mean())
                        except Exception:
                            X, Y = -1.0, -1.0
                        items.append((X, Y, t))
                    break
                emit(s, i, items, bm)
    elif a.cand == "tess_psm11":
        tmp = tempfile.mkdtemp(prefix="tessz_")
        try:
            buf = []

            def flush():
                if not buf:
                    return
                files = []
                for k, (s, i, img) in enumerate(buf):
                    p = os.path.join(tmp, "%04d.png" % k)
                    cv2.imwrite(p, G.prep(img, scale=a.tess_scale))
                    files.append(p)
                lst = os.path.join(tmp, "list.txt")
                with open(lst, "w") as f:
                    f.write("\n".join(files) + "\n")
                base = os.path.join(tmp, "out")
                t0 = time.perf_counter()
                r = _tess(lst, base)
                st["dt"] += time.perf_counter() - t0
                pages = _tsv_pages(base + ".tsv", len(files)) if r.returncode == 0 else None
                if pages is None:
                    pages = []
                    for p in files:
                        with open(lst, "w") as f:
                            f.write(p + "\n")
                        t0 = time.perf_counter()
                        r1 = _tess(lst, base)
                        st["dt"] += time.perf_counter() - t0
                        if r1.returncode == 0:
                            pages += _tsv_pages(base + ".tsv", 1)
                        else:
                            pages.append([])
                            st["crash"] += 1
                for (s, i, _img), pg in zip(buf, pages):
                    sc = float(a.tess_scale)
                    items = [(X / sc, Y / sc, t) for X, Y, t in pg]
                    emit(s, i, items, BandMap(prm[s]))
                    st["nimg"] += 1
                for p in files:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                del buf[:]

            for s in songs:
                arr = np.load(os.path.join(a.frames, s + ".npy"), mmap_mode="r")
                for i in sel[s]:
                    buf.append((s, i, np.asarray(arr[jmap[s][i]])))
                    if len(buf) >= TESS_CHUNK:
                        flush()
            flush()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        raise SystemExit("unknown cand")

    assert len(rows) == ncells, "rows %d != cells %d" % (len(rows), ncells)
    sfx = "" if not a.limit else ".limit%d" % a.limit
    if a.nshards > 1:
        sfx += ".shard%dof%d" % (a.shard, a.nshards)
    name = "Z_%s" % a.cand
    with open(os.path.join(a.out, "pred_%s%s.csv" % (name, sfx)), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "sample_idx", "counter", "pred", "conf"])
        w.writerows(rows)
    meta = {"cond": "Z", "cand": a.cand, "sub": SUB, "cells": ncells,
            "frames": ncells // 6, "frames_full_list": nfr,
            "seconds": round(st["dt"], 3),
            "sec_per_1000_cells": round(1000.0 * st["dt"] / max(1, ncells), 3),
            "sec_per_frame": round(st["dt"] / max(1, ncells // 6), 3),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "stray": st["stray"], "out_of_band": st["out"],
            "out_of_band_digit": st["out_digit"], "inside_band_boxes": st["inside"],
            "images": st["nimg"], "crash_cells": st["crash"],
            "tess_scale": a.tess_scale if a.cand == "tess_psm11" else "",
            "songs": songs, "every": a.every, "shard": a.shard, "nshards": a.nshards,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(meta, open(os.path.join(a.out, "time_%s%s.json" % (name, sfx)), "w"),
              ensure_ascii=False, indent=1)
    print("done %s: %.1fs %.3f s/1000cells out_of_band=%d stray=%d"
          % (name, st["dt"], meta["sec_per_1000_cells"], st["out"], st["stray"]), flush=True)
    if not a.limit:
        hb("Z_LANE_DONE %s%s" % (a.cand, sfx))
        exp = os.environ.get("Z_EXPECT", "").split()
        if exp and all(os.path.exists(os.path.join(a.out, e)) for e in exp):
            hb("Z_DONE")
    return 0


def _tess(lst, base):
    cmd = ["tesseract", lst, base, "--psm", "11", "-l", "eng",
           "-c", "tessedit_char_whitelist=0123456789",
           "-c", "classify_bln_numeric_mode=1", "tsv"]
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
                pages[p].append((float(r["left"]) + float(r["width"]) / 2.0,
                                 float(r["top"]) + float(r["height"]) / 2.0, t))
            except (ValueError, TypeError, KeyError):
                continue
    return pages


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--out", required=True)
    e.add_argument("--songs", default="", help="この曲だけ採取（,区切り）")
    r = sp.add_parser("run")
    r.add_argument("--cand", required=True, choices=CANDS)
    r.add_argument("--frames", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0, help="試走用＝各曲の先頭この枚数だけ")
    r.add_argument("--every", type=int, default=4, help="標本番号 i が この数の倍数のフレームだけ読む（既定4＝sub4 の全部）")
    r.add_argument("--hb-every", type=int, default=25,
                   help="この枚数ごとに heartbeat_z.txt へ進捗を書く（番犬の停滞判定 20分より短くなるように選ぶ）")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    r.add_argument("--tess-scale", type=float, default=1.0,
                   help="Tesseract の前処理の拡大率（A〜C2 は 3・全画面 4K では 1）")
    for p in (e, r):
        p.add_argument("--root", default=ROOT)
        p.add_argument("--video", default=VIDEO)
        p.add_argument("--work", default=WORK)
        p.add_argument("--band", default=BAND)
        p.add_argument("--results", default="results_wao_20261014")
    a = ap.parse_args()
    return cmd_extract(a) if a.cmd == "extract" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
