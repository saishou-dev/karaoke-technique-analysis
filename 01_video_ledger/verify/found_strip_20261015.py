# -*- coding: utf-8 -*-
"""欠落イベント1件につき1枚の「帯の数字の帯状写真（ストリップ）」を逐次デコードで作る（2026-10-15）。

目的は2つ。
 (1) 確定分（found_exact.csv の EXACT）の根拠画像＝旧値 → 変化 → 新値 が1枚で見えること。
 (2) 未確定分（NOT_FOUND）＝本人が後から○×できる1枚（走査範囲を等間隔で並べる）。

シークは使わない（帯動画は passthrough＝VFR でシークがずれる）。動画を頭から1回通し、必要なフレームだけ ROI を貯める。

出力:
  EXACT     → dataset/found_20261015/<key>/evidence/strip_<video>_<counter>_<value>_f<ev>.png
  NOT_FOUND → dataset/found_review_<S>/<video>_<counter>_<値>.png（本人レビュー用・1件1枚）
使い方: python found_strip_20261015.py <key>
"""
import os, sys, csv, json, collections, cv2, numpy as np

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
OUTROOT = os.path.join(DS, "found_20261015")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import SESS  # noqa: E402

MAXTILES = 96


def imwrite(p, img):
    cv2.imencode(".png", img)[1].tofile(p)


def tile(img, lab, k):
    t = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    t = cv2.copyMakeBorder(t, 26, 0, 0, 3, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(t, lab, (2, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return t


def strip(tiles, title, per_row=16):
    """1行 per_row 枚で折り返す（1行に100枚並べると縮小表示で数字が読めない）。"""
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    lines = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        line = np.hstack(row)
        lines.append(line)
    w = max(l.shape[1] for l in lines)
    lines = [cv2.copyMakeBorder(l, 0, 6, 0, w - l.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20)) for l in lines]
    m = np.vstack(lines)
    m = cv2.copyMakeBorder(m, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(m, title, (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
    return m


def run(key):
    cfg = SESS[key]
    stag = str(cfg["session"]) + ("_ai" if cfg["reader"] == "ai" else "")
    outdir = os.path.join(OUTROOT, key)
    rev = os.path.join(DS, "found_review_" + stag)
    os.makedirs(rev, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(outdir, "found_exact.csv"), encoding="utf-8-sig")))
    pts_dir = os.path.join(DS, cfg["pts"])
    jobs = collections.defaultdict(list)   # video -> [(counter,value,a,b,step,kind)]
    for r in rows:
        v, c, val = r["video"], r["counter"], int(r["value"])
        lo = int(r["scan_from"])
        if r["kind"] == "EXACT":
            ev = int(r["event_frame"]); st = int(r["first_stable_frame"])
            a = max(0, ev - 40); b = st + 20
        else:
            a = lo; b = -1   # -1 = 末尾まで
        jobs[v].append([c, val, a, b, r["kind"], r])
    for name in jobs:
        roi = json.load(open(os.path.join(outdir, "roi", name + ".json"), encoding="utf-8"))
        vid = os.path.join(ROOT, cfg["bands"], name + ".mp4")
        cap = cv2.VideoCapture(vid)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pts = [float(x["pts_sec"]) for x in csv.DictReader(open(os.path.join(pts_dir, name + ".csv"), encoding="utf-8"))]
        for j in jobs[name]:
            if j[3] < 0:
                j[3] = N - 1
            j.append(max(1, (j[3] - j[2]) // MAXTILES))   # step
            j.append([])                                   # collected tiles
        f = -1
        while True:
            ok, img = cap.read()
            if not ok:
                break
            f += 1
            for c, val, a, b, kind, r, step, got in jobs[name]:
                if a <= f <= b and (f - a) % step == 0:
                    x1, y1, x2, y2 = roi[c]
                    got.append((f, img[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8].copy()))
        cap.release()
        for c, val, a, b, kind, r, step, got in jobs[name]:
            if not got:
                print("no frames", name, c, val); continue
            k = 3.0
            tl = [tile(im, f"f{ff}", k) for ff, im in got]
            t = pts[min(len(pts) - 1, int(r['event_frame']))] if r["event_frame"] else -1
            if kind == "EXACT":
                ttl = (f"{name} / {c} / value={val} / 増加フレーム f{r['event_frame']} "
                       f"(= {val-1} が最後に読めた f{int(r['event_frame'])-1} の次) / t={t:.3f}s / "
                       f"安定初出 f{r['first_stable_frame']} / step{step}")
                imwrite(os.path.join(outdir, "evidence", f"strip_{name}_{c}_{val}_f{r['event_frame']}.png"), strip(tl, ttl))
            else:
                ttl = (f"【未確定】{name} / {c} / {val} 回目が見つからない / 走査 f{a}-{b} step{step} "
                       f"/ ○×: この帯のどこかで {val-1}→{val} に増えていれば○・そのフレーム番号を書いてください")
                imwrite(os.path.join(rev, f"{name}_{c}_{val}.png"), strip(tl, ttl))
            print("wrote", name[:24], c, val, kind, len(got), "tiles", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])


def window(key, name, c, a, b, step=1, tag="win"):
    """任意区間の等間隔ストリップ（逐次デコード・シークなし）。目視で増加フレームを詰めるため。"""
    cfg = SESS[key]
    outdir = os.path.join(OUTROOT, key)
    roi = json.load(open(os.path.join(outdir, "roi", name + ".json"), encoding="utf-8"))
    x1, y1, x2, y2 = roi[c]
    cap = cv2.VideoCapture(os.path.join(ROOT, cfg["bands"], name + ".mp4"))
    got = []; f = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        if f < a:
            continue
        if f > b:
            break
        if (f - a) % step == 0:
            got.append((f, img[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8].copy()))
    cap.release()
    m = strip([tile(im, f"f{ff}", 3.0) for ff, im in got], f"{name} / {c} / f{a}-{b} step{step}")
    p = os.path.join(outdir, "evidence", f"{tag}_{name}_{c}_f{a}-{b}.png")
    imwrite(p, m); print(p, len(got), "tiles")
