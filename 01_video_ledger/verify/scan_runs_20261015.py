# -*- coding: utf-8 -*-
"""1 曲の帯動画を**逐次デコードで 1 回だけ通し**、指定カウンターを**両極性**（通常＋反転）で全フレーム読み、
   run-length 表（連続して同じ値が読めた区間）と、指定区間のストリップ画像を出す（2026-10-15）。

なぜ必要か＝
  * 帯動画は `-fps_mode passthrough`（VFR）。cv2 のシークはフレームがずれる（10/15 実測 7 フレーム）ので使わない。
  * 読取器 read_counters_*.py の二値化は `roi > T`（明るい数字・暗い背景）だけ。明背景の演出では数字が暗くなり
    不可読になるので、失敗したフレームは `255 - roi` でもう一度読む（found_pin_20261015.py と同じ）。
  * 読み過ぎ（read > csv）の偽イベント特定・欠落イベントの増加フレーム確定は、どちらも
    「この値が画面に出ていた区間はどこか」を全フレームで見ないと決められない。

出力（dataset/found_20261015/<key>/scan/）:
  <video>__<counter>__runs.csv   ... start_frame, n, value, polarity, t_start, t_end（pts 表の秒）
  <video>__<counter>__first.csv  ... 値ごとの初出・最終出現・増加フレーム（＝旧値が最後に読めたフレーム+1）
  strip_<video>__<counter>__f<a>-<b>.png ... --strip で指定した区間の帯状写真（目視用）

使い方:
  python scan_runs_20261015.py <key> "<video>" --counters=kobushi,vibrato,accent [--from=0] [--to=-1]
         [--strip=kobushi:13550-13900:2] [--strip=...] [--ncc=0.60]
"""
import os, sys, csv, json, collections, cv2, numpy as np

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
OUTROOT = os.path.join(DS, "found_20261015")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import SESS, read_val_wao, read_val_ai, GRAY, WAO_COUNTERS, AI_COUNTERS, ROI_WAO_DEFAULT, ROI_AI_DEFAULT  # noqa: E402
from found_strip_20261015 import tile, strip, imwrite  # noqa: E402


def get_roi(cfg, key, name):
    """ROI の実体は found_20261015/<key>/roi/<video>.json（再走査で使ったもの）→ results/<video>/roi.json → 既定。"""
    for p in (os.path.join(OUTROOT, key, "roi", name + ".json"),
              os.path.join(ROOT, cfg["results"], name, "roi.json")):
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    return dict(ROI_WAO_DEFAULT if cfg["reader"] == "wao" else ROI_AI_DEFAULT)


def load_pts(cfg, name):
    p = os.path.join(DS, cfg["pts"], name + ".csv")
    if not os.path.exists(p):
        return []
    return [float(x["pts_sec"]) for x in csv.DictReader(open(p, encoding="utf-8"))]


def scan(key, name, counters, f0=0, f1=-1, ncc=0.60, strips=()):
    cfg = SESS[key]
    rv = read_val_wao if cfg["reader"] == "wao" else read_val_ai
    gray = GRAY[cfg["reader"]]
    roi = get_roi(cfg, key, name)
    pts = load_pts(cfg, name)
    outdir = os.path.join(OUTROOT, key, "scan")
    os.makedirs(outdir, exist_ok=True)
    vid = os.path.join(ROOT, cfg["bands"], name + ".mp4")
    assert os.path.exists(vid), vid
    cap = cv2.VideoCapture(vid)
    cur = {c: "INIT" for c in counters}        # (value, polarity)
    n = {c: 0 for c in counters}
    runs = {c: [] for c in counters}
    want = {}                                   # (counter) -> [(a,b,step,tiles)]
    for c, a, b, st in strips:
        want.setdefault(c, []).append([a, b, st, []])
    f = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        if f < f0:
            continue
        if f1 >= 0 and f > f1:
            break
        g = None
        for c in counters:
            x1, y1, x2, y2 = roi[c]
            if g is None:
                g = gray(img)
            rg = g[y1:y2, x1:x2]
            v = rv(rg, ncc); pol = "+"
            if v is None:
                v = rv(255 - rg, ncc); pol = "-"
            if v is None:
                pol = "."
            k = (v, pol)
            if k == cur[c]:
                n[c] += 1
            else:
                if cur[c] != "INIT":
                    runs[c].append((f - n[c], n[c]) + cur[c])
                cur[c] = k; n[c] = 1
            for w in want.get(c, []):
                if w[0] <= f <= w[1] and (f - w[0]) % w[2] == 0:
                    w[3].append((f, img[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8].copy(),
                                 "" if v is None else f"={v}{pol}"))
    cap.release()
    T = f
    for c in counters:
        if cur[c] != "INIT":
            runs[c].append((T + 1 - n[c], n[c]) + cur[c])
        p = os.path.join(outdir, f"{name}__{c}__runs.csv")
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh); w.writerow(["start_frame", "n", "value", "polarity", "t_start", "t_end"])
            for s, nn, v, pol in runs[c]:
                ts = pts[s] if s < len(pts) else ""
                te = pts[min(s + nn - 1, len(pts) - 1)] if pts else ""
                w.writerow([s, nn, "" if v is None else v, pol, ts, te])
        # 値ごとの初出・最終・増加フレーム
        fr = []
        vals = sorted({v for _, _, v, _ in runs[c] if v is not None})
        for v in vals:
            seg = [(s, nn) for s, nn, vv, _ in runs[c] if vv == v]
            first = min(s for s, _ in seg); last = max(s + nn - 1 for s, nn in seg)
            nfr = sum(nn for _, nn in seg)
            prev = [s + nn - 1 for s, nn, vv, _ in runs[c] if vv == v - 1 and s + nn - 1 < first]
            ev = (max(prev) + 1) if prev else ""
            fr.append(dict(value=v, first_frame=first, last_frame=last, n_frames=nfr, event_frame=ev,
                           event_t=(round(pts[ev], 3) if ev != "" and ev < len(pts) else "")))
        p2 = os.path.join(outdir, f"{name}__{c}__first.csv")
        with open(p2, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["value", "first_frame", "last_frame", "n_frames", "event_frame", "event_t"])
            w.writeheader(); w.writerows(fr)
        print(f"[{c}] runs={len(runs[c])} 読めた値={vals}")
        for r in fr:
            print("   ", r, flush=True)
    for c, lst in want.items():
        for a, b, st, got in lst:
            if not got:
                print("no tiles", c, a, b); continue
            tl = [tile(im, f"f{ff}{lab}", 3.0) for ff, im, lab in got]
            ttl = f"{name} / {c} / f{a}-{b} step{st} / 両極性読取（+通常・-反転・無印=不可読）"
            p = os.path.join(outdir, f"strip_{name}__{c}__f{a}-{b}.png")
            imwrite(p, strip(tl, ttl))
            print("wrote", p, len(got), "tiles", flush=True)


if __name__ == "__main__":
    key = sys.argv[1]; name = sys.argv[2]
    counters = None; f0 = 0; f1 = -1; ncc = 0.60; strips = []
    for a in sys.argv[3:]:
        if a.startswith("--counters="):
            counters = a.split("=", 1)[1].split(",")
        elif a.startswith("--from="):
            f0 = int(a.split("=")[1])
        elif a.startswith("--to="):
            f1 = int(a.split("=")[1])
        elif a.startswith("--ncc="):
            ncc = float(a.split("=")[1])
        elif a.startswith("--strip="):
            s = a.split("=", 1)[1].split(":")
            rng = s[1].split("-")
            strips.append((s[0], int(rng[0]), int(rng[1]), int(s[2]) if len(s) > 2 else 1))
    if counters is None:
        counters = WAO_COUNTERS if SESS[key]["reader"] == "wao" else AI_COUNTERS
    scan(key, name, counters, f0, f1, ncc, strips)
