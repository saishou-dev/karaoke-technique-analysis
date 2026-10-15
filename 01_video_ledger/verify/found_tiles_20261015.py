# -*- coding: utf-8 -*-
"""欠落候補の増加フレームを目視で確定するためのタイル画像生成・全セッション版（2026-10-15・Windows python）。

found_montage_20260915.py（前後8枚）／found_dense_20260915.py（等間隔）／found_refine_20260915.py（NCC 自動絞り込み）を
1本にまとめ、セッションキーで ROI と帯動画を切り替える。ROI の実体は rescan_missing_20261015.py が
dataset/found_20261015/<key>/roi/<video>.json に保存したものを使う。

使い方:
  python found_tiles_20261015.py montage <key>                         ... rescan_missing.csv の FOUND 全件（前後8枚）
  python found_tiles_20261015.py dense <key> <video> <counter> <value> <a> <b> <step>
  python found_tiles_20261015.py refine <key> <video> <counter> <value> <f0> <before_off> <after_off>
  python found_tiles_20261015.py scanvals <key> <video> <counter> <a> <b> [ncc]  ... 区間の読み値を列挙（csv なし・標準出力）
出力: dataset/found_20261015/<key>/evidence/（montage_*.png・dense_*.png・refined_*.png）
"""
import os, sys, csv, json, cv2, numpy as np

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
OUTROOT = os.path.join(DS, "found_20261015")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import SESS, read_val_wao, read_val_ai  # noqa: E402

OFF = [-90, -30, -8, -2, 3, 10, 30, 90]
K = 4


def imwrite(path, img):
    cv2.imencode(".png", img)[1].tofile(path)


def paths(key, name):
    cfg = SESS[key]
    roi = json.load(open(os.path.join(OUTROOT, key, "roi", name + ".json"), encoding="utf-8"))
    vid = os.path.join(ROOT, cfg["bands"], name + ".mp4")
    return cfg, roi, vid


def grab(cap, f, box):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f)); ok, img = cap.read()
    if not ok:
        return None
    x1, y1, x2, y2 = box
    return img[max(0, y1):y2, max(0, x1):x2]


def stack(tiles, title, k=K, labelpos=(4, 20)):
    h = max(t.shape[0] for t in tiles)
    m = np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles])
    m = cv2.copyMakeBorder(m, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(m, title, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return m


def tile(img, lab, k=K, pad=12):
    t = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    t = cv2.copyMakeBorder(t, 28, 0, 0, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(t, lab, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return t


def cmd_montage(key):
    rows = [r for r in csv.DictReader(open(os.path.join(OUTROOT, key, "rescan_missing.csv"), encoding="utf-8-sig")) if r["kind"] == "FOUND"]
    print("FOUND rows:", len(rows))
    for r in rows:
        name = r["video"]; c = r["counter"]; f0 = int(r["frame"])
        cfg, roi, vid = paths(key, name); x1, y1, x2, y2 = roi[c]
        cap = cv2.VideoCapture(vid); tiles = []
        for o in OFF:
            im = grab(cap, f0 + o, [x1 - 12, y1 - 12, x2 + 12, y2 + 12])
            if im is None:
                continue
            tiles.append(tile(im, f"{o:+d}"))
        cap.release()
        m = stack(tiles, f"{name[:30]} {c} value={r['value']} f{f0} t={r['time']}")
        out = os.path.join(OUTROOT, key, "evidence", f"montage_{name}_{c}_{r['value']}_f{f0}.png")
        imwrite(out, m); print("wrote", out)


def cmd_dense(key, name, c, v, a, b, st, k=2.5, pad=6):
    cfg, roi, vid = paths(key, name); x1, y1, x2, y2 = roi[c]
    cap = cv2.VideoCapture(vid); tiles = []
    for f in range(a, b + 1, st):
        im = grab(cap, f, [x1 - pad, y1 - pad, x2 + pad, y2 + pad])
        if im is None:
            break
        tiles.append(tile(im, f"f{f}", k=k))
    cap.release()
    m = stack(tiles, f"{name[:28]} {c} value={v} f{a}-{b} step{st}")
    out = os.path.join(OUTROOT, key, "evidence", f"dense_{name}_{c}_{v}_f{a}-{b}_s{st}.png")
    imwrite(out, m); print(out)


def prep(t):
    g = t.max(axis=2).astype(np.float32)
    return cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, ksize=3)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / max(1e-6, np.linalg.norm(a) * np.linalg.norm(b)))


def cmd_refine(key, name, c, v, f0, bo, ao):
    cfg, roi, vid = paths(key, name); box = roi[c]
    cap = cv2.VideoCapture(vid)
    B = prep(grab(cap, f0 + bo, box)); A = prep(grab(cap, f0 + ao, box))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0 + bo); run = 0; ev = None; trace = []
    f = f0 + bo
    while f <= f0 + ao:
        ok, img = cap.read()
        if not ok:
            break
        x1, y1, x2, y2 = box; P = prep(img[y1:y2, x1:x2])
        sa = ncc(P, A); sb = ncc(P, B); trace.append((f, round(sa, 2), round(sb, 2)))
        run = run + 1 if sa > sb else 0
        if run >= 3 and ev is None:
            ev = f - 2
        f += 1
    if ev is None:
        ev = f0 + ao
    tiles = []
    for o in (-3, -1, 0, 1, 3):
        im = grab(cap, ev + o, [box[0] - 12, box[1] - 12, box[2] + 12, box[3] + 12])
        tiles.append(tile(im, f"{o:+d}"))
    cap.release()
    m = stack(tiles, f"{name[:28]} {c} value={v} refined f{ev} (cand f{f0})")
    imwrite(os.path.join(OUTROOT, key, "evidence", f"refined_{name}_{c}_{v}_f{ev}.png"), m)
    print("REFINED", name, c, v, "cand", f0, "->", ev)
    i = max(0, ev - (f0 + bo) - 3)
    print("  trace:", trace[i:i + 8])
    return ev


def cmd_scanvals(key, name, c, a, b, nccmin=0.60):
    cfg, roi, vid = paths(key, name); x1, y1, x2, y2 = roi[c]
    rv = read_val_wao if cfg["reader"] == "wao" else read_val_ai
    cap = cv2.VideoCapture(vid); cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    f = a; cur = "START"; runs = []
    while f < b:
        ok, img = cap.read()
        if not ok:
            break
        v = rv(img.max(axis=2)[y1:y2, x1:x2], nccmin)
        if v != cur:
            runs.append([f, v, 1]); cur = v
        else:
            runs[-1][2] += 1
        f += 1
    cap.release()
    print(f"{name} {c} f{a}-{b} ncc={nccmin} runs={len(runs)}")
    for st, v, n in runs:
        if n >= 3:
            print(f"  f{st:7d} n={n:5d} v={v}")
    return runs


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "montage":
        cmd_montage(sys.argv[2])
    elif cmd == "dense":
        cmd_dense(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]))
    elif cmd == "refine":
        cmd_refine(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]))
    elif cmd == "scanvals":
        cmd_scanvals(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]),
                     float(sys.argv[7]) if len(sys.argv) > 7 else 0.60)
    else:
        print(__doc__)
