# -*- coding: utf-8 -*-
"""確定した欠落イベントの増加フレームを、両極性の読取で詰める（2026-10-15・第2版）。

第1版（Laplacian-NCC・found_refine_20260915.py と同じ方式）は、数字が読めない区間で背景の色が変わると
そこを「変化」と誤判定した（10/15 実測＝地球儀 shakuri 9 / 想望 shakuri 19 / Mela!(生音) 20260924 shakuri 8 の 3 件で
実際には数字が変わっていないフレームを指した）。原因は NCC が数字でなく背景の変化に反応すること。

真因＝読取器 read_counters_*.py の二値化は `roi > T`（明るい数字・暗い背景）だけを見る。帯の背景が明るくなる演出では
数字のほうが暗くなり、連結成分が取れず「不可読」になる（Mela!(生音) 20260924 shakuri は 593 フレーム＝10 秒が不可読）。
→ 本スクリプトは通常の読取に失敗したフレームで **反転（255-ROI）** をもう一度試す。これで明背景の区間も数字が読める。

増加フレーム＝読取器と同じ規則「旧値が最後に読めたフレーム + 1」を、両極性の読取で求め直したもの。
証拠＝pin の前後 8 枚（step1・逐次デコード・シークなし）。

出力: dataset/found_20261015/<key>/found_pin.csv・evidence/pin_<video>_<counter>_<value>_f<pin>.png
使い方: python found_pin_20261015.py <key>
"""
import os, sys, csv, json, collections, cv2, numpy as np

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
OUTROOT = os.path.join(DS, "found_20261015")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import SESS, read_val_wao, read_val_ai, GRAY  # noqa: E402

MARGIN = 120   # 走査の余白（旧値が読めるフレームまで遡るため）


def imwrite(p, img):
    cv2.imencode(".png", img)[1].tofile(p)


def tile(img, lab, k=4):
    t = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    t = cv2.copyMakeBorder(t, 26, 0, 0, 3, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(t, lab, (2, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return t


def stack(tiles, title, per_row=17):
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    lines = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    w = max(l.shape[1] for l in lines)
    lines = [cv2.copyMakeBorder(l, 0, 6, 0, w - l.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20)) for l in lines]
    m = cv2.copyMakeBorder(np.vstack(lines), 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(m, title, (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
    return m


def run(key):
    cfg = SESS[key]
    rv = read_val_wao if cfg["reader"] == "wao" else read_val_ai
    gray = GRAY[cfg["reader"]]
    outdir = os.path.join(OUTROOT, key)
    rows = [r for r in csv.DictReader(open(os.path.join(outdir, "found_exact.csv"), encoding="utf-8-sig")) if r["kind"] == "EXACT"]
    pts_dir = os.path.join(DS, cfg["pts"])
    jobs = collections.defaultdict(list)
    for r in rows:
        jobs[r["video"]].append(dict(counter=r["counter"], value=int(r["value"]),
                                     ev=int(r["event_frame"]), st=int(r["first_stable_frame"])))
    out = []
    for name in jobs:
        roi = json.load(open(os.path.join(outdir, "roi", name + ".json"), encoding="utf-8"))
        pts = [float(x["pts_sec"]) for x in csv.DictReader(open(os.path.join(pts_dir, name + ".csv"), encoding="utf-8"))]
        lo = max(0, min(j["ev"] for j in jobs[name]) - MARGIN)
        hi = max(j["st"] for j in jobs[name]) + 10
        cap = cv2.VideoCapture(os.path.join(ROOT, cfg["bands"], name + ".mp4"))
        crops = collections.defaultdict(dict)   # counter -> frame -> crop(with pad)
        vals = collections.defaultdict(dict)    # counter -> frame -> (value, polarity)
        f = -1
        while True:
            ok, img = cap.read()
            if not ok:
                break
            f += 1
            if f < lo:
                continue
            if f > hi:
                break
            g = gray(img)
            for j in jobs[name]:
                c = j["counter"]
                if c in crops and f in crops[c]:
                    continue
                x1, y1, x2, y2 = roi[c]
                crops[c][f] = img[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8].copy()
                rg = g[y1:y2, x1:x2]
                v = rv(rg)
                pol = "+"
                if v is None:
                    v = rv(255 - rg); pol = "-"
                vals[c][f] = (v, pol)
        cap.release()
        for j in jobs[name]:
            c = j["counter"]; val = j["value"]; ev = j["ev"]; st = j["st"]
            lastold = [ff for ff, (v, p) in sorted(vals[c].items()) if v == val - 1 and ff < st]
            firstnew = [ff for ff, (v, p) in sorted(vals[c].items()) if v == val]
            if lastold:
                pin = max(lastold) + 1
                how = "両極性読取: 旧値が最後に読めた f%d の次" % max(lastold)
            else:
                pin = ev; how = "旧値が走査範囲で読めず＝読取規則のまま"
            fn = min(firstnew) if firstnew else st
            pol_old = vals[c].get(pin - 1, (None, "?"))[1]
            tl = [tile(crops[c][ff], f"f{ff}" + ("*" if ff == pin else "")) for ff in range(pin - 8, pin + 9) if ff in crops[c]]
            ttl = (f"{name} / {c} / value={val} / PIN f{pin} t={pts[pin]:.3f}s / {how}（極性{pol_old}）/ "
                   f"新値の初出 f{fn} ・読取規則 f{ev} ・安定初出 f{st}")
            imwrite(os.path.join(outdir, "evidence", f"pin_{name}_{c}_{val}_f{pin}.png"), stack(tl, ttl))
            out.append(dict(video=name, counter=c, value=val, pin_frame=pin, first_new_frame=fn,
                            rule_frame=ev, first_stable_frame=st, how=how, time_sec=round(pts[pin], 3)))
            print(f"{name[:24]:26s} {c:10s} {val:3d} rule f{ev} stable f{st} -> PIN f{pin} (new初出 f{fn})", flush=True)
    p = os.path.join(outdir, "found_pin.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "counter", "value", "pin_frame", "first_new_frame", "rule_frame", "first_stable_frame", "how", "time_sec"])
        w.writeheader(); w.writerows(out)
    print("WROTE", p, len(out), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
