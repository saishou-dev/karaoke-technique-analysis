# -*- coding: utf-8 -*-
"""欠落候補の増加フレームを**逐次デコード**で frame 単位に確定する（2026-10-15・Windows python）。

なぜ必要か＝帯動画は `-fps_mode passthrough`（VFR）なので cv2 の `CAP_PROP_POS_FRAMES` シークはフレームがずれる。
found_montage/found_refine（シーク式）は「増えたこと」の確認には使えるが、増加フレームの確定には使えない
（10/15 実測＝I LOVE... shakuri 15 でシーク tile と逐次走査が 7 フレームずれ）。本スクリプトは帯動画を先頭から
1 回だけ通して読むので、frame 番号は読取器・帯 pts 表と同じ基準になる。

走査の範囲＝rescan_missing_20261015.py と同じ「読取器が最後に増分を読めたフレーム以降」。
それ以前のフレームはデコードだけして読み取りを行わない（誤読の混入を防ぐ・高速化）。

出力:
  dataset/found_20261015/<key>/found_exact.csv  ... video,counter,value,event_frame,prev_frame,kind
  dataset/found_20261015/<key>/runs_<video>_<counter>.csv ... 走査範囲の run-length（n>=2）
  dataset/found_20261015/<key>/evidence/exact_<video>_<counter>_<value>_f<ev>.png ... 逐次走査で撮った前後5枚

使い方: python found_exact_20261015.py <key> [--ncc=0.60] [--run=3]
"""
import os, sys, csv, json, collections, cv2, numpy as np

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
OUTROOT = os.path.join(DS, "found_20261015")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import SESS, read_val_wao, read_val_ai, GRAY, WAO_COUNTERS, AI_COUNTERS  # noqa: E402


def imwrite(path, img):
    cv2.imencode(".png", img)[1].tofile(path)


def tile(img, lab, k=4):
    t = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    t = cv2.copyMakeBorder(t, 28, 0, 0, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(t, lab, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return t


def stack(tiles, title):
    h = max(t.shape[0] for t in tiles)
    m = np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles])
    m = cv2.copyMakeBorder(m, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(m, title, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return m


def run(key, nccmin=0.60, RUN=3):
    cfg = SESS[key]
    rv = read_val_wao if cfg["reader"] == "wao" else read_val_ai
    gray = GRAY[cfg["reader"]]
    outdir = os.path.join(OUTROOT, key)
    evd = os.path.join(outdir, "evidence"); os.makedirs(evd, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(outdir, "rescan_missing.csv"), encoding="utf-8-sig")))
    need = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        if r["kind"] in ("FOUND", "NOT_FOUND") and r["value"]:
            need[r["video"]][r["counter"]].add(int(r["value"]))
    out = []
    for name in need:
        roi = json.load(open(os.path.join(outdir, "roi", name + ".json"), encoding="utf-8"))
        ce = list(csv.DictReader(open(os.path.join(ROOT, cfg["results"], name, "counter_events.csv"), encoding="utf-8")))
        cs = list(need[name].keys())
        lo = {}
        for c in cs:
            evs = [e for e in ce if e["counter"] == c and int(e["delta"]) > 0]
            lo[c] = int(evs[-1]["frame"]) if evs else 0
        start = min(lo.values())
        vid = os.path.join(ROOT, cfg["bands"], name + ".mp4")
        cap = cv2.VideoCapture(vid)
        buf = {c: collections.deque(maxlen=24) for c in cs}
        runs = {c: [] for c in cs}
        cur = {c: "INIT" for c in cs}
        n = {c: 0 for c in cs}
        first = {c: {} for c in cs}
        pend = []
        f = -1
        while True:
            ok, img = cap.read()
            if not ok:
                break
            f += 1
            if f < start:
                continue
            g = None
            for c in cs:
                if f < lo[c]:
                    continue
                if g is None:
                    g = gray(img)
                x1, y1, x2, y2 = roi[c]
                buf[c].append((f, img[max(0, y1 - 10):y2 + 10, max(0, x1 - 10):x2 + 10].copy()))
                v = rv(g[y1:y2, x1:x2], nccmin)
                if v == cur[c]:
                    n[c] += 1
                else:
                    if cur[c] != "INIT":
                        runs[c].append((f - n[c], n[c], cur[c]))
                    cur[c] = v; n[c] = 1
                if v is not None and n[c] == RUN and v not in first[c]:
                    first[c][v] = f - (RUN - 1)
                    if v in need[name][c]:
                        # タイルは「旧値が最後に読めたフレーム+1」のあたりを見たいので、安定初出の 10 フレーム前を中心にする
                        pend.append([c, v, f - (RUN - 1), []])
            for p in pend[:]:
                c, v, ev, got = p
                seen = set(x[0] for x in got)
                got.extend([(ff, im) for ff, im in buf[c] if ev - 14 <= ff <= ev + 4 and ff not in seen])
                if any(ff >= ev + 4 for ff, _ in got):
                    got.sort(key=lambda x: x[0])
                    m = stack([tile(im, f"f{ff}" + ("*" if ff == ev else "")) for ff, im in got],
                              f"{name[:30]} {c} value={v} EXACT f{ev} (seq ncc{nccmin} run>={RUN})")
                    imwrite(os.path.join(evd, f"exact_{name}_{c}_{v}_f{ev}.png"), m)
                    pend.remove(p)
        for p in pend:
            c, v, ev, got = p
            if got:
                got.sort(key=lambda x: x[0])
                m = stack([tile(im, f"f{ff}" + ("*" if ff == ev else "")) for ff, im in got],
                          f"{name[:30]} {c} value={v} EXACT f{ev} (末尾・前後が足りない)")
                imwrite(os.path.join(evd, f"exact_{name}_{c}_{v}_f{ev}.png"), m)
        for c in cs:
            if cur[c] != "INIT":
                runs[c].append((f + 1 - n[c], n[c], cur[c]))
            with open(os.path.join(outdir, f"runs_{name}_{c}.csv"), "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh); w.writerow(["start_frame", "n", "value"])
                for s, nn, v in runs[c]:
                    w.writerow([s, nn, "" if v is None else v])
            for v in sorted(need[name][c]):
                st = first[c].get(v); prev = first[c].get(v - 1)
                # 増加フレーム＝読取器 read_counters_*.py と同じ規則＝「旧値が最後に読めたフレーム + 1」
                ev = ""
                if st is not None:
                    last_old = [s + nn - 1 for s, nn, vv in runs[c] if vv == v - 1 and s + nn - 1 < st]
                    ev = (max(last_old) + 1) if last_old else st
                out.append(dict(video=name, counter=c, value=v, event_frame=ev,
                                first_stable_frame=st if st is not None else "",
                                prev_frame=prev if prev is not None else "", scan_from=lo[c],
                                kind="EXACT" if st is not None else "NOT_FOUND", note=f"seq ncc{nccmin} run>={RUN} ev=last(v-1)+1"))
                print(f"{name[:26]:28s} {c:10s} v={v:3d} from f{lo[c]} stable={st} ev={ev}", flush=True)
        cap.release()
    p = os.path.join(outdir, "found_exact.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "counter", "value", "event_frame", "first_stable_frame", "prev_frame", "scan_from", "kind", "note"])
        w.writeheader(); w.writerows(out)
    print("WROTE", p, len(out), dict(collections.Counter(r["kind"] for r in out)), flush=True)


if __name__ == "__main__":
    key = sys.argv[1]; ncc = 0.60; RUN = 3
    for a in sys.argv[2:]:
        if a.startswith("--ncc="):
            ncc = float(a.split("=")[1])
        elif a.startswith("--run="):
            RUN = int(a.split("=")[1])
    run(key, ncc, RUN)
