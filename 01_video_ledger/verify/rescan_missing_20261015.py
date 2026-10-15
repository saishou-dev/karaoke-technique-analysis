# -*- coding: utf-8 -*-
"""欠落した増分の部分再走査・全セッション版（2026-10-15・Windows python＝cv2 5.0.0）。

rescan_missing_20260911.py／rescan_missing_20260915.py の一般化。verify_*.csv が DIFF の曲・カウンターについて、
読取器が最後に読めたイベント以降（および増分の飛び＝delta>1 の直前 2 秒）を全フレーム走査し、緩い閾値
（WAO＝NCC 0.60・しきい値 8 段／Ai＝NCC 0.60・ROI 輝度ゲイン）で f+1..T の各値の初出（3 フレーム以上連続）
を探して追記候補を出す。証拠として ROI のタイル画像も保存する。G1 へは未反映（append_found_20261015.py が反映）。

使い方: python rescan_missing_20261015.py [<sess-key> ...]   （既定＝全キー・優先順）
  sess-key: 20261006wao / 20261006ai / 20260924 / 20260915r / 20260911
出力: dataset/found_20261015/<key>/rescan_missing.csv・evidence/*.png・roi/<video>.json（使った ROI の実体）

規則（本人指示 2026-10-15）：読み過ぎ（read > csv）が 1 件でもある曲は OVERCOUNT として記録し、その曲の
欠落候補は一切出さない（追記対象から外す）。20261014 は対象外（別件調査中）。
"""
import os, sys, csv, json, numpy as np, cv2

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
OUTROOT = os.path.join(DS, "found_20261015")

WAO_COUNTERS = ["shakuri", "kobushi", "fall", "vibrato", "accent", "hammering"]
AI_COUNTERS = ["shakuri", "kobushi", "fall", "vibrato"]

# セッション定義（優先順＝本人指示 10/6 → 9/24 → 9/15 → 9/11）
SESS = {
    "20261006wao": dict(reader="wao", verify="verify_20261006_wao.csv", results="results_wao_20261006",
                        bands="bands_wao", pts="pts_bands_20261006", roi="results", g1="G1_events_20261006_wao.csv",
                        session=20261006, pts_source="pts_bands_20261006"),
    "20261006ai":  dict(reader="ai",  verify="verify_20261006_ai.csv",  results="results_ai_20261006",
                        bands="bands_ai",  pts="pts_bands_ai_20261006", roi="ai_default", g1="G1_events_20261006_ai.csv",
                        session=20261006, pts_source="pts_bands_ai_20261006"),
    "20260924":    dict(reader="wao", verify="verify_20260924.csv",     results="results_wao_20260924",
                        bands="bands_wao", pts="pts_bands_20260924", roi="results", g1="G1_events_20260924.csv",
                        session=20260924, pts_source="pts_bands_20260924"),
    "20260915r":   dict(reader="wao", verify="verify_20260915_rescan.csv", results="results_wao_20260915_rescan",
                        bands="bands_wao", pts="pts_bands_20260915", roi=os.path.join(DS, "rescan_20260915", "roi"),
                        g1="G1_events_20260915_rescan.csv", session=20260915, pts_source="pts_bands_20260915"),
    # 9/11 の台帳の正本＝_bandpts_rerun_20260914（G1_events_all の 20260911 行の video_time_sec は 1470/1470 でこちらと一致。
    # 旧 G1_events_20260911.csv は帯 pts 表の作り直し前の時刻で、_all には入っていない＝追記してはいけない）
    "20260911":    dict(reader="wao", verify="verify_20260911.csv",     results="results_wao_20260911",
                        bands="bands_wao", pts="pts_bands_20260911", roi="default",
                        g1="G1_events_20260911_bandpts_rerun_20260914.csv",
                        session=20260911, pts_source="pts_bands_20260911"),
}
ORDER = ["20261006wao", "20261006ai", "20260924", "20260915r", "20260911"]

# ---------- WAO 読取（rescan_missing_20260911.py と同一） ----------
ROI_WAO_DEFAULT = json.load(open(os.path.join(ROOT, "templates/wao_20260816/roi_wao.json"), encoding="utf-8"))
LAB_WAO = {int(k): v for k, v in json.load(open(os.path.join(ROOT, "templates/wao_20260816/cluster_reps_wao_ext_labels.json"), encoding="utf-8")).items()}
_reps = np.load(os.path.join(ROOT, "templates/wao_20260816/cluster_reps_wao_ext.npy"))
RF_WAO = np.stack([(r.ravel() - r.mean()) / max(1e-6, np.linalg.norm(r.ravel() - r.mean())) for r in _reps])
THR_WAO = (228, 215, 200, 185, 170, 150, 130, 110)

# ---------- Ai 読取（read_counters_ai_20261006ai.py の ROI・ゲイン・ラベル） ----------
ROI_AI_DEFAULT = {"shakuri": [245, 85, 350, 148], "kobushi": [565, 85, 690, 148],
                  "fall": [925, 85, 1020, 148], "vibrato": [1305, 85, 1395, 148]}
LAB_AI = {0: 0, 1: 2, 2: 3, 3: 1, 4: 2, 5: 6, 6: 1, 7: 2, 8: 1, 9: 7,
          10: 4, 11: 5, 12: 6, 13: 9, 14: None, 15: 5, 16: 0, 17: 0, 18: None,
          19: 2, 20: 1, 21: 6, 22: 1, 23: 0, 24: 2, 25: 2, 26: 2, 27: 2,
          28: None, 29: 6, 30: 2, 31: 8, 32: 2, 33: 4, 34: 0, 35: 4,
          36: None, 37: 7, 38: None, 39: 8}
_repsa = np.load(os.path.join(ROOT, "templates/ai_20260915ai/cluster_reps.npy"))
RF_AI = np.stack([(r.ravel() - r.mean()) / max(1e-6, np.linalg.norm(r.ravel() - r.mean())) for r in _repsa])
THR_AI = (228, 215, 200, 185, 170, 150, 130)
NORM = (28, 40)


def roi_gain(rg, target=235.0, gmax=3.0):
    p = float(np.percentile(rg, 99.5)); k = min(gmax, target / max(p, 40.0))
    return rg if k <= 1.0 else np.clip(rg.astype(np.float32) * k, 0, 255).astype(np.uint8)


def boxes(rg, thr, hmin=26, hmax=60, wmin=3, wmax=46, amin=40, junkmax=1500, gapmax=16):
    best1 = best2 = None
    for T in thr:
        m = (rg > T).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        bx = []; junk = 0
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if hmin <= h <= hmax and wmin <= w <= wmax and a >= amin:
                bx.append((x, y, w, h))
            elif a > 30:
                junk += a
        if not (1 <= len(bx) <= 2 and junk <= junkmax):
            continue
        bx.sort(key=lambda b: b[0])
        if len(bx) == 2:
            gap = bx[1][0] - (bx[0][0] + bx[0][2])
            if gap > gapmax or gap < -4 or abs(bx[0][3] - bx[1][3]) > 9 or abs(bx[0][1] - bx[1][1]) > 7:
                continue
            if best2 is None:
                best2 = bx
        elif best1 is None:
            best1 = bx
    return best2 if best2 is not None else best1


def classify(rg, b, RF, LAB, ncc):
    x, y, w, h = b; p = 2
    g = rg[max(0, y - p):y + h + p, max(0, x - p):x + w + p].astype(np.float32)
    g = (g - g.min()) / max(1.0, g.max() - g.min()); g = cv2.resize(g, NORM)
    v = g.ravel() - g.mean(); n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    sc = RF @ (v / n); bi = int(np.argmax(sc))
    return LAB.get(bi) if sc[bi] >= ncc else None


def read_val_wao(rg, ncc=0.60):
    bx = boxes(rg, THR_WAO)
    if not bx:
        return None
    ds = []
    for b in bx:
        d = classify(rg, b, RF_WAO, LAB_WAO, ncc)
        if d is None:
            return None
        ds.append(d)
    return ds[0] if len(ds) == 1 else ds[0] * 10 + ds[1]


def gray_wao(img):
    """WAO の読取器 read_counters_wao_20260915.py は V 値（max over BGR）を使う（同スクリプト 13 行の置換）。"""
    return img.max(axis=2)


def gray_ai(img):
    """Ai の読取器 read_counters_ai_20261006ai.py は cv2.COLOR_BGR2GRAY を使う（同 107 行）。"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


GRAY = {"wao": gray_wao, "ai": gray_ai}


def read_val_ai(rg, ncc=0.60):
    rg = roi_gain(rg)
    bx = boxes(rg, THR_AI, hmin=28, hmax=58, wmin=4, wmax=44, amin=60, junkmax=1200, gapmax=14)
    if not bx:
        return None
    ds = []
    for b in bx:
        d = classify(rg, b, RF_AI, LAB_AI, ncc)
        if d is None:
            return None
        ds.append(d)
    return ds[0] if len(ds) == 1 else ds[0] * 10 + ds[1]


def get_roi(cfg, name):
    r = cfg["roi"]
    if r == "default":
        return dict(ROI_WAO_DEFAULT)
    if r == "ai_default":
        return dict(ROI_AI_DEFAULT)
    if r == "results":
        p = os.path.join(ROOT, cfg["results"], name, "roi.json")
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else dict(ROI_WAO_DEFAULT)
    p = os.path.join(r, name + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else dict(ROI_WAO_DEFAULT)


def run(key):
    cfg = SESS[key]
    counters = WAO_COUNTERS if cfg["reader"] == "wao" else AI_COUNTERS
    read_val = read_val_wao if cfg["reader"] == "wao" else read_val_ai
    gray = GRAY[cfg["reader"]]
    outdir = os.path.join(OUTROOT, key)
    os.makedirs(os.path.join(outdir, "evidence"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "roi"), exist_ok=True)
    ver = list(csv.DictReader(open(os.path.join(DS, cfg["verify"]), encoding="utf-8-sig")))
    out = []
    for r in ver:
        if r["status"] != "DIFF":
            continue
        name = r["video"]
        over = [c for c in counters if int(r["read_" + c]) > int(r["csv_" + c])]
        if over:
            for c in over:
                out.append(dict(video=name, counter=c, kind="OVERCOUNT", value="", frame="", time="",
                                note=f"read {r['read_'+c]} > csv {r['csv_'+c]}: 読み過ぎ＝この曲は追記対象外（本人指示 10/15）"))
            print(f"[SKIP overcount] {name} {over}", flush=True)
            continue
        roi = get_roi(cfg, name)
        json.dump(roi, open(os.path.join(outdir, "roi", name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        evp = os.path.join(ROOT, cfg["results"], name, "counter_events.csv")
        ev = list(csv.DictReader(open(evp, encoding="utf-8"))) if os.path.exists(evp) else []
        vid = os.path.join(ROOT, cfg["bands"], name + ".mp4")
        if not os.path.exists(vid):
            out.append(dict(video=name, counter="", kind="NO_BAND", value="", frame="", time="", note=vid))
            print("[NO BAND]", vid, flush=True); continue
        cap = cv2.VideoCapture(vid)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
        for c in counters:
            target = int(r["csv_" + c]); final = int(r["read_" + c])
            if final >= target:
                continue
            evs = [e for e in ev if e["counter"] == c and int(e["delta"]) > 0]
            last_fr = int(evs[-1]["frame"]) if evs else 0
            ranges = [(last_fr, N)]
            for e in evs:
                if int(e["delta"]) > 1:
                    ranges.append((max(0, int(e["frame"]) - int(2 * fps)), int(e["frame"])))
            x1, y1, x2, y2 = roi[c]; found = {}
            for (a, b) in ranges:
                cap.set(cv2.CAP_PROP_POS_FRAMES, a); run_v = None; run_n = 0; f = a
                while f < b:
                    ok, img = cap.read()
                    if not ok:
                        break
                    v = read_val(gray(img)[y1:y2, x1:x2])
                    if v is not None and v == run_v:
                        run_n += 1
                    else:
                        run_v = v; run_n = 1
                    if v is not None and final < v <= target and run_n >= 3 and v not in found:
                        found[v] = f - 2
                        tile = img[max(0, y1 - 10):y2 + 10, max(0, x1 - 10):x2 + 10]
                        cv2.imencode(".png", tile)[1].tofile(os.path.join(outdir, "evidence", f"{name}_{c}_{v}_f{f-2}.png"))
                    f += 1
            for v in range(final + 1, target + 1):
                if v in found:
                    out.append(dict(video=name, counter=c, kind="FOUND", value=v, frame=found[v],
                                    time=round(found[v] / fps, 3), note="rescan ncc0.60 run>=3"))
                else:
                    out.append(dict(video=name, counter=c, kind="NOT_FOUND", value=v, frame="", time="",
                                    note="要目視（帯フェード/演出遮蔽の可能性）"))
            print(f"{key} {name[:28]:30s} {c:10s} {final} -> {target} {found}", flush=True)
        cap.release()
    p = os.path.join(outdir, "rescan_missing.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["video", "counter", "kind", "value", "frame", "time", "note"])
        w.writeheader(); w.writerows(out)
    kinds = {}
    for o in out:
        kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
    print("WROTE", p, len(out), kinds, flush=True)
    return kinds


if __name__ == "__main__":
    keys = sys.argv[1:] or ORDER
    summary = {}
    for k in keys:
        print("=" * 70); print("SESSION", k, flush=True)
        summary[k] = run(k)
    print("=" * 70)
    for k, v in summary.items():
        print(k, v)
