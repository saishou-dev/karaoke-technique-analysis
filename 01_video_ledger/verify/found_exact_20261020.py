# -*- coding: utf-8 -*-
"""欠落候補の増加フレームを**逐次デコード**で frame 単位に確定する・**全候補＋曲末側から**版（2026-10-20）。

found_exact_20261015.py の後継。中身の読み取りは同じで、**候補の出し方だけ**を変えた。

なぜ作り直したか（2026-10-20・きらり フォール1 の取りこぼし）
  旧版は `if v is not None and n[c] == RUN and v not in first[c]` ＝**その値の最初の安定区間 1 件だけ**を
  候補にしていた。フォールのように直前の値が 0（読取器の初期値）の技法では走査開始が動画の先頭になるため、
  先頭付近の誤読（きらり＝f492 の「0」を緩い閾値で「1」と読んだもの）が唯一の候補になり、目視で正しく却下した
  のに**本物の f13880（234.50 秒・曲の終わり 1.25 秒前）は候補に上がらないまま「取得不能」と記録された**。
  実害＝10/14 の曲単位の完全一致が 27/30 のまま（本当は 28/30）。

変えたこと（本人指示 2026-10-20・恒久）
  1. **全候補を出す**＝走査範囲内にあるその値の安定区間を 1 件残らず行にする（rank_from_end 付き）。
  2. **曲末側から並べる**＝出力は event_frame の降順。目視は曲末側＝rank_from_end=1 から始める。
  3. **候補 0 件でも曲末を必ず見る**＝(video, counter) ごとに「数字の枠が最後に見つかったフレーム」までの
     帯状写真 `tail_*.png` を必ず作る。これを見ずに「取得不能」と記録しない。
  4. **2桁ガード（v0.8e と同じ gap>=5px）を目視補完の読みにも効かせる**。
     旧版が使う rescan_missing_20261015.boxes() には本番読取器 v0.8e の 2桁ガードが入っていないため、
     1文字の「1」が 2 本の縦線に割れたフレームは **値 11** と読まれ、**値 1 の候補として一度も出てこない**
     （2026-10-20 実測＝きらり fall の f13880 は旧版の走査で value=11・n=11 の安定区間になっていた）。
     つまり 1. の「最初の1件だけ」と合わせて**二重に取りこぼす**作りだった。`--gap2=0` で旧挙動に戻せる。

出力（dataset/found_20261015/<key>/ ＝旧版と同じ場所・ファイル名だけ分ける）
  found_exact_all.csv  ... video,counter,value,rank_from_end,n_candidates,event_frame,first_stable_frame,
                           run_n,prev_frame,scan_from,last_box_frame,kind,note
  evidence/exact2_<video>_<counter>_<value>_r<rank>_f<ev>.png ... 候補ごとの前後 1 フレーム刻み
  evidence/tail_<video>_<counter>_f<a>-<b>.png                ... 表示領域の末尾（必ず作る）

使い方: python found_exact_20261020.py <key> [--ncc=0.60] [--run=3] [--ev=3] [--gap2=5] [--targets=<csv>]
  --ev=N       候補ごとの根拠画像を曲末側から N 件だけ作る（0＝作らない・既定 3）
  --gap2=N     2桁として採用する桁間の最小隙間(px)・既定 5（v0.8e と同じ）・0＝旧 found_exact の挙動
  --targets=   video,counter,value の CSV を直接渡す（既定＝<key>/rescan_missing.csv の FOUND/NOT_FOUND）
"""
import os, sys, csv, json, collections, cv2, numpy as np

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
OUTROOT = os.path.join(DS, "found_20261015")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import (SESS, GRAY, get_roi, classify, roi_gain, INVERT_FB,  # noqa: E402
                                     RF_WAO, LAB_WAO, THR_WAO, RF_AI, LAB_AI, THR_AI)

PRE, POST = 8, 6          # 候補の根拠画像で見る前後フレーム数
TAIL_N, TAIL_STEP = 15, 2  # 曲末の帯状写真のタイル数と刻み
MIN_GAP2 = 5              # 2桁として採用する桁間の最小隙間(px)＝v0.8e と同じ既定（--gap2 で変える）


def boxes2(rg, thr, hmin=26, hmax=60, wmin=3, wmax=46, amin=40, junkmax=1500, gapmax=16):
    """rescan_missing_20261015.boxes() に v0.8e の 2桁ガード（gap>=MIN_GAP2）を足した版。

    gap が小さい2箱＝1文字が2つの連結成分に割れた形なので2桁として採用せず、
    他の閾値の1箱候補へ委ねる（v0.8e read_counters_wao_v08e_20261015.py の (A) と同じ規則）。
    """
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
            if gap > gapmax or gap < MIN_GAP2 or abs(bx[0][3] - bx[1][3]) > 9 or abs(bx[0][1] - bx[1][1]) > 7:
                continue
            if best2 is None:
                best2 = bx
        elif best1 is None:
            best1 = bx
    return best2 if best2 is not None else best1


def _read(rg, ncc, reader):
    if reader == "ai":
        rg = roi_gain(rg)
        bx = boxes2(rg, THR_AI, hmin=28, hmax=58, wmin=4, wmax=44, amin=60, junkmax=1200, gapmax=14)
        RF, LAB = RF_AI, LAB_AI
    else:
        bx = boxes2(rg, THR_WAO)
        RF, LAB = RF_WAO, LAB_WAO
    if not bx:
        return None
    ds = []
    for b in bx:
        d = classify(rg, b, RF, LAB, ncc)
        if d is None:
            return None
        ds.append(d)
    return ds[0] if len(ds) == 1 else ds[0] * 10 + ds[1]


def read_val(rg, ncc, reader):
    v = _read(rg, ncc, reader)
    if v is None and INVERT_FB:          # KV_FOUND_INVERT=1＝明背景を反転して読み直す（10/15 と同じ）
        v = _read(255 - rg, ncc, reader)
    return v


def has_box(rg, reader):
    """数字の枠が見つかるか＝「表示領域に数字が出ているか」の判定（gap ガードなしの広い条件）。"""
    if reader == "ai":
        return boxes2(roi_gain(rg), THR_AI, hmin=28, hmax=58, wmin=4, wmax=44, amin=60, junkmax=1200, gapmax=14) is not None
    return boxes2(rg, THR_WAO) is not None


def imwrite(path, img):
    ok, buf = cv2.imencode(".png", img)
    assert ok, path
    with open(path, "wb") as fh:          # 日本語パスで cv2.imwrite は黙って失敗する（2026-10-20 実測）
        fh.write(buf.tobytes())
    assert os.path.getsize(path) > 0, path


def tile(img, lab, k=3):
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


def targets_of(outdir, path):
    """{video: {counter: {value, ...}}}"""
    p = path or os.path.join(outdir, "rescan_missing.csv")
    need = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if not r.get("value"):
            continue
        if path is None and r.get("kind") not in ("FOUND", "NOT_FOUND"):
            continue
        need[r["video"]][r["counter"]].add(int(r["value"]))
    return need


def scan(cfg, name, cs, lo, nccmin, roi):
    """pass1＝逐次デコード 1 周。run-length と「枠が見つかった最後のフレーム」を返す。"""
    rdr = cfg["reader"]
    gray = GRAY[rdr]
    cap = cv2.VideoCapture(os.path.join(ROOT, cfg["bands"], name + ".mp4"))
    runs = {c: [] for c in cs}
    cur = {c: "INIT" for c in cs}
    n = {c: 0 for c in cs}
    lastbox = {c: -1 for c in cs}
    start = min(lo.values())
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
            sub = g[y1:y2, x1:x2]
            if has_box(sub, rdr):
                lastbox[c] = f
            v = read_val(sub, nccmin, rdr)
            if v == cur[c]:
                n[c] += 1
            else:
                if cur[c] != "INIT":
                    runs[c].append((f - n[c], n[c], cur[c]))
                cur[c] = v
                n[c] = 1
    for c in cs:
        if cur[c] != "INIT":
            runs[c].append((f + 1 - n[c], n[c], cur[c]))
    cap.release()
    return runs, lastbox, f


def candidates(runs_c, value, RUN):
    """その値の安定区間を**全部**返す（曲末側から＝event_frame 降順）。"""
    out = []
    for i, (s, nn, v) in enumerate(runs_c):
        if v != value or nn < RUN:
            continue
        # 増加フレーム＝読取器と同じ規則＝「旧値（value-1）が最後に読めたフレーム + 1」。
        # 無ければ安定区間の先頭をそのまま使う。
        last_old = [s2 + n2 - 1 for s2, n2, v2 in runs_c if v2 == value - 1 and s2 + n2 - 1 < s]
        ev = (max(last_old) + 1) if last_old else s
        out.append(dict(event_frame=ev, first_stable_frame=s, run_n=nn))
    # 同じ増加フレームに落ちる安定区間（点滅や薄れで途切れた同一の表示）はまとめる
    by = {}
    for d in out:
        k = d["event_frame"]
        if k not in by or d["run_n"] > by[k]["run_n"]:
            by[k] = d
        by[k]["n_runs"] = by[k].get("n_runs", 0) + 1
    out = sorted(by.values(), key=lambda d: -d["event_frame"])
    return out


def grab(cfg, name, wanted, roi):
    """pass2＝必要なフレームの ROI だけをもう 1 周で取る。wanted={(counter): set(frames)}"""
    cap = cv2.VideoCapture(os.path.join(ROOT, cfg["bands"], name + ".mp4"))
    got = {c: {} for c in wanted}
    f = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        for c, fr in wanted.items():
            if f in fr:
                x1, y1, x2, y2 = roi[c]
                got[c][f] = img[max(0, y1 - 10):y2 + 10, max(0, x1 - 10):x2 + 10].copy()
    cap.release()
    return got


def run(key, nccmin=0.60, RUN=3, nev=3, targets=None, gap2=MIN_GAP2):
    global MIN_GAP2
    MIN_GAP2 = gap2
    cfg = SESS[key]
    outdir = os.path.join(OUTROOT, key)
    evd = os.path.join(outdir, "evidence")
    os.makedirs(evd, exist_ok=True)
    need = targets_of(outdir, targets)
    out = []
    for name in need:
        roi = get_roi(cfg, name)
        rp = os.path.join(outdir, "roi", name + ".json")
        if os.path.exists(rp):
            roi = json.load(open(rp, encoding="utf-8"))
        ce = list(csv.DictReader(open(os.path.join(ROOT, cfg["results"], name, "counter_events.csv"), encoding="utf-8")))
        cs = list(need[name].keys())
        lo = {}
        for c in cs:
            evs = [e for e in ce if e["counter"] == c and int(e["delta"]) > 0]
            lo[c] = int(evs[-1]["frame"]) if evs else 0
        runs, lastbox, nframe = scan(cfg, name, cs, lo, nccmin, roi)
        # ---- 候補表（全件・曲末側から）
        picked = collections.defaultdict(set)   # counter -> 画像を作るフレーム
        for c in cs:
            with open(os.path.join(outdir, f"runs2_{name}_{c}.csv"), "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh); w.writerow(["start_frame", "n", "value"])
                for s, nn, v in runs[c]:
                    w.writerow([s, nn, "" if v is None else v])
            for value in sorted(need[name][c]):
                cands = candidates(runs[c], value, RUN)
                for i, d in enumerate(cands):
                    out.append(dict(video=name, counter=c, value=value, rank_from_end=i + 1,
                                    n_candidates=len(cands), event_frame=d["event_frame"],
                                    first_stable_frame=d["first_stable_frame"], run_n=d["run_n"],
                                    prev_frame="", scan_from=lo[c], last_box_frame=lastbox[c],
                                    kind="EXACT", note=f"seq ncc{nccmin} run>={RUN} gap2={MIN_GAP2} ev=last(v-1)+1 全候補・曲末側から"))
                    if i < nev:
                        picked[c] |= set(range(d["event_frame"] - PRE, d["event_frame"] + POST + 1))
                if not cands:
                    out.append(dict(video=name, counter=c, value=value, rank_from_end=0, n_candidates=0,
                                    event_frame="", first_stable_frame="", run_n="", prev_frame="",
                                    scan_from=lo[c], last_box_frame=lastbox[c], kind="NOT_FOUND",
                                    note=f"seq ncc{nccmin} run>={RUN} gap2={MIN_GAP2} 候補0件＝tail 画像で曲末まで目視すること"))
                print(f"{name[:26]:28s} {c:10s} v={value:3d} from f{lo[c]} 候補{len(cands)}件 "
                      f"曲末側={[d['event_frame'] for d in cands[:5]]} last_box=f{lastbox[c]}", flush=True)
            # 曲末の帯状写真は候補の有無によらず必ず作る
            if lastbox[c] >= 0:
                picked[c] |= set(range(max(0, lastbox[c] - (TAIL_N - 1) * TAIL_STEP), lastbox[c] + 1, TAIL_STEP))
        # ---- pass2＝画像
        wanted = {c: {f for f in fr if 0 <= f <= nframe} for c, fr in picked.items() if fr}
        if wanted:
            got = grab(cfg, name, wanted, roi)
            for c in wanted:
                for value in sorted(need[name][c]):
                    for i, d in enumerate(candidates(runs[c], value, RUN)[:nev]):
                        ev = d["event_frame"]
                        fr = sorted(f for f in got[c] if ev - PRE <= f <= ev + POST)
                        if not fr:
                            continue
                        m = stack([tile(got[c][f], f"f{f}" + ("*" if f == ev else "")) for f in fr],
                                  f"{name[:30]} {c} value={value} 候補{i+1}/{d['run_n']}f EXACT f{ev} (曲末側から{i+1}番目)")
                        imwrite(os.path.join(evd, f"exact2_{name}_{c}_{value}_r{i+1}_f{ev}.png"), m)
                if lastbox[c] >= 0:
                    fr = sorted(f for f in got[c] if f <= lastbox[c] and f >= lastbox[c] - (TAIL_N - 1) * TAIL_STEP)
                    if fr:
                        m = stack([tile(got[c][f], f"f{f}") for f in fr],
                                  f"{name[:30]} {c} 表示領域の末尾 f{fr[0]}-f{fr[-1]}（枠の最後＝f{lastbox[c]} / 全{nframe+1}f）")
                        imwrite(os.path.join(evd, f"tail_{name}_{c}_f{fr[0]}-{fr[-1]}.png"), m)
    out.sort(key=lambda r: (r["video"], r["counter"], r["value"], r["rank_from_end"]))
    p = os.path.join(outdir, "found_exact_all.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "counter", "value", "rank_from_end", "n_candidates",
                                           "event_frame", "first_stable_frame", "run_n", "prev_frame",
                                           "scan_from", "last_box_frame", "kind", "note"])
        w.writeheader(); w.writerows(out)
    print("WROTE", p, len(out), dict(collections.Counter(r["kind"] for r in out)), flush=True)


if __name__ == "__main__":
    key = sys.argv[1]
    ncc, RUN, nev, tg = 0.60, 3, 3, None
    for a in sys.argv[2:]:
        if a.startswith("--ncc="):
            ncc = float(a.split("=")[1])
        elif a.startswith("--run="):
            RUN = int(a.split("=")[1])
        elif a.startswith("--ev="):
            nev = int(a.split("=")[1])
        elif a.startswith("--targets="):
            tg = a.split("=", 1)[1]
    run(key, ncc, RUN, nev, tg)
