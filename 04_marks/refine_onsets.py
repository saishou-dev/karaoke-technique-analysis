# -*- coding: utf-8 -*-
"""出現時刻の精緻化（逆探索）。
目視サンプルで、v4 の「出現」が印の初出ではなく、ヒット演出のキラキラで一度隠れた後の
「再出現」を拾っている例が多かった（−6f のコマに既に印がある）。そこで、対応付け済みの各イベントについて
  1. 印の位置（mark_x, mark_y）を中心に ±160px の小領域を [mark−2.6s, mark+0.35s] で復号し、
  2. mark+8f のグレー画像を「その印自身のテンプレート」として、各フレームでマスク付き正規化相関（位置固定）を計算、
  3. mark から過去へ辿り、相関 ≥ thr のフレームを最後に、相関 < thr が 12 フレーム（≈0.2s）続いた所で打ち切り、
     最も早い「相関 ≥ thr」のフレームを真の出現時刻とする（キラキラによる短い隠れは跨ぐ）。
  4. 青ファミリ（こぶし／ハンマリング）は、こぶし・ハンマリングのテンプレートとの相関で種類を判定する。
入力: marks_20260924.csv（rematch 済み）  出力: marks_20260924.csv（mark_time_raw / mark_time_sec(精緻化) / lead_sec / kind_ncc 列を追加）
証拠: evidence/refined/<video>/<tech>_<n>_T<T>.png（初出−6f｜初出｜v4出現｜v4+8f）"""
import os, sys, time, argparse, traceback
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_marks as d

HALF = 160                # 小領域の半径 [px]
BACK, FWD = 2.6, 0.35     # 逆探索の範囲 [s]
NCC_THR = 0.55            # 「印がある」とみなす相関
GAP_MAX = 12              # 跨いでよい低相関の連続フレーム数（≈0.2s）
PAD = 14                  # テンプレート周囲の余白
NCC_SELF_MIN = 0.5        # 自己相関ゲート（これ未満は誤検出として無効化）
CLS_MIN_GREEN, CLS_MIN_HAMMER = 0.30, 0.40  # 種類テンプレート相関ゲート（ビブラート M／ハンマリング S）。9/24 は 0.45/0.5 で処理、8/18 で真の M が 0.3 前後だったため 9/29 に緩和。こぶし・赤は対象外

def refine_event(video, r, cache):
    path = os.path.join(d.VID_DIR, video + ".mp4")
    x, y = int(r.mark_x), int(r.mark_y)
    x0, y0 = max(0, x - HALF), max(0, y - HALF)
    x1, y1 = min(3840, x + HALF), min(2160, y + HALF)
    box = (x0, y0, (x1 - x0) - (x1 - x0) % 2, (y1 - y0) - (y1 - y0) % 2)
    t0 = max(0.0, r.mark_time_sec - BACK)
    n = int((BACK + FWD) * d.FPS_NOMINAL) + 4
    frames = []
    for idx, fr in d.decode_segment(path, t0, n, box, hwaccel="", threads=3):
        frames.append(fr)
    pts = d.read_pts(d.decode_segment.last_errpath)
    n = min(len(frames), len(pts)); frames, pts = frames[:n], pts[:n]
    if n < 30: return None, "復号フレーム不足"
    pts = np.array(pts)
    jm = int(np.argmin(np.abs(pts - r.mark_time_sec)))
    jt = min(n - 1, jm + 8)
    fam = d.FAMILY[r.counter][0]
    # その印自身のテンプレート（mark+8f の、印の外接矩形＋余白）
    m = d.classify(frames[jt])
    fm = cv2.dilate(((m & fam) > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    cx, cy = x - x0, y - y0
    nlab, lab, st, cen = cv2.connectedComponentsWithStats(fm, 8)
    # 印の位置に最も近い成分
    best_k, best_d = -1, 1e9
    for k in range(1, nlab):
        if st[k, cv2.CC_STAT_AREA] < 40: continue
        dd = (cen[k][0] - cx) ** 2 + (cen[k][1] - cy) ** 2
        if dd < best_d: best_d, best_k = dd, k
    if best_k < 0 or best_d > 60 ** 2: return None, "印の成分が見つからない"
    bx, by, bw, bh, ba = st[best_k]
    tx0, ty0 = max(0, bx - PAD), max(0, by - PAD); tx1, ty1 = min(box[2], bx + bw + PAD), min(box[3], by + bh + PAD)
    tpl_bgr = frames[jt][ty0:ty1, tx0:tx1]
    tpl = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2HSV)
    mt = d.classify(tpl_bgr)
    icon = ((mt & fam) > 0) | ((hsv[..., 2] >= 180) & (hsv[..., 1] < 100)) | ((mt & d.B_YEL) > 0)
    icon = cv2.dilate(icon.astype(np.uint8), np.ones((5, 5), np.uint8))
    msk = (icon > 0).astype(np.float32)
    # 位置固定のマスク付き相関（±4px の遊びを許す）
    def ncc_at(j):
        g = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
        gy0, gy1 = max(0, ty0 - 4), min(box[3], ty1 + 4); gx0, gx1 = max(0, tx0 - 4), min(box[2], tx1 + 4)
        r_ = d.masked_ncc(g[gy0:gy1, gx0:gx1], tpl, msk)
        r_ = np.nan_to_num(r_, nan=-1.0, posinf=-1.0, neginf=-1.0)
        v = float(r_.max()); return v if v <= 1.05 else -1.0
    ncc = np.full(n, np.nan)
    for j in range(n): ncc[j] = ncc_at(j)
    # 逆探索（3 フレーム最小値フィルタ：3 フレーム連続で相関が高い所だけを「印がある」とみなす）
    ncc_f = ncc.copy()
    for j in range(n): ncc_f[j] = np.nanmin(ncc[j:min(n, j + 3)])
    j_on = jm; gap = 0; j = jm - 1
    while j >= 0:
        if ncc_f[j] >= NCC_THR: j_on = j; gap = 0
        else:
            gap += 1
            if gap >= GAP_MAX: break
        j -= 1
    hit_start = j_on == 0   # 窓の先頭まで印があった＝2.6s より前から存在（真の出現は不明）
    # 種類判定：そのファミリの種類テンプレートとの相関（全ファミリ）。cls_ncc = 最大値
    kind_ncc, kind, cls_ncc = "", "", np.nan
    if True:
        tpls = d.load_templates().get(fam, [])
        g = cv2.cvtColor(frames[jt], cv2.COLOR_BGR2GRAY)
        scores = {}
        for k, tg, tm in tpls:
            if g.shape[0] < tg.shape[0] or g.shape[1] < tg.shape[1]: continue
            rr = np.nan_to_num(d.masked_ncc(g, tg, tm), nan=-1.0, posinf=-1.0, neginf=-1.0)
            v = float(rr.max()); scores[k] = v if v <= 1.05 else -1.0
        if scores:
            kind = max(scores, key=scores.get); kind_ncc = ";".join(f"{k}={v:.2f}" for k, v in scores.items())
            cls_ncc = scores.get(r.counter, max(scores.values()))   # 自分の技法のテンプレートがあればその値
    # 証拠画像：初出−6f｜初出｜v4出現｜v4+8f（小領域全体を 0.75 倍）
    tiles = []
    for jj, lab_ in ((max(0, j_on - 6), "onset-6f"), (j_on, "ONSET"), (jm, "v4"), (jt, "v4+8f")):
        t = cv2.resize(frames[jj], None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (int(tx0 * .75), int(ty0 * .75)), (int(tx1 * .75), int(ty1 * .75)), (0, 255, 255) if lab_ == "ONSET" else (255, 255, 255), 1)
        cv2.putText(t, f"{lab_} {pts[jj]:.3f} ncc={ncc[jj]:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(t)
    strip = np.hstack(tiles)
    return dict(mark_time_refined=round(float(pts[j_on]), 3), ncc_at_v4=round(float(ncc[jm]), 3), ncc_min_run=round(float(np.nanmin(ncc[j_on:jm + 1])), 3),
                back_frames=int(jm - j_on), hit_start=int(hit_start), kind=kind, kind_ncc=kind_ncc, cls_ncc=round(float(cls_ncc), 3) if cls_ncc == cls_ncc else np.nan, strip=strip), ""

def worker(args):
    video, session = args
    try:
        d.configure(session)
        df = pd.read_csv(d.MARKS_CSV)
        sub = df[(df.video == video) & df.mark_time_sec.notna()]
        out = []
        ev_dir = os.path.join(d.OUT_DIR, "evidence", "refined", video); os.makedirs(ev_dir, exist_ok=True)
        t_start = time.time()
        for _, r in sub.iterrows():
            try:
                res, err = refine_event(video, r, None)
            except Exception as e:
                res, err = None, f"例外 {e}"
            row = dict(video=video, counter=r.counter, count_value=int(r.count_value))
            if res is None:
                row.update(refine_note=err)
            else:
                d.imwrite_u(os.path.join(ev_dir, f"{r.counter}_{int(r.count_value):03d}_T{r.video_time_sec:.3f}.png"), res.pop("strip"))
                row.update(res); row["refine_note"] = ""
            out.append(row)
        pd.DataFrame(out).to_csv(os.path.join(d.PARTIAL_DIR, video + "_refined.csv"), index=False, encoding="utf-8-sig")
        d.log(f"[{video}] 精緻化 {len(out)}件 {time.time()-t_start:.0f}s 遡り中央値 {np.median([o.get('back_frames', np.nan) for o in out if 'back_frames' in o]) if out else 0:.0f}f")
    except Exception as e:
        d.log(f"[{video}] 精緻化失敗 {e}\n{traceback.format_exc()}")

def merge():
    df = pd.read_csv(d.MARKS_CSV)
    parts = [pd.read_csv(p) for p in sorted(__import__("glob").glob(os.path.join(d.PARTIAL_DIR, "*_refined.csv")))]
    rf = pd.concat(parts, ignore_index=True)
    if "mark_time_raw" not in df.columns: df["mark_time_raw"] = df.mark_time_sec
    df = df.drop(columns=[c for c in ["mark_time_refined", "ncc_at_v4", "ncc_min_run", "back_frames", "hit_start", "kind", "kind_ncc", "cls_ncc", "refine_note"] if c in df.columns])
    df = df.merge(rf, on=["video", "counter", "count_value"], how="left")
    # 自己相関ゲート：v4 出現フレームの印が +8f の自分自身と相関 0.5 未満＝安定した表示物ではない → 誤検出として無効化
    ref = df.mark_time_refined.notna()
    bad_self = ref & (df.ncc_at_v4 < NCC_SELF_MIN)
    gf = getattr(d, "GATE_FACTOR", 1.0)
    bad_cls = ref & (((df.counter == "vibrato") & (df.cls_ncc < CLS_MIN_GREEN * gf)) | ((df.counter == "hammering") & (df.cls_ncc < CLS_MIN_HAMMER * gf)))
    bad_start = ref & (df.hit_start == 1) & (df.counter != "vibrato")
    df["note"] = df.note.fillna("").astype(str)
    df.loc[bad_self, "note"] += "；自己相関低（誤検出の疑い・無効化）"
    df.loc[bad_cls & ~bad_self, "note"] += "；種類テンプレート相関低（背景・演出の疑い・無効化）"
    df.loc[bad_start & ~bad_self & ~bad_cls, "note"] += "；2.6s 前から存在（静止背景の疑い・無効化）"
    bad = bad_self | bad_cls | bad_start
    df.loc[bad, ["mark_time_sec", "lead_sec"]] = np.nan
    has = df.mark_time_refined.notna() & ~bad
    df.loc[has, "mark_time_sec"] = df.loc[has, "mark_time_refined"]
    df.loc[has, "lead_sec"] = (df.video_time_sec - df.mark_time_sec).round(3)[has]
    df["method"] = np.where(has, "v4_hsv_persist+backtrack", df.method)
    df.to_csv(d.MARKS_CSV, index=False, encoding="utf-8-sig")
    d.log(f"[{d.SESSION}] 精緻化を結合 {int(has.sum())}件 無効化{int(bad.sum())}件（自己相関{int(bad_self.sum())}・種類相関{int(bad_cls.sum())}・窓先頭{int(bad_start.sum())}） 遡り中央値 {df.back_frames.median():.0f}f 窓先頭到達 {int(df.hit_start.fillna(0).sum())}件")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--videos", nargs="*"); ap.add_argument("--merge", action="store_true")
    ap.add_argument("--session", default="20260924")
    a = ap.parse_args(); d.configure(a.session)
    if not a.merge:
        vids = a.videos or sorted(pd.read_csv(d.MARKS_CSV).video.unique())
        d.log(f"精緻化開始 曲数{len(vids)} 並列{a.workers}")
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for _ in pool.imap_unordered(worker, [(v, a.session) for v in vids]): pass
    merge()
