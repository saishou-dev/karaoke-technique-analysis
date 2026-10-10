# -*- coding: utf-8 -*-
"""partial/<video>_cands.csv（検出済み候補）から対応付けだけをやり直す（復号なし・数秒）。
変更点: 赤ファミリの相関が 1.05 を超える候補（平坦なパッチで正規化が破綻したもの）を無効にする。
出力: marks_20260924.csv を上書き（列は detect_marks.py と同じ）。"""
import os, sys, glob, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_marks as d

NCC_MAX_VALID = 1.05

def rematch_video(video):
    g1 = pd.read_csv(d.G1_CSV)
    ev = g1[(g1.session == d.SESSION) & (g1.multi_increment == 0) & (g1.video == video)].sort_values("video_time_sec")
    band = pd.read_csv(d.BAND_CSV); crop = d.roi_for(video, band)
    cp = os.path.join(d.PARTIAL_DIR, video + "_cands.csv")
    c = pd.read_csv(cp) if os.path.exists(cp) else pd.DataFrame()
    if len(c):
        c = c[~((c.fam == d.B_RED) & (c.ncc > NCC_MAX_VALID))].reset_index(drop=True)
        c["kind"] = c["kind"].fillna("")
    old = pd.read_csv(os.path.join(d.PARTIAL_DIR, video + ".csv"))
    flash_per_win = dict(zip(zip(old.counter, old.count_value), old.n_flash_in_win))
    rows, taken = [], set()
    for _, r in ev.iterrows():
        T = float(r.video_time_sec); tech = r.counter
        fam, core_req, prior, lmax = d.FAMILY[tech]
        pool = []
        for i, cc in c.iterrows():
            if cc.fam != fam or i in taken: continue
            if not (T - lmax <= cc.t <= T + 0.5): continue
            if fam == d.B_RED and cc.kind != tech: continue
            pool.append((i, cc))
        nf = flash_per_win.get((tech, int(r.count_value)), 0)
        best, note = None, ""
        if pool:
            usable = [(i, cc) for i, cc in pool if not cc.near_cursor]
            if not usable: note = "候補がカーソル近傍のみ（バー塗り進みの可能性）"
            else:
                usable.sort(key=lambda ic: -(ic[1].conf - 0.10 * abs((T - ic[1].t) - prior)))
                best = usable[0]
        else:
            note = "候補なし" + (f"（全面変化フレーム{nf}）" if nf else "")
            other = c[(c.fam == fam) & (c.t >= T - lmax) & (c.t <= T + 0.5)]
            if len(other): note += f"；同色候補{len(other)}件は種類不一致"
        row = dict(video=video, counter=tech, count_value=int(r.count_value), video_time_sec=T,
                   mark_time_sec=np.nan, lead_sec=np.nan, conf=np.nan, method="v4_hsv_persist", note=note,
                   mark_x=np.nan, mark_y=np.nan, area=np.nan, core="", after_flash=np.nan, n_cand_in_win=len(pool), n_flash_in_win=nf)
        if best is not None:
            i, cc = best; taken.add(i)
            row.update(mark_time_sec=round(cc.t, 3), lead_sec=round(T - cc.t, 3), conf=round(min(cc.conf, 1.0), 3),
                       mark_x=int(cc.x * d.DIV + crop[0]), mark_y=int(cc.y * d.DIV + crop[1]), area=int(cc.area), core=cc.core,
                       after_flash=int(cc.after_flash),
                       note=("代替候補あり" if len(pool) > 1 else "") + ("；直前にフラッシュ演出" if cc.after_flash else ""))
        rows.append(row)
    return rows

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--session", default="20260924"); a = ap.parse_args(); d.configure(a.session)
    g1 = pd.read_csv(d.G1_CSV)
    vids = sorted(g1[g1.session == d.SESSION].video.unique())
    allrows = []
    for v in vids:
        if not os.path.exists(os.path.join(d.PARTIAL_DIR, v + ".csv")):
            d.log(f"[{d.SESSION}] rematch: partial なし（帯較正なし等で未処理）→ 飛ばす: {v}"); continue   # 10/08: 1006 は帯較正が 21/29 曲
        allrows += rematch_video(v)
    df = pd.DataFrame(allrows)
    df.to_csv(d.MARKS_CSV, index=False, encoding="utf-8-sig")
    d.log(f"[{d.SESSION}] rematch 完了 {len(df)}行 抽出率 {df.mark_time_sec.notna().mean():.3f}（赤ファミリ ncc>1.05 を無効化）")
