# -*- coding: utf-8 -*-
"""案D（ページ表示版）手順1＝対応件のマーク出現時刻の前後で再生カーソル x を毎フレーム追う・2026-11-05 Opus
各曲で [mark_time−PRE, mark_time+POST] を併合した区間だけ復号（NVDEC）し、detect_marks.py と同じ ROI を 1/2 に縮めて
detect_marks.find_cursor（1/2 縮小基準の判定）でカーソル x を出す。x は 4K 原画素に戻す（×2＋crop 原点）。
出力＝OUT/tracks/<session>__<video>.csv（pts_sec, cursor_x_4k）。既存スクリプトは import して関数を使うだけ。
使い方: python cursor_track_20261105.py SESSION [VIDEO ...]（VIDEO 省略＝対応件のある全曲）
"""
import os, sys, re, subprocess, csv, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_marks as dm

BASE = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
OUT = os.path.join(BASE, "out", "mark_pos_time_20261105")
PRE, POST = 6.0, 0.6

def load_pairs(session):
    """対応件（mark_status=='mark'・multi_increment=0）に mark_x を付けて返す"""
    if int(session) == 20260924:
        g = pd.read_csv(os.path.join(BASE, "G1_events_v2_20260924.csv"))
        g = g[g.mark_status == "mark"].copy()
    else:
        e = pd.read_csv(os.path.join(BASE, f"mark_lead_events_{session}.csv"))
        m = pd.read_csv(os.path.join(BASE, f"marks_{session}.csv"))[["video", "counter", "count_value", "mark_x", "mark_y"]]
        g = e[e.mark_status == "mark"].merge(m, on=["video", "counter", "count_value"], how="left")
    g["session"] = int(session)
    return g

def intervals(ts):
    iv = []
    for t in sorted(ts):
        a, b = max(0.0, t - PRE), t + POST
        if iv and a <= iv[-1][1]: iv[-1][1] = max(iv[-1][1], b)
        else: iv.append([a, b])
    return iv

def track(session, video, mts):
    dm.configure(session)
    crop = dm.roi_for(video, pd.read_csv(dm.BAND_CSV)); cx, cy, cw, ch = crop
    ow, oh = cw // 2, ch // 2
    path = os.path.join(dm.VID_DIR, video + ".mp4")
    rows = []
    for a, b in intervals(mts):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-hwaccel", "cuda", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}",
               "-copyts", "-i", path, "-an", "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={ow}:{oh}:flags=area,showinfo",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, bufsize=10**7)
        import threading
        err = []
        th = threading.Thread(target=lambda: err.append(p.stderr.read())); th.start()
        fs = ow * oh * 3; xs = []
        while True:
            buf = p.stdout.read(fs)
            if len(buf) < fs: break
            f = np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
            c = dm.find_cursor(dm.classify(f))
            xs.append(None if c is None else c * 2 + cx)
        p.wait(); th.join()
        pts = [float(x) for x in re.findall(rb"pts_time:([0-9.]+)", err[0] if err else b"")]
        for t, x in zip(pts, xs):
            rows.append((round(t, 4), "" if x is None else x))
    return rows

if __name__ == "__main__":
    ses = sys.argv[1]
    pairs = load_pairs(ses)
    vids = sys.argv[2:] or sorted(pairs.video.unique())
    if vids and vids[0].startswith("@"):   # @リスト（UTF-8・1行1曲）＝シェル経由の日本語引数の化け対策
        vids = [l.strip() for l in open(vids[0][1:], encoding="utf-8") if l.strip()]
    os.makedirs(os.path.join(OUT, "tracks"), exist_ok=True)
    for v in vids:
        fn = os.path.join(OUT, "tracks", f"{ses}__{v}.csv")
        if os.path.exists(fn): print("skip", v); continue
        t0 = time.time()
        try:
            rows = track(ses, v, pairs[pairs.video == v].mark_time_sec.astype(float).tolist())
        except Exception as ex:
            print(f"FAIL {v} {ex!r}", flush=True); continue
        tmp = fn + ".part"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["pts_sec", "cursor_x_4k"]); w.writerows(rows)
        os.replace(tmp, fn)
        n_ok = sum(1 for r in rows if r[1] != "")
        print(f"{v} frames={len(rows)} cursor={n_ok} sec={time.time() - t0:.1f}", flush=True)
