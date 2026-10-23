# -*- coding: utf-8 -*-
"""WAO 3セッション（9/02・8/12・8/18）のお手本読取を「評価の段」まで通せるようにする下ごしらえ（2026-10-23）。

なぜ必要か。
10/23 の WAO64 連結ジョブは「読み取りの段」は成功しているが、「評価の段」が全曲 rc=1 で落ちていた。
原因は2つの入力が無いことだけである。
  (1) 曲ごとの同期表 dataset/sync/sync_offsets_<session>.csv が無い（親の sync_offsets.csv には 64 曲ぶん揃っている）。
  (2) 録音の pYIN f0 dataset/pitch_pyin_<session>/ が無い。
お手本の絶対音高はページごとの半音オフセットを歌唱 F0 との一致で決めるため、評価の段を通らないと <video>_ref.csv が書かれない。
<video>_ref.csv が無い曲は build_refdiff_feats_20261023.py が飛ばすので、読み取りの段だけでは特徴量が1曲も増えない。

すること（順番）。
1. dataset/sync/sync_offsets.csv から該当セッションの行だけを抜き、列名を video・wav・offset_sec に直して
   dataset/sync/sync_offsets_<session>.csv を新しく書く（既存の親ファイルには触らない）。
2. dataset/pitch_pyin_<session>/ に pYIN f0 を書く（scripts/pitch_pyin_20261014.py と同一パラメータ＝
   SR22050・hop220・fmin70・fmax800・frame2048。既にある曲は飛ばす）。
3. 既に frames.npz がある曲について read_pitchbar.py --stage eval を回し、<video>_ref.csv・_eval.json・_overlay.png を書く。

使い方:
  python prep_eval_wao_20261023.py --sessions 20260902                # 下ごしらえ＋評価の段まで
  python prep_eval_wao_20261023.py --sessions 20260902 --stage prep   # 1と2だけ
  python prep_eval_wao_20261023.py --sessions 20260812,20260818 --workers 5
"""
import os
import sys
import csv
import glob
import time
import argparse
import subprocess
import concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
WAV_ROOT = r"G:\研究アーカイブ\録音"
OUT = os.path.join(ROOT, "e5_refdiff_20261023", "out")
SR = 22050
HOP = 220


def log(s):
    print(f"{time.strftime('%H:%M:%S')} {s}", flush=True)


def make_sync(session):
    """親の同期表から該当セッションだけを抜き、読取側が読める列名で書く。"""
    src = os.path.join(ROOT, "dataset", "sync", "sync_offsets.csv")
    dst = os.path.join(ROOT, "dataset", "sync", f"sync_offsets_{session}.csv")
    if os.path.exists(dst):
        log(f"[{session}] 同期表は既にある（触らない） {dst}")
        return sum(1 for _ in csv.DictReader(open(dst, encoding="utf-8-sig")))
    rows = [r for r in csv.DictReader(open(src, encoding="utf-8-sig")) if r["session"] == session]
    if not rows:
        log(f"[{session}] !! 親の同期表に行が無い＝下ごしらえ不能")
        return 0
    tmp = dst + ".part"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["video", "wav", "offset_sec", "method", "confidence", "note"])
        for r in rows:
            w.writerow([r["video_id"], os.path.basename(r["wav_file"]), r["offset_sec"],
                        r.get("method", ""), r.get("confidence", ""),
                        "sync_offsets.csv から抜き出し 2026-10-23"])
    os.replace(tmp, dst)
    log(f"[{session}] 同期表を作成 {len(rows)}曲 {dst}")
    return len(rows)


def pyin_one(p):
    import numpy as np
    import librosa
    session = pyin_one.session
    out = os.path.join(ROOT, "dataset", f"pitch_pyin_{session}")
    stem = os.path.splitext(os.path.basename(p))[0]
    dst = os.path.join(out, stem + "_f0.csv")
    if os.path.exists(dst):
        return stem, "exists"
    t0 = time.time()
    y, _ = librosa.load(p, sr=SR, mono=True)
    f0, vflag, vprob = librosa.pyin(y, fmin=70, fmax=800, sr=SR, frame_length=2048, hop_length=HOP)
    times = librosa.times_like(f0, sr=SR, hop_length=HOP)
    tmp = dst + ".part"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["audio_time_sec", "f0_hz", "conf", "voiced"])
        for t, fz, vf, vp in zip(times, f0, vflag, vprob):
            w.writerow([f"{t:.3f}", "0.00" if np.isnan(fz) else f"{fz:.2f}", f"{vp:.3f}", int(bool(vf))])
    os.replace(tmp, dst)
    return stem, f"ok voiced={float(np.mean(vflag)):.0%} {time.time()-t0:.0f}s"


def make_pyin(session, workers):
    out = os.path.join(ROOT, "dataset", f"pitch_pyin_{session}")
    os.makedirs(out, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(WAV_ROOT, f"songs_{session}_raw", "*.wav")))
    log(f"[{session}] pYIN 開始 {len(wavs)}本 workers={workers}")
    pyin_one.session = session
    done = 0
    with cf.ProcessPoolExecutor(max_workers=workers,
                                initializer=_init_session, initargs=(session,)) as ex:
        for stem, st in ex.map(pyin_one, wavs):
            done += 1
            log(f"  {stem}: {st}")
    log(f"[{session}] pYIN 終了 {done}本")
    return done


def _init_session(session):
    pyin_one.session = session


def eval_one(args):
    session, video = args
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, "read_pitchbar.py"),
                        session, video, "--stage", "eval"],
                       env=env, capture_output=True, text=True, errors="replace")
    tail = (p.stdout or "").strip().splitlines()[-1:] + (p.stderr or "").strip().splitlines()[-1:]
    return video, p.returncode, int(time.time() - t0), " | ".join(tail)


def run_eval(session, workers):
    vids = sorted(os.path.basename(x)[:-len("_frames.npz")]
                  for x in glob.glob(os.path.join(OUT, f"*_{session}_*_frames.npz")))
    todo = [v for v in vids if not os.path.exists(os.path.join(OUT, v + "_ref.csv"))]
    log(f"[{session}] 評価の段 読取済み{len(vids)}曲 うち未評価{len(todo)}曲 workers={workers}")
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for video, rc, sec, tail in ex.map(eval_one, [(session, v) for v in todo]):
            ok += (rc == 0)
            log(f"  {video}: eval rc={rc} {sec}s {tail}")
    log(f"[{session}] 評価の段 終了 成功{ok}/{len(todo)}")
    return ok, len(todo)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="20260902,20260812,20260818")
    ap.add_argument("--stage", default="all", choices=["prep", "eval", "all"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--wait_log", default="", help="このログに印が出るまで待つ（前段の読取の完了待ち）")
    ap.add_argument("--wait_mark", default="WAO64 DONE")
    ap.add_argument("--max_wait_hours", type=float, default=12.0)
    a = ap.parse_args()

    if a.wait_log:
        t0 = time.time()
        log(f"前段待ち 開始 印='{a.wait_mark}' ログ={a.wait_log}")
        while time.time() - t0 < a.max_wait_hours * 3600:
            try:
                if a.wait_mark in open(a.wait_log, encoding="utf-8", errors="replace").read():
                    log(f"前段の完了を確認")
                    break
            except FileNotFoundError:
                pass
            log(f"待機中（前段の印 '{a.wait_mark}' を待っています）")
            time.sleep(120)
        else:
            log("待ち時間の上限に達した。読み取り済みの曲だけで評価の段を回す")

    for s in [x for x in a.sessions.split(",") if x]:
        if a.stage in ("prep", "all"):
            if make_sync(s) == 0:
                continue
            make_pyin(s, a.workers)
        if a.stage in ("eval", "all"):
            run_eval(s, a.workers)
    log("PREP EVAL WAO DONE")
