# -*- coding: utf-8 -*-
"""スマホ録画の音声トラックだけで歌声のピッチが取れるかの検証（2026-09-09・卒論 考察用）
   同期済みの曲について、録画音声（伴奏＋歌声＋室内）とSSL2のWAV（マイク直結）に同じpYINをかけ、
   同一時間軸で f0 を比較する。比較は「WAV側が有声で信頼度>0.5」のフレームのみ。
   出力: dataset/phone_audio_pitch_test_20260909.csv（曲別）＋ 同名 _frames.csv（フレーム別・任意）
"""
import csv, os, subprocess, sys, time
import numpy as np, librosa, soundfile as sf
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; SRC = r"G:\研究アーカイブ\ビデオ素材"; REC = r"G:\研究アーカイブ\録音"
FF = os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "WinGet", "Packages",
                  "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe", "ffmpeg-9.0.1-full_build", "bin", "ffmpeg.exe")
WORK = os.path.join(ROOT, "dataset", "phone_audio_test"); os.makedirs(WORK, exist_ok=True)
SR = 22050; HOP = 220; SEG = 60.0  # 各曲60秒（曲の中央）
MAXPER = 2  # セッション毎に上位2曲（同期信頼度順）

rows = list(csv.DictReader(open(os.path.join(ROOT, "dataset", "sync", "sync_offsets.csv"), encoding="utf-8-sig")))
rows = [r for r in rows if r["method"] != "none" and r["wav_file"] and r["confidence"]]
rows.sort(key=lambda r: -float(r["confidence"]))
pick = []; per = {}
for r in rows:
    s = r["session"]
    if per.get(s, 0) >= MAXPER: continue
    src = os.path.join(SRC, s, r["video_file"]); wav = os.path.join(REC, r["wav_file"])
    if os.path.exists(src) and os.path.exists(wav):
        pick.append(r); per[s] = per.get(s, 0) + 1
print("picked", [(r["video_id"], r["confidence"]) for r in pick], flush=True)

ev = list(csv.DictReader(open(os.path.join(ROOT, "dataset", "G1_events_synced.csv"), encoding="utf-8-sig")))

def pyin(y):
    f0, vflag, vprob = librosa.pyin(y, fmin=70, fmax=1100, sr=SR, frame_length=2048, hop_length=HOP)
    return np.nan_to_num(f0, nan=0.0), np.nan_to_num(vprob, nan=0.0)

out = []
for r in pick:
    t0 = time.time(); vid = r["video_id"]; s = r["session"]
    src = os.path.join(SRC, s, r["video_file"]); wav = os.path.join(REC, r["wav_file"]); off = float(r["offset_sec"])
    info = sf.info(wav); wav_len = info.duration
    # 動画時間軸の中央60秒（WAV側の範囲内に収める）
    vstart = max(0.0, (wav_len - off) / 2 - SEG / 2); vend = vstart + SEG
    ph = os.path.join(WORK, vid + "_phone.wav")
    if not os.path.exists(ph):
        subprocess.run([FF, "-y", "-v", "error", "-ss", f"{vstart:.3f}", "-i", src, "-t", f"{SEG:.3f}", "-vn", "-ac", "1", "-ar", str(SR), ph], check=True)
    yp, _ = librosa.load(ph, sr=SR, mono=True)
    yw, _ = librosa.load(wav, sr=SR, mono=True, offset=vstart + off, duration=SEG)
    n = min(len(yp), len(yw)); yp, yw = yp[:n], yw[:n]
    fp, cp = pyin(yp); fw, cw = pyin(yw)
    m = min(len(fp), len(fw)); fp, cp, fw, cw = fp[:m], cp[:m], fw[:m], cw[:m]
    tt = vstart + np.arange(m) * HOP / SR
    ref = (fw > 0) & (cw > 0.5)
    both = ref & (fp > 0)
    cents = 1200 * np.log2(np.where(both, fp, 1) / np.where(both, fw, 1))
    c = cents[both]
    def stats(mask, c_):
        nref = int(mask.sum());
        if nref == 0: return dict(n_ref=0)
        nb = int((mask & (fp > 0)).sum()); cc = c_
        return dict(n_ref=nref, phone_voiced_rate=round(nb / nref, 3),
                    within50=round(float(np.mean(np.abs(cc) <= 50)) if len(cc) else 0, 3),
                    within100=round(float(np.mean(np.abs(cc) <= 100)) if len(cc) else 0, 3),
                    octave_err=round(float(np.mean((np.abs(np.abs(cc) - 1200) <= 60))) if len(cc) else 0, 3),
                    median_abs_cents=round(float(np.median(np.abs(cc))) if len(cc) else 0, 1))
    allst = stats(ref, c)
    # 技法イベント近傍（±0.5s）だけの成績
    evt = [float(e["video_time_sec"]) for e in ev if e["video"] == vid and e["video_time_sec"]]
    near = np.zeros(m, bool)
    for t in evt:
        if vstart <= t <= vend: near |= (np.abs(tt - t) <= 0.5)
    nm = ref & near; nb = nm & (fp > 0)
    cn = 1200 * np.log2(np.where(nb, fp, 1) / np.where(nb, fw, 1))[nb]
    evst = stats(nm, cn)
    # 伴奏の影響の目安：スマホ音声のうちWAVが無声の区間で「有声」と判定された率（＝伴奏を歌声と誤認）
    unv = (fw == 0) | (cw <= 0.3); false_voiced = round(float(np.mean(fp[unv] > 0)) if unv.sum() else 0, 3)
    o = dict(video=vid, session=s, sync_conf=r["confidence"], seg_start=round(vstart, 1), n_events_in_seg=int(near.any() and len([t for t in evt if vstart <= t <= vend])),
             **{"all_" + k: v for k, v in allst.items()}, **{"ev_" + k: v for k, v in evst.items()}, phone_false_voiced_rate=false_voiced, sec=round(time.time() - t0))
    out.append(o); print(o, flush=True)
    with open(os.path.join(WORK, vid + "_frames.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["video_time_sec", "f0_wav", "conf_wav", "f0_phone", "conf_phone"])
        for i in range(m): w.writerow([round(tt[i], 3), round(fw[i], 1), round(cw[i], 2), round(fp[i], 1), round(cp[i], 2)])
keys = sorted({k for o in out for k in o}, key=lambda k: (k not in ("video", "session"), k))
with open(os.path.join(ROOT, "dataset", "phone_audio_pitch_test_20260909.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(o) for o in out]
print("DONE", len(out), flush=True)
