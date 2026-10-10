# -*- coding: utf-8 -*-
"""スマホ音声のみモードのセルフテスト（2026-09-15）。9/11 の CHEERS_20260911_105621（動画 1.6GB・録音 T07 あり・同期表 offset=-21.13）で、
   ① extract：WAV が 22050Hz/1ch、長さが動画コンテナ長と ±0.05s
   ② pyin：f0 表が生成され、行数≒長さ×100
   ③ オフセット0の検証：抽出WAV × SSL2 WAV（T07）の相互相関（sync_20260915.py と同じ RMS包絡 8kHz）で得る lag が、
      録画×録音の既知オフセット（dataset/sync/sync_offsets_20260911.csv の -21.13）と ±0.05s で一致 ＝ 抽出WAVの時間軸＝動画時間軸
      （併せて f0 の一致率＝WAV基準 有声・conf>0.5 のフレームで ±50cent 以内の率を参考記録）
   ④ 同期表なしで G1 を生成（audio_time 空欄）→ annotate で wav=<video>.wav・audio_time_sec=video_time_sec・audio_source=phone になる
   ⑤ スマホ同期表（offset 0）を置いて G1 を生成 → wav/audio_time_sec が pipeline 経路で埋まり、④と同一。g3_predict_wav を抽出WAVに回すと照合表が出る
   出力はすべて dataset/_selftest_20260915phone/（本番ファイルは触らない）。使い方: python scripts/selftest_phone_20260915.py
"""
import os, sys, csv, shutil, subprocess, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, PYTHON, ROOT, WAV_ROOT

S = "20260911"; V = "CHEERS_20260911_105621"; TW = "T07_20260911_105934.wav"
T = os.path.join(ROOT, "dataset", "_selftest_20260915phone"); os.makedirs(T, exist_ok=True)
SC = os.path.join(ROOT, "scripts"); EX = os.path.join(SC, "extract_phone_audio_20260915.py")
AUD = os.path.join(T, f"phone_audio_{S}"); PY = os.path.join(T, f"pitch_pyin_phone_{S}"); SYNC = os.path.join(T, "sync", f"sync_offsets_{S}.csv")
LOG = os.path.join(T, "selftest.log"); ENV = dict(os.environ, PYTHONIOENCODING="utf-8", KV_SESSION=S)
res = []
def ok(name, cond, detail=""):
    res.append((name, bool(cond), detail)); print(("PASS " if cond else "FAIL ") + name + (" — " + detail if detail else ""), flush=True)
def run(args, env=None):
    p = subprocess.run([PYTHON, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env or ENV)
    open(LOG, "a", encoding="utf-8").write(f"### {' '.join(args)} rc={p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-1500:]}\n"); return p
common = [S, "--videos", V, "--out-base", T, "--sync-csv", SYNC, "--pipe-log", os.path.join(T, "pipeline.log")]

# ① extract
t0 = time.time(); p = run([EX, *common, "--step", "extract"]); wav = os.path.join(AUD, V + ".wav")
ex = {r["video"]: r for r in csv.DictReader(open(os.path.join(AUD, "_extract.csv"), encoding="utf-8-sig"))} if os.path.exists(os.path.join(AUD, "_extract.csv")) else {}
r = ex.get(V, {})
import soundfile as sf
wi = sf.info(wav) if os.path.exists(wav) else None
ok("①抽出WAV 22050Hz/1ch・長さ＝動画±0.05s", p.returncode == 0 and wi is not None and wi.samplerate == 22050 and wi.channels == 1 and abs(float(r.get("dur_diff", "9"))) <= 0.05,
   f"rc={p.returncode} status={r.get('status')} video={r.get('video_dur')}s wav={r.get('wav_dur')}s diff={r.get('dur_diff')}s audio_start={r.get('audio_start')} src={r.get('src_codec')}/{r.get('src_sr')}/{r.get('src_ch')}ch {time.time()-t0:.0f}s")

# ② pyin
t0 = time.time(); p = run([EX, *common, "--step", "pyin"]); f0p = os.path.join(PY, V + "_f0.csv")
F = pd.read_csv(f0p, encoding="utf-8-sig") if os.path.exists(f0p) else pd.DataFrame()
ok("②pYIN f0 表（同パラメータ・audio_source=phone 列）", p.returncode == 0 and len(F) > 0 and abs(len(F) - wi.duration * 22050 / 220) <= 5 and set(F["audio_source"]) == {"phone"},
   f"rc={p.returncode} rows={len(F)} expected≈{wi.duration*22050/220:.0f} voiced={F['voiced'].mean():.0%} median_f0={F.loc[F['f0_hz']>0,'f0_hz'].median():.0f}Hz {time.time()-t0:.0f}s" if len(F) else f"rc={p.returncode}")

# ③ オフセット0の検証（既知の録画×録音オフセットを抽出WAVで再現）
import librosa
SR8 = 8000; HOP8 = 80
def env8(y):
    e = librosa.feature.rms(y=y, frame_length=400, hop_length=HOP8)[0]; return (e - e.mean()) / (e.std() + 1e-9)
def xcorr(a, b):
    n = len(a) + len(b); N = 1 << (n - 1).bit_length(); c = np.fft.irfft(np.fft.rfft(a, N) * np.conj(np.fft.rfft(b, N)), N)
    c = np.concatenate([c[-(len(b) - 1):], c[:len(a)]]); lags = np.arange(-(len(b) - 1), len(a)); k = int(np.argmax(c)); return lags[k] * HOP8 / SR8, float(c[k] / min(len(a), len(b)))
yp, _ = librosa.load(wav, sr=SR8, mono=True); yw, _ = librosa.load(os.path.join(WAV_ROOT, f"songs_{S}_raw", TW), sr=SR8, mono=True)
lag, rr = xcorr(env8(yp), env8(yw)); ref = float([x for x in csv.DictReader(open(Cfg(S).sync_csv, encoding="utf-8-sig")) if x["video"] == V][0]["offset_sec"])
got = -lag  # sync_20260915.py と同じ符号規約（offset_sec = wav_time - video_time）
# f0 一致率（参考）：WAV側 f0（dataset/pitch_pyin_20260911/T07_f0.csv）を video 時間軸へ（t_video = t_wav - offset）
W = pd.read_csv(os.path.join(ROOT, "dataset", "pitch_pyin_20260911", TW[:-4] + "_f0.csv"), encoding="utf-8-sig")
tw = W["audio_time_sec"].values - ref; fw = W["f0_hz"].values; cw = W["conf"].values
fp = np.interp(tw, F["audio_time_sec"].values, F["f0_hz"].values, left=0, right=0)
m = (fw > 0) & (cw > 0.5) & (tw >= 0) & (tw <= F["audio_time_sec"].max()); both = m & (fp > 0)
cents = 1200 * np.log2(fp[both] / fw[both]); w50 = float(np.mean(np.abs(cents) <= 50)) if both.sum() else 0; oct_ = float(np.mean(np.abs(np.abs(cents) - 1200) <= 60)) if both.sum() else 0
ok("③抽出WAV×SSL2 WAV の相互相関が既知オフセットと一致（＝動画時間軸・オフセット0）", abs(got - ref) <= 0.05,
   f"xcorr offset={got:+.3f}s ref(sync_offsets_20260911)={ref:+.3f}s diff={got-ref:+.3f}s r={rr:.3f}｜参考 f0一致(±50cent)={w50:.0%} オクターブ誤り={oct_:.0%} n_ref={int(m.sum())} phone有声率={both.sum()/max(1,m.sum()):.0%}")

# ④ 同期表なしで G1（Ai 版スクリプト＝KV_AI_BASE で同期表の位置をテスト側に置ける・時刻計算は WAO 版と同一）→ annotate
RD = os.path.join(T, f"results_wao_{S}"); os.makedirs(os.path.join(RD, V), exist_ok=True)
shutil.copy(os.path.join(ROOT, f"results_wao_{S}", V, "counter_events.csv"), os.path.join(RD, V, "counter_events.csv"))
if os.path.exists(SYNC): os.remove(SYNC)
G1 = os.path.join(T, f"G1_events_{S}.csv")
for f in (G1, G1[:-4] + "_pre_phone.bak.csv"):
    if os.path.exists(f): os.remove(f)
env = dict(ENV, KV_AI_BASE=T, KV_RESULTS_DIR=RD, KV_PTS_BAND_DIR=os.path.join(ROOT, "dataset", f"pts_bands_{S}"), KV_G1_OUT=G1, KV_PIPE_DIR=os.path.join(T, "pipeline"), KV_BLIND_DIR=os.path.join(T, f"blind_{S}"))
p = run([os.path.join(SC, "g1_and_match_20260915ai.py"), S], env)
G0 = pd.read_csv(G1, encoding="utf-8-sig") if os.path.exists(G1) else pd.DataFrame()
blank = len(G0) > 0 and G0["wav"].isna().all() and G0["audio_time_sec"].isna().all()
p2 = run([EX, *common, "--g1", G1, "--step", "annotate"])
G = pd.read_csv(G1, encoding="utf-8-sig") if os.path.exists(G1) else pd.DataFrame()
c4 = p.returncode == 0 and blank and p2.returncode == 0 and len(G) == len(G0) and (G["wav"] == V + ".wav").all() and (G["audio_source"] == "phone").all() \
     and (G["audio_time_sec"] - G["video_time_sec"]).abs().max() <= 0.0005 and os.path.exists(G1[:-4] + "_pre_phone.bak.csv")
ok("④同期表なし→G1 は audio_time 空欄→annotate で wav/audio_time(=video_time)/audio_source=phone", c4,
   f"g1 rc={p.returncode} blank_before={blank} annotate rc={p2.returncode} rows={len(G)} max|audio-video|={(G['audio_time_sec']-G['video_time_sec']).abs().max() if len(G) else 'NA'} audio_source={G['audio_source'].value_counts().to_dict() if len(G) else {}}")
# 9/14 の帯pts版 G1 と video_time が一致（同じ表を引いている確認）
R = pd.read_csv(os.path.join(ROOT, "dataset", "_selftest_20260914", "G1_selftest.csv"), encoding="utf-8-sig"); R = R[R["video"] == V]
M = G.merge(R, on=["video", "counter", "count_value"], suffixes=("_new", "_ref")) if len(G) else pd.DataFrame()
ok("④' video_time_sec が 9/14 セルフテスト G1 と一致", len(M) == len(R) == len(G) and (M["video_time_sec_new"] - M["video_time_sec_ref"]).abs().max() <= 0.0015,
   f"rows new={len(G)} ref={len(R)} merged={len(M)} max|Δ|={(M['video_time_sec_new']-M['video_time_sec_ref']).abs().max() if len(M) else 'NA'}")

# ⑤ スマホ同期表 → G1（pipeline 経路）＋ g3_predict_wav → 照合
p3 = run([EX, *common, "--step", "synctable"])
BL = env["KV_BLIND_DIR"]; os.makedirs(BL, exist_ok=True)
t0 = time.time(); p4 = run([os.path.join(SC, "g3_predict_wav.py"), wav, "--out", BL])
G1b = os.path.join(T, f"G1_events_{S}_via_synctable.csv"); env5 = dict(env, KV_G1_OUT=G1b)
p5 = run([os.path.join(SC, "g1_and_match_20260915ai.py"), S], env5)
Gb = pd.read_csv(G1b, encoding="utf-8-sig") if os.path.exists(G1b) else pd.DataFrame()
same = len(Gb) == len(G) and (Gb["wav"] == V + ".wav").all() and (Gb["audio_time_sec"].values - G["audio_time_sec"].values).__abs__().max() <= 0.0005 if len(Gb) and len(G) else False
ms = os.path.join(BL, f"match_summary_{S}.csv"); MS = pd.read_csv(ms, encoding="utf-8-sig") if os.path.exists(ms) else pd.DataFrame()
ok("⑤スマホ同期表→G1（pipeline 経路）が④と同一・g3_predict_wav→照合表が出る", p3.returncode == 0 and p4.returncode == 0 and p5.returncode == 0 and same and len(MS) > 0,
   f"synctable rc={p3.returncode} predict rc={p4.returncode} ({time.time()-t0:.0f}s) g1 rc={p5.returncode} same_as_annotate={same} match_summary rows={len(MS)}"
   + (("｜参考(tol=1.0) " + "; ".join(f"{r.type}: true={r.true} pred={r.pred} R={r.recall} P={r.precision}" for r in MS[MS['tol'] == 1.0].itertuples())) if len(MS) else ""))

with open(os.path.join(T, "SELFTEST_RESULT.txt"), "w", encoding="utf-8") as f:
    for n, c, d in res: f.write(("PASS " if c else "FAIL ") + n + (" — " + d if d else "") + "\n")
    f.write(f"ALL {'PASS' if all(c for _, c, _ in res) else 'FAIL'} ({sum(c for _, c, _ in res)}/{len(res)})\n")
print("ALL", "PASS" if all(c for _, c, _ in res) else "FAIL")
