# -*- coding: utf-8 -*-
"""スマホ動画の音声トラックを WAV 化し、録音WAV（SSL2）の代わりに使う「スマホ音声のみモード」（2026-09-15・Windows python）。
   背景：9/15 は SSL2 の USB データケーブル忘れで録音WAVなし。動画（4K60・AAC 48kHz 2ch）の音声だけがある。
   方式：
     extract  ffmpeg で 各動画 → dataset/phone_audio_<S>/<video>.wav（22050Hz・モノラル・16bit）。
              `-af aresample=async=1:first_pts=0` で音声ストリームの start_time（9/11実測 0.002s）を無音で埋め、
              WAV の時間軸＝動画コンテナの時間軸（＝帯pts表の軸）にする。よって同期オフセットは定義上 0。
     pyin     pitch_pyin_20260911.py と同じパラメータ（SR22050・hop220・fmin70/fmax800・frame2048）で f0 → dataset/pitch_pyin_phone_<S>/<video>_f0.csv。
              伴奏が混ざるため f0 は参考値（9/9 の検証＝WAV基準で ±50cent 一致 19〜71%・伴奏を有声と誤認 51〜80%）。
     synctable 同期表 dataset/sync/sync_offsets_<S>.csv を「offset_sec=0・wav=<video>.wav・method=phone_audio_offset0・audio_source=phone」で書く
              （録音WAV由来の表が既にあれば触らない）。これで pipeline の sync 工程は SKIP、g1 工程は audio_time_sec=video_time_sec を書く。
     annotate G1（dataset/G1_events_<S>.csv）に audio_source 列を付ける（phone／ssl2／空）。wav が空の行はスマホWAVで埋める（audio_time_sec=video_time_sec）。
              初回だけ G1_events_<S>_pre_phone.bak.csv を残す。冪等。
   使い方（既定＝all＝extract→pyin→synctable→annotate）:
     python scripts/extract_phone_audio_20260915.py 20260915
     python scripts/extract_phone_audio_20260915.py 20260915 --step extract      # 抽出だけ（監督と並行可・監督より先に走らせる）
     python scripts/extract_phone_audio_20260915.py 20260915 --step annotate     # G1 生成後に列を付ける
     --machine wao|ai（ログの書き先＝pipeline_<S>／pipeline_ai_<S>・既定 wao。G1・同期表のパスは両機種で同じ）
   ブラインド照合：python scripts/g3_predict_wav.py dataset/phone_audio_<S>/*.wav --out dataset/blind_<S> → pipeline --from g1（<video>_pred.csv の stem＝G1 の wav 列 stem）。
   セルフテスト用の差し替え：--videos <stem…>／--vid-dir／--out-base／--sync-csv／--g1／--pipe-log。
"""
import os, sys, csv, glob, json, time, argparse, subprocess, shutil, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, FFMPEG, FFPROBE, ROOT

SR = 22050; HOP = 220; FMIN = 70; FMAX = 800; FRAME = 2048   # pitch_pyin_20260911.py と同一
STEPS = ["extract", "pyin", "synctable", "annotate"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", default=None)
    ap.add_argument("--step", default="all", choices=STEPS + ["all"])
    ap.add_argument("--machine", default="wao", choices=["wao", "ai"])
    ap.add_argument("--workers", type=int, default=3, help="ffmpeg 並列数（pyin は --pyin-workers）")
    ap.add_argument("--pyin-workers", type=int, default=6)
    ap.add_argument("--videos", nargs="*", default=None, help="対象の動画 stem を限定（セルフテスト用）")
    ap.add_argument("--vid-dir", default=None); ap.add_argument("--out-base", default=None, help="phone_audio_<S>／pitch_pyin_phone_<S> の親（既定 dataset/）")
    ap.add_argument("--sync-csv", default=None); ap.add_argument("--g1", default=None); ap.add_argument("--pipe-log", default=None)
    ap.add_argument("--force-sync", action="store_true", help="既存の同期表がスマホ由来なら書き直す（録音WAV由来の表は何があっても触らない）")
    return ap.parse_args()


class P:
    """パス束。本番は Cfg（機種で pipeline ログ先だけ変わる）。"""
    def __init__(self, a):
        if a.machine == "ai":
            from kv_config_20260915ai import CfgAi
            self.C = CfgAi(a.session)
        else:
            self.C = Cfg(a.session)
        C = self.C; S = C.S; self.S = S
        self.vid_dir = a.vid_dir or C.vid_dir
        base = a.out_base or os.path.join(ROOT, "dataset")
        self.audio_dir = os.path.join(base, f"phone_audio_{S}")
        self.pyin_dir = os.path.join(base, f"pitch_pyin_phone_{S}")
        self.extract_csv = os.path.join(self.audio_dir, "_extract.csv")
        self.log_path = os.path.join(self.audio_dir, "phone_audio.log")
        self.sync_csv = a.sync_csv or C.sync_csv
        self.g1_csv = a.g1 or C.g1_csv
        self.pipe_log = a.pipe_log or C.pipe_log
        os.makedirs(self.audio_dir, exist_ok=True)

    def log(self, msg):
        self.C.log("PHONE " + msg, self.log_path)
        try: self.C.log("PHONE " + msg, self.pipe_log)
        except Exception: pass

    def videos(self, only=None):
        if not os.path.isdir(self.vid_dir): return []
        vs = sorted(os.path.splitext(f)[0] for f in os.listdir(self.vid_dir) if f.lower().endswith(".mp4") and not f.startswith("VID_"))
        if only: vs = [v for v in vs if v in set(only)]
        return vs

    def wav_of(self, stem): return os.path.join(self.audio_dir, stem + ".wav")


# ---------- extract ----------
def probe(path):
    p = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,sample_rate,channels,start_time,duration",
                        "-of", "json", path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    j = json.loads(p.stdout or "{}"); out = dict(fmt_duration=float(j.get("format", {}).get("duration", 0) or 0), audio=None, video=None)
    for s in j.get("streams", []):
        if s.get("codec_type") == "audio" and out["audio"] is None: out["audio"] = s
        if s.get("codec_type") == "video" and out["video"] is None: out["video"] = s
    return out


def wav_info(path):
    import soundfile as sf
    i = sf.info(path); return dict(sr=i.samplerate, ch=i.channels, dur=float(i.duration), frames=i.frames)


def extract_one(args):
    src, dst = args
    if os.path.exists(dst): return dst, "exists", None
    pr = probe(src)
    if pr["audio"] is None: return dst, "err no audio stream", pr
    tmp = dst + ".part"
    p = subprocess.run([FFMPEG, "-y", "-v", "error", "-i", src, "-vn", "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", str(SR),
                        "-c:a", "pcm_s16le", "-f", "wav", tmp], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 1000:
        if os.path.exists(tmp): os.remove(tmp)
        return dst, "err ffmpeg " + p.stderr[-200:].replace("\n", " "), pr
    os.replace(tmp, dst); return dst, "ok", pr


def step_extract(P, a):
    vids = P.videos(a.videos)
    if not vids: P.log(f"EXTRACT STOP: 曲名付き動画が無い {P.vid_dir}"); return 2
    jobs = [(os.path.join(P.vid_dir, v + ".mp4"), P.wav_of(v)) for v in vids]
    rows = {}
    if os.path.exists(P.extract_csv):
        rows = {r["video"]: r for r in csv.DictReader(open(P.extract_csv, encoding="utf-8-sig"))}
    t0 = time.time(); n_ok = n_err = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for (src, dst), (_, st, pr) in zip(jobs, ex.map(extract_one, jobs)):
            v = os.path.splitext(os.path.basename(src))[0]
            if st.startswith("err"):
                n_err += 1; rows[v] = dict(video=v, wav="", status=st, video_dur="", wav_dur="", dur_diff="", audio_start="", src_codec="", src_sr="", src_ch="")
                P.log(f"EXTRACT {v}: {st}"); continue
            if pr is None: pr = probe(src)
            wi = wav_info(dst); vd = pr["fmt_duration"]; au = pr["audio"] or {}
            diff = wi["dur"] - vd; flag = "" if abs(diff) <= 0.05 else " WARN 長さ差>0.05s"
            rows[v] = dict(video=v, wav=os.path.basename(dst), status=st + flag.strip(), video_dur=f"{vd:.3f}", wav_dur=f"{wi['dur']:.3f}", dur_diff=f"{diff:+.3f}",
                           audio_start=au.get("start_time", ""), src_codec=au.get("codec_name", ""), src_sr=au.get("sample_rate", ""), src_ch=au.get("channels", ""))
            ok = wi["sr"] == SR and wi["ch"] == 1
            if not ok: rows[v]["status"] = f"err wav {wi['sr']}Hz/{wi['ch']}ch"; n_err += 1
            else: n_ok += 1
            P.log(f"EXTRACT {v}: {rows[v]['status']} video={vd:.3f}s wav={wi['dur']:.3f}s diff={diff:+.3f}s audio_start={au.get('start_time','')} src={au.get('codec_name','')}/{au.get('sample_rate','')}Hz/{au.get('channels','')}ch")
    fields = ["video", "wav", "status", "video_dur", "wav_dur", "dur_diff", "audio_start", "src_codec", "src_sr", "src_ch"]
    with open(P.extract_csv + ".part", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows[k] for k in sorted(rows))
    os.replace(P.extract_csv + ".part", P.extract_csv)
    P.log(f"EXTRACT DONE ok={n_ok} err={n_err} total={len(vids)} {time.time()-t0:.0f}s → {P.audio_dir}")
    return 0 if n_err == 0 else 1


# ---------- pyin ----------
def pyin_one(args):
    src, dst = args
    import numpy as np, librosa
    if os.path.exists(dst): return dst, "exists"
    t0 = time.time(); y, _ = librosa.load(src, sr=SR, mono=True)
    f0, vflag, vprob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=FRAME, hop_length=HOP)
    times = librosa.times_like(f0, sr=SR, hop_length=HOP); tmp = dst + ".part"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["audio_time_sec", "f0_hz", "conf", "voiced", "audio_source"])
        for t, fz, vf, vp in zip(times, f0, vflag, vprob):
            w.writerow([f"{t:.3f}", "0.00" if np.isnan(fz) else f"{fz:.2f}", f"{vp:.3f}", int(bool(vf)), "phone"])
    os.replace(tmp, dst)
    med = float(np.nanmedian(f0)) if np.isfinite(np.nanmedian(f0)) else 0.0
    return dst, f"ok voiced={float(np.mean(vflag)):.0%} median={med:.0f}Hz {time.time()-t0:.0f}s"


def step_pyin(P, a):
    os.makedirs(P.pyin_dir, exist_ok=True)
    vids = P.videos(a.videos); jobs = [(P.wav_of(v), os.path.join(P.pyin_dir, v + "_f0.csv")) for v in vids if os.path.exists(P.wav_of(v))]
    if not jobs: P.log("PYIN STOP: スマホWAVが無い（--step extract が先）"); return 2
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.pyin_workers) as ex:
        for (src, _), (dst, st) in zip(jobs, ex.map(pyin_one, jobs)):
            P.log(f"PYIN {os.path.splitext(os.path.basename(src))[0]}: {st}")
    P.log(f"PYIN DONE n={len(jobs)} {time.time()-t0:.0f}s → {P.pyin_dir}（伴奏混入・参考値）")
    return 0


# ---------- synctable ----------
SYNC_FIELDS = ["video", "wav", "offset_sec", "method", "r_main", "r_onset", "lag_main", "lag_onset", "agree", "peak_ratio", "played_at", "audio_source"]


def step_synctable(P, a):
    vids = [v for v in P.videos(a.videos) if os.path.exists(P.wav_of(v))]
    if not vids: P.log("SYNCTABLE STOP: スマホWAVが無い（--step extract が先）"); return 2
    if os.path.exists(P.sync_csv):
        old = list(csv.DictReader(open(P.sync_csv, encoding="utf-8-sig")))
        if any(r.get("audio_source", "") != "phone" for r in old):
            P.log(f"SYNCTABLE skip: 録音WAV由来の同期表がある {P.sync_csv}（触らない）"); return 0
        if not a.force_sync:
            P.log(f"SYNCTABLE skip: スマホ由来の同期表が既にある {P.sync_csv}（--force-sync で書き直し）"); return 0
    played = {}
    if os.path.exists(P.C.rename_csv):
        for r in csv.DictReader(open(P.C.rename_csv, encoding="utf-8-sig")):
            if r.get("renamed"): played[os.path.splitext(r["renamed"])[0]] = r.get("played_at", "")
    rows = [dict(video=v, wav=v + ".wav", offset_sec=0.0, method="phone_audio_offset0", r_main="", r_onset="", lag_main=0.0, lag_onset=0.0, agree=True,
                 peak_ratio="", played_at=played.get(v, ""), audio_source="phone") for v in vids]
    os.makedirs(os.path.dirname(P.sync_csv), exist_ok=True)
    with open(P.sync_csv + ".part", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SYNC_FIELDS); w.writeheader(); w.writerows(rows)
    os.replace(P.sync_csv + ".part", P.sync_csv)
    P.log(f"SYNCTABLE WROTE {P.sync_csv} rows={len(rows)} offset=0（スマホ音声＝動画時間軸）")
    return 0


# ---------- annotate ----------
def step_annotate(P, a):
    import pandas as pd
    if not os.path.exists(P.g1_csv): P.log(f"ANNOTATE WARN: G1 が未生成 {P.g1_csv}（pipeline の g1 工程後に --step annotate）"); return 1
    G = pd.read_csv(P.g1_csv, encoding="utf-8-sig", dtype={"wav": str})
    G["wav"] = G["wav"].fillna("")
    phone = {v for v in set(G["video"]) if os.path.exists(P.wav_of(v))}
    bak = P.g1_csv[:-4] + "_pre_phone.bak.csv"
    if not os.path.exists(bak): shutil.copy(P.g1_csv, bak)
    filled = 0
    for i in G.index[(G["wav"] == "") & (G["video"].isin(phone))]:
        G.at[i, "wav"] = G.at[i, "video"] + ".wav"; G.at[i, "audio_time_sec"] = G.at[i, "video_time_sec"]; filled += 1
    def src(r):
        if r["wav"] == "": return ""
        return "phone" if (r["video"] in phone and r["wav"] == r["video"] + ".wav") else "ssl2"
    G["audio_source"] = G.apply(src, axis=1)
    G.to_csv(P.g1_csv, index=False, encoding="utf-8-sig")
    vc = G["audio_source"].value_counts().to_dict()
    P.log(f"ANNOTATE WROTE {P.g1_csv} rows={len(G)} filled={filled} audio_source={vc}（phone＝伴奏混入・参考値／backup={os.path.basename(bak)}）")
    return 0


if __name__ == "__main__":
    a = parse(); P = P(a); steps = STEPS if a.step == "all" else [a.step]
    P.log(f"START session={P.S} machine={a.machine} steps={steps} vid_dir={P.vid_dir}")
    rc = 0
    for s in steps:
        r = {"extract": step_extract, "pyin": step_pyin, "synctable": step_synctable, "annotate": step_annotate}[s](P, a)
        rc = max(rc, r)
        if r == 2: break
    P.log(f"END rc={rc}"); print("PHONE_AUDIO_DONE" if rc == 0 else f"PHONE_AUDIO_RC={rc}", flush=True)
    raise SystemExit(rc)
