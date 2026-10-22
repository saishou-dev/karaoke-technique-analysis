# -*- coding: utf-8 -*-
r"""C3 整合表の下ごしらえ：録音 WAV から 100 fps のピッチ曲線を作って npz に保存する（2026-10-23）。

なぜ必要か＝C3（曲単位の整合）で使う「録音から測ったビブラート合計秒」は 100 fps のピッチ曲線が前提だが、
g3v2_20260928/feats には 2026-10-14 までのセッションしか無い。10/21 以降の録音を同じ形式で足すための道具。

出力の形式は g3v2_20260928/feats/*.npz と同じ（st=MIDI 半音・voiced=0/1・conf=有声確率・rms・t0=0.0・hop=220/22050）。
logmel は C3 では使わないので作らない（CRNN 予測に使う場合は g3v2 側のパイプラインで作ること）。

定義：
  f0   = librosa.pyin（fmin 70 / fmax 800 Hz・frame_length 2048・hop 220・sr 22050 → 100.2 fps）
  st   = 69 + 12*log2(f0/440)（有声フレームのみ・無声は直前値を保持せず 0）
  voiced = pyin の voiced_flag（f0 が得られたフレーム）

使い方:
  python c3_pitch_feats.py --session 20261021 [--wavdir ...] [--out ...] [--jobs 8]
"""
import argparse, os, sys, glob, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")

SR = 22050
HOP = 220
FMIN, FMAX = 70.0, 800.0


def one(args):
    path, out = args
    import soundfile as sf, librosa
    base = os.path.splitext(os.path.basename(path))[0]
    dst = os.path.join(out, base + ".npz")
    if os.path.exists(dst):
        return (base, "skip", 0.0)
    t0 = time.time()
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    f0, vflag, vprob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR,
                                    frame_length=2048, hop_length=HOP)
    f0 = np.nan_to_num(f0, nan=0.0)
    v = (vflag & (f0 > FMIN * 0.9)).astype(np.float32)
    st = np.where(v > 0, 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-6) / 440.0), 0.0).astype(np.float32)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0].astype(np.float32)
    n = min(len(st), len(rms))
    np.savez_compressed(dst, st=st[:n], voiced=v[:n],
                        conf=np.nan_to_num(vprob, nan=0.0)[:n].astype(np.float32),
                        rms=rms[:n], t0=np.float64(0.0), hop=np.float64(HOP / SR),
                        src=np.array("wav"))
    return (base, "ok", time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--wavdir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--jobs", type=int, default=8, help="並列プロセス数（他の作業と分け合うため既定は論理コアの半分程度）")
    a = ap.parse_args()
    wavdir = a.wavdir or f"G:/研究アーカイブ/録音/songs_{a.session}_raw"
    out = a.out or f"G:/研究アーカイブ/成果物/動画解析/c3_consistency_20261023/feats_{a.session}"
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(wavdir, "*.wav")))
    print(f"session={a.session} wav={len(files)}本 jobs={a.jobs} out={out}", flush=True)
    if not files:
        print("WAV が見つからない", flush=True)
        return 1
    import multiprocessing as mp
    t0 = time.time()
    done = 0
    with mp.Pool(max(1, a.jobs)) as pool:
        for base, how, sec in pool.imap_unordered(one, [(f, out) for f in files]):
            done += 1
            print(f"[{done}/{len(files)}] {how} {sec:.1f}s {base}", flush=True)
    print(f"完了 {len(files)}本 {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
