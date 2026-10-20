# -*- coding: utf-8 -*-
"""帯（カウンター表示）が画面から消えたフレームを曲ごとに測る（2026-10-20）。

なぜ必要か＝観測区間の表 dataset/song_valid_range.csv の last_frame_pts は**帯動画 pts 表の末尾**であって、
「帯の表示が消えたフレーム」ではない。帯は曲の途中で消えることがある（10/14 Habit＝曲の 2 割が残ったまま）。
その区間を「観測できた区間」に入れたままにすると、そこに出た予測はすべて誤検出（FP）として数えられる。

測り方（何をもって「消えた」とするか）
  1. 帯動画（bands_wao/<video>.mp4 など＝カウンター帯だけを切り出した 2480x120 の動画）を**逐次デコード**する
     （帯動画は VFR なので cv2 のシークはずれる＝10/15 の教訓）。
  2. 各フレームをグレースケール化し、幅 400px へ縮小し、**9x9 の移動平均を引いて高域だけ残す**
     （帯は半透明で、背景の映像がそのまま透けて明るさが変わる。低域を落として「文字と枠の輪郭」だけを見る）。
  3. 帯が確実に映っている区間（曲の 35%〜60% のフレームを 10 枚おきに取り、画素ごとの中央値）を
     その曲の**基準画像**とする。
  4. 各フレームと基準画像の **ZNCC（平均を引いてから正規化した相互相関）** を帯スコアとする。
     帯があるフレームは字の中身が変わってもレイアウト（技法名のラベル・枠・背景の濃淡）が同じなので高く、
     帯が消えて背景映像だけになると低い。
  5. スコアを**幅 15 フレーム（約 0.25 秒）の移動中央値**で均す（加点フラッシュの 1〜2 枚を拾わないため）。
  6. 均したスコアが **THR 以上**のフレームを「帯あり」とし、その最後のフレームを band_last_frame、
     その次のフレーム（帯が消えた最初のフレーム）を band_gone_frame とする。
     最後まで帯があるときは band_gone_frame は空にする。

THR の決め方＝--probe で各曲のスコア分布（分位点）と、既知の 3 曲の答え合わせを出して決める。
  既知＝Habit(10/14) f11435 付近・共犯_102428(10/14) f13687 付近・かくれんぼ(9/11) f16491 で帯が消える
  （いずれも 10/20 午前に 1 フレーム刻みの帯状写真で目視確定したもの）。

使い方
  python band_gone_20261020.py <jobs.tsv> <out.csv> [--thr 0.55] [--probe]
    jobs.tsv ... 1行＝ session <TAB> video <TAB> 帯動画のパス <TAB> pts表のパス（タブ区切り・utf-8）
    --probe  ... スコアの分位点も出力に足す（しきい値を決めるための下見）
"""
import os, sys, csv, time
import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")

W = 400          # 縮小後の幅（高さは縦横比を保つ）
SMOOTH = 15      # 移動中央値の幅（フレーム）＝約 0.25 秒
THR = 0.30       # 帯ありと判定する ZNCC の下限（--thr で変える）
MINRUN = 30      # 「帯あり」として認める最短の連続長（フレーム）＝約 0.5 秒


def read_small(path):
    """帯動画を逐次デコードして、縮小グレースケールの列（uint8・N×h×w）を返す。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("開けない: " + path)
    out = []
    h = w = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if h is None:
            H, Wd = img.shape[:2]
            w = W if Wd > W else Wd
            h = max(1, int(round(H * w / Wd)))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not out:
        raise SystemExit("0 フレーム: " + path)
    return np.stack(out)


def highpass(F):
    """9x9 の移動平均を引いて高域（文字・枠の輪郭）だけ残す。半透明の帯を透ける背景から切り離すため。"""
    return np.stack([x - cv2.blur(x, (9, 9)) for x in F.astype(np.float32)])


def zncc_series(F):
    """各フレームと基準画像（曲の 35〜60% の中央値）との ZNCC（高域版）。"""
    n = len(F)
    a, b = int(n * 0.35), int(n * 0.60)
    if b - a < 20:
        a, b = 0, n
    F = highpass(F)
    ref = np.median(F[a:b:max(1, (b - a) // 200)], axis=0)
    r = ref.ravel(); r = r - r.mean()
    rn = np.linalg.norm(r)
    X = F.reshape(n, -1)
    X = X - X.mean(axis=1, keepdims=True)
    xn = np.linalg.norm(X, axis=1)
    s = (X @ r) / np.maximum(xn * rn, 1e-6)
    return s


def medfilt(x, k):
    if k <= 1:
        return x
    p = k // 2
    xp = np.pad(x, (p, p), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=-1)


def last_run_end(present, minrun):
    """True の連続区間のうち、長さ minrun 以上のものの **最後の区間の末尾**の添字。無ければ None。"""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], present.view(np.int8), [0]))))
    runs = [(idx[i], idx[i + 1] - 1) for i in range(0, len(idx), 2)]
    runs = [r for r in runs if r[1] - r[0] + 1 >= minrun]
    if not runs:
        return None, None
    return runs[0][0], runs[-1][1]


def pts_list(path):
    if not path or not os.path.exists(path):
        return None
    return [float(r["pts_sec"]) for r in csv.DictReader(open(path, encoding="utf-8"))]


def one(ses, video, band, pts, thr, probe):
    t0 = time.time()
    F = read_small(band)
    s = zncc_series(F)
    sm = medfilt(s, SMOOTH)
    present = sm >= thr
    f0, f1 = last_run_end(present, MINRUN)
    P = pts_list(pts)
    n = len(F)
    rec = dict(session=ses, video=video, n_frames_video=n, n_frames_pts=(len(P) if P else ""),
               band_first_frame=("" if f0 is None else int(f0)),
               band_last_frame=("" if f1 is None else int(f1)),
               band_gone_frame=("" if f1 is None or f1 >= n - 1 else int(f1 + 1)),
               band_tail_frames=("" if f1 is None else int(n - 1 - f1)),
               band_thr=thr, band_src=os.path.basename(band), sec=round(time.time() - t0, 1))
    for k, f in (("band_first_pts", f0), ("band_last_pts", f1),
                 ("band_gone_pts", (None if f1 is None or f1 >= n - 1 else f1 + 1))):
        rec[k] = (round(P[f], 3) if (P and f is not None and f < len(P)) else "")
    if probe:
        q = np.percentile(sm, [1, 5, 10, 25, 50, 75, 95])
        rec.update({f"q{int(p)}": round(float(v), 4) for p, v in zip([1, 5, 10, 25, 50, 75, 95], q)})
        rec["frac_ge_thr"] = round(float(present.mean()), 4)
    return rec


COLS = ["session", "video", "n_frames_video", "n_frames_pts", "band_first_frame", "band_last_frame",
        "band_gone_frame", "band_tail_frames", "band_first_pts", "band_last_pts", "band_gone_pts",
        "band_thr", "band_src", "sec"]
PCOLS = ["q1", "q5", "q10", "q25", "q50", "q75", "q95", "frac_ge_thr"]


def main():
    jobs = sys.argv[1]; out = sys.argv[2]
    thr = THR; probe = False
    for a in sys.argv[3:]:
        if a.startswith("--thr"):
            thr = float(a.split("=", 1)[1]) if "=" in a else float(sys.argv[sys.argv.index(a) + 1])
        elif a == "--probe":
            probe = True
    rows = []
    cols = COLS + (PCOLS if probe else [])
    with open(jobs, encoding="utf-8") as f:
        J = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    for i, (ses, video, band, pts) in enumerate(J, 1):
        try:
            r = one(ses, video, band, pts, thr, probe)
        except SystemExit as e:
            print(f"  [{i}/{len(J)}] NG {video}: {e}", flush=True); continue
        rows.append(r)
        print(f"  [{i}/{len(J)}] {video:<45} 帯 {r['band_first_frame']}..{r['band_last_frame']} / {r['n_frames_video']}"
              f"  消失 f{r['band_gone_frame']}  残り {r['band_tail_frames']} 枚  {r['sec']}s", flush=True)
        with open(out, "w", newline="", encoding="utf-8-sig") as g:
            w = csv.DictWriter(g, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print("WROTE", out, len(rows))


if __name__ == "__main__":
    main()
