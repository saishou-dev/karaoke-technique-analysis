# -*- coding: utf-8 -*-
"""速度逆算法の前提検証＝x–t スライス（時空間スライス解析）。

標準名＝時空間スライス解析（spatio-temporal slice analysis）。
一次出典＝C.-W. Ngo, T.-C. Pong, H.-J. Zhang, "Motion analysis and segmentation through
spatio-temporal slices processing," IEEE Trans. Image Processing, 12(3), 341–355, 2003.
DOI 10.1109/TIP.2003.809020（x–t スライス上では一定速度の運動は直線になり、傾きが速度になる）。

やること:
  1. 音程ガイド表示（音程バー）の領域を横 192 列に割った断面を、全フレーム積み上げて 1 枚の画像にする
     （横＝画面の x・縦＝時刻）。これが x–t スライス。
  2. 流れる線の傾きがスクロール速度 v。傾きは「せん断して時間方向に足す」操作で測る
     （候補の速度ごとに画像を斜めにずらして縦に積み、いちばん鋭い山が立つ速度を選ぶ＝Radon 変換と同じ考え方）。
  3. 窓（既定 6 秒・3 秒ずらし）ごとに v を測り、曲内で一定か・曲ごとに違うかを数値と画像で示す。

入力＝2段目で保存した升目 grid_<video>.npz（復号はしない）。
出力:
  <OUT_DIR>/xt/<video>_xt.png        曲全体の x–t スライス（根拠画像）
  <OUT_DIR>/xt/<video>_xt_zoom.png   30 秒ぶんを等倍で拡大したもの（傾きが目で見える）
  <OUT_DIR>/xt_speed.csv             窓ごとの速度
"""
import os, sys, argparse
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P
import detect_grid as D

V_LO, V_HI, V_STEP = 100.0, 560.0, 5.0     # 候補の速度 [px/秒]
WIN_SEC, HOP_SEC = 6.0, 3.0
CH = {"white": 0, "gold": 1, "bright": 2}

def xt(video, ch="bright"):
    """x–t スライスを作る。横＝列（画面の x）・縦＝フレーム（時刻）。"""
    g, t, crop = D.load(video)
    a = g[:, CH[ch]].sum(axis=1).astype(np.float32)     # (N, ncol) 縦を合計
    # 行ごとに正規化（画面全体の明暗の変化を消す＝線の形だけを残す）
    m = a.mean(axis=1, keepdims=True); s = a.std(axis=1, keepdims=True) + 1e-6
    z = (a - m) / s
    return t, z, crop

def best_v(z, t, i0, i1, px_per_col, vs, min_frac=0.6):
    """窓 [i0, i1) で、候補の速度ごとにせん断して縦に足し、いちばん鋭い山が立つ速度を返す。

    2 つの落とし穴に対処している。
    ① 音程バーは静止しているので x–t スライスでは**縦の縞**になり、そのままだと速度 0 がいつも最強に出る。
       → 窓の中で列ごとの時間中央値を引き、静止した構造を消してから測る（動いているものだけ残す）。
    ② せん断すると右側の列ほど足し合わせる回数が減るので、回数で割らないと遅い速度が有利になる。
       → 列ごとの回数を数えて割り、十分な回数がある列だけで鋭さを測る。
    鋭さ＝(profile の最大値 − 中央値) ÷ 標準偏差。線が 1 本に集まるほど大きくなる。
    """
    blk = z[i0:i1].copy()
    n, C = blk.shape
    if n < 20: return None
    blk -= np.median(blk, axis=0, keepdims=True)        # ①静止した縦縞を消す
    dt = t[i0:i1] - t[i0]
    best = (None, -1e18)
    for v in vs:
        shift = np.round(v * dt / px_per_col).astype(int)
        acc = np.zeros(C, np.float32); cnt = np.zeros(C, np.float32)
        for j in range(n):
            s = int(shift[j])
            if s >= C: continue
            acc[:C - s] += blk[j, s:]; cnt[:C - s] += 1.0   # ②列ごとに回数を数える
        ok = cnt >= min_frac * n
        if ok.sum() < 20: continue
        prof = acc[ok] / cnt[ok]
        sd = float(prof.std())
        if sd <= 1e-6: continue
        sc = (float(prof.max()) - float(np.median(prof))) / sd
        if sc > best[1]: best = (v, sc)
    return best

def measure(video, ch="bright", win=WIN_SEC, hop=HOP_SEC):
    t, z, crop = xt(video, ch)
    ncol = z.shape[1]; px_per_col = crop[2] / ncol
    vs = np.arange(V_LO, V_HI + 1e-9, V_STEP)
    fps = P.FPS_NOMINAL
    nw, nh = int(win * fps), int(hop * fps)
    rows = []
    for i0 in range(0, len(t) - nw, nh):
        r = best_v(z, t, i0, i0 + nw, px_per_col, vs)
        if r is None or r[0] is None: continue
        rows.append(dict(video=video, t0=round(float(t[i0]), 2), v=float(r[0]), 鋭さ=round(r[1], 3)))
    return pd.DataFrame(rows), t, z, crop

def save_images(video, t, z, crop, zoom_t0=60.0, zoom_sec=30.0):
    """x–t スライスを画像として書き出す（根拠画像）。横＝画面の x・縦＝時刻（下へ進む）。"""
    d = os.path.join(P.OUT_DIR, "xt")
    img = np.clip((z * 60 + 128), 0, 255).astype(np.uint8)
    # 全体＝時間方向を 8 コマに 1 つ間引き、横を 4 倍に伸ばして見やすくする
    full = cv2.resize(img[::8], (img.shape[1] * 4, img.shape[0] // 8), interpolation=cv2.INTER_NEAREST)
    full = cv2.applyColorMap(full, cv2.COLORMAP_BONE)
    cv2.putText(full, f"{video}  x-t slice (yoko=gamen no x / tate=jikoku)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    P.imwrite_u(os.path.join(d, f"{video}_xt.png"), full)
    i0 = int(np.searchsorted(t, zoom_t0)); i1 = int(np.searchsorted(t, zoom_t0 + zoom_sec))
    zm = cv2.resize(img[i0:i1], (img.shape[1] * 4, (i1 - i0) // 2), interpolation=cv2.INTER_NEAREST)
    zm = cv2.applyColorMap(zm, cv2.COLORMAP_BONE)
    cv2.putText(zm, f"{video}  {zoom_t0:.0f}-{zoom_t0+zoom_sec:.0f}s", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    P.imwrite_u(os.path.join(d, f"{video}_xt_zoom.png"), zm)

def page_period(video, ch="bright", lo=2.5, hi=14.0):
    """ページの周期＝カーソルが左端へ戻る間隔。
    傾きの自動推定は当てにならなかったが、周期は左端の列の明るさの自己相関で安定して測れる。
    「ページごとに速度が違う」なら周期もばらつく。"""
    t, z, crop = xt(video, ch)
    sig = z[:, :8].mean(axis=1)
    sig = sig - np.median(sig)
    fps = P.FPS_NOMINAL
    n = len(sig)
    ac = np.correlate(sig, sig, mode="full")[n - 1:]
    ac /= (ac[0] + 1e-9)
    i0, i1 = int(lo * fps), min(int(hi * fps), len(ac) - 1)
    k = i0 + int(np.argmax(ac[i0:i1]))
    return k / fps, float(ac[k])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--ch", default="bright")
    a = ap.parse_args()
    allr = []
    summ = []
    for v in (a.videos or P.VIDEOS):
        df, t, z, crop = measure(v, a.ch)
        save_images(v, t, z, crop)
        allr.append(df)
        q = df.v
        summ.append(dict(video=v, 窓の数=len(df), 中央値=round(float(q.median()), 1),
                         第1四分位=round(float(q.quantile(.25)), 1), 第3四分位=round(float(q.quantile(.75)), 1),
                         最小=round(float(q.min()), 1), 最大=round(float(q.max()), 1),
                         中央値まわり20パーセント以内の窓=round(float(((q - q.median()).abs() <= 0.2 * q.median()).mean()), 3)))
        P.log(f"[{v}] x-t スライス 窓{len(df)}本 v中央値{q.median():.1f}px/s")
    pd.concat(allr, ignore_index=True).to_csv(os.path.join(P.OUT_DIR, "xt_speed.csv"),
                                              index=False, encoding="utf-8-sig")
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(P.OUT_DIR, "xt_speed_summary.csv"), index=False, encoding="utf-8-sig")
    print(s.to_string(index=False))
