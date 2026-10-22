# -*- coding: utf-8 -*-
"""6段目：精緻化の後段の歯止め3つの閾値を振り、件数と時刻精度のトレードオフを表にする。

**復号はしない。既存スクリプトも動かさない。** 精緻化が書いた中間ファイル
`partial_20261014\*_refined.csv`（自己相関・種類相関・窓先頭到達・さかのぼり量が入っている）と、
精緻化前の退避 `marks_20261014.pre_refine_20261022.bak.csv` を読んで、結合だけをやり直す。

歯止め（`refine_onsets.py` の merge() と同じ条件）:
  ① 自己相関   ncc_at_v4 < NCC_SELF_MIN           → 無効化（安定した表示物ではない）
  ② 種類相関   ビブラート cls_ncc < CLS_GREEN
               ハンマリング cls_ncc < CLS_HAMMER   → 無効化（背景・演出の疑い）
  ③ 窓の先頭   hit_start==1                        → 無効化（2.6 秒前から存在＝静止背景の疑い）

振る値（現行は `refine_onsets.py` の既定）:
  ① 0.40（緩め）／0.50（現行）／0.60（厳しめ）
  ② (0.20,0.30) 緩め ／ (0.30,0.40) 現行 ／ (0.45,0.50) 厳しめ（9/24 の当初値）
  ③ none（無効化しない・緩め）／ except_vibrato（ビブラート以外を無効化・現行）／ all（全技法・厳しめ）
  3×3×3＝27 通り。

列:
  抽出・抽出率／残差の中央値・±0.5秒以内の割合（技法ごとの中央値で補正）／
  ビブラートの 2秒超の割合・最大・90%点

さらに、いちばん大きい外れ値（ビブラート 5.224 秒）が加点演出の火花への食いつきかを、
2段目で保存した升目（`grid_<video>.npz` の白い画素）から判定する。印の位置の升目の白い面積が
主張された出現時刻で跳ねていれば、火花に食いついた疑いが濃い。

出力:
  <OUT_DIR>/gates_sweep.csv        27 通りの表
  <OUT_DIR>/gates_outlier_check.csv 外れ値の火花判定
  <OUT_DIR>/refine_evidence/sparkle_<...>.png  根拠画像（白の面積の時間変化＋印の4コマ）
"""
import os, sys, glob, itertools, argparse
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P

MARKS_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
PARTIAL = os.path.join(MARKS_DIR, "partial_20261014")
PRE = os.path.join(MARKS_DIR, "marks_20261014.pre_refine_20261022.bak.csv")
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール",
      "vibrato": "ビブラート", "accent": "アクセント", "hammering": "ハンマリング"}
SELFS = [0.40, 0.50, 0.60]
CLSS = [(0.20, 0.30), (0.30, 0.40), (0.45, 0.50)]
STARTS = ["none", "except_vibrato", "all"]
LABEL = {0.40: "緩め", 0.50: "現行", 0.60: "厳しめ",
         (0.20, 0.30): "緩め", (0.30, 0.40): "現行", (0.45, 0.50): "厳しめ",
         "none": "緩め", "except_vibrato": "現行", "all": "厳しめ"}

CSV_0924 = os.path.join(MARKS_DIR, "marks_20260924.csv")

def base(session=20261014):
    """歯止めの入力（自己相関・種類相関・窓先頭到達）が入った表を作る。復号も再実行もしない。

    10/14＝精緻化前の退避 + 中間ファイル `partial_20261014\\*_refined.csv` を結合。
    9/24 ＝`marks_20260924.csv` に精緻化の列が既に結合済みなので、それをそのまま読む。
           精緻化前の印の時刻は `mark_time_raw` 列に残っている。"""
    if int(session) == 20260924:
        df = pd.read_csv(CSV_0924)
        for c in ("video_time_sec", "mark_time_raw", "mark_time_refined",
                  "ncc_at_v4", "cls_ncc", "hit_start", "mark_x", "mark_y"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["mark_time_sec"] = df["mark_time_raw"]      # 精緻化前の印の時刻
        return df
    pre = pd.read_csv(PRE)
    for c in ("mark_time_sec", "video_time_sec", "mark_x", "mark_y"):
        pre[c] = pd.to_numeric(pre[c], errors="coerce")
    ref = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PARTIAL, "*_refined.csv")))],
                    ignore_index=True)
    df = pre.merge(ref, on=["video", "counter", "count_value"], how="left")
    return df

def apply_gates(df, self_min, cls_pair, start_mode):
    cg, ch = cls_pair
    ref = df.mark_time_refined.notna()
    bad_self = ref & (df.ncc_at_v4 < self_min)
    bad_cls = ref & (((df.counter == "vibrato") & (df.cls_ncc < cg)) |
                     ((df.counter == "hammering") & (df.cls_ncc < ch)))
    if start_mode == "none":
        bad_start = ref & False
    elif start_mode == "all":
        bad_start = ref & (df.hit_start == 1)
    else:
        bad_start = ref & (df.hit_start == 1) & (df.counter != "vibrato")
    bad = bad_self | bad_cls | bad_start
    has = ref & ~bad
    out = df.copy()
    out["ok"] = has
    out["lead"] = np.where(has, out.video_time_sec - out.mark_time_refined, np.nan)
    return out, int(bad_self.sum()), int(bad_cls.sum()), int(bad_start.sum())

def by_counter_gates(out, counter="vibrato"):
    """ある技法だけの裾（件数・2秒超・90%点・最大）"""
    g = out[out.ok & (out.counter == counter)].lead
    return dict(件数=len(g),
                超2秒=round(float((g > 2.0).mean()), 3) if len(g) else np.nan,
                p90=round(float(g.quantile(.9)), 3) if len(g) else np.nan,
                最大=round(float(g.max()), 3) if len(g) else np.nan,
                中央値=round(float(g.median()), 3) if len(g) else np.nan)

def metrics(out):
    n_proc = len(out)
    n_ok = int(out.ok.sum())
    res_all = []
    for c, g in out[out.ok].groupby("counter"):
        med = float(g.lead.median())
        res_all.append((g.lead - med).abs())
    res = pd.concat(res_all) if res_all else pd.Series(dtype=float)
    vib = out[out.ok & (out.counter == "vibrato")].lead
    frac05 = float((res <= 0.5).mean()) if len(res) else np.nan
    return dict(抽出=n_ok, 抽出率=round(n_ok / n_proc, 3),
                # 有効件数＝補正後の時刻が ±0.5 秒に収まった件数。件数と精度の積で、両方を同時に見る指標。
                有効件数=int(round(n_ok * frac05)) if len(res) else 0,
                残差の中央値=round(float(res.median()), 3) if len(res) else np.nan,
                残差05秒以内=round(frac05, 3) if len(res) else np.nan,
                ビブラート件数=len(vib),
                ビブラート2秒超=round(float((vib > 2.0).mean()), 3) if len(vib) else np.nan,
                ビブラートp90=round(float(vib.quantile(.9)), 3) if len(vib) else np.nan,
                ビブラート最大=round(float(vib.max()), 3) if len(vib) else np.nan)

def sweep(df, label=""):
    rows = []
    for s, c, st in itertools.product(SELFS, CLSS, STARTS):
        out, b1, b2, b3 = apply_gates(df, s, c, st)
        r = dict(自己相関=f"{s:.2f}({LABEL[s]})",
                 種類相関=f"{c[0]:.2f}/{c[1]:.2f}({LABEL[c]})",
                 窓先頭=f"{st}({LABEL[st]})",
                 無効化=b1 + b2 + b3, 内訳=f"自己{b1}/種類{b2}/先頭{b3}")
        if label: r = dict(セッション=label, **r)
        r.update(metrics(out))
        rows.append(r)
    return pd.DataFrame(rows)

# ---- 外れ値が火花への食いつきかを判定する ----
CH = {"white": 0, "gold": 1, "bright": 2}

def sparkle_check(video, mark_x, mark_y, t_onset, t_v4, span=8.0, ch="bright"):
    """2段目の升目（1/2縮小・16×192）から、印の位置の升目の明るい画素の面積の時間変化を取る。
    主張された出現時刻で跳ねていれば、加点演出の火花に食いついた疑いが濃い。

    指標に「明るい画素（V≥200）」を使う。無彩色の白（S<60 かつ V≥230）はこの場所でほぼ 0 で、
    加点演出の火花が背景映像の色に染まっていて白と判定されないため（実測で区間の最大 5 画素）。"""
    z = np.load(os.path.join(P.OUT_DIR, f"grid_{video}.npz"))
    g = z["grid"].astype(np.float32); t = z["t"]; crop = z["crop"]
    nrow, ncol = int(z["nrow"]), int(z["ncol"])
    cx = (mark_x - crop[0]) / crop[2] * ncol
    cy = (mark_y - crop[1]) / crop[3] * nrow
    if not (0 <= cx < ncol and 0 <= cy < nrow): return None
    c0, c1 = max(0, int(cx) - 3), min(ncol, int(cx) + 4)
    r0, r1 = max(0, int(cy) - 1), min(nrow, int(cy) + 2)
    w = g[:, CH[ch], r0:r1, c0:c1].sum(axis=(1, 2))
    wh = g[:, CH["white"], r0:r1, c0:c1].sum(axis=(1, 2))
    m = (t >= t_onset - span) & (t <= t_v4 + 1.0)
    return dict(t=t[m], w=w[m], white=wh[m], cell=(r0, r1, c0, c1), ch=ch)

def plot_curve(d, t_onset, t_v4, title, W=960, H=260):
    t, w = d["t"], d["w"]
    img = np.full((H, W, 3), 24, np.uint8)
    lo, hi = float(np.min(w)), float(np.max(w)) + 1e-6
    x = ((t - t[0]) / (t[-1] - t[0]) * (W - 80) + 60).astype(int)
    y = (H - 40 - (w - lo) / (hi - lo) * (H - 80)).astype(int)
    cv2.polylines(img, [np.stack([x, y], 1)], False, (180, 220, 255), 1)
    for tt, col, lab in ((t_onset, (0, 255, 255), "ONSET"), (t_v4, (0, 200, 0), "v4")):
        xi = int((tt - t[0]) / (t[-1] - t[0]) * (W - 80) + 60)
        cv2.line(img, (xi, 20), (xi, H - 30), col, 2)
        cv2.putText(img, lab, (xi + 4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.putText(img, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img, f"white px  max={hi:.0f}  min={lo:.0f}", (8, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, f"{t[0]:.1f}s", (50, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, f"{t[-1]:.1f}s", (W - 110, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img

TECH = ["vibrato", "shakuri", "kobushi", "accent", "hammering", "fall"]

def rank_table(session, label, min_n=20):
    """27 通りそれぞれで「裾がいちばん長い技法」を求める。
    割合そのものは閾値で動くので、**順位が安定かどうか**を見るための表。
    件数 min_n 未満の技法は順位の判定から外す（10/14 のビブラートのように数件では読めないため）。"""
    df = base(session)
    rows = []
    for s, c, st in itertools.product(SELFS, CLSS, STARTS):
        out, *_ = apply_gates(df, s, c, st)
        d = dict(セッション=label, 自己相関=f"{s:.2f}", 種類相関=f"{c[0]:.2f}/{c[1]:.2f}", 窓先頭=st)
        v2, v90 = {}, {}
        for t in TECH:
            m = by_counter_gates(out, t)
            d[JP[t] + "_n"] = m["件数"]; d[JP[t] + "_超2秒"] = m["超2秒"]; d[JP[t] + "_p90"] = m["p90"]
            if m["件数"] >= min_n:
                v2[t] = m["超2秒"]; v90[t] = m["p90"]
        d["超2秒の最大の技法"] = JP[max(v2, key=v2.get)] if v2 else "-"
        d["p90の最大の技法"] = JP[max(v90, key=v90.get)] if v90 else "-"
        rows.append(d)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--session", type=int, default=20261014)
    ap.add_argument("--rank", action="store_true", help="27通りで裾がいちばん長い技法を数える")
    a = ap.parse_args()
    if a.rank:
        for sess, lab in ((20260924, "9/24"), (20261014, "10/14")):
            t = rank_table(sess, lab)
            t.to_csv(os.path.join(P.OUT_DIR, f"gates_rank_{sess}.csv"), index=False, encoding="utf-8-sig")
            print(f"=== {lab}：27通りのうち、裾がいちばん長い技法（件数20以上の技法だけで判定） ===")
            print("  2秒超が最大 :", t["超2秒の最大の技法"].value_counts().to_dict())
            print("  p90 が最大  :", t["p90の最大の技法"].value_counts().to_dict())
            print("  ビブラート 件数 %d〜%d / 2秒超 %.3f〜%.3f / p90 %.3f〜%.3f"
                  % (t["ビブラート_n"].min(), t["ビブラート_n"].max(),
                     t["ビブラート_超2秒"].min(), t["ビブラート_超2秒"].max(),
                     t["ビブラート_p90"].min(), t["ビブラート_p90"].max()))
            print()
        sys.exit(0)
    if a.session == 20260924:
        df = base(20260924)
        sw = sweep(df, "9/24")
        sw.to_csv(os.path.join(P.OUT_DIR, "gates_sweep_0924.csv"), index=False, encoding="utf-8-sig")
        cols = ["自己相関", "種類相関", "窓先頭", "抽出", "抽出率", "有効件数", "残差の中央値",
                "残差05秒以内", "ビブラート件数", "ビブラート2秒超", "ビブラートp90", "ビブラート最大"]
        print("=== 9/24・27通り（ビブラートの2秒超が高い順） ===")
        print(sw.sort_values("ビブラート2秒超", ascending=False)[cols].to_string(index=False))
        print()
        print("=== ビブラートの裾の振れ幅 ===")
        for k in ("ビブラート件数", "ビブラート2秒超", "ビブラートp90", "ビブラート最大"):
            print("  %-14s 最小 %8.3f  最大 %8.3f  幅 %8.3f" % (k, sw[k].min(), sw[k].max(), sw[k].max() - sw[k].min()))
        sys.exit(0)
    df = base()
    print(f"精緻化の対象 {len(df)} 行／精緻化が値を返した {int(df.mark_time_refined.notna().sum())} 件")
    sw = sweep(df)
    sw.to_csv(os.path.join(P.OUT_DIR, "gates_sweep.csv"), index=False, encoding="utf-8-sig")
    print()
    print("=== 歯止めの閾値 27 通り（抽出率の高い順・上位） ===")
    print(sw.sort_values("抽出率", ascending=False).head(a.top).to_string(index=False))
    print()
    print("=== 現行の組み合わせ ===")
    cur = sw[(sw.自己相関.str.contains("現行")) & (sw.種類相関.str.contains("現行")) & (sw.窓先頭.str.contains("現行"))]
    print(cur.to_string(index=False))

    # 外れ値の火花判定
    rows = []
    checks = [("RPG(クレヨンしんちゃんアニメバージョン)_20261014_114902", "vibrato", 13),
              ("怪獣の花唄_20261014_104344", "accent", 11)]
    for v, c, cv_ in checks:
        r = df[(df.video == v) & (df.counter == c) & (df.count_value == cv_)]
        if len(r) == 0: continue
        r = r.iloc[0]
        d = sparkle_check(v, float(r.mark_x), float(r.mark_y),
                          float(r.mark_time_refined), float(r.mark_time_sec))
        if d is None: continue
        t, w = d["t"], d["w"]
        i_on = int(np.argmin(np.abs(t - r.mark_time_refined)))
        i_v4 = int(np.argmin(np.abs(t - r.mark_time_sec)))
        base_med = float(np.median(w)); base_mad = float(np.median(np.abs(w - base_med))) * 1.4826 or 1.0
        z_on = (float(w[i_on]) - base_med) / base_mad
        z_v4 = (float(w[i_v4]) - base_med) / base_mad
        rows.append(dict(video=v, 技法=JP.get(c, c), count_value=cv_,
                         カウンター増分=round(float(r.video_time_sec), 3),
                         出現=round(float(r.mark_time_refined), 3),
                         基本検出=round(float(r.mark_time_sec), 3),
                         遅れ=round(float(r.video_time_sec - r.mark_time_refined), 3),
                         指標=d["ch"],
                         出現時の明るい画素=round(float(w[i_on]), 1),
                         区間の中央値=round(base_med, 1),
                         区間の最大=round(float(w.max()), 1),
                         出現のz=round(z_on, 2), 基本検出のz=round(z_v4, 2),
                         判定="火花への食いつきの疑い" if z_on >= 3.0 else "火花の跳ねなし"))
        img = plot_curve(d, float(r.mark_time_refined), float(r.mark_time_sec),
                         f"{v[:18]} {c}#{cv_}  bright area around mark (cell r{d['cell'][0]}-{d['cell'][1]} c{d['cell'][2]}-{d['cell'][3]})")
        strip = os.path.join(MARKS_DIR, "evidence", "refined", v,
                             f"{c}_{cv_:03d}_T{float(r.video_time_sec):.3f}.png")
        if os.path.exists(strip):
            s = cv2.imdecode(np.fromfile(strip, np.uint8), cv2.IMREAD_COLOR)
            s = cv2.resize(s, (img.shape[1], int(s.shape[0] * img.shape[1] / s.shape[1])))
            img = np.vstack([img, s])
        P.imwrite_u(os.path.join(P.OUT_DIR, "refine_evidence", f"sparkle_{JP.get(c,c)}_{cv_:03d}.png"), img)
    oc = pd.DataFrame(rows)
    oc.to_csv(os.path.join(P.OUT_DIR, "gates_outlier_check.csv"), index=False, encoding="utf-8-sig")
    print()
    print("=== 外れ値の火花判定（印の位置の升目の白い画素） ===")
    print(oc.to_string(index=False))
