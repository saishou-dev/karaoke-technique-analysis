# -*- coding: utf-8 -*-
"""C2（入力を変える比較）用の特徴量づくり＝「歌った音程 − お手本の音程」の差を列として足す（2026-10-23）。

何をするか。
録画から読み取ったお手本の音程（基準メロディ列・E4 の <video>_ref.csv）を、検出器が使う 10ms きざみの時間軸に合わせます。
そのうえで「そのとき歌った音程（pYIN の推定値）」から「そのときのお手本の音程」を引いた差（単位＝セント・100 セント＝半音1つ）を作ります。
お手本の音程が読めていないフレーム（お手本の表示が無い・読取に失敗した区間）には、差の値を 0 として置き、別の列に「欠損の印」を立てます。

出力。
既存の特徴量ファイル（feats/<video>.npz）は上書きせず、列を足した別名のファイル（feats_refdiff/<video>.npz）を新しく書きます。
あわせて、どの曲でお手本がどれだけ読めたかの一覧（feats_refdiff/_coverage.csv）を書きます。

使い方:
  python build_refdiff_feats_20261023.py                 # index.json の全曲（お手本が無い曲は飛ばす）
  python build_refdiff_feats_20261023.py --sessions 20260924,20261014
  python build_refdiff_feats_20261023.py --force         # 既に作った別名ファイルも作り直す
"""
import os
import sys
import csv
import json
import glob
import argparse

import numpy as np
import pandas as pd

ROOT = "G:/研究アーカイブ/成果物/動画解析"
G3 = f"{ROOT}/g3v2_20260928"
OUTDIR = f"{G3}/feats_refdiff"
# お手本音程バーの読取結果を探す場所（先に見つかったものを使う）。10/08 が正本・10/23 が学習側セッションの追加分。
REF_DIRS = [f"{ROOT}/pitchbar_read_20261008/out", f"{ROOT}/e5_refdiff_20261023/out"]
CLIP_SEMI = 12.0      # 差の切り詰め（±12 半音＝±1オクターブ）。読み違いの外れ値で学習が壊れるのを防ぐ。
MATCH_TOL = 0.02      # 時間軸を合わせるときの許容（秒）。これより離れたフレームは「お手本なし」とする。


def find_ref(video):
    """<video>_ref.csv と <video>_eval.json の所在を返す（無ければ (None, None)）。"""
    for d in REF_DIRS:
        p = os.path.join(d, video + "_ref.csv")
        if os.path.exists(p):
            e = os.path.join(d, video + "_eval.json")
            return p, (e if os.path.exists(e) else None)
    return None, None


def grid_ref(ref_csv, t):
    """お手本の音程を、検出器の時間軸 t（秒の配列）へ最近傍で並べ直す。戻り＝(ref_midi, 有効フラグ)。"""
    p = pd.read_csv(ref_csv, encoding="utf-8-sig")
    at = p["audio_time_sec"].values.astype(float)
    rm = p["ref_midi"].values.astype(float)
    ok = np.isfinite(at) & np.isfinite(rm)
    at, rm = at[ok], rm[ok]
    if len(at) == 0:
        return np.zeros(len(t), np.float32), np.zeros(len(t), np.float32)
    order = np.argsort(at)
    at, rm = at[order], rm[order]
    j = np.clip(np.searchsorted(at, t), 0, len(at) - 1)
    jm = np.maximum(j - 1, 0)
    pick = np.where(np.abs(at[jm] - t) < np.abs(at[j] - t), jm, j)
    near = np.abs(at[pick] - t) <= MATCH_TOL
    return np.where(near, rm[pick], 0.0).astype(np.float32), near.astype(np.float32)


def build(d, force=False):
    """1曲分。戻り＝記録用の1行（dict）。"""
    v = d["video"]
    row = dict(video=v, session=d["session"], status="", n_frames=0, ref_frac=0.0, voiced_frac=0.0,
               both_frac=0.0, med_abs_diff_cent="", ref_csv="", agree100="")
    base = f"{G3}/feats/{v}.npz"
    if not os.path.exists(base):
        row["status"] = "特徴量なし"
        return row
    ref_csv, eval_json = find_ref(v)
    if ref_csv is None:
        row["status"] = "お手本の読取結果なし"
        return row
    row["ref_csv"] = ref_csv
    if eval_json:
        try:
            row["agree100"] = round(float(json.load(open(eval_json, encoding="utf-8"))["agree100"]), 3)
        except Exception:
            pass
    out = f"{OUTDIR}/{v}.npz"
    z = dict(np.load(base))
    T = z["logmel"].shape[1]
    hop = float(z["hop"]) if "hop" in z else 220 / 22050
    t = np.arange(T) * hop + float(z["t0"]) if "t0" in z else np.arange(T) * hop
    row["n_frames"] = int(T)
    if os.path.exists(out) and not force:
        zz = np.load(out)
        rv = zz["refvalid"]
        vo = zz["voiced"]
        row.update(status="既にあり", ref_frac=round(float(rv.mean()), 3), voiced_frac=round(float(vo.mean()), 3),
                   both_frac=round(float(((rv > 0) & (vo > 0)).mean()), 3))
        return row
    ref_midi, refok = grid_ref(ref_csv, t)
    st = z["st"]
    voiced = z["voiced"]
    valid = ((refok > 0) & (voiced > 0)).astype(np.float32)   # お手本があり、かつ声が出ているフレームだけ差が意味を持つ
    diff_cent = np.where(valid > 0, np.clip(st - ref_midi, -CLIP_SEMI, CLIP_SEMI) * 100.0, 0.0).astype(np.float32)
    z["refdiff_cent"] = diff_cent
    z["refvalid"] = valid
    z["ref_midi"] = ref_midi
    z["refsrc"] = os.path.basename(ref_csv)
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(out, **z)
    m = diff_cent[valid > 0]
    row.update(status="ok", ref_frac=round(float((refok > 0).mean()), 3), voiced_frac=round(float(voiced.mean()), 3),
               both_frac=round(float(valid.mean()), 3),
               med_abs_diff_cent=(round(float(np.median(np.abs(m))), 1) if len(m) else ""))
    return row


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index.json")
    ap.add_argument("--sessions", default="", help="対象セッション（カンマ区切り・既定＝全部）")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    index = json.load(open(f"{G3}/{a.index}", encoding="utf-8"))
    if a.sessions:
        keep = set(a.sessions.split(","))
        index = [d for d in index if d["session"] in keep]
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for d in index:
        r = build(d, a.force)
        rows.append(r)
        print(f"{r['video']}\t{r['status']}\tお手本あり率={r['ref_frac']}\t声とお手本の両方={r['both_frac']}\t差の中央値={r['med_abs_diff_cent']}セント", flush=True)
    keys = list(rows[0].keys())
    with open(f"{OUTDIR}/_coverage.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] in ("ok", "既にあり")]
    print(f"\n作成 {len(ok)}曲 / 対象 {len(rows)}曲", flush=True)
    for s in ("お手本の読取結果なし", "特徴量なし"):
        n = [r for r in rows if r["status"] == s]
        if n:
            print(f"除外 {s}: {len(n)}曲 例={[r['video'] for r in n[:3]]}", flush=True)
    print("DONE", f"{OUTDIR}/_coverage.csv", flush=True)
