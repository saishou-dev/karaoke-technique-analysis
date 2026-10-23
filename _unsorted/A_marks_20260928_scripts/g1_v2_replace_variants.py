# -*- coding: utf-8 -*-
"""正解時刻 v2 の置換のしかたを数通り作り、件数と精度を並べて比べる。

入力＝`g1_v2_from_marks.py` が書いた候補 CSV（`dataset/g1_v2_candidate_<session>.csv`）。
正本 G1（`dataset/G1_events_all.csv`）は読むだけで変更しない。

出す案（材料41 の裁定(3)）
  (A)  混合＝方式Aで取れた行は v2・取れなかった行は v1 のまま。
  (A') 混合（アクセントだけ v1）＝アクセントは抽出率・対照差が低いので v1 に戻す案。
  (B)  取れた行だけ＝取れなかった行は正解から落とす（件数は減るが時刻は全部 v2）。

混合の副作用＝「同一曲・同一技法で、カウンターの番号順に対して時刻が逆行する行」。
v1 は番号順に単調増加するが、v2 は行ごとに動く量が違うため逆転が起きうる。
これに対し 2 通りを作って件数を並べる（採否は後）。
  逆行-除外   ＝ 逆行に関わる行だけ v1 に戻す（単調になるまで繰り返す）。
  逆行-並び替え＝ 時刻の集まりは変えず、同一曲・同一技法の中で昇順に並べ直して番号に割り当てる。

使い方
  python g1_v2_replace_variants.py --session 20261014
"""
import os, argparse
import numpy as np, pandas as pd

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール",
      "vibrato": "ビブラート", "accent": "アクセント", "hammering": "ハンマリング"}
TECH_ORDER = ["accent", "vibrato", "shakuri", "kobushi", "hammering", "fall"]


def load(session):
    p = os.path.join(ROOT, "dataset", f"g1_v2_candidate_{session}.csv")
    df = pd.read_csv(p)
    df = df[df.multi_increment == 0].copy()          # 件数の標準（方式Aの処理対象）
    df["v2_ok"] = df.v2_ok.astype(str).str.lower().isin(["true", "1"])
    for c in ("video_time_sec", "v2_time_sec", "v2_resid_sec", "count_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, p


def count_inversions(df, timecol):
    """同一曲・同一技法で、番号順に対して時刻が逆行する行の数。"""
    n, detail = 0, []
    for (v, c), g in df.groupby(["video", "counter"]):
        g = g.sort_values("count_value")
        t = g[timecol].values
        bad = int((np.diff(t) < 0).sum())
        if bad:
            n += bad
            detail.append((v, c, bad))
    return n, detail


def fix_exclude(df, timecol="mix"):
    """逆行に関わる行を v1 に戻す（単調になるまで繰り返す）。戻した行数を返す。"""
    out = df.copy()
    reverted = 0
    for _ in range(50):
        changed = 0
        for (v, c), g in out.groupby(["video", "counter"]):
            g = g.sort_values("count_value")
            idx = g.index.values
            t = g[timecol].values
            for i in range(1, len(t)):
                if t[i] < t[i - 1]:
                    # 置換した行のうち後ろ側を v1 に戻す（v1 なら前側を戻す）
                    j = idx[i] if out.at[idx[i], "v2_ok_eff"] else idx[i - 1]
                    if out.at[j, "v2_ok_eff"]:
                        out.at[j, timecol] = out.at[j, "video_time_sec"]
                        out.at[j, "v2_ok_eff"] = False
                        reverted += 1
                        changed += 1
                        break
        if not changed:
            break
    return out, reverted


def fix_sort(df, timecol="mix"):
    """時刻の集まりは変えず、同一曲・同一技法の中で昇順に並べ直す。動いた行数を返す。"""
    out = df.copy()
    moved = 0
    for (v, c), g in out.groupby(["video", "counter"]):
        g = g.sort_values("count_value")
        idx = g.index.values
        t = np.sort(g[timecol].values)
        before = g[timecol].values
        moved += int((before != t).sum())
        out.loc[idx, timecol] = t
    return out, moved


def shift_stats(df, timecol="mix"):
    """v1 からの動き（置換された行だけ）。"""
    s = df.loc[df.v2_ok_eff, timecol].values - df.loc[df.v2_ok_eff, "video_time_sec"].values
    if not len(s):
        return {}
    return dict(置換行=len(s), 中央値=round(float(np.median(s)), 3), 平均=round(float(s.mean()), 3),
                p10=round(float(np.percentile(s, 10)), 3), p90=round(float(np.percentile(s, 90)), 3),
                最小=round(float(s.min()), 3), 最大=round(float(s.max()), 3),
                早い側=int((s < 0).sum()), 超1秒=int((np.abs(s) > 1).sum()),
                超2秒=int((np.abs(s) > 2).sum()))


def variant(df, name, keep_v1_tech=()):
    """案を 1 つ組む。keep_v1_tech の技法は v1 に戻す。"""
    out = df.copy()
    out["v2_ok_eff"] = out.v2_ok & ~out.counter.isin(keep_v1_tech)
    out["mix"] = np.where(out.v2_ok_eff, out.v2_time_sec, out.video_time_sec)
    inv, det = count_inversions(out, "mix")
    row = dict(案=name, 母数=len(out))
    row.update(shift_stats(out))
    row["逆行"] = inv
    ex, rev = fix_exclude(out)
    inv_ex, _ = count_inversions(ex, "mix")
    so, mv = fix_sort(out)
    inv_so, _ = count_inversions(so, "mix")
    row.update(逆行除外_戻した行=rev, 逆行除外_残り逆行=inv_ex,
               逆行並び替え_動いた行=mv, 逆行並び替え_残り逆行=inv_so)
    return row, det, out, ex, so


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="20261014")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    df, src = load(a.session)
    outdir = a.outdir or os.path.join(ROOT, "marks_20260928", f"v2_{a.session}")
    os.makedirs(outdir, exist_ok=True)
    print(f"入力 {src}（multi_increment=0 の {len(df)} 行・v2 で取れた {int(df.v2_ok.sum())} 行）")
    print()

    rows, dets = [], {}
    for name, keep in [("(A) 混合", ()), ("(A') 混合・アクセントは v1", ("accent",))]:
        r, det, mix, ex, so = variant(df, name, keep)
        rows.append(r); dets[name] = det
        tag = "A" if not keep else "Aprime"
        mix[["video", "counter", "count_value", "video_time_sec", "v2_time_sec",
             "v2_ok_eff", "mix"]].to_csv(os.path.join(outdir, f"v2_mix_{tag}_{a.session}.csv"),
                                         index=False, encoding="utf-8-sig")
        ex[["video", "counter", "count_value", "video_time_sec", "v2_ok_eff", "mix"]].to_csv(
            os.path.join(outdir, f"v2_mix_{tag}_inv_exclude_{a.session}.csv"), index=False, encoding="utf-8-sig")
        so[["video", "counter", "count_value", "video_time_sec", "v2_ok_eff", "mix"]].to_csv(
            os.path.join(outdir, f"v2_mix_{tag}_inv_sort_{a.session}.csv"), index=False, encoding="utf-8-sig")

    # (B) 取れた行だけ
    b = df[df.v2_ok].copy()
    b["v2_ok_eff"] = True; b["mix"] = b.v2_time_sec
    rb = dict(案="(B) 取れた行だけ", 母数=len(b))
    rb.update(shift_stats(b))
    inv_b, det_b = count_inversions(b, "mix")
    rb["逆行"] = inv_b
    rb.update(逆行除外_戻した行=0, 逆行除外_残り逆行=inv_b, 逆行並び替え_動いた行=0, 逆行並び替え_残り逆行=inv_b)
    rows.append(rb); dets["(B) 取れた行だけ"] = det_b
    b[["video", "counter", "count_value", "video_time_sec", "mix"]].to_csv(
        os.path.join(outdir, f"v2_only_B_{a.session}.csv"), index=False, encoding="utf-8-sig")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(outdir, f"v2_variants_{a.session}.csv"), index=False, encoding="utf-8-sig")
    print("=== 案ごとの件数と動き ===")
    print(tbl.to_string(index=False))
    print()

    print("=== 逆行の内訳（曲・技法・件数） ===")
    for k, det in dets.items():
        print(f"[{k}] 合計 {sum(x[2] for x in det)}件")
        for v, c, n in sorted(det, key=lambda x: -x[2]):
            print(f"   {JP.get(c, c)} {n}件  {v}")
    print()

    # (B) の精度は g1_v2_from_marks.py の全体行と同じ値になる（取れた行だけの残差）
    r = df.loc[df.v2_ok, "v2_resid_sec"].dropna()
    print(f"(B) の精度＝残差中央値 {r.median():.3f} 秒・±0.5秒 {(r <= .5).mean():.3f}・±1.0秒 {(r <= 1.0).mean():.3f}・件数 {len(r)}")
    rb2 = df.loc[df.v2_ok & (df.counter != 'accent'), "v2_resid_sec"].dropna()
    print(f"(B) アクセント除く＝残差中央値 {rb2.median():.3f} 秒・±0.5秒 {(rb2 <= .5).mean():.3f}・件数 {len(rb2)}")
