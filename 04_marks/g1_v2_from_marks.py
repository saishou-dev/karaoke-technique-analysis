# -*- coding: utf-8 -*-
"""正解時刻 v2（技法加点表示の出現時刻）を作り、精度を表にする。

やること（復号なし・既存スクリプトも動かさない）
  1. `marks_<session>.csv`（detect → rematch → refine まで済み）を読む。
  2. 精緻化の後段の歯止め（自己相関・種類相関・窓の先頭）を **指定の閾値で** 当て直す。
     既定＝材料39 の 6段目の推奨（自己相関 0.40／種類相関 0.20-0.30／窓先頭は現行）。
  3. 曲ごと・技法ごとに「抽出率・残差（中央値／90%点／最大）・±0.5秒／±1秒の割合・対照（偶然一致率）」を出す。
  4. 正解時刻 v2 の候補 CSV を作る（G1 の各行に v2 時刻・取れたか・残差を付ける。正本 G1 は変更しない）。
  5. 検出器評価用に、イベント時刻を v2 に置き換えた index の写しを書く（任意・--write_index）。

件数の数え方（材料41 で 1 つに決めた標準）
  **multi_increment=1（同一コマで複数のカウンターが増えた行）を除いた G1 行数**を母数にする。
  方式A はこの行を処理対象から外す作りなので、方式A の抽出率の分母と一致する。

残差の定義（材料39 の 5-4・6段目と同じ）
  残差 = | 表示先行時間 − その技法の表示先行時間の中央値 |
  ＝技法ごとの遅れの中央値で補正したあとに残るずれ。

対照（偶然一致率）の定義
  方式A は加点時刻 T の周り（窓 [T − lmax, T + 0.5]・lmax はビブラート 3.0 秒・他 2.2 秒）だけを探す。
  そこで「窓の中から一様乱数で時刻を 1 つ選んだ場合」に同じ手続き（技法ごとの中央値で補正 → 残差）を通した値を
  対照とする。--control_iter 回の平均。窓幅 W なら ±0.5 秒の割合はおよそ 1.0/W になる。

使い方
  python g1_v2_from_marks.py --session 20261014
  python g1_v2_from_marks.py --session 20260924 --write_index
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_marks as d

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
G1_CSV = os.path.join(ROOT, "dataset", "G1_events_all.csv")
INDEX_JSON = os.path.join(ROOT, "g3v2_20260928", "index.json")
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール",
      "vibrato": "ビブラート", "accent": "アクセント", "hammering": "ハンマリング"}
TECH_ORDER = ["accent", "vibrato", "shakuri", "kobushi", "hammering", "fall"]


# ---------------- 歯止め ----------------
def apply_gates(df, self_min, cls_green, cls_hammer, start_mode):
    """`refine_onsets.py` の merge() と同じ条件を、指定の閾値で当て直す。"""
    ref = df.mark_time_refined.notna()
    bad_self = ref & (df.ncc_at_v4 < self_min)
    bad_cls = ref & (((df.counter == "vibrato") & (df.cls_ncc < cls_green)) |
                     ((df.counter == "hammering") & (df.cls_ncc < cls_hammer)))
    if start_mode == "none":
        bad_start = ref & False
    elif start_mode == "all":
        bad_start = ref & (df.hit_start == 1)
    else:                                    # except_vibrato ＝ 現行
        bad_start = ref & (df.hit_start == 1) & (df.counter != "vibrato")
    bad = bad_self | bad_cls | bad_start
    out = df.copy()
    out["ok"] = ref & ~bad
    out["v2_time_sec"] = np.where(out.ok, out.mark_time_refined, np.nan)
    out["lead"] = np.where(out.ok, out.video_time_sec - out.mark_time_refined, np.nan)
    return out, int(bad_self.sum()), int(bad_cls.sum()), int(bad_start.sum())


def load_marks(session, gates):
    """marks_<session>.csv を読み、歯止めを当て直した表を返す。"""
    d.configure(session)
    df = pd.read_csv(d.MARKS_CSV)
    for c in ("video_time_sec", "mark_time_sec", "mark_time_raw", "mark_time_refined",
              "ncc_at_v4", "cls_ncc", "hit_start", "mark_x", "mark_y", "count_value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "mark_time_refined" not in df.columns:
        sys.exit(f"精緻化の列が無い（refine_onsets.py を通していない）: {d.MARKS_CSV}")
    return apply_gates(df, *gates)


# ---------------- 残差と対照 ----------------
def residual(out):
    """技法ごとの表示先行時間の中央値で補正したあとの残差（絶対値）。取れた行だけ値が入る。"""
    med = out[out.ok].groupby("counter").lead.median()
    r = (out.lead - out.counter.map(med)).abs()
    return r.where(out.ok), med


def control_rates(out, n_iter=200, seed=20261023):
    """偶然一致率。窓の中から一様乱数で時刻を選び、同じ補正と残差を通す。"""
    sub = out[out.ok]
    if not len(sub):
        return dict(対照05=np.nan, 対照10=np.nan, 対照残差中央値=np.nan)
    lmax = sub.counter.map({k: v[3] for k, v in d.FAMILY.items()}).values.astype(float)
    tech = sub.counter.values
    rng = np.random.default_rng(seed)
    f05, f10, medr = [], [], []
    for _ in range(n_iter):
        lead = rng.uniform(-0.5, lmax)                  # T − 一様乱数の時刻
        s = pd.Series(lead, index=tech)
        med = s.groupby(level=0).median()
        res = (s - s.index.map(med)).abs().values
        f05.append(float((res <= 0.5).mean())); f10.append(float((res <= 1.0).mean()))
        medr.append(float(np.median(res)))
    return dict(対照05=round(float(np.mean(f05)), 3), 対照10=round(float(np.mean(f10)), 3),
                対照残差中央値=round(float(np.mean(medr)), 3))


def summarize(out, n_proc, label, n_iter=200):
    """1 つの集まり（全体・曲・技法）の 1 行分をつくる。"""
    res, _ = residual(out)
    r = res.dropna()
    ctl = control_rates(out, n_iter)
    n_ok = int(out.ok.sum())
    row = dict(区分=label, 処理=int(n_proc), 抽出=n_ok,
               抽出率=round(n_ok / n_proc, 3) if n_proc else np.nan)
    if len(r):
        row.update(残差中央値=round(float(r.median()), 3),
                   残差p90=round(float(r.quantile(.9)), 3),
                   残差最大=round(float(r.max()), 3),
                   残差05以内=round(float((r <= 0.5).mean()), 3),
                   残差10以内=round(float((r <= 1.0).mean()), 3))
    else:
        row.update(残差中央値=np.nan, 残差p90=np.nan, 残差最大=np.nan,
                   残差05以内=np.nan, 残差10以内=np.nan)
    row.update(ctl)
    row["差05"] = round(row["残差05以内"] - row["対照05"], 3) if row["残差05以内"] == row["残差05以内"] else np.nan
    row["差10"] = round(row["残差10以内"] - row["対照10"], 3) if row["残差10以内"] == row["残差10以内"] else np.nan
    row["有効件数"] = int(round(n_ok * row["残差05以内"])) if row["残差05以内"] == row["残差05以内"] else 0
    row["先行中央値"] = round(float(out[out.ok].lead.median()), 3) if n_ok else np.nan
    return row


# ---------------- 正解 v2 の候補 ----------------
def build_candidate(session, out):
    """G1 の各行に v2 時刻・取れたか・残差を付けた CSV（正本 G1 は変更しない）。"""
    g1 = pd.read_csv(G1_CSV)
    g1 = g1[g1.session == int(session)].copy()
    res, med = residual(out)
    o = out.assign(残差=res)[["video", "counter", "count_value", "ok", "v2_time_sec",
                              "lead", "残差", "ncc_at_v4", "cls_ncc", "hit_start", "note"]]
    o = o.rename(columns={"ok": "v2_ok", "lead": "v2_lead_sec", "残差": "v2_resid_sec",
                          "note": "v2_note"})
    m = g1.merge(o, on=["video", "counter", "count_value"], how="left")
    m["v2_ok"] = m.v2_ok.fillna(False).astype(bool)
    m["v2_対象外理由"] = np.where(m.multi_increment == 1, "同一コマ複数増分（方式Aの処理対象外）",
                            np.where(m.v2_time_sec.notna(), "", "方式Aで時刻が取れず"))
    # 正解 v2 として使う時刻（取れた行は v2・取れない行は v1 のまま）
    m["v2_video_time_sec"] = np.where(m.v2_ok, m.v2_time_sec, m.video_time_sec)
    if "audio_time_sec" in m.columns:
        m["v2_audio_time_sec"] = np.where(m.v2_ok, m.audio_time_sec - m.v2_lead_sec, m.audio_time_sec)
    m["技法補正中央値"] = m.counter.map(med)
    return m


def write_index_v2(session, cand, dst):
    """検出器評価用：index.json のイベント時刻を v2 に置き換えた写しを書く（指定セッションだけ）。"""
    idx = json.load(open(INDEX_JSON, encoding="utf-8"))
    # index.json のイベントは multi_increment=0 の行だけ（9/24 は 1,194 件・10/14 は 1,253 件）。
    # 件数を v1 と完全に揃えるため、ここでも同じ行だけを使う（揃えないと F 値の差に件数の差が混ざる）。
    rep = {}
    for v, g in cand[cand.multi_increment == 0].groupby("video"):
        rep[v] = [(str(r.counter), float(r.v2_audio_time_sec)) for r in g.itertuples()]
    n_song = 0
    for e in idx:
        if e["video"] in rep:
            ev = sorted(rep[e["video"]], key=lambda x: x[1])
            e["events"] = [list(x) for x in ev]
            e["n"] = len(ev)
            # span（採点区間）は v1 のまま据え置く。v2 は時刻が 0.8 秒ほど前へ動くので
            # span を作り直すと予測の切り取り範囲まで変わり、F 値の差に混ざる。
            n_song += 1
    json.dump(idx, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    return n_song, dst


# ---------------- 本体 ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="20261014")
    ap.add_argument("--self_min", type=float, default=0.40, help="自己相関の歯止め（推奨 0.40・現行 0.50）")
    ap.add_argument("--cls_green", type=float, default=0.20, help="ビブラートの種類相関（推奨 0.20・現行 0.30）")
    ap.add_argument("--cls_hammer", type=float, default=0.30, help="ハンマリングの種類相関（推奨 0.30・現行 0.40）")
    ap.add_argument("--start_mode", default="except_vibrato", choices=["none", "except_vibrato", "all"])
    ap.add_argument("--control_iter", type=int, default=200)
    ap.add_argument("--subset", nargs="*", default=None, help="この曲だけに絞った行も出す（材料39 の 3 曲の再現確認用）")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--write_index", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    gates = (a.self_min, a.cls_green, a.cls_hammer, a.start_mode)
    out, b_self, b_cls, b_start = load_marks(a.session, gates)
    outdir = a.outdir or os.path.join(ROOT, "marks_20260928", f"v2_{a.session}")
    os.makedirs(outdir, exist_ok=True)

    g1 = pd.read_csv(G1_CSV)
    g1s = g1[g1.session == int(a.session)]
    n_g1 = len(g1s); n_g1_single = int((g1s.multi_increment == 0).sum())
    print(f"=== セッション {a.session} ===")
    print(f"G1 行数 {n_g1}／multi_increment=0 の行数 {n_g1_single}（＝件数の標準）／方式A が処理した行 {len(out)}")
    print(f"歯止め 自己相関{a.self_min} 種類相関{a.cls_green}/{a.cls_hammer} 窓先頭{a.start_mode}"
          f" → 無効化 {b_self + b_cls + b_start}件（自己{b_self}・種類{b_cls}・先頭{b_start}）")
    print()

    rows = [summarize(out, len(out), "全体", a.control_iter)]
    for t in TECH_ORDER:
        s = out[out.counter == t]
        if len(s):
            rows.append(summarize(s, len(s), "技法:" + JP[t], a.control_iter))
    tech_tbl = pd.DataFrame(rows)
    tech_tbl.to_csv(os.path.join(outdir, "v2_by_counter.csv"), index=False, encoding="utf-8-sig")
    print("=== 全体と技法ごと ===")
    print(tech_tbl.to_string(index=False))
    print()

    song_rows = []
    for v, s in out.groupby("video"):
        song_rows.append(summarize(s, len(s), v, a.control_iter))
    song_tbl = pd.DataFrame(song_rows).sort_values("抽出率", ascending=False)
    song_tbl.to_csv(os.path.join(outdir, "v2_by_song.csv"), index=False, encoding="utf-8-sig")
    print("=== 曲ごと ===")
    print(song_tbl.to_string(index=False))
    print()

    if a.subset:
        sub = out[out.video.isin(a.subset)]
        r = summarize(sub, len(sub), "材料39 の 3 曲", a.control_iter)
        print("=== 材料39（3 曲）の再現確認 ===")
        print(pd.DataFrame([r]).to_string(index=False))
        print()

    cand = build_candidate(a.session, out)
    cp = os.path.join(ROOT, "dataset", f"g1_v2_candidate_{a.session}.csv")
    cand.to_csv(cp, index=False, encoding="utf-8-sig")
    print(f"正解時刻 v2 の候補 → {cp}（{len(cand)}行・v2 で置換 {int(cand.v2_ok.sum())}行）")

    if a.write_index:
        dst = os.path.join(ROOT, "g3v2_20260928", f"index_v2_{a.session}.json")
        n, p = write_index_v2(a.session, cand, dst)
        print(f"検出器評価用の index の写し → {p}（{n}曲のイベント時刻を v2 に置換）")
