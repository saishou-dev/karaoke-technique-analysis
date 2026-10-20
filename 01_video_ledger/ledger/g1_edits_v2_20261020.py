# -*- coding: utf-8 -*-
"""台帳（セッション G1 ＋ 統合 G1_events_all）へ **宣言的な編集リスト**を冪等に当てる道具。

版＝v2（2026-10-20 午前の報告 ③誤り1 への対処＝本人○）。v1 は g1_edits_20261020.py（legacy/ へ）。

  op=add      欠落イベントの追記（時刻＝帯 pts 表の pts_sec[event_frame]）
  op=relabel  読み過ぎ（read > csv）で値がずれた行の付け替え（例: こぶし 11 回目＝実際は 10 回目）
  op=retime   既存行の時刻の打ち直し（9/15 のシーク式の目視追記＝フレームがずれている）
  op=fixaudio 追記行の wav／audio_time_sec を同じ video の既存行から引き直す
  op=rechain  （v2 で追加）delta 連鎖だけを「先頭行＝count_value −（録画開始前の加点）」で組み直す

編集リスト（CSV・utf-8-sig）の列:
  op, key, video, counter, value, new_value, event_frame, why
    key        ... rescan_missing_20261015.py の SESS キー（20260924 / 20260915r / ...）
    value      ... 対象の count_value（add のときは追記する値・rechain のときは空でよい）
    new_value  ... relabel のときの新しい count_value
    event_frame... add / retime のときの増加フレーム（＝旧値が最後に読めたフレーム + 1）
    why        ... 根拠（目視した画像のファイル名を必ず書く）

規則
  * 時刻は帯 pts 表 dataset/<pts>/<video>.csv の pts_sec[event_frame]（既存行と同じ基準）。
  * audio_time_sec と wav は **同じ video の既存行**から引く（wav＝最頻値・オフセット＝中央値）。
  * delta 連鎖は、触った (video, counter)／(session, video, take, counter) だけ組み直す。
  * 冪等＝既に当たっている編集は skip する。2 回流しても行は増えないし値も動かない。

----------------------------------------------------------------------------------------
v2 の変更（2026-10-20・④-1 本人○）＝**新しく先頭になる追記行の delta**
----------------------------------------------------------------------------------------
v1 までの rebuild_delta は「先頭行の delta は読取器の初期値を含むのでそのまま」という規則だった。
これは**既存の先頭行**（読取器が書いた初期値入り）には正しいが、**その技法の 1 件目（count_value=1）を
新規に追記した行**には当てはまらない（追記時に delta=0 で初期化し、そのまま先頭として据え置かれる）。
実測＝統合台帳の「各 (曲, テイク, 技法) の最初の行」756 行はすべて delta=1 で、delta=0 は 10/20 に
追記した 2 行（きらり 10/14 fall 1・かくれんぼ 9/11 fall 1）だけ＝規約から外れていた。

v2 の規則：
  **追記した行が、その (video, counter)（統合台帳では take も含む）の最小 count_value になるときは、
    delta = count_value −（song_valid_range.csv の録画開始前の加点 pre_record_counts[counter]）で書く。**
  （録画開始前の加点が 0 の曲なら delta = count_value。9/11 かくれんぼ・10/14 きらりはどちらも fall=0 → delta=1）
  既存の先頭行しかないとき（追記していないとき）の挙動は v1 と同じ＝そのまま。

既に書いてしまった行を直すために op=rechain を足した（delta 連鎖だけを上の規則で組み直す・冪等）。
評価側（build_ledgers_20261008.py・build_index.py）は delta を使わないので F値 への影響は 0。
効くのは multi_increment（delta>1 の印）の判定だけ。

----------------------------------------------------------------------------------------
v1（g1_edits_20261020.py）からの引き継ぎ
  * 退避名の日付と note のタグは実行日（datetime.date.today()）から取る。--note-tag=YYYYMMDD で明示指定可。
  * 実行日は標準出力の先頭 `[実行日] DATETAG=…` に出る。

使い方: python g1_edits_v2_20261020.py <edits.csv> --tag=<退避名> [--note-tag=<日付>] [--dry]
        退避＝<台帳>.pre_<tag>_<実行日YYYYMMDD>.bak（初回のみ・既にあれば作り直さない）
"""
import os, sys, csv, shutil, hashlib, collections, datetime
import numpy as np, pandas as pd

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
ALL = os.path.join(DS, "G1_events_all.csv")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import SESS  # noqa: E402
from song_valid_range_20261015 import load as svr_load  # noqa: E402  （録画開始前の加点）

# 統合台帳（G1_events_all.csv）に**まだ結合されていない**セッションは、セッション台帳だけを編集し、
# 統合台帳には1行も触らない（結合は merge_g1_all_*.py の役目）。
FORBIDDEN_SESSIONS = set()

# 退避名と note に入れる日付＝**実行日**（スクリプト名の日付で固定しない）。
DATETAG = datetime.date.today().strftime("%Y%m%d")


def rowhash(df):
    s = df.astype(str).apply(lambda r: "\x1f".join(map(str, r.tolist())), axis=1)
    return list(s) if len(s) else []


def interp(xs, ys, x):
    o = np.argsort(xs); xs = np.asarray(xs, float)[o]; ys = np.asarray(ys, float)[o]
    if len(xs) == 1 or x <= xs[0]:
        return float(ys[0] + (x - xs[0]))
    if x >= xs[-1]:
        return float(ys[-1] + (x - xs[-1]))
    return float(np.interp(x, xs, ys))


def pts_of(cfg, video):
    p = os.path.join(DS, cfg["pts"], video + ".csv")
    return [float(r["pts_sec"]) for r in csv.DictReader(open(p, encoding="utf-8"))]


def pre_record(video, counter, log=None):
    """song_valid_range.csv の「録画開始前の加点」（pre_record_counts）から技法 1 つ分を引く。
       表に無い曲は 0 として扱い、警告を残す（表は dataset/song_valid_range.csv が正本）。"""
    r = svr_load().get(video)
    if not r:
        if log is not None:
            log.append(f"  注意: song_valid_range.csv に {video} が無い＝録画開始前の加点を 0 とみなす")
        return 0
    for kv in (r.get("pre_record_counts") or "").split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k == counter:
                return int(v)
    return 0


def rebuild_delta(df, mask, log, label, base=None):
    """count_value 昇順に delta を組み直す。
       base=None  ... 先頭行の delta はそのまま（＝読取器の初期値を保つ・v1 と同じ）
       base=<int> ... 先頭行が**新しく先頭になった追記行**のとき。delta = count_value − base
                      （base＝録画開始前の加点＝song_valid_range.csv）。v2 で追加。
       戻り値＝delta を書き換えた行数。"""
    sub = df.loc[mask].sort_values("count_value")
    if not len(sub):
        return 0
    idx = sub.index.tolist(); vals = sub["count_value"].astype(int).tolist()
    prev = int(base) if base is not None else vals[0] - int(sub["delta"].iloc[0])
    n = 0
    for i, ix in enumerate(idx):
        d = vals[i] - prev
        if int(df.at[ix, "delta"]) != d:
            log.append(f"  delta 連鎖: {label} {vals[i]} 回目 delta {int(df.at[ix,'delta'])} -> {d}")
            n += 1
        df.at[ix, "delta"] = d
        df.at[ix, "multi_increment"] = int(d > 1)
        prev = vals[i]
    return n


def audio_ref(df, video):
    """同じ video の既存行から (wav の最頻値, audio-video オフセットの中央値) を引く。"""
    e = df[df["video"] == video]
    wav = ""
    if "wav" in df.columns and len(e):
        w = e["wav"].dropna()
        wav = collections.Counter(w).most_common(1)[0][0] if len(w) else ""
    off = 0.0
    if len(e) and "audio_time_sec" in df.columns:
        off = float(np.median(e["audio_time_sec"].astype(float) - e["video_time_sec"].astype(float)))
    return wav, round(off, 3)


def main(path, tag, dry=False):
    edits = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["op"].strip()]
    keys = sorted({r["key"] for r in edits})
    A0 = pd.read_csv(ALL, encoding="utf-8-sig")
    A = A0.copy()
    acols = list(A.columns)
    log = [f"[実行日] DATETAG={DATETAG}（退避＝.pre_<tag>_{DATETAG}.bak・note＝found_visual_{DATETAG}）",
           "[版] g1_edits_v2_20261020.py（新しく先頭になる追記行の delta＝count_value −録画開始前の加点）"]
    touched_all = set(); base_all = {}
    nchg = 0
    for key in keys:
        cfg = SESS[key]; ses = cfg["session"]
        assert ses not in FORBIDDEN_SESSIONS, ses
        in_all = bool((A["session"] == ses).any())   # False＝未結合（統合台帳は触らない）
        if not in_all:
            log.append(f"[{key}] 統合台帳に session={ses} の行が無い＝セッション台帳だけを編集する（結合は merge_g1_all_*.py の役目）")
        g1p = os.path.join(DS, cfg["g1"])
        G0 = pd.read_csv(g1p, encoding="utf-8-sig")
        G = G0.copy()
        if "note" not in G.columns:
            G["note"] = ""
        touched_g = set(); base_g = {}; added_g = collections.defaultdict(list)
        for e in [x for x in edits if x["key"] == key]:
            v, c = e["video"], e["counter"]
            why = e.get("why", "")
            gm = (G["video"] == v) & (G["counter"] == c)
            am = (A["session"] == ses) & (A["video"] == v) & (A["counter"] == c)
            if e["op"] == "add":
                pts = pts_of(cfg, v)
                val = int(e["value"]); f = int(e["event_frame"]); t = round(pts[f], 3)
                if (G.loc[gm, "count_value"].astype(int) == val).any():
                    log.append(f"skip(既存) add {key} {v} {c} {val}"); continue
                wav, off = audio_ref(G, v)
                r = {k: "" for k in G.columns}
                r.update(video=v, counter=c, count_value=val, delta=0, frame=f, event_frame_est=f,
                         video_time_sec=t, stable_read_time_sec=t, wav=wav,
                         audio_time_sec=round(t + off, 3), multi_increment=0, note=f"found_visual_{DATETAG}")
                if "pts_source" in G.columns:
                    r["pts_source"] = cfg["pts_source"]
                if "audio_source" in G.columns:
                    s = G.loc[G["video"] == v, "audio_source"].dropna()
                    r["audio_source"] = s.iloc[0] if len(s) else ""
                G = pd.concat([G, pd.DataFrame([r])], ignore_index=True)
                touched_g.add((v, c)); added_g[(v, c)].append(val); nchg += 1
                log.append(f"ADD  {key} {v} {c} {val} f={f} t={t:.3f} audio={r['audio_time_sec']}  {why}")
                if not in_all:
                    continue
                # 統合台帳（take ごとに 1 行）
                ex = A[(A["session"] == ses) & (A["video"] == v)]
                assert len(ex), f"統合台帳に {ses} {v} が無い"
                if ((A["session"] == ses) & (A["video"] == v) & (A["counter"] == c)
                        & (A["count_value"].astype(int) == val)).any():
                    log.append(f"  skip(既存・統合) {val}")
                else:
                    new = []
                    for tk in sorted(ex["take"].unique()):
                        q = ex[ex["take"] == tk]
                        # v2: この take の既存の最小 count_value より小さい＝新しく先頭になる行
                        old = A[(A["session"] == ses) & (A["video"] == v) & (A["take"] == tk) & (A["counter"] == c)]
                        oldmin = int(old["count_value"].astype(int).min()) if len(old) else None
                        if oldmin is None or val < oldmin:
                            base_all[(ses, v, tk, c)] = pre_record(v, c, log)
                        new.append(dict(session=ses, video=v, counter=c, count_value=val, delta=0,
                                        video_time_sec=t, stable_read_time_sec=t, take=tk,
                                        audio_time_sec=round(interp(q["video_time_sec"], q["audio_time_sec"], t), 3),
                                        video_time_idx_sec=round(interp(q["video_time_sec"], q["video_time_idx_sec"], t), 3),
                                        stable_read_time_idx_sec=round(interp(q["stable_read_time_sec"], q["stable_read_time_idx_sec"], t), 3),
                                        multi_increment=0))
                        touched_all.add((ses, v, tk, c))
                    A = pd.concat([A, pd.DataFrame(new)[acols]], ignore_index=True)
                    log.append(f"  統合台帳 +{len(new)} 行（take {sorted(ex['take'].unique())}）")
            elif e["op"] == "rechain":
                # v2 追加：delta 連鎖だけを「先頭行＝count_value −（録画開始前の加点）」で組み直す。
                # 既に v1 で追記してしまい delta=0 のままになっている先頭行を直すための op。
                assert gm.any(), f"対象行が無い {key} {v} {c}"
                pre = pre_record(v, c, log)
                base_g[(v, c)] = pre; touched_g.add((v, c))
                log.append(f"RECHAIN {key} {v} {c}  先頭行 delta = count_value - {pre}（録画開始前の加点）  {why}")
                if in_all and am.any():
                    for tk in sorted(A.loc[am, "take"].unique()):
                        base_all[(ses, v, tk, c)] = pre; touched_all.add((ses, v, tk, c))
            elif e["op"] == "relabel":
                val = int(e["value"]); nv = int(e["new_value"])
                m = gm & (G["count_value"].astype(int) == val)
                if not m.any():
                    log.append(f"skip(済) relabel {key} {v} {c} {val}->{nv}"); continue
                G.loc[m, "count_value"] = nv
                G.loc[m, "note"] = (G.loc[m, "note"].fillna("").astype(str) + ("; " if G.loc[m, "note"].iloc[0] else "")
                                    + f"relabel_{DATETAG}: {val}->{nv} {why}")
                touched_g.add((v, c)); nchg += 1
                log.append(f"RELABEL {key} {v} {c} {val} -> {nv}  {why}")
                ma = am & (A["count_value"].astype(int) == val)
                if ma.any():
                    A.loc[ma, "count_value"] = nv
                    for tk in A.loc[ma, "take"].unique():
                        touched_all.add((ses, v, tk, c))
                    log.append(f"  統合台帳 {int(ma.sum())} 行を {val} -> {nv}")
            elif e["op"] == "retime":
                pts = pts_of(cfg, v)
                val = int(e["value"]); f = int(e["event_frame"]); t = round(pts[f], 3)
                m = gm & (G["count_value"].astype(int) == val)
                assert m.any(), f"対象行が無い {key} {v} {c} {val}"
                old_t = float(G.loc[m, "video_time_sec"].iloc[0])
                if int(G.loc[m, "frame"].iloc[0]) == f and abs(old_t - t) < 1e-6:
                    log.append(f"skip(済) retime {key} {v} {c} {val} f={f}"); continue
                dt = round(t - old_t, 3)
                wav, off = audio_ref(G, v)
                G.loc[m, "frame"] = f; G.loc[m, "event_frame_est"] = f
                G.loc[m, "video_time_sec"] = t
                G.loc[m, "stable_read_time_sec"] = t
                G.loc[m, "audio_time_sec"] = round(t + off, 3)
                G.loc[m, "note"] = (G.loc[m, "note"].fillna("").astype(str) + "; "
                                    + f"timefix_{DATETAG}: f{int(G0.loc[m,'frame'].iloc[0])}->f{f} ({dt:+.3f}s) {why}")
                nchg += 1
                log.append(f"RETIME {key} {v} {c} {val}: f{int(G0.loc[m,'frame'].iloc[0])} -> f{f}  t {old_t:.3f} -> {t:.3f} ({dt:+.3f}s)  {why}")
                ma = am & (A["count_value"].astype(int) == val)
                if ma.any():
                    for col in ("video_time_sec", "stable_read_time_sec", "audio_time_sec",
                                "video_time_idx_sec", "stable_read_time_idx_sec"):
                        A.loc[ma, col] = (A.loc[ma, col].astype(float) + dt).round(3)
                    log.append(f"  統合台帳 {int(ma.sum())} 行の時刻を {dt:+.3f}s")
            elif e["op"] == "fixaudio":
                val = int(e["value"])
                m = gm & (G["count_value"].astype(int) == val)
                assert m.any(), f"対象行が無い {key} {v} {c} {val}"
                wav, off = audio_ref(G[G["count_value"].astype(str) != str(val)], v)
                t = float(G.loc[m, "video_time_sec"].iloc[0])
                cur_a = float(G.loc[m, "audio_time_sec"].iloc[0]); cur_w = str(G.loc[m, "wav"].iloc[0])
                if abs(cur_a - (t + off)) < 1e-6 and cur_w == wav:
                    log.append(f"skip(済) fixaudio {key} {v} {c} {val}"); continue
                G.loc[m, "wav"] = wav
                G.loc[m, "audio_time_sec"] = round(t + off, 3)
                nchg += 1
                log.append(f"FIXAUDIO {key} {v} {c} {val}: wav {cur_w} -> {wav} / audio {cur_a:.3f} -> {t+off:.3f} ({off:+.3f}s)")
            else:
                raise SystemExit("未知の op: " + e["op"])
        # ---- セッション台帳の delta 連鎖 ----
        for (v, c) in touched_g:
            base = base_g.get((v, c))
            if base is None and added_g.get((v, c)):
                # v2: 今回の追記行がその (video, counter) の最小 count_value になったか
                mask_now = (G["video"] == v) & (G["counter"] == c)
                newmin = int(G.loc[mask_now, "count_value"].astype(int).min())
                old = G0[(G0["video"] == v) & (G0["counter"] == c)]
                oldmin = int(old["count_value"].astype(int).min()) if len(old) else None
                if newmin in added_g[(v, c)] and (oldmin is None or newmin < oldmin):
                    base = pre_record(v, c, log)
                    log.append(f"  先頭行が新規追記＝delta の基準を録画開始前の加点から取る: {v[:20]} {c} base={base}")
            nchg += rebuild_delta(G, (G["video"] == v) & (G["counter"] == c), log, f"{v[:20]} {c}", base)
        G = G.sort_values(["video", "video_time_sec", "counter"], kind="stable").reset_index(drop=True)
        h0, h1 = set(rowhash(G0)), set(rowhash(G))
        log.append(f"[{key}] {os.path.basename(g1p)}: {len(G0)} -> {len(G)} 行 / 消えた既存行 {len(h0-h1)} / 新しい行 {len(h1-h0)}")
        if not dry:
            bak = g1p + f".pre_{tag}_{DATETAG}.bak"
            if not os.path.exists(bak):
                shutil.copy(g1p, bak); log.append(f"  退避 {os.path.basename(bak)}")
            G.to_csv(g1p, index=False, encoding="utf-8-sig")
    # ---- 統合台帳の delta 連鎖 ----
    for (s, v, tk, c) in touched_all:
        nchg += rebuild_delta(A, (A["session"] == s) & (A["video"] == v) & (A["take"] == tk) & (A["counter"] == c),
                              log, f"[all] {v[:20]} {tk} {c}", base_all.get((s, v, tk, c)))
    A["_s"] = A["session"]
    A = A.sort_values("_s", kind="stable").drop(columns="_s").reset_index(drop=True)
    h0, h1 = set(rowhash(A0)), set(rowhash(A))
    log.append(f"[統合] G1_events_all.csv: {len(A0)} -> {len(A)} 行 / 消えた既存行 {len(h0-h1)} / 新しい行 {len(h1-h0)}")
    n14a = int((A0["session"] == 20261014).sum()); n14b = int((A["session"] == 20261014).sum())
    s0 = set(rowhash(A0[A0["session"] == 20261014])); s1 = set(rowhash(A[A["session"] == 20261014]))
    log.append(f"[検算] 20261014 行 {n14a} -> {n14b} / 変化した行 {len(s0 - s1)}")
    if n14a and 20261014 in {SESS[k]["session"] for k in keys}:
        log.append("[検算] 20261014 は編集対象セッション＝上の ADD/RELABEL/RETIME/RECHAIN の行が変更の全部")
    else:
        assert s0 == s1, "20261014 の行が変わった"
    if not dry:
        bak = ALL + f".pre_{tag}_{DATETAG}.bak"
        if not os.path.exists(bak):
            shutil.copy(ALL, bak); log.append(f"  退避 {os.path.basename(bak)}")
        A.to_csv(ALL, index=False, encoding="utf-8-sig")
    print("\n".join(log))
    print(("dry-run: " if dry else "") + f"編集 {nchg} 件")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = "edits"
    for a in sys.argv[1:]:
        if a.startswith("--tag="):
            tag = a.split("=", 1)[1]
        elif a.startswith("--note-tag="):
            DATETAG = a.split("=", 1)[1]
            assert len(DATETAG) == 8 and DATETAG.isdigit(), "--note-tag= は YYYYMMDD"
    main(args[0], tag, "--dry" in sys.argv)
