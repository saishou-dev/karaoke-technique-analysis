# -*- coding: utf-8 -*-
"""台帳（セッション G1 ＋ 統合 G1_events_all）へ **宣言的な編集リスト**を冪等に当てる道具（2026-10-15）。

append_found_20261015.py は「追記」しかできない。10/15 の裁定で必要になったのは次の 3 種類なので、
3 つとも 1 つの表で書けるようにした（どれも「どの行を・どう変えたか」を md5 で検算して残す）。

  op=add      欠落イベントの追記（時刻＝帯 pts 表の pts_sec[event_frame]）
  op=relabel  読み過ぎ（read > csv）で値がずれた行の付け替え（例: こぶし 11 回目＝実際は 10 回目）
  op=retime   既存行の時刻の打ち直し（9/15 のシーク式の目視追記＝フレームがずれている）

編集リスト（CSV・utf-8-sig）の列:
  op, key, video, counter, value, new_value, event_frame, why
    key        ... rescan_missing_20261015.py の SESS キー（20260924 / 20260915r / ...）
    value      ... 対象の count_value（add のときは追記する値）
    new_value  ... relabel のときの新しい count_value
    event_frame... add / retime のときの増加フレーム（＝旧値が最後に読めたフレーム + 1）
    why        ... 根拠（目視した画像のファイル名を必ず書く）

規則
  * 時刻は帯 pts 表 dataset/<pts>/<video>.csv の pts_sec[event_frame]（既存行と同じ基準）。
  * audio_time_sec と wav は **同じ video の既存行**から引く（wav＝最頻値・オフセット＝中央値）。
    append_found_20261015.py は wav を "<video>.wav"・audio_time_sec を video_time_sec と同値で書いていたが、
    9/24 のように録音が別ファイル（T06_...wav・オフセット +1.61s）のセッションではこれが誤りになる（10/15 実測）。
  * delta 連鎖は、触った (video, counter)／(session, video, take, counter) だけ組み直す（先頭行の delta は
    読取器の初期値を含むのでそのまま＝録画開始前の加点は song_valid_range.csv が持つ）。
  * 冪等＝既に当たっている編集は skip する。2 回流しても行は増えないし値も動かない。
  * 20261014 の行には触らない（本人指示 10/15・別件調査中）。

使い方: python g1_edits_20261015.py <edits.csv> --tag=<退避名> [--dry]
        退避＝<台帳>.pre_<tag>_20261015.bak（初回のみ・既にあれば作り直さない）
"""
import os, sys, csv, shutil, hashlib, collections
import numpy as np, pandas as pd

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
ALL = os.path.join(DS, "G1_events_all.csv")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rescan_missing_20261015 import SESS  # noqa: E402

# 統合台帳（G1_events_all.csv）に**まだ結合されていない**セッションは、セッション台帳だけを編集し、
# 統合台帳には1行も触らない（結合は merge_g1_all_*.py の役目）。
# 2026-10-15 11:50 の変更：以前は 20261014 を一律禁止していたが、v0.8e で作り直した 20261014 の
# セッション台帳へ目視追記が必要になったため「未結合なら統合台帳を触らない」という規則に置き換えた。
# 統合台帳側の 20261014 行が1行も変わらないことは、従来どおり最後の assert（s0 == s1）で担保する。
FORBIDDEN_SESSIONS = set()


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


def rebuild_delta(df, mask, log, label):
    """count_value 昇順に delta を組み直す（先頭行の delta はそのまま＝読取器の初期値を保つ）。"""
    sub = df.loc[mask].sort_values("count_value")
    if not len(sub):
        return
    idx = sub.index.tolist(); vals = sub["count_value"].astype(int).tolist()
    prev = vals[0] - int(sub["delta"].iloc[0])
    for i, ix in enumerate(idx):
        d = vals[i] - prev
        if int(df.at[ix, "delta"]) != d:
            log.append(f"  delta 連鎖: {label} {vals[i]} 回目 delta {int(df.at[ix,'delta'])} -> {d}")
        df.at[ix, "delta"] = d
        df.at[ix, "multi_increment"] = int(d > 1)
        prev = vals[i]


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
    log = []
    touched_all = set()
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
        touched_g = set()
        for e in [x for x in edits if x["key"] == key]:
            v, c = e["video"], e["counter"]
            why = e.get("why", "")
            pts = pts_of(cfg, v)
            gm = (G["video"] == v) & (G["counter"] == c)
            am = (A["session"] == ses) & (A["video"] == v) & (A["counter"] == c)
            if e["op"] == "add":
                val = int(e["value"]); f = int(e["event_frame"]); t = round(pts[f], 3)
                if (G.loc[gm, "count_value"].astype(int) == val).any():
                    log.append(f"skip(既存) add {key} {v} {c} {val}"); continue
                wav, off = audio_ref(G, v)
                r = {k: "" for k in G.columns}
                r.update(video=v, counter=c, count_value=val, delta=0, frame=f, event_frame_est=f,
                         video_time_sec=t, stable_read_time_sec=t, wav=wav,
                         audio_time_sec=round(t + off, 3), multi_increment=0, note="found_visual_20261015")
                if "pts_source" in G.columns:
                    r["pts_source"] = cfg["pts_source"]
                if "audio_source" in G.columns:
                    s = G.loc[G["video"] == v, "audio_source"].dropna()
                    r["audio_source"] = s.iloc[0] if len(s) else ""
                G = pd.concat([G, pd.DataFrame([r])], ignore_index=True)
                touched_g.add((v, c)); nchg += 1
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
                        new.append(dict(session=ses, video=v, counter=c, count_value=val, delta=0,
                                        video_time_sec=t, stable_read_time_sec=t, take=tk,
                                        audio_time_sec=round(interp(q["video_time_sec"], q["audio_time_sec"], t), 3),
                                        video_time_idx_sec=round(interp(q["video_time_sec"], q["video_time_idx_sec"], t), 3),
                                        stable_read_time_idx_sec=round(interp(q["stable_read_time_sec"], q["stable_read_time_idx_sec"], t), 3),
                                        multi_increment=0))
                        touched_all.add((ses, v, tk, c))
                    A = pd.concat([A, pd.DataFrame(new)[acols]], ignore_index=True)
                    log.append(f"  統合台帳 +{len(new)} 行（take {sorted(ex['take'].unique())}）")
            elif e["op"] == "relabel":
                val = int(e["value"]); nv = int(e["new_value"])
                m = gm & (G["count_value"].astype(int) == val)
                if not m.any():
                    log.append(f"skip(済) relabel {key} {v} {c} {val}->{nv}"); continue
                G.loc[m, "count_value"] = nv
                G.loc[m, "note"] = (G.loc[m, "note"].fillna("").astype(str) + ("; " if G.loc[m, "note"].iloc[0] else "")
                                    + f"relabel_20261015: {val}->{nv} {why}")
                touched_g.add((v, c)); nchg += 1
                log.append(f"RELABEL {key} {v} {c} {val} -> {nv}  {why}")
                ma = am & (A["count_value"].astype(int) == val)
                if ma.any():
                    A.loc[ma, "count_value"] = nv
                    for tk in A.loc[ma, "take"].unique():
                        touched_all.add((ses, v, tk, c))
                    log.append(f"  統合台帳 {int(ma.sum())} 行を {val} -> {nv}")
            elif e["op"] == "retime":
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
                                    + f"timefix_20261015: f{int(G0.loc[m,'frame'].iloc[0])}->f{f} ({dt:+.3f}s) {why}")
                nchg += 1
                log.append(f"RETIME {key} {v} {c} {val}: f{int(G0.loc[m,'frame'].iloc[0])} -> f{f}  t {old_t:.3f} -> {t:.3f} ({dt:+.3f}s)  {why}")
                ma = am & (A["count_value"].astype(int) == val)
                if ma.any():
                    for col in ("video_time_sec", "stable_read_time_sec", "audio_time_sec",
                                "video_time_idx_sec", "stable_read_time_idx_sec"):
                        A.loc[ma, col] = (A.loc[ma, col].astype(float) + dt).round(3)
                    log.append(f"  統合台帳 {int(ma.sum())} 行の時刻を {dt:+.3f}s")
            elif e["op"] == "fixaudio":
                # append_found_20261015.py が wav="<video>.wav"・audio_time_sec=video_time_sec で書いた行を、
                # 同じ video の既存行から引いた wav とオフセットで直す（セッション台帳のみ・統合台帳は補間済みで正しい）。
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
        for (v, c) in touched_g:
            rebuild_delta(G, (G["video"] == v) & (G["counter"] == c), log, f"{v[:20]} {c}")
        G = G.sort_values(["video", "video_time_sec", "counter"], kind="stable").reset_index(drop=True)
        h0, h1 = set(rowhash(G0)), set(rowhash(G))
        log.append(f"[{key}] {os.path.basename(g1p)}: {len(G0)} -> {len(G)} 行 / 消えた既存行 {len(h0-h1)} / 新しい行 {len(h1-h0)}")
        if not dry:
            bak = g1p + f".pre_{tag}_20261015.bak"
            if not os.path.exists(bak):
                shutil.copy(g1p, bak); log.append(f"  退避 {os.path.basename(bak)}")
            G.to_csv(g1p, index=False, encoding="utf-8-sig")
    for (s, v, tk, c) in touched_all:
        rebuild_delta(A, (A["session"] == s) & (A["video"] == v) & (A["take"] == tk) & (A["counter"] == c),
                      log, f"[all] {v[:20]} {tk} {c}")
    A["_s"] = A["session"]
    A = A.sort_values("_s", kind="stable").drop(columns="_s").reset_index(drop=True)
    h0, h1 = set(rowhash(A0)), set(rowhash(A))
    log.append(f"[統合] G1_events_all.csv: {len(A0)} -> {len(A)} 行 / 消えた既存行 {len(h0-h1)} / 新しい行 {len(h1-h0)}")
    n14a = int((A0["session"] == 20261014).sum()); n14b = int((A["session"] == 20261014).sum())
    s0 = set(rowhash(A0[A0["session"] == 20261014])); s1 = set(rowhash(A[A["session"] == 20261014]))
    log.append(f"[検算] 20261014 行 {n14a} -> {n14b} / 変化した行 {len(s0 - s1)}（0 であること）")
    if n14a and 20261014 in {SESS[k]["session"] for k in keys}:
        # 結合後に 20261014 自身を編集する場合だけは、変更行を列挙して通す（それ以外は1行も変えない）
        log.append("[検算] 20261014 は編集対象セッション＝上の ADD/RELABEL/RETIME の行が変更の全部")
    else:
        assert s0 == s1, "20261014 の行が変わった"
    if not dry:
        bak = ALL + f".pre_{tag}_20261015.bak"
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
    main(args[0], tag, "--dry" in sys.argv)
