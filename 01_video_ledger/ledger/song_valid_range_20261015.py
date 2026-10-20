# -*- coding: utf-8 -*-
"""曲ごとの「観測できた区間」＝帯動画が映っていた範囲と、録画開始前の加点数の表を作る／読む（2026-10-15）。

なぜ必要か＝採点機のカウンターは**曲の頭から**数えているが、こちらの録画（帯動画）は曲の途中から始まることがある。
その場合 G1 台帳には「録画が始まる前の加点」がイベントとして入らない（最終値は合うので従来の突合では見えない）。
10/15 実測＝9/15 5 曲 25 件・9/11 2 曲 16 件・10/6(WAO) 2 曲 2 件。
検出器の評価でこの区間を正解なしのまま使うと、その区間の予測がすべて誤検出（FP）として数えられてしまうので、
評価は**観測できた区間だけ**で行う（既定）。切替は各評価スクリプトの --valid_range オプション。

表 dataset/song_valid_range.csv の列
  session, machine, video, first_frame_pts, last_frame_pts, pre_record_counts, note
  （＋ pre_record_total, first_read_pts, n_frames, src）
  * first_frame_pts / last_frame_pts ... 帯動画 pts 表（dataset/<pts dir>/<video>.csv）の先頭・末尾の pts_sec。
    帯動画の時刻＝G1 台帳の video_time_sec と同じ基準。録音WAV時刻へは + offset（sync_offsets）で変換する。
  * pre_record_counts ... 読取器が**最初に読めたフレーム**での値（技法別・"shakuri=2;vibrato=1" 形式）。
    これが 0 より大きい＝その数だけ録画開始前（正確には最初に読めたフレームより前）に加点されている。
  * first_read_pts ... その最初に読めたフレームの pts_sec（帯の演出で数秒読めないことがあるため別に持つ）。
G1 の列は変えない（台帳の正本は G1 のまま・この表は別表）。

使い方
  python song_valid_range_20261015.py            # 表を作り直す（dataset/song_valid_range.csv）
  python song_valid_range_20261015.py --check     # 作らずに内訳だけ出す
  from song_valid_range_20261015 import load, clip_span   # 評価スクリプトから使う
"""
import os, sys, csv, collections

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
DS = os.path.join(ROOT, "dataset")
OUT = os.path.join(DS, "song_valid_range.csv")
ALL = os.path.join(DS, "G1_events_all.csv")

WAO_COUNTERS = ["shakuri", "kobushi", "fall", "vibrato", "accent", "hammering"]
AI_COUNTERS = ["shakuri", "kobushi", "fall", "vibrato"]

# セッション → 読取結果ディレクトリ（先に見つかった方）・帯 pts 表ディレクトリ
SRC = {
    "20260628": (["results"], ["pts"]),
    "20260807": (["results"], ["pts"]),
    "20260812": (["results_wao_20260812", "results"], ["pts"]),
    "20260818": (["results_wao_20260818", "results"], ["pts"]),
    "20260902": (["results_wao_20260902", "results"], ["pts"]),
    "20260911": (["results_wao_20260911", "results_wao_20260911_v2"], ["pts_bands_20260911", "pts"]),
    "20260915": (["results_wao_20260915_rescan", "results_wao_20260915"], ["pts_bands_20260915", "pts"]),
    "20260924": (["results_wao_20260924"], ["pts_bands_20260924", "pts"]),
    "20261006": (["results_wao_20261006", "results_ai_20261006"], ["pts_bands_20261006", "pts_bands_ai_20261006"]),
    # 2026-10-20 追加＝10/14（WAO 30 曲）。10/15 に G1_events_all へ結合済みだがこの表に入っておらず、
    # 「セッション定義なし」で 30 曲まるごと落ちていた（表 207 曲）。帯が曲の途中で消える曲（Habit・共犯_123700）が
    # あるので、入れないと評価の区間が最後まで伸びたままになる。読取結果は v0.8e の正式結果を見る。
    "20261014": (["results_wao_20261014"], ["pts_bands_20261014", "pts"]),
}


def _find(dirs, rel):
    for d in dirs:
        p = os.path.join(ROOT, d, rel)
        if os.path.exists(p):
            return d, p
    return None, None


def _find_by_stamp(dirs, video, suffix):
    """8/07 のように台帳の video 名（VID_20260807_124936）と帯 pts 表の名前（Answer_20260807_124936.csv）が
       違うセッションのための後付け照合。末尾の _YYYYMMDD_HHMMSS が一致する 1 件だけ採用する。"""
    st = video[-16:]
    if not (len(st) == 16 and st[0] == "_"):
        return None, None, ""
    for d in dirs:
        dd = os.path.join(ROOT, d)
        if not os.path.isdir(dd):
            continue
        def base(n):
            return n[:-len(suffix)] if suffix else n
        hit = [n for n in os.listdir(dd) if n.endswith(suffix) and base(n).endswith(st)]
        if len(hit) == 1:
            return d, os.path.join(dd, hit[0]), base(hit[0])
    return None, None, ""


def build(check=False):
    vids = collections.OrderedDict()
    for r in csv.DictReader(open(ALL, encoding="utf-8-sig")):
        vids.setdefault((r["session"], r["video"]), 0)
        vids[(r["session"], r["video"])] += 1
    rows = []; warn = []
    for (ses, v) in vids:
        if ses not in SRC:
            warn.append(f"セッション定義なし: {ses} {v}"); continue
        rdirs, pdirs = SRC[ses]
        rd, rp = _find(rdirs, os.path.join(v, "counter_events.csv"))
        pd_, pp = _find([os.path.join("dataset", x) for x in pdirs], v + ".csv")
        note = []
        if pp is None:
            pd_, pp, alt = _find_by_stamp([os.path.join("dataset", x) for x in pdirs], v, ".csv")
            if pp:
                note.append(f"帯pts表は曲名ファイル {alt}（時刻一致で対応付け）")
        if rp is None:
            rd2, rp2, alt2 = _find_by_stamp(rdirs, v, "")
            if rp2 and os.path.exists(os.path.join(rp2, "counter_events.csv")):
                rd, rp = rd2, os.path.join(rp2, "counter_events.csv")
                note.append(f"読取結果は曲名フォルダ {alt2}")
        if pp is None:
            warn.append(f"帯 pts 表なし: {ses} {v}"); note.append("pts表なし")
            first = last = nfr = ""
        else:
            P = [float(x["pts_sec"]) for x in csv.DictReader(open(pp, encoding="utf-8"))]
            first, last, nfr = round(P[0], 3), round(P[-1], 3), len(P)
        machine = "ai" if (rd or "").startswith("results_ai") else "wao"
        counters = AI_COUNTERS if machine == "ai" else WAO_COUNTERS
        pre = {}; fread = ""
        if rp is None:
            warn.append(f"読取結果なし: {ses} {v}"); note.append("読取結果なし")
        else:
            seen = {}
            for e in csv.DictReader(open(rp, encoding="utf-8")):
                seen.setdefault(e["counter"], e)
            for c in counters:
                if c in seen:
                    pre[c] = int(seen[c]["value"])
            fr = [int(e["frame"]) for e in seen.values()]
            if fr and pp is not None:
                i = min(fr)
                fread = round(P[i], 3) if i < len(P) else ""
        tot = sum(pre.values())
        if tot:
            note.append(f"録画開始前の加点 {tot} 件")
        rows.append(dict(session=ses, machine=machine, video=v,
                         first_frame_pts=first, last_frame_pts=last,
                         pre_record_counts=";".join(f"{c}={pre[c]}" for c in counters if c in pre),
                         note="／".join(note), pre_record_total=tot, first_read_pts=fread,
                         n_frames=nfr, src=(rd or "") + "|" + (pd_ or "")))
    rows.sort(key=lambda r: (r["session"], r["video"]))
    print(f"曲 {len(rows)}（{len(set(r['session'] for r in rows))} セッション）")
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["pre_record_total"]:
            agg[(r["session"], r["machine"])][0] += 1
            agg[(r["session"], r["machine"])][1] += r["pre_record_total"]
    print("録画開始前の加点があるセッション（曲数・件数）")
    for k in sorted(agg):
        print(f"  {k[0]} {k[1]:4s}  {agg[k][0]} 曲 {agg[k][1]} 件")
    for w in warn:
        print("  注意:", w)
    if check:
        return rows
    cols = ["session", "machine", "video", "first_frame_pts", "last_frame_pts", "pre_record_counts", "note",
            "pre_record_total", "first_read_pts", "n_frames", "src"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("WROTE", OUT, len(rows))
    return rows


# ---------------- 評価スクリプトから使う API ----------------
_CACHE = None


def load(path=OUT):
    """{video: {...}} を返す。表が無ければ空＝素通し。"""
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        if os.path.exists(path):
            for r in csv.DictReader(open(path, encoding="utf-8-sig")):
                _CACHE[r["video"]] = r
    return _CACHE


def clip_span(video, span, offset=0.0, mode="exclude", path=OUT):
    """評価区間 span=[lo,hi]（録音WAV時刻）を、観測できた区間と重ねて狭める。
       mode="keep" なら何もしない。帯 pts は帯動画の時刻なので + offset で WAV 時刻にする。"""
    if mode != "exclude":
        return span, False
    r = load(path).get(video)
    if not r or r["first_frame_pts"] == "":
        return span, False
    lo = max(float(span[0]), float(r["first_frame_pts"]) + offset)
    hi = min(float(span[1]), float(r["last_frame_pts"]) + offset)
    return [lo, hi], (lo != float(span[0]) or hi != float(span[1]))


if __name__ == "__main__":
    build("--check" in sys.argv)
