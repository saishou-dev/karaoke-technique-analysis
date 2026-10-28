# -*- coding: utf-8 -*-
r"""手順2＝「60fps だからこその読み取り規則」を何通りも当てて加点を取り出す（2026-10-28）。

1フレームごとの読み（dumps\<source>__<video>.npz）に、次の3つの選択を組み合わせた規則を当てる。
OCR はやり直さないので、候補を全部実行できる。

(a) 入力  … g60（固定60fps・複製あり）／g5933（固定59.3354fps）／blend60（固定60fps・混ぜて作る・複製なし）／vfr（現行A）
(b) 下ごしらえ … none（そのまま）／dedup:θ（直前フレームとの画素差が θ 未満のフレーム＝複製を読みから外す）
(c) 安定の数え方 … frames:6（現行＝同じ値が6回続いたら確定）
                 ／time:T（同じ値が T 秒以上続いたら確定。T は「最初に読めた所から最後に読めた所まで」の長さ）
(d) 時刻     … index（フレーム番号 ÷ 固定レート）／pts（表示時刻の表・現行A だけ）

安定ランの重み（どの軌跡を本線に選ぶかの重み付き最長非減少部分列＝WLIS で使う）は既定で「読めた回数」。
time 規則では「続いた長さ（フレーム数換算）」を重みにする版も別候補として置く（weight=span）。

出力＝cfr_native_20261028\results\<cand>__<video>\counter_events.csv（現行と同じ列）
      ＋ report\rules.log
使い方＝python rules_20261028.py
"""
import bisect
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_common_20261028 import COUNTERS, OUT, SOURCES, VIDEOS, dump_path, ensure_out, log

STABLE_MIN = 6
BAND_MIN = 0.5
BAND_W = 150
# 複製と見なす画素差の上限（既定）。report\dedup_sweep.csv で 0.05〜1.00 を振って確かめる。
# 2026-10-28 実測＝固定60fps の帯動画で、直前フレームとの平均絶対差は最小 0.031・分布に切れ目が無い
# （共犯＝0.05 未満 2 枚／0.1 未満 28 枚／0.2 未満 123 枚／0.3 未満 182 枚。理論上の複製は 168 枚）。
# つまり「複製」と「数字も背景も動いていない本物のフレーム」は画素差では分けられない。
# 既定は 0.20（理論上の複製数に最も近く、かつ超えない側）。
DEDUP_T = 0.20

# 候補の定義。pre=("dedup", θ) / ("none",)  stab=("frames", 6) / ("time", 秒, 重み)
CANDS = {
    "A_現行":        dict(src="vfr",     pre=("none",),          stab=("frames", 6),              tm="pts",
                        label="A 現行＝可変のまま＋表示時刻の表＋6フレーム安定"),
    "P1_60_複製除去":  dict(src="g60",     pre=("dedup", DEDUP_T), stab=("frames", 6),              tm="index",
                        label="(1) 固定60fps＋複製フレーム除去＋6フレーム安定（時刻＝番号÷60）"),
    "P2_60_時間安定":  dict(src="g60",     pre=("none",),          stab=("time", 0.100, "count"),   tm="index",
                        label="(2) 固定60fps＋100ms 以上続いたら確定（時刻＝番号÷60）"),
    "P2b_60_時間101": dict(src="g60",     pre=("none",),          stab=("time", 6 / (450000 / 7584), "count"), tm="index",
                        label="(2b) 固定60fps＋101.1ms＝元録画の6フレーム分以上（追加）"),
    "P2s_60_時間長重み": dict(src="g60",   pre=("none",),          stab=("time", 0.100, "span"),    tm="index",
                        label="(2s) (2) の重みを『続いた長さ』に変えた版（追加）"),
    "P3_5933_複製除去": dict(src="g5933",  pre=("dedup", DEDUP_T), stab=("frames", 6),              tm="index",
                        label="(3) 固定59.3354fps＋複製フレーム除去＋6フレーム安定（時刻＝番号÷59.3354）"),
    "P4_混ぜ60":      dict(src="blend60", pre=("none",),          stab=("frames", 6),              tm="index",
                        label="(4) 固定60fps＋前後の中間で補間（複製なし）＋6フレーム安定（時刻＝番号÷60）"),
    "P5_60_除去＋時間": dict(src="g60",    pre=("dedup", DEDUP_T), stab=("time", 0.100, "count"),   tm="index",
                        label="(5) (1)＋(2) の併用（追加＝複製除去と時間安定の両方）"),
    "P6_混ぜ60_時間":  dict(src="blend60", pre=("none",),          stab=("time", 0.100, "count"),   tm="index",
                        label="(6) (4)＋(2) の併用（追加）"),
    # ---- 2026-10-28 追加の候補（足した理由を各行に書く）----
    # (7)＝指示(4) の字面どおり「抜けフレームだけ中間で埋める」版。P4（全フレーム混ぜ）が効いた理由が
    #      「複製が消えたこと」なのか「抜けの埋め方」なのかを分けるため。
    "P7_抜けだけ中間": dict(src="gapblend60", pre=("none",), stab=("frames", 6), tm="index",
                       label="(7) 固定60fps＋抜けフレームだけ前後の中間（複製は残る）＋6フレーム安定"),
    # (8)＝混ぜ方は P4 と同じで格子を録画の実測レートにした版。「混ぜる効果」と「60 という数字」を分けるため。
    "P8_混ぜ5933": dict(src="blend5933", pre=("none",), stab=("frames", 6), tm="index",
                     label="(8) 固定59.3354fps＋前後の中間で混ぜて作る＋6フレーム安定（時刻＝番号÷59.3354）"),
    # (9)＝可変のまま読み、各フレームに 60fps の格子番号を割り当てて時刻を「番号÷60」で出す版。
    #      「フレーム番号＝時刻」を 60 の格子で成り立たせたときの丸め代（最大 8.3 ミリ秒）だけを測る上限の目安。
    #      ただし格子番号を作るのに表示時刻の表が要るので、表を捨てる方法ではない。
    "P9_格子番号化": dict(src="vfr", pre=("none",), stab=("frames", 6), tm="grid60",
                     label="(9) 可変のまま読み、時刻だけ 60fps の格子に丸める（番号÷60・表示時刻の表は必要）"),
    # (4s)＝P4 の重みを「続いた長さ」に変えた版（time 規則の重みの効き方を見るため）。
    "P4s_混ぜ60_時間長重み": dict(src="blend60", pre=("none",), stab=("time", 0.100, "span"), tm="index",
                        label="(4s) (4) に時間安定＋長さ重みを足した版（追加）"),
    # (7t)＝(7) に時間安定を足した版。(7) は「再エンコードを挟まない固定60fps＋複製あり」なので、
    #       これと (7) を比べると「時間安定の効き」だけが、(7) と (0) を比べると「再エンコードの効き」だけが出る。
    "P7t_抜けだけ中間_時間": dict(src="gapblend60", pre=("none",), stab=("time", 0.100, "count"), tm="index",
                       label="(7t) (7)＋100ms 以上続いたら確定（追加）"),
    # (4f)＝(4) をファイルに書き出してから読んだ版。x264 を通しても (4) の利点が残るかを見る（実運用の条件）。
    "P4f_混ぜ60_ファイル": dict(src="blend60file", pre=("none",), stab=("frames", 6), tm="index",
                       label="(4f) (4) を x264 で書き出したファイルを読む＋6フレーム安定"),
    # (4ft)＝(4f) に時間安定を足した版。
    "P4ft_混ぜ60_ファイル_時間": dict(src="blend60file", pre=("none",), stab=("time", 0.100, "count"), tm="index",
                       label="(4ft) (4f)＋100ms 以上続いたら確定"),
    # (7f)＝(7) をファイルに書き出してから読んだ版＝現行の帯動画から固定60fps を作り直す実運用そのまま。
    "P7f_作り直し60_ファイル": dict(src="gapblend60file", pre=("none",), stab=("frames", 6), tm="index",
                       label="(7f) 現行の帯動画から固定60fps を作り直したファイル＋6フレーム安定"),
    "P7ft_作り直し60_ファイル_時間": dict(src="gapblend60file", pre=("none",), stab=("time", 0.100, "count"), tm="index",
                       label="(7ft) (7f)＋100ms 以上続いたら確定"),
    # (A2)＝現行A の入力に時間基準の安定だけを当てた版。
    #       「時間基準は固定fps のための細工ではなく、現行に入れても害が無いか」を確かめるため（2026-10-28 追加）。
    "A2_現行_時間安定": dict(src="vfr", pre=("none",), stab=("time", 0.100, "count"), tm="pts",
                      label="(A2) 現行A の入力＋100ms 以上続いたら確定（表示時刻の表のまま）"),
    "P0_60_現行規則":  dict(src="g60",     pre=("none",),          stab=("frames", 6),              tm="index",
                        label="(0) 固定60fps＋現行の規則そのまま＝10/22 の B2（足場・再現の確認用）"),
    "P3b_5933_現行規則": dict(src="g5933", pre=("none",),          stab=("frames", 6),             tm="index",
                        label="(3b) 固定59.3354fps＋現行の規則そのまま＝10/22 の D2（足場）"),
}


def load_dump(source, video):
    z = np.load(dump_path(source, video), allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    vals = {k: list(zip(z["fr_" + k].tolist(), z["val_" + k].tolist())) for k in COUNTERS}
    return meta, vals, z["diff_prev"]


def band_frame_range(values, nframes):
    """読取器 v0.8e(B) と同じ帯消失ガード。"""
    cnt = {}
    for name in values:
        for fr, _v in values[name]:
            cnt[fr] = cnt.get(fr, 0) + 1
    cum = [0] * (nframes + 1)
    for i in range(nframes):
        cum[i + 1] = cum[i] + cnt.get(i, 0)

    def sm(i):
        a = max(0, i - BAND_W)
        b = min(nframes, i + BAND_W + 1)
        return (cum[b] - cum[a]) / max(1, b - a)

    bs = 0
    while bs < nframes and sm(bs) < BAND_MIN:
        bs += 1
    be = nframes - 1
    while be >= 0 and sm(be) < BAND_MIN:
        be -= 1
    return bs, be


def make_runs(seq):
    """同じ値が続いた区間＝[最初のフレーム, 値, 読めた回数, 最後のフレーム]。"""
    runs = []
    run_v = None
    for fr, v in seq:
        if run_v is not None and v == run_v:
            runs[-1][2] += 1
            runs[-1][3] = fr
        else:
            runs.append([fr, v, 1, fr])
            run_v = v
    return runs


def detect_increments(seq, stab, rate):
    """安定ランを選び、重み付き最長非減少部分列で軌跡を1本選び、増えた点＝加点を返す。"""
    runs = make_runs(seq)
    if stab[0] == "frames":
        keep = [r for r in runs if r[2] >= stab[1]]
        wts = [float(r[2]) for r in keep]
    else:
        T = stab[1]
        keep = [r for r in runs if (r[3] - r[0] + 1) / rate >= T]
        if stab[2] == "span":
            wts = [float(r[3] - r[0] + 1) for r in keep]
        else:
            wts = [float(r[2]) for r in keep]
    n = len(keep)
    accepted = []
    if n:
        dp = list(wts)
        prev = [-1] * n
        for i in range(n):
            for j in range(i):
                if keep[j][1] <= keep[i][1] and dp[j] + wts[i] > dp[i]:
                    dp[i] = dp[j] + wts[i]
                    prev[i] = j
        best = int(np.argmax(dp))
        path = []
        while best != -1:
            path.append(best)
            best = prev[best]
        path.reverse()
        cur = None
        for idx in path:
            fr, v = keep[idx][0], keep[idx][1]
            if cur is None:
                accepted.append((fr, v, 0))
                cur = v
            elif v > cur:
                accepted.append((fr, v, v - cur))
                cur = v
    return accepted, n


def run_candidate(cand, video, meta, vals, diff, pts=None):
    C = CANDS[cand]
    rate = SOURCES[C["src"]]["rate"] or meta["fps"]
    nframes = meta["nframes_read"]
    values = {k: list(v) for k, v in vals.items()}
    n_removed = 0
    if C["pre"][0] == "dedup":
        th = C["pre"][1]
        dup = diff < th
        dup[0] = False
        n_removed = int(dup.sum())
        for k in COUNTERS:
            values[k] = [(f, v) for f, v in values[k] if not (f < len(dup) and dup[f])]
    bs, be = band_frame_range(values, nframes)
    if bs <= be:
        for k in COUNTERS:
            values[k] = [(f, v) for f, v in values[k] if bs <= f <= be]
    out = {}
    stats = {}
    for k in COUNTERS:
        acc, nruns = detect_increments(values[k], C["stab"], rate)
        out[k] = acc
        stats[k] = dict(runs=nruns, final=(acc[-1][1] if acc else None),
                        increments=sum(d for _, _, d in acc), n_events=max(0, len(acc) - 1),
                        n_single=sum(1 for _, _, d in acc if d == 1))
    rows = []
    for name, acc in out.items():
        seq = values[name]
        for i, (fr, v, d) in enumerate(acc):
            ev_fr = fr
            if i > 0:
                pv = acc[i - 1][1]
                last_old = None
                for f2, v2 in seq:
                    if f2 >= fr:
                        break
                    if v2 == pv:
                        last_old = f2
                if last_old is not None:
                    ev_fr = last_old + 1
            if C["tm"] == "pts":
                t = pts[min(ev_fr, len(pts) - 1)]
            elif C["tm"] == "grid60":
                t = round(pts[min(ev_fr, len(pts) - 1)] * 60.0) / 60.0
            else:
                t = ev_fr / rate
            rows.append(dict(counter=name, frame=fr, value=v, delta=d,
                             event_frame_est=ev_fr, event_time_est=round(t, 4)))
    rows.sort(key=lambda r: (r["frame"], r["counter"]))
    return rows, stats, dict(rate=rate, nframes=nframes, n_removed=n_removed,
                             band_start=bs, band_end=be,
                             ocr_sec=meta.get("elapsed_sec"))


if __name__ == "__main__":
    ensure_out()
    LOG = os.path.join(OUT, "report", "rules.log")
    t_all = time.time()
    meta_rows = []
    sweep_rows = []
    diffstat_rows = []
    for video in VIDEOS:
        cache = {}
        for src in set(C["src"] for C in CANDS.values()):
            p = dump_path(src, video)
            if os.path.exists(p):
                cache[src] = load_dump(src, video)
        pts = None
        pp = SOURCES["vfr"]["pts"] % video
        if os.path.exists(pp):
            pts = [float(r["pts_sec"]) for r in csv.DictReader(open(pp, encoding="utf-8"))]
        for cand, C in CANDS.items():
            if C["src"] not in cache:
                log("[skip] %s %s（読み込みが無い）" % (cand, video), LOG)
                continue
            meta, vals, diff = cache[C["src"]]
            t0 = time.time()
            rows, stats, info = run_candidate(cand, video, meta, vals, diff, pts)
            dt = time.time() - t0
            d = os.path.join(OUT, "results", "%s__%s" % (cand, video))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "counter_events.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["counter", "frame", "time_sec", "value", "delta", "event_frame_est", "event_time_est"])
                for r in rows:
                    w.writerow([r["counter"], r["frame"], "%.3f" % (r["frame"] / info["rate"]),
                                r["value"], r["delta"], r["event_frame_est"], "%.4f" % r["event_time_est"]])
            json.dump(dict(cand=cand, video=video, label=C["label"], info=info,
                           stats={k: stats[k] for k in COUNTERS}),
                      open(os.path.join(d, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            meta_rows.append(dict(cand=cand, video=video, label=C["label"], src=C["src"],
                                  rate=round(info["rate"], 4), nframes=info["nframes"],
                                  removed=info["n_removed"], band="%d..%d" % (info["band_start"], info["band_end"]),
                                  ocr_sec=info["ocr_sec"], rule_sec=round(dt, 2),
                                  n_events=sum(stats[k]["n_events"] for k in COUNTERS),
                                  n_single=sum(stats[k]["n_single"] for k in COUNTERS),
                                  runs=sum(stats[k]["runs"] for k in COUNTERS),
                                  **{"final_" + k: stats[k]["final"] for k in COUNTERS}))
            log("[rule] %-16s %-14s 加点 %d（単独 %d）除去 %d ラン %d  %.2fs"
                % (cand, video[:14], meta_rows[-1]["n_events"], meta_rows[-1]["n_single"],
                   info["n_removed"], meta_rows[-1]["runs"], dt), LOG)
        # 複製と見なす画素差のしきい値を振る（候補(1) のしきい値依存を見る）
        if "g60" in cache:
            meta, vals, diff = cache["g60"]
            for th in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
                C = dict(src="g60", pre=("dedup", th), stab=("frames", 6), tm="index",
                         label="sweep")
                CANDS["_sweep"] = C
                rows, stats, info = run_candidate("_sweep", video, meta, vals, diff, pts)
                sweep_rows.append(dict(video=video, theta=th, removed=info["n_removed"],
                                       n_events=sum(stats[k]["n_events"] for k in COUNTERS),
                                       runs=sum(stats[k]["runs"] for k in COUNTERS),
                                       **{"final_" + k: stats[k]["final"] for k in COUNTERS}))
            CANDS.pop("_sweep", None)
        # 画素差の分布と、理論上の複製枚数（pts 表から最近傍割当で数えたもの）を記録する
        for src in sorted(cache):
            meta, vals, diff = cache[src]
            d = np.sort(diff[1:])
            rate = SOURCES[src]["rate"]
            theo = ""
            if rate and pts:
                used = {}
                N = len(pts)
                for k in range(int(pts[-1] * rate) + 1):
                    t = k / rate
                    j = bisect.bisect_left(pts, t)
                    i = min([x for x in (j - 1, j, j + 1) if 0 <= x < N], key=lambda x: abs(pts[x] - t))
                    used[i] = used.get(i, 0) + 1
                theo = sum(c - 1 for c in used.values() if c > 1)
            diffstat_rows.append(dict(
                video=video, source=src, nframes=meta["nframes_read"],
                theoretical_dup=theo, diff_min=round(float(d[0]), 4),
                **{("lt_%.2f" % t): int((diff[1:] < t).sum()) for t in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)},
                diff_q01=round(float(np.quantile(d, 0.01)), 4),
                diff_q05=round(float(np.quantile(d, 0.05)), 4),
                diff_median=round(float(np.quantile(d, 0.5)), 4)))

    REP = os.path.join(OUT, "report")
    if diffstat_rows:
        with open(os.path.join(REP, "diffstat.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(diffstat_rows[0].keys()))
            w.writeheader()
            for r in diffstat_rows:
                w.writerow(r)
    with open(os.path.join(REP, "rules_meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        w.writeheader()
        for r in meta_rows:
            w.writerow(r)
    if sweep_rows:
        with open(os.path.join(REP, "dedup_sweep.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            w.writeheader()
            for r in sweep_rows:
                w.writerow(r)
    log("DONE rules（%.0f 秒）" % (time.time() - t_all), LOG)
