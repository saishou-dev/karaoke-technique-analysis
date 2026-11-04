# -*- coding: utf-8 -*-
r"""正解データ（G1）の時刻を技法マーク側へ補正した「写し」を作る（2026-11-04・研究 D2 の一歩目）。

なぜ必要か。
  技法カウンター（加点回数表示）の時刻は、音程バーの技法マークより中央値で約 0.84 秒遅い（材料48）。
  この遅れぶん正解の時刻を前へ寄せた写しで C2（検出器）を学習し直し、現行と F値を比べるための入力を作る。

補正量。
  技法別の中央値＝9/24 と 10/14 の2収録で「対応」した件（multi_increment=0・mark_status=mark）を合わせた中央値（材料48 2-2）。
  読むもの＝marks_20260928\G1_events_v2_20260924.csv（9/24 の正本）・marks_20260928\mark_lead_events_20261014.csv（10/14 行単位）。
  表に無い技法は、2収録を合わせた全体の中央値を使う（今回は6技法とも表にある）。
  材料48 の件数（9/24 484件・10/14 570件）と技法別の中央値を再現できなければ止める。

作るもの（out\c2_shift_20261104\labels\ だけ・元のファイルは読むだけ）。
  shift_table_20261104.csv          ＝技法別の補正量と根拠の件数
  G1_events_all_shift_20261104.csv  ＝dataset\G1_events_all.csv の写し。video_time_sec・audio_time_sec・video_time_idx_sec を
                                      補正量だけ前へ（stable_read_* は読取の時刻なので変えない）。末尾に shift_sec・元の時刻3列を足す。
  index_shift_20261104.json         ＝学習台本が読む index.json の写し。events の時刻だけを補正量だけ前へ（span は元のまま＝
                                      学習の範囲と評価の範囲を元とそろえる）。
  check_20261104.txt                ＝件数・技法別件数の一致の確認結果

使い方:  python make_g1_shifted_20261104.py
"""
import csv
import json
import os
import statistics
import sys
import collections

if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # g3v2_20260928
BASE = os.path.dirname(ROOT)                       # 動画解析
OUT = os.path.join(ROOT, "out", "c2_shift_20261104", "labels")
MARKS = os.path.join(BASE, "marks_20260928")
SRC_0924 = os.path.join(MARKS, "G1_events_v2_20260924.csv")
SRC_1014 = os.path.join(MARKS, "mark_lead_events_20261014.csv")
G1 = os.path.join(BASE, "dataset", "G1_events_all.csv")
INDEX = os.path.join(ROOT, "index.json")
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
# 材料48 2-2 の値（再現の確認に使う）
EXPECT_N = {"20260924": 484, "20261014": 570}
EXPECT_MED = {"20260924": dict(vibrato=0.809, shakuri=0.977, kobushi=0.860, accent=0.784, hammering=0.843, fall=1.163),
              "20261014": dict(vibrato=0.859, shakuri=0.859, kobushi=0.826, accent=0.741, hammering=0.809, fall=0.961)}
TIME_COLS = ["video_time_sec", "audio_time_sec", "video_time_idx_sec"]


def matched(path):
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["multi_increment"] != "0" or r["mark_status"] != "mark" or r["mark_dt_sec"] == "":
            continue
        out[r["counter"]].append(float(r["mark_dt_sec"]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    log = []
    say = lambda s: (print(s), log.append(s))
    a, b = matched(SRC_0924), matched(SRC_1014)
    for ses, m in (("20260924", a), ("20261014", b)):
        n = sum(len(v) for v in m.values())
        if n != EXPECT_N[ses]:
            raise SystemExit(f"止める＝{ses} の対応件数 {n} が材料48 の {EXPECT_N[ses]} と違う")
        for c, e in EXPECT_MED[ses].items():
            got = round(statistics.median(m[c]), 3)
            if abs(got - e) > 0.0011:
                raise SystemExit(f"止める＝{ses} {c} の中央値 {got} が材料48 の {e} と違う")
    say("材料48 2-2 の再現＝対応件数 9/24 484・10/14 570・技法別の中央値12個とも一致")
    allv = [x for m in (a, b) for v in m.values() for x in v]
    overall = statistics.median(allv)
    shift, rows = {}, []
    for c in TYPES:
        v = a.get(c, []) + b.get(c, [])
        if v:
            shift[c] = statistics.median(v); src = "2収録の対応件の中央値"
        else:
            shift[c] = overall; src = "表に無い＝全体の中央値"
        rows.append([c, len(a.get(c, [])), len(b.get(c, [])), len(v), round(shift[c], 4), src])
    rows.append(["ALL", sum(len(x) for x in a.values()), sum(len(x) for x in b.values()), len(allv), round(overall, 4), "参考（使っていない）"])
    with open(os.path.join(OUT, "shift_table_20261104.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["技法", "対応件数_9/24", "対応件数_10/14", "対応件数_合計", "補正量_秒（前へ）", "出所"]); w.writerows(rows)
    say("補正量（秒・前へ）＝" + "・".join(f"{c} {shift[c]:.4f}" for c in TYPES) + f"（全体 {overall:.4f}・参考）")

    # --- G1 の写し ---
    with open(G1, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f); fields = rd.fieldnames; g = list(rd)
    extra = ["shift_sec"] + [f"orig_{c}" for c in TIME_COLS]
    out_rows, unknown = [], collections.Counter()
    for r in g:
        r2 = dict(r); s = shift.get(r["counter"])
        if s is None:
            unknown[r["counter"]] += 1; s = 0.0
        for c in TIME_COLS:
            r2[f"orig_{c}"] = r[c]
            if r[c] not in ("", None):
                r2[c] = f"{float(r[c]) - s:.4f}"
        r2["shift_sec"] = f"{s:.4f}"
        out_rows.append(r2)
    gpath = os.path.join(OUT, "G1_events_all_shift_20261104.csv")
    with open(gpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields + extra); w.writeheader(); w.writerows(out_rows)
    g2 = list(csv.DictReader(open(gpath, encoding="utf-8-sig")))
    c1 = collections.Counter((r["session"], r["counter"]) for r in g); c2 = collections.Counter((r["session"], r["counter"]) for r in g2)
    ok_g = len(g) == len(g2) and c1 == c2
    say(f"G1 写し＝元 {len(g)} 行・写し {len(g2)} 行・収録×技法の件数 {'一致' if c1 == c2 else '不一致'}・技法外の行 {dict(unknown) or 0}")
    say("G1 技法別＝" + "・".join(f"{c} {sum(v for (s, cc), v in c2.items() if cc == c)}" for c in TYPES))

    # --- index の写し ---
    ix = json.load(open(INDEX, encoding="utf-8"))
    ix2, bad = [], 0
    for d in ix:
        d2 = dict(d)
        d2["events"] = [[c, round(t - shift.get(c, 0.0), 4)] for c, t in d["events"]]
        lo, hi = d["span"]
        bad += sum(1 for c, t in d2["events"] if not (lo <= t <= hi))
        d2["shift_note"] = "c2_shift_20261104: events = 元の時刻 − 技法別中央値（span は元のまま）"
        ix2.append(d2)
    ipath = os.path.join(OUT, "index_shift_20261104.json")
    json.dump(ix2, open(ipath, "w", encoding="utf-8"), ensure_ascii=False)
    ix3 = json.load(open(ipath, encoding="utf-8"))
    e1 = collections.Counter(c for d in ix for c, _ in d["events"]); e2 = collections.Counter(c for d in ix3 for c, _ in d["events"])
    pair_ok = all(len(d["events"]) == len(d3["events"]) and all(c == c3 and abs((t - t3) - shift[c]) < 1e-3
                  for (c, t), (c3, t3) in zip(d["events"], d3["events"])) for d, d3 in zip(ix, ix3))
    ok_i = len(ix) == len(ix3) and e1 == e2 and pair_ok
    say(f"index 写し＝曲 {len(ix)}→{len(ix3)}・イベント {sum(e1.values())}→{sum(e2.values())}・技法別 {'一致' if e1 == e2 else '不一致'}・"
        f"1対1の補正量 {'一致' if pair_ok else '不一致'}・補正後に span の外へ出た件 {bad}")
    say("index 技法別＝" + "・".join(f"{c} {e2[c]}" for c in TYPES))
    open(os.path.join(OUT, "check_20261104.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    if not (ok_g and ok_i):
        raise SystemExit("止める＝件数の一致が取れない")
    print("OK")


if __name__ == "__main__":
    main()
