# -*- coding: utf-8 -*-
r"""正解データ（G1）の時刻を「印の位置の時刻（D）」側へずらした写しを作る（2026-11-05・c2-shift2・make_g1_shifted_20261104.py と同型）。

ずらす量 M＝marks_20260928\out\mark_pos_time_20261105\D_times_20261105.csv の a_counter_minus_D（カウンター時刻−D）の
  技法別中央値（9/24 と 10/14 の合算・D がある件だけ）。6技法とも（ビブラートも点としてずらす）。
作るもの（out\c2_shift2_20261105\labels\ だけ・元は読むだけ）
  shift_table_20261105.csv / G1_events_all_shift2_20261105.csv / index_shift2_20261105.json / check_20261105.txt
span は元のまま（学習と評価の範囲を元とそろえる）。span の外へ出た件が 1% を超えたら止める。
使い方: python make_g1_shifted_20261105.py
"""
import csv, json, os, statistics, sys, collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "out", "c2_shift2_20261105", "labels")
DT = os.path.join(BASE, "marks_20260928", "out", "mark_pos_time_20261105", "D_times_20261105.csv")
G1 = os.path.join(BASE, "dataset", "G1_events_all.csv")
INDEX = os.path.join(ROOT, "index.json")
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
REF = dict(shakuri=1.64, kobushi=1.94, fall=1.90, accent=1.62, hammering=1.62)   # 1010 報告の参考値
TIME_COLS = ["video_time_sec", "audio_time_sec", "video_time_idx_sec"]


def main():
    os.makedirs(OUT, exist_ok=True)
    log = []
    say = lambda s: (print(s), log.append(s))
    v = collections.defaultdict(list); ns = collections.Counter()
    for r in csv.DictReader(open(DT, encoding="utf-8-sig")):
        if r["D"] == "" or r["a_counter_minus_D"] == "":
            continue
        v[r["counter"]].append(float(r["a_counter_minus_D"])); ns[(r["session"], r["counter"])] += 1
    shift, rows = {}, []
    for c in TYPES:
        if not v[c]:
            raise SystemExit(f"止める＝{c} に D のある件が無い")
        shift[c] = statistics.median(v[c])
        ref = REF.get(c)
        diff = "" if ref is None else round(shift[c] - ref, 4)
        rows.append([c, ns[("20260924", c)], ns[("20261014", c)], len(v[c]), round(shift[c], 4), "" if ref is None else ref, diff])
    allv = [x for c in TYPES for x in v[c]]
    rows.append(["ALL", sum(ns[("20260924", c)] for c in TYPES), sum(ns[("20261014", c)] for c in TYPES), len(allv), round(statistics.median(allv), 4), "", "参考（使っていない）"])
    with open(os.path.join(OUT, "shift_table_20261105.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["技法", "件数_9/24", "件数_10/14", "件数_合計", "M_秒（前へ）", "参考値_1010報告", "差"]); w.writerows(rows)
    say("M（秒・前へ）＝" + "・".join(f"{c} {shift[c]:.4f}（{len(v[c])}件）" for c in TYPES))
    big = [c for c in TYPES if c in REF and abs(shift[c] - REF[c]) >= 0.05]
    say(f"参考値との差 0.05秒以上＝{big or 'なし'}")

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
    gpath = os.path.join(OUT, "G1_events_all_shift2_20261105.csv")
    with open(gpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields + extra); w.writeheader(); w.writerows(out_rows)
    g2 = list(csv.DictReader(open(gpath, encoding="utf-8-sig")))
    c1 = collections.Counter((r["session"], r["counter"]) for r in g); c2 = collections.Counter((r["session"], r["counter"]) for r in g2)
    ok_g = len(g) == len(g2) and c1 == c2
    say(f"G1 写し＝元 {len(g)} 行・写し {len(g2)} 行・収録×技法の件数 {'一致' if c1 == c2 else '不一致'}・技法外の行 {dict(unknown) or 0}")

    ix = json.load(open(INDEX, encoding="utf-8"))
    ix2, bad, bad_t = [], 0, collections.Counter()
    for d in ix:
        d2 = dict(d)
        d2["events"] = [[c, round(t - shift.get(c, 0.0), 4)] for c, t in d["events"]]
        lo, hi = d["span"]
        for c, t in d2["events"]:
            if not (lo <= t <= hi):
                bad += 1; bad_t[c] += 1
        d2["shift_note"] = "c2_shift2_20261105: events = 元の時刻 − M（カウンター−D の技法別中央値・span は元のまま）"
        ix2.append(d2)
    ipath = os.path.join(OUT, "index_shift2_20261105.json")
    json.dump(ix2, open(ipath, "w", encoding="utf-8"), ensure_ascii=False)
    ix3 = json.load(open(ipath, encoding="utf-8"))
    e1 = collections.Counter(c for d in ix for c, _ in d["events"]); e2 = collections.Counter(c for d in ix3 for c, _ in d["events"])
    pair_ok = all(len(d["events"]) == len(d3["events"]) and all(c == c3 and abs((t - t3) - shift[c]) < 1e-3
                  for (c, t), (c3, t3) in zip(d["events"], d3["events"])) for d, d3 in zip(ix, ix3))
    ok_i = len(ix) == len(ix3) and e1 == e2 and pair_ok
    n_ev = sum(e1.values()); frac = bad / n_ev if n_ev else 0
    say(f"index 写し＝曲 {len(ix)}→{len(ix3)}・イベント {n_ev}→{sum(e2.values())}・技法別 {'一致' if e1 == e2 else '不一致'}・"
        f"1対1の M {'一致' if pair_ok else '不一致'}・span の外へ出た件 {bad}（{frac * 100:.2f}%・技法別 {dict(bad_t) or 0}）")
    say("index 技法別＝" + "・".join(f"{c} {e2[c]}" for c in TYPES))
    open(os.path.join(OUT, "check_20261105.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    if not (ok_g and ok_i):
        raise SystemExit("止める＝件数の一致が取れない")
    if frac > 0.01:
        raise SystemExit(f"止める＝span の外 {frac * 100:.2f}% が 1% 超")
    print("OK")


if __name__ == "__main__":
    main()
