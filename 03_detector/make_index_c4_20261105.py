# -*- coding: utf-8 -*-
r"""c4-compare の学習に渡す正解の一覧を作る（2026-11-05 Opus・指示書 2026-11-05_1806）。元は全て読むだけ。
作る物（out/c4_compare_20261105/labels/）
  index_c4_shift0_5tech_20261105.json  ＝遅れ 0（付け直す前＝カウンターが増えた時刻そのまま）＝index.json から vibrato を外した写し。
      make_g1_shifted_global_20261105.py の作り（index.json の時刻−遅れ）を遅れ 0 で回したのと同じ。
  index_c4_pertech_5tech_20261105.json ＝技法ごとの遅れ＝out/c2_shift2_20261105/labels/index_shift2_20261105.json から vibrato を外した写し。
  index_c4_lcNNNN_20261105.json        ＝学習の曲を入れ子で減らした写し（正解は全体1つの遅れ＝index_c3_5tech_20261105.json）。
      学習の曲（dev 20260911・試験 20260924 の外・特徴がある曲）を video 名で並べ、random.Random(0) で1回混ぜ、先頭から n 曲。
      n＝学習イベント（5技法・全件）の累計が目標（1,000・2,000・4,000・6,000）に最も近い曲数。dev と試験の曲は全部入れる。
  学習量の内訳_c4_2026-11-05.csv・学習曲の割り当て_c4_2026-11-05.csv・check_c4_20261105.txt
確かめる＝曲数 208・技法別の件数・遅れの差（1件ずつ）・span の外へ出た件。
使い方: python make_index_c4_20261105.py
"""
import csv, json, os, random, sys, collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out", "c4_compare_20261105", "labels")
TYPES5 = ["shakuri", "kobushi", "fall", "accent", "hammering"]
DEV, TEST = "20260911", "20260924"
SRC0 = os.path.join(ROOT, "index.json")
SRCG = os.path.join(ROOT, "out", "c3_5tech_global_20261105", "labels", "index_c3_5tech_20261105.json")
SRCP = os.path.join(ROOT, "out", "c2_shift2_20261105", "labels", "index_shift2_20261105.json")
GM = 1.6722
# 6,000 は学習の曲の全件（5,717件）を超える＝全部の曲＝基準と同じになる。代わりの点として 5,000 も作る（判断待ちに書く）。
TARGETS = [1000, 2000, 4000, 5000, 6000]


def strip5(ix, note):
    out = []
    for d in ix:
        d2 = dict(d); d2["events"] = [[c, t] for c, t in d["events"] if c in TYPES5]; d2["c4_note"] = note; out.append(d2)
    return out


def cnt(ix):
    return collections.Counter(c for d in ix for c, _ in d["events"])


def outside(ix):
    n = collections.Counter()
    for d in ix:
        lo, hi = d["span"]
        for c, t in d["events"]:
            if not (lo <= t <= hi): n[c] += 1
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    log = []
    say = lambda s: (print(s), log.append(s))
    ix0 = json.load(open(SRC0, encoding="utf-8")); ixg = json.load(open(SRCG, encoding="utf-8")); ixp = json.load(open(SRCP, encoding="utf-8"))
    ok = len(ix0) == len(ixg) == len(ixp) == 208 and all(a["video"] == b["video"] == c["video"] and a["span"] == b["span"] == c["span"] for a, b, c in zip(ix0, ixg, ixp))
    say(f"元＝index.json {len(ix0)}曲・全体1つ {len(ixg)}曲・技法ごと {len(ixp)}曲・曲の順と span {'一致' if ok else '不一致'}")

    s0 = strip5(ix0, "c4_compare_20261105: index.json（遅れ 0＝付け直す前）から vibrato を外した写し・span は元のまま")
    sp = strip5(ixp, "c4_compare_20261105: index_shift2_20261105（技法ごとの遅れ）から vibrato を外した写し・span は元のまま")
    for name, ix in (("index_c4_shift0_5tech_20261105.json", s0), ("index_c4_pertech_5tech_20261105.json", sp)):
        json.dump(ix, open(os.path.join(OUT, name), "w", encoding="utf-8"), ensure_ascii=False)
    r0 = json.load(open(os.path.join(OUT, "index_c4_shift0_5tech_20261105.json"), encoding="utf-8"))
    rp = json.load(open(os.path.join(OUT, "index_c4_pertech_5tech_20261105.json"), encoding="utf-8"))
    c0, cg, cp, call = cnt(r0), cnt(ixg), cnt(rp), cnt(ix0)
    say("件数（技法別）＝遅れ 0 " + "・".join(f"{c} {c0[c]}" for c in TYPES5) + f"・計 {sum(c0.values())}")
    say("件数（技法別）＝全体1つ " + "・".join(f"{c} {cg[c]}" for c in TYPES5) + f"・計 {sum(cg.values())}")
    say("件数（技法別）＝技法ごと " + "・".join(f"{c} {cp[c]}" for c in TYPES5) + f"・計 {sum(cp.values())}")
    ok = ok and all(c0[c] == cg[c] == cp[c] == call[c] for c in TYPES5) and set(c0) <= set(TYPES5) and set(cp) <= set(TYPES5)
    # 1件ずつの遅れの差
    dg = collections.defaultdict(set); dp = collections.defaultdict(set); bad = 0
    for a, g, p in zip(r0, ixg, rp):
        if not (len(a["events"]) == len(g["events"]) == len(p["events"])):
            bad += 1; continue
        for (ca, ta), (cg_, tg), (cp_, tp) in zip(a["events"], g["events"], p["events"]):
            if not (ca == cg_ == cp_):
                bad += 1; continue
            dg[ca].add(round(ta - tg, 3)); dp[ca].add(round(ta - tp, 3))
    say(f"1件ずつ（遅れ 0 − 全体1つ）＝" + "・".join(f"{c} {sorted(dg[c])}" for c in TYPES5) + f"・技法か並びの不一致 {bad}")
    say(f"1件ずつ（遅れ 0 − 技法ごと）＝" + "・".join(f"{c} {sorted(dp[c])}" for c in TYPES5))
    ok = ok and bad == 0 and all(len(dg[c]) == 1 and abs(next(iter(dg[c])) - GM) < 0.001 for c in TYPES5) and all(len(dp[c]) == 1 for c in TYPES5)
    for name, ix in (("遅れ 0", r0), ("全体1つ", ixg), ("技法ごと", rp)):
        o = outside(ix)
        say(f"span の外の件（{name}）＝{sum(o.values())}（{dict(o) or 0}）")
        for ses in (DEV, TEST):
            cc = collections.Counter(c for d in ix if d["session"] == ses for c, t in d["events"] if d["span"][0] <= t <= d["span"][1])
            say(f"  {ses} の span 内＝" + "・".join(f"{c} {cc[c]}" for c in TYPES5) + f"・計 {sum(cc.values())}")

    # 学習の曲を入れ子で減らす（全体1つの遅れの正解）
    have = [d for d in ixg if os.path.exists(os.path.join(ROOT, "feats", d["video"] + ".npz"))]
    train = sorted([d for d in have if d["session"] not in (DEV, TEST)], key=lambda d: d["video"])
    order = list(train); random.Random(0).shuffle(order)
    cum, cin = [], []
    s = si = 0
    for d in order:
        lo, hi = d["span"]
        s += len(d["events"]); si += sum(1 for _, t in d["events"] if lo <= t <= hi)
        cum.append(s); cin.append(si)
    say(f"学習の曲＝{len(train)}曲・学習イベント（5技法・全件）{cum[-1]}件・span 内 {cin[-1]}件")
    rows, conds = [], []
    rank = {d["video"]: i for i, d in enumerate(order)}
    for tg in TARGETS:
        n = min(range(1, len(order) + 1), key=lambda k: (abs(cum[k - 1] - tg), k))
        keep = set(d["video"] for d in order[:n])
        sub = [d for d in ixg if d["session"] in (DEV, TEST) or d["video"] in keep]
        name = f"index_c4_lc{tg}_20261105.json"
        full = n == len(order)
        if not full:
            json.dump(sub, open(os.path.join(OUT, name), "w", encoding="utf-8"), ensure_ascii=False)
        per = collections.Counter(c for d in order[:n] for c, _ in d["events"])
        conds.append((tg, n, cum[n - 1], cin[n - 1], per, "" if full else name))
        say(f"目標 {tg}件→学習 {n}曲・学習イベント {cum[n - 1]}件（span 内 {cin[n - 1]}件）・一覧の曲 {len(sub)}" + ("・全部の曲＝基準と同じ＝作らない" if full else f"→{name}"))
    per = collections.Counter(c for d in order for c, _ in d["events"])
    conds.append(("全部（基準）", len(order), cum[-1], cin[-1], per, "index_c3_5tech_20261105.json"))
    with open(os.path.join(OUT, "学習量の内訳_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["目標", "学習曲数", "学習イベント件数", "span内の件数"] + TYPES5 + ["index"])
        for tg, n, a, b, per, nm in conds:
            w.writerow([tg, n, a, b] + [per[c] for c in TYPES5] + [nm])
    with open(os.path.join(OUT, "学習曲の割り当て_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["video", "session", "混ぜた後の順番", "件数", "累計"] + [f"lc{t}" for t in TARGETS])
        for d in order:
            r = rank[d["video"]]
            w.writerow([d["video"], d["session"], r, len(d["events"]), cum[r]] + [1 if r < n else 0 for _, n, *_ in conds[:-1]])
    say("判定＝" + ("OK" if ok else "NG"))
    open(os.path.join(OUT, "check_c4_20261105.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    if not ok:
        raise SystemExit("止める＝件数か遅れの差が合わない")


if __name__ == "__main__":
    main()
