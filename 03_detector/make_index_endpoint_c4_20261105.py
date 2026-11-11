# -*- coding: utf-8 -*-
r"""手順7a＝終わりの時刻の正解から vibrato を外した5技法の一覧を作る（2026-11-05 Opus・c4-compare 追補2）。
元＝out/endpoint_all_20261105/labels/index_endpoint_all_20261105.json（別の部下 endpoint-all の台帳・読むだけ）。
作る物＝out/c4_compare_20261105/labels/index_c4_endpoint_5tech_20261105.json ＋ check_endpoint_c4_20261105.txt。
確かめる＝曲数 208・曲の順と span が基準（index_c3_5tech_20261105）と同じ・技法ごとの件数が基準と一致・span の外の件。
加えて基準と時刻が違う件の数（技法ごと・1件ずつ並べて数える）と、差の中央値。
使い方: python make_index_endpoint_c4_20261105.py
"""
import collections, json, os, statistics, sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "out", "endpoint_all_20261105", "labels", "index_endpoint_all_20261105.json")
BASE = os.path.join(ROOT, "out", "c3_5tech_global_20261105", "labels", "index_c3_5tech_20261105.json")
OUT = os.path.join(ROOT, "out", "c4_compare_20261105", "labels")
TYPES5 = ["shakuri", "kobushi", "fall", "accent", "hammering"]


def main():
    log = []
    say = lambda s: (print(s), log.append(s))
    ix = json.load(open(SRC, encoding="utf-8")); bx = json.load(open(BASE, encoding="utf-8"))
    out = []
    for d in ix:
        d2 = dict(d); d2["events"] = [[c, t] for c, t in d["events"] if c in TYPES5]
        d2["c4_note"] = "c4_compare_20261105: index_endpoint_all_20261105（終わりの時刻の台帳）から vibrato を外した写し・span は元のまま"
        out.append(d2)
    path = os.path.join(OUT, "index_c4_endpoint_5tech_20261105.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    r = json.load(open(path, encoding="utf-8"))
    ce = collections.Counter(c for d in r for c, _ in d["events"]); cb = collections.Counter(c for d in bx for c, _ in d["events"])
    c_src = collections.Counter(c for d in ix for c, _ in d["events"])
    same = len(r) == len(bx) == 208 and all(a["video"] == b["video"] and a["span"] == b["span"] and a["session"] == b["session"] for a, b in zip(r, bx))
    say(f"元＝{SRC}・{len(ix)}曲・{sum(c_src.values())}件（vibrato {c_src['vibrato']}）")
    say(f"写し＝{len(r)}曲・{sum(ce.values())}件・曲の順と span と収録＝基準と{'一致' if same else '不一致'}")
    say("技法別（写し／基準）＝" + "・".join(f"{c} {ce[c]}／{cb[c]}" for c in TYPES5))
    ok = same and all(ce[c] == cb[c] for c in TYPES5) and set(ce) <= set(TYPES5)
    diff = collections.Counter(); dv = collections.defaultdict(list); pair_bad = 0
    for a, b in zip(r, bx):
        ea = collections.defaultdict(list); eb = collections.defaultdict(list)
        for c, t in a["events"]: ea[c].append(t)
        for c, t in b["events"]: eb[c].append(t)
        for c in TYPES5:
            if len(ea[c]) != len(eb[c]):
                pair_bad += 1; continue
            for x, y in zip(ea[c], eb[c]):
                if abs(x - y) > 1e-3:
                    diff[c] += 1; dv[c].append(x - y)
    say(f"技法ごとに曲の中の並びで1件ずつ並べた＝件数の合わない曲×技法 {pair_bad}")
    say("基準と時刻が違う件（写し−基準・秒）＝" + "・".join(f"{c} {diff[c]}件" + (f"（中央値 {statistics.median(dv[c]):+.3f}）" if dv[c] else "") for c in TYPES5))
    o = collections.Counter(c for d in r for c, t in d["events"] if not (d["span"][0] <= t <= d["span"][1]))
    say(f"span の外の件＝{sum(o.values())}（{dict(o) or 0}）")
    for ses in ("20260911", "20260924"):
        cc = collections.Counter(c for d in r if d["session"] == ses for c, t in d["events"] if d["span"][0] <= t <= d["span"][1])
        say(f"  {ses} の span 内＝" + "・".join(f"{c} {cc[c]}" for c in TYPES5) + f"・計 {sum(cc.values())}")
    say("判定＝" + ("OK" if ok else "NG"))
    open(os.path.join(OUT, "check_endpoint_c4_20261105.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    if not ok:
        raise SystemExit("止める＝件数が基準と合わない")


if __name__ == "__main__":
    main()
