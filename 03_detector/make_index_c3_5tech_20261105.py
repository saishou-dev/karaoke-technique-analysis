# -*- coding: utf-8 -*-
r"""5技法だけの正解の一覧を作る（2026-11-05 Opus・c3-5tech・指示書 2026-11-05_1630）。
元＝out\c2_shift3_global_20261105\labels\index_shift3_global_20261105.json（遅れ 1.6722秒・全技法共通・208曲・10,200件）＝読むだけ。
作るもの＝out\c3_5tech_global_20261105\labels\index_c3_5tech_20261105.json（vibrato の件だけ外した写し・span と他の項目は元のまま）
          ＋ check_c3_5tech_20261105.txt。
確かめる＝曲数 208 のまま・5技法の件数が元と一致・vibrato 0・1件ずつの [技法, 時刻] が元の順のまま一致。合わなければ止める。
使い方: python make_index_c3_5tech_20261105.py
"""
import json, os, sys, collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "out", "c2_shift3_global_20261105", "labels", "index_shift3_global_20261105.json")
OUT = os.path.join(ROOT, "out", "c3_5tech_global_20261105", "labels")
TYPES6 = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
TYPES5 = ["shakuri", "kobushi", "fall", "accent", "hammering"]


def main():
    os.makedirs(OUT, exist_ok=True)
    log = []
    say = lambda s: (print(s), log.append(s))
    ix = json.load(open(SRC, encoding="utf-8"))
    ix2 = []
    for d in ix:
        d2 = dict(d)
        d2["events"] = [[c, t] for c, t in d["events"] if c != "vibrato"]
        d2["c3_note"] = "c3_5tech_20261105: index_shift3_global_20261105 から vibrato の件だけ外した写し（遅れ 1.6722秒・span は元のまま）"
        ix2.append(d2)
    path = os.path.join(OUT, "index_c3_5tech_20261105.json")
    json.dump(ix2, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    ix3 = json.load(open(path, encoding="utf-8"))
    e1 = collections.Counter(c for d in ix for c, _ in d["events"])
    e3 = collections.Counter(c for d in ix3 for c, _ in d["events"])
    s1 = collections.Counter(d["session"] for d in ix); s3 = collections.Counter(d["session"] for d in ix3)
    same_order = all(d["video"] == d3["video"] and d["span"] == d3["span"]
                     and [e for e in d["events"] if e[0] != "vibrato"] == d3["events"] for d, d3 in zip(ix, ix3))
    say(f"元＝{SRC}")
    say(f"曲＝元 {len(ix)}・写し {len(ix3)}・収録別 {'一致' if s1 == s3 else '不一致'}（{dict(sorted(s3.items()))}）")
    say(f"件数＝元 {sum(e1.values())}・写し {sum(e3.values())}（差 {sum(e1.values()) - sum(e3.values())}＝元の vibrato {e1['vibrato']}）")
    say("技法別（元→写し）＝" + "・".join(f"{c} {e1[c]}→{e3[c]}" for c in TYPES6))
    ok = (len(ix) == len(ix3) == 208 and s1 == s3 and e3["vibrato"] == 0 and all(e1[c] == e3[c] for c in TYPES5)
          and set(e3) <= set(TYPES5) and same_order)
    say(f"1件ずつの [技法, 時刻]・span・曲の順＝{'一致' if same_order else '不一致'}")
    # 試験（20260924）と dev（20260911）の span 内の件数（採点の母数の見込み）
    for ses in ("20260924", "20260911"):
        cc = collections.Counter(c for d in ix3 if d["session"] == ses for c, t in d["events"] if d["span"][0] <= t <= d["span"][1])
        say(f"{ses} の span 内＝" + "・".join(f"{c} {cc[c]}" for c in TYPES5) + f"・計 {sum(cc.values())}")
    say("判定＝" + ("OK" if ok else "NG"))
    open(os.path.join(OUT, "check_c3_5tech_20261105.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    if not ok:
        raise SystemExit("止める＝件数が合わない")


if __name__ == "__main__":
    main()
