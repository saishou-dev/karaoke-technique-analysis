# -*- coding: utf-8 -*-
r"""4技法（フォールを外す）の正解の一覧を作る（2026-11-09 Opus・fall-4tech・指示書 2026-11-09_1406）。
make_index_c3_5tech_20261105.py の写し。違い＝元を5技法の一覧にして、フォール（fall）の件だけ外す。
元＝out\c3_5tech_global_20261105\labels\index_c3_5tech_20261105.json（11/5・遅れ 1.6722秒・全技法共通・208曲・7,856件）＝読むだけ。
    その元＝out\c2_shift3_global_20261105\labels\index_shift3_global_20261105.json（208曲・10,200件）。
作るもの＝out\fall_4tech_20261109\labels\index_fall4tech_20261109.json（fall の件だけ外した写し・span と他の項目は元のまま）
          ＋ check_fall4tech_20261109.txt。
確かめる＝曲数 208 のまま・4技法の件数が元と一致・fall 0・1件ずつの [技法, 時刻] が元の順のまま一致・
          学習の曲（20260911・20260924 以外）の件数＝5技法の件数−学習のフォール（101件の見込み）。合わなければ止める。
使い方: python make_index_fall4tech_20261109.py
"""
import json, os, sys, collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "out", "c3_5tech_global_20261105", "labels", "index_c3_5tech_20261105.json")
OUT = os.path.join(ROOT, "out", "fall_4tech_20261109", "labels")
TYPES5 = ["shakuri", "kobushi", "fall", "accent", "hammering"]
TYPES4 = ["shakuri", "kobushi", "accent", "hammering"]
DEV, TEST = "20260911", "20260924"


def main():
    os.makedirs(OUT, exist_ok=True)
    log = []
    say = lambda s: (print(s), log.append(s))
    ix = json.load(open(SRC, encoding="utf-8"))
    ix2 = []
    for d in ix:
        d2 = dict(d)
        d2["events"] = [[c, t] for c, t in d["events"] if c != "fall"]
        d2["fall4tech_note"] = "fall_4tech_20261109: index_c3_5tech_20261105 から fall の件だけ外した写し（遅れ 1.6722秒・span は元のまま）"
        ix2.append(d2)
    path = os.path.join(OUT, "index_fall4tech_20261109.json")
    json.dump(ix2, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    ix3 = json.load(open(path, encoding="utf-8"))
    e1 = collections.Counter(c for d in ix for c, _ in d["events"])
    e3 = collections.Counter(c for d in ix3 for c, _ in d["events"])
    s1 = collections.Counter(d["session"] for d in ix); s3 = collections.Counter(d["session"] for d in ix3)
    same_order = all(d["video"] == d3["video"] and d["span"] == d3["span"]
                     and [e for e in d["events"] if e[0] != "fall"] == d3["events"] for d, d3 in zip(ix, ix3))
    say(f"元＝{SRC}")
    say(f"曲＝元 {len(ix)}・写し {len(ix3)}・収録別 {'一致' if s1 == s3 else '不一致'}（{dict(sorted(s3.items()))}）")
    say(f"件数（208曲の全部）＝元 {sum(e1.values())}・写し {sum(e3.values())}（差 {sum(e1.values()) - sum(e3.values())}＝元の fall {e1['fall']}）")
    say("技法別（元→写し）＝" + "・".join(f"{c} {e1[c]}→{e3[c]}" for c in TYPES5))
    # 学習の曲（dev・試験 以外）の件数＝学習台本が読む件数（特徴の npz がある曲だけ）
    feats = lambda d: os.path.exists(os.path.join(ROOT, "feats", d["video"] + ".npz"))
    tr = lambda ixx: [d for d in ixx if d["session"] not in (DEV, TEST) and feats(d)]
    t1 = collections.Counter(c for d in tr(ix) for c, _ in d["events"]); t3 = collections.Counter(c for d in tr(ix3) for c, _ in d["events"])
    say(f"学習の曲＝{len(tr(ix3))}曲・5技法 {sum(t1.values())}件・フォール {t1['fall']}件・4技法 {sum(t3.values())}件"
        f"（検算 {sum(t1.values())}−{t1['fall']}＝{sum(t1.values()) - t1['fall']}・{'一致' if sum(t1.values()) - t1['fall'] == sum(t3.values()) else '不一致'}）")
    ok = (len(ix) == len(ix3) == 208 and s1 == s3 and e3["fall"] == 0 and all(e1[c] == e3[c] for c in TYPES4)
          and set(e3) <= set(TYPES4) and same_order and sum(e1.values()) - e1["fall"] == sum(e3.values())
          and sum(t1.values()) - t1["fall"] == sum(t3.values()))
    say(f"1件ずつの [技法, 時刻]・span・曲の順＝{'一致' if same_order else '不一致'}")
    for ses in (TEST, DEV):
        cc = collections.Counter(c for d in ix3 if d["session"] == ses for c, t in d["events"] if d["span"][0] <= t <= d["span"][1])
        c5 = collections.Counter(c for d in ix if d["session"] == ses for c, t in d["events"] if d["span"][0] <= t <= d["span"][1])
        say(f"{ses} の span 内＝" + "・".join(f"{c} {cc[c]}" for c in TYPES4) + f"・計 {sum(cc.values())}（5技法の時 {sum(c5.values())}・フォール {c5['fall']}）")
    say("判定＝" + ("OK" if ok else "NG"))
    open(os.path.join(OUT, "check_fall4tech_20261109.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    if not ok:
        raise SystemExit("止める＝件数が合わない")


if __name__ == "__main__":
    main()
