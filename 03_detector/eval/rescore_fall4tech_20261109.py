# -*- coding: utf-8 -*-
r"""fall-4tech の数え直しと比べの表（2026-11-09 Opus・指示書 2026-11-09_1406・学習しない）。
rescore_step8_c4_20261105.py の作法の写し（8a・8b の決め方は同じ・照合は common_c4_20261105 の peaks と match）。
4技法（フォールを外して学習）＝out/fall_4tech_20261109/f4_global_8ep_s{0,1,2}/probs_{dev,test}.npz（学習台本が採用の重みで書いた確率の列）。
  今の境目＝metrics.json の thr（dev の F 最大）。8a＝dev（20260911・30曲）で出す数を正解の数に合わせた境目（技法ごと・0.01〜0.99・同じ差なら dev の F が高い方・なお同じなら低い境目）。
  8b＝試験の28曲の山を確率の高い順に正解の数と同じ件数だけ出した時の当てた数。許容±0.5秒。正解＝index_fall4tech_20261109（全体1つの遅れ）。
5技法まとめ（基準）＝種0 c3_5tech_global_20261105/c3_5tech_8ep_s0・種1・2 c4_compare_20261105/c4_global_8ep_s1・s2。
  今の境目＝各 metrics.json の test_tol0.5。8a・8b＝手順8_境目をそろえた数え直し_c4_2026-11-05.json の base・s1・s2（own・0.5）。読むだけ。
自己確認＝4技法の本は確率の列から今の境目で数えた 正解・出した・当てた が metrics.json と一致すること。合わなければ止める。
         4技法の試験の正解の数が5技法の同じ技法と一致すること（同じ正解からフォールを外しただけ）。
比べ＝技法ごとに5技法まとめ 種3本の幅（最小〜最大）と4技法 種3本の幅。重ならなければ 上／下・重なれば 差なし。合計＝フォールを除く4技法の和。
終わっていない本は飛ばす（中間の表にも使う）。
使い方: python rescore_fall4tech_20261109.py
出力＝out/fall_4tech_20261109/フォール外し_4技法_2026-11-09.md ・ フォール外し_4技法_2026-11-09.csv ・ .json
"""
import csv, json, os, re, sys, time
import numpy as np
from common_c4_20261105 import ROOT, peaks, match, row, DEV, TEST

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
T4 = ["shakuri", "kobushi", "accent", "hammering"]
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "accent": "アクセント", "hammering": "ハンマリング", "ALL": "4技法の合計（フォールを除く）"}
OUT = os.path.join(ROOT, "out", "fall_4tech_20261109")
IX4 = "out/fall_4tech_20261109/labels/index_fall4tech_20261109.json"
IX5 = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
S8 = os.path.join(ROOT, "out", "c4_compare_20261105", "手順8_境目をそろえた数え直し_c4_2026-11-05.json")
BASE = [  # 種, 手順8 の鍵, モデルの所
    (0, "base", os.path.join(ROOT, "out", "c3_5tech_global_20261105", "c3_5tech_8ep_s0")),
    (1, "s1", os.path.join(ROOT, "out", "c4_compare_20261105", "c4_global_8ep_s1")),
    (2, "s2", os.path.join(ROOT, "out", "c4_compare_20261105", "c4_global_8ep_s2")),
]
NEW = [(s, os.path.join(OUT, f"f4_global_8ep_s{s}")) for s in (0, 1, 2)]
FINE = [round(x, 2) for x in np.arange(0.01, 0.995, 0.01)]
CHAIN = os.path.join(OUT, "fall4tech_chain_20261109.log")


def truth4(index, session):
    out = {}
    for d in index:
        if d["session"] != session or not os.path.exists(os.path.join(ROOT, "feats", d["video"] + ".npz")):
            continue
        lo, hi = d["span"]
        out[d["video"]] = (d["span"], {c: [t for cc, t in d["events"] if cc == c and lo <= t <= hi] for c in T4 + ["fall"]})
    return out


def load_pk(path):
    z = np.load(path)
    return {v: [peaks(z[v][k]) for k in range(len(T4))] for v in z.files}


def count_thr(pk, truth, thr, tol, only=None):
    res = {}
    for c in (only or T4):
        k = T4.index(c); h = npd = nt = 0
        for v, (span, tr) in truth.items():
            lo, hi = span
            pr = [t for t, p in pk[v][k] if p >= thr[c] and lo <= t <= hi]
            h += match(pr, tr[c], tol); npd += len(pr); nt += len(tr[c])
        res[c] = row(h, npd, nt)
    return res


def choose_ii(pk, truth):
    thr = {}
    for c in T4:
        best = None
        for th in FINE:
            e = count_thr(pk, truth, {c: th}, 0.5, only=[c])[c]
            key = (abs(e["pred"] - e["true"]), -e["f1"], th)
            if best is None or key < best[0]:
                best = (key, th)
        thr[c] = best[1]
    return thr


def topn(pk, truth, tol):
    res = {}
    for c in T4:
        k = T4.index(c); cand = []
        for v, (span, tr) in truth.items():
            lo, hi = span
            cand += [(-p, v, t) for t, p in pk[v][k] if lo <= t <= hi]
        nt = sum(len(tr[c]) for _, tr in truth.values())
        cand.sort(); sel = {}
        for _, v, t in cand[:nt]:
            sel.setdefault(v, []).append(t)
        h = sum(match(sel.get(v, []), tr[c], tol) for v, (span, tr) in truth.items())
        res[c] = row(h, min(nt, len(cand)), nt)
    return res


def add_all(res):
    h = sum(res[c]["hit"] for c in T4); p = sum(res[c]["pred"] for c in T4); t = sum(res[c]["true"] for c in T4)
    res["ALL"] = row(h, p, t)
    return res


def chain_minutes(tag):
    if not os.path.exists(CHAIN):
        return None
    m = re.findall(rf"=== {re.escape(tag)} 終了 rc=(\S+) 所要 [\d.]+秒（([\d.]+)分）", open(CHAIN, encoding="utf-8", errors="replace").read())
    return (m[-1][0], float(m[-1][1])) if m else None


def rng(xs):
    return (min(xs), max(xs)) if xs else None


def verdict(r5, r4):
    if r5 is None or r4 is None:
        return "—"
    if r4[0] > r5[1]:
        return "上"
    if r4[1] < r5[0]:
        return "下"
    return "差なし"


def main():
    ix4, ix5 = json.load(open(os.path.join(ROOT, IX4), encoding="utf-8")), json.load(open(os.path.join(ROOT, IX5), encoding="utf-8"))
    tdev, ttest, ttest5 = truth4(ix4, DEV), truth4(ix4, TEST), truth4(ix5, TEST)
    for c in T4:   # 正解の数の確認（5技法の同じ技法と一致）
        n4 = sum(len(tr[c]) for _, tr in ttest.values()); n5 = sum(len(tr[c]) for _, tr in ttest5.values())
        if n4 != n5:
            raise SystemExit(f"止める＝試験の正解の数が5技法と違う {c} {n4}／{n5}")
    nfall4 = sum(len(tr["fall"]) for _, tr in ttest.values()); nfall5 = sum(len(tr["fall"]) for _, tr in ttest5.values())
    s8 = json.load(open(S8, encoding="utf-8"))
    # 5技法まとめ
    B = {}
    for seed, key, md in BASE:
        mj = json.load(open(os.path.join(md, "metrics.json"), encoding="utf-8"))
        now = {c: dict(mj["metrics"]["test_tol0.5"][c]) for c in T4 + ["fall"]}
        a8 = s8[key]["res"]["own"]["0.5"]["a"]; b8 = s8[key]["res"]["own"]["0.5"]["b"]
        B[seed] = dict(now=add_all({c: row(now[c]["hit"], now[c]["pred"], now[c]["true"]) for c in T4}),
                       a=add_all({c: row(a8[c]["hit"], a8[c]["pred"], a8[c]["true"]) for c in T4}),
                       b=add_all({c: row(b8[c]["hit"], b8[c]["pred"], b8[c]["true"]) for c in T4}),
                       fall=dict(now=now["fall"], a=a8["fall"], b=b8["fall"]),
                       all5=dict(now=mj["metrics"]["test_tol0.5"]["ALL"], a=a8["ALL"], b=b8["ALL"]),
                       best_ep=mj["best"][1], dev_f=mj["best"][0], min=round(mj["elapsed_sec"] / 60, 1), thr8a=s8[key]["thr"], md=md)
    # 4技法
    N, notes, waiting = {}, [], []
    for seed, md in NEW:
        if not all(os.path.exists(os.path.join(md, f)) for f in ("metrics.json", "probs_dev.npz", "probs_test.npz")):
            waiting.append(seed); continue
        mj = json.load(open(os.path.join(md, "metrics.json"), encoding="utf-8"))
        pkd, pkt = load_pk(os.path.join(md, "probs_dev.npz")), load_pk(os.path.join(md, "probs_test.npz"))
        now = count_thr(pkt, ttest, mj["thr"], 0.5)
        for c in T4:
            mm = mj["metrics"]["test_tol0.5"][c]
            if (now[c]["true"], now[c]["pred"], now[c]["hit"]) != (mm["true"], mm["pred"], mm["hit"]):
                raise SystemExit(f"止める＝自己確認の不一致 種{seed} {c} {now[c]} ／ {mm}")
        notes.append(f"種{seed}＝今の境目で metrics.json と一致（4技法）")
        thr = choose_ii(pkd, tdev)
        devc = add_all(count_thr(pkd, tdev, thr, 0.5))
        tag = os.path.basename(md); cm = chain_minutes(tag)
        N[seed] = dict(now=add_all(now), a=add_all(count_thr(pkt, ttest, thr, 0.5)), b=add_all(topn(pkt, ttest, 0.5)),
                       best_ep=mj["best"][1], dev_f=mj["best"][0], min=round(mj["elapsed_sec"] / 60, 1), chain_min=cm, thr8a=thr,
                       thr_now=mj["thr"], dev8a=devc, md=md)
        print(f"種{seed}: 今 " + " ".join(f"{c[:4]} {N[seed]['now'][c]['hit']}/{N[seed]['now'][c]['pred']}" for c in T4 + ["ALL"])
              + "｜8a " + " ".join(f"{c[:4]} {N[seed]['a'][c]['hit']}/{N[seed]['a'][c]['pred']}" for c in T4 + ["ALL"])
              + "｜8b " + " ".join(f"{c[:4]} {N[seed]['b'][c]['hit']}" for c in T4 + ["ALL"]), flush=True)

    # 比べ
    MODES = [("now", "今の境目"), ("a", "8a（出す数を正解の数に合わせる）"), ("b", "8b（境目に依らない数）")]
    CMP = {}
    for m, _ in MODES:
        CMP[m] = {}
        for c in T4 + ["ALL"]:
            r5 = rng([B[s][m][c]["hit"] for s in B]); r4 = rng([N[s][m][c]["hit"] for s in N])
            CMP[m][c] = dict(r5=r5, r4=r4, v=verdict(r5, r4) if len(N) == 3 else ("—（3本そろっていない）" if N else "—"))
    ntrue = {c: B[0]["a"][c]["true"] for c in T4 + ["ALL"]}
    stamp = time.strftime("%Y-%m-%d %H:%M")
    L = ["# フォール外し＝4技法のモデルと5技法まとめの比べ（2026-11-09・fall-4tech）", "", "**Opus 参考値・本人未検証**", "",
         f"作成 {stamp}。試験＝20260924 の28曲。許容±0.5秒。照合＝span の内側・技法ごとの貪欲一致（11/5 と同じ）。",
         "正解＝out/c2_shift3_global_20261105/labels（全体1つの遅れ 1.6722秒）から作った一覧。5技法まとめ＝index_c3_5tech_20261105・4技法＝index_fall4tech_20261109（フォールの件だけ外した写し）。",
         "学習＝学習150曲・調整30曲（20260911）・8エポック・採用＝調整の曲で F 最大のエポック。形＝畳み込み3層→双方向 GRU 2層・幅128。",
         "幅＝種0・1・2 の3本の当てた数の最小〜最大。重ならなければ 上／下（4技法が5技法まとめより）・重なれば 差なし。", ""]
    if waiting:
        L += ["まだ終わっていない4技法の本＝種" + "・".join(str(s) for s in waiting) + "（表に無い）", ""]
    L += ["## 比べの表（当てた数・種3本の幅）", "",
          "| 技法 | 正解の数 | 8a 5技法まとめ | 8a 4技法 | 8a 判定 | 8b 5技法まとめ | 8b 4技法 | 8b 判定 | 今の境目 5技法まとめ | 今の境目 4技法 | 今の境目 判定 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda r: "—" if not r else f"{r[0]}〜{r[1]}"
    for c in T4 + ["ALL"]:
        L.append(f"| {JP[c]} | {ntrue[c]} | " + " | ".join(f"{f(CMP[m][c]['r5'])} | {f(CMP[m][c]['r4'])} | {CMP[m][c]['v']}" for m in ("a", "b", "now")) + " |")
    L += ["", "主の列＝8a。8b＝試験の曲の正解の数を使う物差し（境目を選ばない比べ用）。今の境目＝調整の曲で F 最大の境目（出す数が本ごとに大きく揺れる）。",
          f"フォールの正解（試験）＝{nfall5}件（4技法の一覧では {nfall4}件）。5技法まとめのフォールの当てた数＝今の境目 " + "・".join(str(B[s]['fall']['now']['hit']) for s in B)
          + "／8a " + "・".join(str(B[s]['fall']['a']['hit']) for s in B) + "／8b " + "・".join(str(B[s]['fall']['b']['hit']) for s in B) + "（種0・1・2）。", ""]
    L += ["## 本ごと（当てた数／出した数）", "",
          "| 本 | 数え方 | " + " | ".join(JP[c] for c in T4) + " | 4技法の合計 |", "|---|---|" + "---|" * 5]
    for lab, D in (("5技法まとめ", B), ("4技法", N)):
        for s in sorted(D):
            for m, mn in MODES:
                e = D[s][m]
                L.append(f"| {lab} 種{s} | {mn} | " + " | ".join(f"{e[c]['hit']}／{e[c]['pred']}" for c in T4) + f" | {e['ALL']['hit']}／{e['ALL']['pred']} |")
    L += ["", "## 学習の記録", "", "| 本 | 採用エポック | 調整の曲の F（採用時） | 学習の分（特徴の読み込みを除く） | 連結ジョブで見た所要（分） | 所 |", "|---|---|---|---|---|---|"]
    for s in sorted(B):
        L.append(f"| 5技法まとめ 種{s} | {B[s]['best_ep']} | {B[s]['dev_f']} | {B[s]['min']} | — | {os.path.relpath(B[s]['md'], ROOT)} |")
    for s in sorted(N):
        cm = N[s]["chain_min"]
        L.append(f"| 4技法 種{s} | {N[s]['best_ep']} | {N[s]['dev_f']} | {N[s]['min']} | {cm[1] if cm else '—'} | {os.path.relpath(N[s]['md'], ROOT)} |")
    L += ["", "## 境目", ""]
    for s in sorted(N):
        L.append(f"- 4技法 種{s}＝今 " + "・".join(f"{JP[c]} {N[s]['thr_now'][c]:.2f}" for c in T4) + "／8a " + "・".join(f"{JP[c]} {N[s]['thr8a'][c]:.2f}" for c in T4)
                 + f"（8a の dev＝正解 {N[s]['dev8a']['ALL']['true']}・出した {N[s]['dev8a']['ALL']['pred']}）")
    for s in sorted(B):
        L.append(f"- 5技法まとめ 種{s}＝8a " + "・".join(f"{JP[c]} {B[s]['thr8a'][c]:.2f}" for c in T4) + "（手順8 の json から）")
    L += ["", "自己確認＝" + ("／".join(notes) if notes else "なし（4技法の本がまだ無い）") + "／試験の正解の数＝4技法すべて5技法の同じ技法と一致"]
    open(os.path.join(OUT, "フォール外し_4技法_2026-11-09.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "フォール外し_4技法_2026-11-09.csv"), "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo); w.writerow(["本", "種", "数え方", "技法", "正解の数", "出した数", "当てた数", "余計に出した数", "見落とした数"])
        for lab, D in (("5技法まとめ", B), ("4技法", N)):
            for s in sorted(D):
                for m, mn in MODES:
                    for c in T4 + ["ALL"]:
                        e = D[s][m][c]; w.writerow([lab, s, mn, JP[c], e["true"], e["pred"], e["hit"], e["extra"], e["miss"]])
    js = dict(made=stamp, waiting=waiting, cmp={m: {c: dict(r5=v["r5"], r4=v["r4"], v=v["v"]) for c, v in x.items()} for m, x in CMP.items()},
              base={s: {k: B[s][k] for k in ("now", "a", "b", "fall", "all5", "best_ep", "dev_f", "min", "thr8a")} for s in B},
              new={s: {k: N[s][k] for k in ("now", "a", "b", "best_ep", "dev_f", "min", "chain_min", "thr8a", "thr_now", "dev8a")} for s in N})
    json.dump(js, open(os.path.join(OUT, "フォール外し_4技法_2026-11-09.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for m, mn in MODES:
        print(mn + "＝" + "・".join(f"{JP[c][:4]} {f(CMP[m][c]['r5'])} 対 {f(CMP[m][c]['r4'])} {CMP[m][c]['v']}" for c in T4 + ["ALL"]))
    print("飛ばした（まだ）＝", waiting)


if __name__ == "__main__":
    main()
