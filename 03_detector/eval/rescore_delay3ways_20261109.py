# -*- coding: utf-8 -*-
r"""delay-3ways の数え直しと表（2026-11-09 Opus・学習しない）。数え方＝11/5 手順8 と同じ（common_delay3ways）。
本＝今（全体1つの遅れ・種0／1／2＝11/5 の3本）・案B（規則の対応付けの正解・種0）・案A（弱いラベル・種0／1／2）・案D（反復の置き直し・1〜3回目・種0）。
数える正解＝全体1つの遅れの正解（glo＝1.6722秒を引いた時刻＝今と同じ物差し）が主。案B は自分の正解（own）でも数える。
  今の境目＝metrics.json の thr（dev の F 最大）。8a＝dev（自分が学習した正解）で出した数を正解の数に合わせた境目。8b＝確率の高い順に正解の数だけ。
  許容＝±0.5秒（主）と ±1.0秒。
自己確認＝今の境目・自分の正解・±0.5秒の 正解・出した・当てた が metrics.json と一致（合わなければ止める）。
遅れの分布＝今＝固定 1.6722。案A＝学習曲で [tc−3, tc] の中の確率の最大の時刻と tc の差（学習が選んだ遅れ）。案B＝対応した件の tc−t候補。案D＝その回の正解の tc−t（置き直した件と全件）。
  参考＝今の種0 にも案A と同じ取り方（区間の中の最大）を当てた値を付ける。
使い方: python rescore_delay3ways_20261109.py [--interim N --note "…"]
出力＝out/delay_3ways_20261109/遅れの3案_2026-11-09.md・.json（・--interim の時は routine_inbox/2026-11-09_HHMM_delay-3ways_中間N.md）
"""
import argparse, json, os, sys, time
import numpy as np
from common_delay3ways_20261109 import ROOT, TYPES, JP, DEV, TEST, FR, SHIFT, IXG, OUT, truth_of, load_index, pk_of, count_thr, choose_ii, topn, qstat

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
INBOX = "G:/hisho/routine_inbox" if os.name == "nt" else "/mnt/g/hisho/routine_inbox"
C4P = os.path.join(ROOT, "out", "c4_compare_20261105", "probs")
L = "out/delay_3ways_20261109/labels/"
RUNS = [  # key, 名, 群, dev の確率, 試験の確率, 学習曲の確率, モデルの所在, 自分の正解
    ("now0", "今 種0", "今", os.path.join(C4P, "probs_base_dev.npz"), os.path.join(C4P, "probs_base_test.npz"), os.path.join(OUT, "probs", "probs_base_all.npz"),
     os.path.join(ROOT, "out", "c3_5tech_global_20261105", "c3_5tech_8ep_s0"), IXG),
    ("now1", "今 種1", "今", os.path.join(C4P, "probs_s1_dev.npz"), os.path.join(C4P, "probs_s1_test.npz"), None,
     os.path.join(ROOT, "out", "c4_compare_20261105", "c4_global_8ep_s1"), IXG),
    ("now2", "今 種2", "今", os.path.join(C4P, "probs_s2_dev.npz"), os.path.join(C4P, "probs_s2_test.npz"), None,
     os.path.join(ROOT, "out", "c4_compare_20261105", "c4_global_8ep_s2"), IXG),
]
for key, nm, grp, tag, ix in [("A0", "案A 種0", "A", "dA_weak_8ep_s0", IXG), ("A1", "案A 種1", "A", "dA_weak_8ep_s1", IXG), ("A2", "案A 種2", "A", "dA_weak_8ep_s2", IXG),
                              ("Ax", "案A 変種（区間の重さ 30）種0", "Ax", "dA_weak_wp30_8ep_s0", IXG),
                              ("B", "案B 種0", "B", "dB_rule_8ep_s0", L + "index_B_rule_20261109.json"),
                              ("D1", "案D 1回目", "D", "dD_r1_8ep_s0", L + "index_D_r1_20261109.json"), ("D2", "案D 2回目", "D", "dD_r2_8ep_s0", L + "index_D_r2_20261109.json"),
                              ("D3", "案D 3回目", "D", "dD_r3_8ep_s0", L + "index_D_r3_20261109.json")]:
    md = os.path.join(OUT, tag)
    RUNS.append((key, nm, grp, os.path.join(md, "probs_dev.npz"), os.path.join(md, "probs_test.npz"), os.path.join(md, "probs_train.npz"), md, ix))
ORDER = ["now0", "now1", "now2", "A0", "A1", "A2", "Ax", "B", "D1", "D2", "D3"]


def learned_delay(probs_train, glo):
    """学習曲の各加点で [tc−3, tc] の中の確率の最大の時刻と tc の差（技法別）。"""
    z = np.load(probs_train); out = {c: [] for c in TYPES}
    for d in glo:
        if d["session"] in (DEV, TEST) or d["video"] not in z.files:
            continue
        P = z[d["video"]]; T = P.shape[1]
        for c, t in d["events"]:
            if c not in TYPES:
                continue
            k = TYPES.index(c); tc = t + SHIFT
            f0, f1 = max(0, int(round((tc - 3.0) * FR))), min(T - 1, int(round(tc * FR)))
            if f1 <= f0:
                continue
            f = f0 + int(np.argmax(P[k, f0:f1 + 1])); out[c].append(tc - f / FR)
    res = {c: qstat(out[c]) for c in TYPES}; res["ALL"] = qstat(sum(out.values(), []))
    return res, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--interim", type=int, default=0); ap.add_argument("--note", default="")
    a = ap.parse_args(); t0 = time.time()
    glo = load_index(IXG); tg_dev, tg_test = truth_of(glo, DEV), truth_of(glo, TEST)
    R, notes, raw_delay = {}, [], {}
    for key, nm, grp, pdv, pts, ptr, md, ix in RUNS:
        if not (os.path.exists(pdv) and os.path.exists(pts) and os.path.exists(os.path.join(md, "metrics.json"))):
            continue
        own = load_index(ix); to_dev, to_test = truth_of(own, DEV), truth_of(own, TEST)
        mj = json.load(open(os.path.join(md, "metrics.json"), encoding="utf-8"))
        pkd, pkt = pk_of(np.load(pdv)), pk_of(np.load(pts))
        thr0 = mj["thr"]
        e = count_thr(pkt, to_test, thr0, 0.5)
        for c in TYPES:
            mm = mj["metrics"]["test_tol0.5"][c]
            if (e[c]["true"], e[c]["pred"], e[c]["hit"]) != (mm["true"], mm["pred"], mm["hit"]):
                raise SystemExit(f"止める＝自己確認の不一致 {key} {c} {e[c]} ／ {mm}")
        notes.append(f"{key}＝今の境目で metrics.json と一致（5技法）")
        thr8 = choose_ii(pkd, to_dev)
        res = {}
        for which, tt in (("glo", tg_test), ("own", to_test)):
            if which == "own" and ix == IXG:
                continue
            res[which] = {str(tol): dict(cur=count_thr(pkt, tt, thr0, tol), a=count_thr(pkt, tt, thr8, tol), b=topn(pkt, tt, tol)) for tol in (0.5, 1.0)}
        x = dict(name=nm, grp=grp, thr_cur=thr0, thr8a=thr8, dev8a=count_thr(pkd, to_dev, thr8, 0.5), res=res, best_ep=mj["best"][1], elapsed=mj.get("elapsed_sec"))
        if ptr and os.path.exists(ptr):
            x["learned_delay"], raw_delay[key] = learned_delay(ptr, glo)
        R[key] = x
        g = res["glo"]["0.5"]
        print(f"{key}: 今の境目 {g['cur']['ALL']['hit']}/{g['cur']['ALL']['pred']} 8a {g['a']['ALL']['hit']}/{g['a']['ALL']['pred']} 8b {g['b']['ALL']['hit']}", flush=True)
    # 遅れの分布（案ごと）
    DL = {"now": {c: dict(n=None, median=SHIFT, q1=SHIFT, q3=SHIFT) for c in TYPES + ["ALL"]}}
    pb = os.path.join(OUT, "B", "summary_rule_20261109.json")
    if os.path.exists(pb):
        sb = json.load(open(pb, encoding="utf-8")); DL["B"] = sb["all"]; DL["B_model"] = None
        pm = os.path.join(OUT, "B", "summary_model_20261109.json")
        if os.path.exists(pm):
            DL["B_model"] = json.load(open(pm, encoding="utf-8"))["all"]
    for r in (1, 2, 3):
        p = os.path.join(OUT, "D", f"relabel_r{r}_20261109.json")
        if os.path.exists(p):
            DL[f"D{r}"] = json.load(open(p, encoding="utf-8"))
    json.dump(dict(R=R, DL=DL, notes=notes), open(os.path.join(OUT, "遅れの3案_2026-11-09.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({k: {c: [round(x, 3) for x in v] for c, v in d.items()} for k, d in raw_delay.items()},
              open(os.path.join(OUT, "learned_delay_raw_20261109.json"), "w", encoding="utf-8"))

    def dstr(e):
        if not e or e.get("median") is None:
            return "—"
        return f"中央 {e['median']:.2f}・四分位 {e['q1']:.2f}〜{e['q3']:.2f}"

    def delay_of(key):
        if key.startswith("now"):
            return "固定 1.6722" + (f"（参考＝区間の中の最大 {dstr(R[key]['learned_delay']['ALL'])}）" if "learned_delay" in R[key] else "")
        if key.startswith("A"):
            return "学習が選んだ " + dstr(R[key].get("learned_delay", {}).get("ALL"))
        if key == "B":
            e = DL.get("B", {}).get("ALL")
            return f"対応した件 {dstr(e)}（対応 {e['rate'] * 100:.1f}%）" if e else "—"
        if key.startswith("D"):
            e = DL.get(key, {}).get("per", {}).get("ALL")
            return f"山へ置いた件 {dstr(e['delay_moved'])}（{e['rate'] * 100:.1f}%）・全件 {dstr(e['delay_all'])}" if e else "—"
        return "—"

    T = ["# 遅れの3案＝今・案A・案B・案D の比べ（2026-11-09）", "", "**Opus 参考値・本人未検証**", "",
         "試験＝20260924 の28曲。数える正解＝全体1つの遅れ（1.6722秒を引いた時刻＝今と同じ物差し）。照合＝span の内・技法ごとの貪欲一致。正解の数 982。",
         "今の境目＝dev の F 最大（学習の時の境目）。8a＝dev で出した数を正解の数に合わせた境目（主）。8b＝確率の高い順に正解の数だけ出した時（境目を選ばない比べ用）。", ""]
    T += ["## 当てた数（±0.5秒・主）", "", "| 本 | 今の境目 当てた／出した | 8a 当てた／出した | 8b 当てた | ±1.0秒 8a 当てた | 採用エポック | 遅れの分布（秒） |", "|---|---|---|---|---|---|---|"]
    for k in ORDER:
        if k not in R:
            continue
        x = R[k]; g = x["res"]["glo"]
        T.append(f"| {x['name']} | {g['0.5']['cur']['ALL']['hit']}／{g['0.5']['cur']['ALL']['pred']} | {g['0.5']['a']['ALL']['hit']}／{g['0.5']['a']['ALL']['pred']} | "
                 f"{g['0.5']['b']['ALL']['hit']} | {g['1.0']['a']['ALL']['hit']} | {x['best_ep']} | {delay_of(k)} |")
    # 群ごとの幅
    T += ["", "## 群ごとの幅（8a・±0.5秒・当てた数）", ""]
    for grp in ("今", "A", "B", "D"):
        hs = [R[k]["res"]["glo"]["0.5"]["a"]["ALL"]["hit"] for k in ORDER if k in R and R[k]["grp"] == grp]
        hb = [R[k]["res"]["glo"]["0.5"]["b"]["ALL"]["hit"] for k in ORDER if k in R and R[k]["grp"] == grp]
        if hs:
            T.append(f"- {grp}＝8a {min(hs)}〜{max(hs)}（{len(hs)}本）・8b {min(hb)}〜{max(hb)}")
    T += ["", "## 技法ごと（8a・±0.5秒・当てた／出した）", "", "| 本 | " + " | ".join(JP[c] for c in TYPES) + " |", "|---|" + "---|" * 5]
    for k in ORDER:
        if k in R:
            e = R[k]["res"]["glo"]["0.5"]["a"]
            T.append(f"| {R[k]['name']} | " + " | ".join(f"{e[c]['hit']}／{e[c]['pred']}" for c in TYPES) + " |")
    if "B" in R:
        o = R["B"]["res"]["own"]
        T += ["", f"案B を自分の正解（対応した件＝規則の時刻）で数えた参考＝今の境目 {o['0.5']['cur']['ALL']['hit']}／{o['0.5']['cur']['ALL']['pred']}・8a {o['0.5']['a']['ALL']['hit']}／{o['0.5']['a']['ALL']['pred']}・8b {o['0.5']['b']['ALL']['hit']}"]
    T += ["", "## 遅れの分布（技法別・秒・tc−t）", "", "| 案 | " + " | ".join(JP[c] for c in TYPES) + " | 合計 |", "|---|" + "---|" * 6]
    if "B" in DL:
        T.append("| 案B 規則の候補（対応した件） | " + " | ".join(f"{dstr(DL['B'][c])}（{(DL['B'][c]['rate'] or 0) * 100:.0f}%）" for c in TYPES + ["ALL"]) + " |")
    if DL.get("B_model"):
        T.append("| 案B 今のモデル種0 の山（対応した件） | " + " | ".join(f"{dstr(DL['B_model'][c])}（{(DL['B_model'][c]['rate'] or 0) * 100:.0f}%）" for c in TYPES + ["ALL"]) + " |")
    for k in ("now0", "A0", "A1", "A2", "Ax"):
        if k in R and "learned_delay" in R[k]:
            T.append(f"| {R[k]['name']} 区間の中の最大 | " + " | ".join(dstr(R[k]["learned_delay"][c]) for c in TYPES + ["ALL"]) + " |")
    for r in (1, 2, 3):
        if f"D{r}" in DL:
            p = DL[f"D{r}"]["per"]
            T.append(f"| 案D {r}回目 山へ置いた件 | " + " | ".join(f"{dstr(p[c]['delay_moved'])}（{(p[c]['rate'] or 0) * 100:.0f}%）" for c in TYPES + ["ALL"]) + " |")
    T += ["", "8a の境目", ""]
    for k in ORDER:
        if k in R:
            T.append(f"- {R[k]['name']}＝" + "・".join(f"{JP[c]} {R[k]['thr8a'][c]:.2f}" for c in TYPES) + f"（dev の正解 {R[k]['dev8a']['ALL']['true']}・出した {R[k]['dev8a']['ALL']['pred']}）")
    T += ["", "自己確認＝" + "／".join(notes), f"所要 {time.time() - t0:.0f}秒"]
    open(os.path.join(OUT, "遅れの3案_2026-11-09.md"), "w", encoding="utf-8").write("\n".join(T) + "\n")
    if a.interim:
        hm = time.strftime("%H%M")
        I = ["<!-- unread -->", f"# delay-3ways 中間{a.interim}（{time.strftime('%Y-%m-%d %H:%M')}・Opus）", ""]
        if a.note:
            I += [x for x in a.note.split("|")] + [""]
        for nmB, kB in (("規則の候補", "B"), ("今のモデル種0 の山", "B_model")):
            if DL.get(kB):
                I.append(f"案B 対応付け（{nmB}）＝" + "・".join(f"{JP[c]} 対応 {(DL[kB][c]['rate'] or 0) * 100:.0f}% 遅れ {dstr(DL[kB][c])}" for c in TYPES + ["ALL"]))
        I += ["", "数＝試験28曲・全体1つの遅れの正解・±0.5秒・当てた数（正解 982）。", "Opus 参考値・本人未検証。", ""]
        for k in ORDER:
            if k in R:
                g = R[k]["res"]["glo"]["0.5"]
                I.append(f"{R[k]['name']}＝今の境目 {g['cur']['ALL']['hit']}・8a {g['a']['ALL']['hit']}・8b {g['b']['ALL']['hit']}・遅れ {delay_of(k)}")
        I += ["", "表＝out/delay_3ways_20261109/遅れの3案_2026-11-09.md"]
        open(os.path.join(INBOX, f"2026-11-09_{hm}_delay-3ways_中間{a.interim}.md"), "w", encoding="utf-8").write("\n".join(I) + "\n")
        print("中間報告＝", os.path.join(INBOX, f"2026-11-09_{hm}_delay-3ways_中間{a.interim}.md"))
    print(f"所要 {time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()
