# -*- coding: utf-8 -*-
"""読取器の版を**同条件で**比べる（2026-10-15）。

同じ帯動画・同じ ROI（`results_wao_<S>/<video>/roi.json`）で版だけを変えた読取結果のディレクトリを並べ、
(1) 採点CSVとの突合（PERFECT 数・過剰読み・欠落・誤差絶対値計・技法別 read/csv）と
(2) 先頭に並べた版との**増加イベント単位の差**（完全一致・時刻だけ動いた・消えた・増えた）を1枚の表にする。

既存の verify_*.csv・results_wao_<S>・台帳はいっさい書き換えない（読むだけ）。出力＝標準出力（Markdown 表）＋ --out で CSV。

使い方:
  python compare_readers_20261015.py 20261014 --machine wao \
      --dirs "v0.8d=results_wao_20261014.pre_v08e_20261015,v0.8e=results_wao_20261014,v0.9=results_wao_20261014_v09"
  python compare_readers_20261015.py 20261006 --machine wao \
      --dirs "v0.8d=results_wao_20261006,v0.8e=results_wao_20261006_v08e,v0.9=results_wao_20261006_v09"
"""
import os, sys, csv
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
WAO_M = {"shakuri": "shakuriCount", "kobushi": "kobushiCount", "fall": "fallCount",
         "vibrato": "vibratoCount", "accent": "accentCount", "hammering": "hammeringOnCount"}

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")


def get_cfg(session, machine):
    """セッションに応じた設定（採点CSV・rename_mapping・突合する技法）を返す。"""
    if session == "20261014":
        from kv_config_20261014 import Cfg1014
        C = Cfg1014(session)
        assert machine == "wao", "10/14 は WAO のみ"
        return C, WAO_M
    if session == "20261006":
        from kv_config_20261006 import Cfg1006, CfgAi1006
        C = Cfg1006(session) if machine == "wao" else CfgAi1006(session)
        return C, (WAO_M if machine == "wao" else C.csv_map)
    raise SystemExit("未対応のセッション: " + session)


def verify_rows(C, M, machine, resdir):
    d = pd.read_csv(C.dam_csv, encoding="utf-8-sig", low_memory=False)
    d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    ren = {r["renamed"][:-4]: r["played_at"] for r in C.mapping_rows()
           if r.get("played_at") and r.get("machine") == machine and r.get("how") == "TIME_WINDOW"}
    rows = []
    for name, pa in sorted(ren.items(), key=lambda kv: kv[1]):
        ce = os.path.join(resdir, name, "counter_events.csv")
        if not os.path.exists(ce):
            rows.append(dict(video=name, status="NOT_READ")); continue
        hit = d[d["dt"] == pd.Timestamp(pa)]
        if not len(hit):
            rows.append(dict(video=name, status="NO_CSV_ROW")); continue
        ev = pd.read_csv(ce)
        fin = {c: int(ev[ev["counter"] == c]["value"].max()) if (ev["counter"] == c).any() else 0 for c in M}
        row = hit.iloc[0]; diff = {c: fin[c] - int(row[M[c]]) for c in M}
        rows.append(dict(video=name, status="PERFECT" if all(v == 0 for v in diff.values()) else "DIFF",
                         **{f"read_{c}": fin[c] for c in M}, **{f"csv_{c}": int(row[M[c]]) for c in M},
                         diff=";".join(f"{c}{v:+d}" for c, v in diff.items() if v)))
    return pd.DataFrame(rows)


def events(resdir):
    """増加イベントの集合＝{(video, counter, value): event_frame_est}（delta>0 の行だけ）。"""
    out = {}
    for v in sorted(os.listdir(resdir)):
        p = os.path.join(resdir, v, "counter_events.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if int(r["delta"]) > 0:
                out[(v, r["counter"], int(r["value"]))] = int(r["event_frame_est"])
    return out


def main():
    session = sys.argv[1]
    machine = sys.argv[sys.argv.index("--machine") + 1] if "--machine" in sys.argv else "wao"
    spec = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--dirs=")), None)
    if spec is None and "--dirs" in sys.argv:
        spec = sys.argv[sys.argv.index("--dirs") + 1]
    out_csv = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--out=")), "")
    assert spec, "--dirs が必要"
    pairs = []
    for part in spec.split(","):
        lab, d = part.split("=", 1)
        pairs.append((lab.strip(), os.path.join(ROOT, d.strip()) if not os.path.isabs(d.strip()) else d.strip()))
    C, M = get_cfg(session, machine)
    counters = list(M)

    V = {}
    for lab, d in pairs:
        assert os.path.isdir(d), "ディレクトリが無い: " + d
        V[lab] = verify_rows(C, M, machine, d)
        n_read = int((V[lab]["status"] != "NOT_READ").sum())
        print(f"{lab:8s} {os.path.basename(d):42s} 読取 {n_read} 曲")

    print(f"\n### {session} {machine}：版ごとの突合（同じ帯動画・同じ ROI）\n")
    print("| 版 | PERFECT | DIFF | 過剰読み（偽陽性） | 欠落 | 誤差絶対値計 | 増加イベント数 |")
    print("|---|---|---|---|---|---|---|")
    base = pairs[0][0]
    summ = []
    for lab, d in pairs:
        df = V[lab]; sub = df[df["status"] != "NOT_READ"]
        over = sum((sub[f"read_{c}"] - sub[f"csv_{c}"]).clip(lower=0).sum() for c in counters)
        under = sum((sub[f"csv_{c}"] - sub[f"read_{c}"]).clip(lower=0).sum() for c in counters)
        ab = sum((sub[f"read_{c}"] - sub[f"csv_{c}"]).abs().sum() for c in counters)
        nev = len(events(d))
        print(f"| {lab} | **{int((df['status']=='PERFECT').sum())}/{len(sub)}** | "
              f"{int((df['status']=='DIFF').sum())} | +{int(over)} | −{int(under)} | {int(ab)} | {nev} |")
        summ.append(dict(session=session, machine=machine, reader=lab, dir=os.path.basename(d),
                         perfect=int((df["status"] == "PERFECT").sum()), n=len(sub),
                         over=int(over), under=int(under), abs_err=int(ab), events=nev))

    print(f"\n### 技法別 read/csv（過剰読み／欠落）\n")
    print("| 技法 | " + " | ".join(lab for lab, _ in pairs) + " |")
    print("|---" * (len(pairs) + 1) + "|")
    for c in counters:
        cells = []
        for lab, _ in pairs:
            sub = V[lab][V[lab]["status"] != "NOT_READ"]
            r, cv = int(sub[f"read_{c}"].sum()), int(sub[f"csv_{c}"].sum())
            ov = int((sub[f"read_{c}"] - sub[f"csv_{c}"]).clip(lower=0).sum())
            un = int((sub[f"csv_{c}"] - sub[f"read_{c}"]).clip(lower=0).sum())
            cells.append(f"{r}/{cv}={r/cv:.3f}（+{ov}／−{un}）" if cv else f"{r}/0（+{ov}）")
        print(f"| {c} | " + " | ".join(cells) + " |")

    print(f"\n### 増加イベント単位の差（基準＝{base}）\n")
    print("| 比較 | 完全一致 | 時刻だけ動いた | 消えた | 増えた |")
    print("|---|---|---|---|---|")
    E0 = events(pairs[0][1])
    for lab, d in pairs[1:]:
        E = events(d)
        same = sum(1 for k, v in E0.items() if k in E and E[k] == v)
        moved = sum(1 for k, v in E0.items() if k in E and E[k] != v)
        gone = sum(1 for k in E0 if k not in E)
        added = sum(1 for k in E if k not in E0)
        print(f"| {base} → {lab} | {same} | {moved} | {gone} | {added} |")
        summ.append(dict(session=session, machine=machine, reader=f"{base}->{lab}", dir="(events)",
                         perfect="", n="", over="", under="", abs_err="",
                         events=f"same={same};moved={moved};gone={gone};added={added}"))

    print(f"\n### 曲ごとの誤差絶対値計（0＝PERFECT）\n")
    print("| 曲 | " + " | ".join(lab for lab, _ in pairs) + " |")
    print("|---" * (len(pairs) + 1) + "|")
    for name in V[base]["video"]:
        row = V[base][V[base]["video"] == name].iloc[0]
        if row["status"] == "NOT_READ":
            continue
        cells = []
        for lab, _ in pairs:
            q = V[lab][V[lab]["video"] == name]
            if not len(q) or q.iloc[0]["status"] == "NOT_READ":
                cells.append("—"); continue
            q = q.iloc[0]
            cells.append(str(sum(abs(int(q[f"read_{c}"]) - int(q[f"csv_{c}"])) for c in counters)))
        if len(set(cells)) > 1 or cells[0] != "0":
            print(f"| {name.split('_2026')[0]} | " + " | ".join(cells) + " |")

    if out_csv:
        pd.DataFrame(summ).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("\nWROTE", out_csv)


if __name__ == "__main__":
    main()
