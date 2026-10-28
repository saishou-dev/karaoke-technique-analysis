# -*- coding: utf-8 -*-
"""C2（検出器）の学習曲線用に、学習曲を 25%・50%・75% へ減らした index ファイルを作る（2026-10-28）。

なぜ必要か。
「目標200曲・1万件は適正か」を客観的に答えるには、学習データ量だけを変えて
試験F値がまだ伸びているか頭打ちかを見る（学習曲線）。
train_crnn_refdiff_20261023.py には曲の部分集合を指定する引数が無いが、
--index で読む index ファイルを差し替えられる（分割は index の session 列から決まる）ため、
学習曲だけを間引いた index を別名で作って渡す＝学習台本は1行も書き換えない。

決め方。
  学習曲＝session が dev(20260911)・test(20260924) 以外の曲（既定で150本）。
  並びを video 名でそろえてから random.Random(0) で1回だけ混ぜ、先頭から n 本を取る
  ＝25% ⊂ 50% ⊂ 75% ⊂ 100% の入れ子になる（小さい集合は大きい集合の部分集合）。
  dev 30本・test 28本はそのまま全部入れる（しきい値の決め方と未見の試験を基準と同じに保つ）。

出力（out/c2_learning_curve_20261028/）。
  index_lc25.json / index_lc50.json / index_lc75.json ＝学習台本へ渡す index
  学習量の内訳_2026-10-28.csv ＝条件ごとの学習曲数・学習イベント件数・技法別の件数
  学習曲の割り当て_2026-10-28.csv ＝曲ごとにどの条件から入るか（再現用）

使い方:
  python make_c2_lc_index_20261028.py
"""
import csv
import json
import os
import random
import sys

if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "out", "c2_learning_curve_20261028")
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
DEV, TEST = "20260911", "20260924"
FRACS = [(25, 38), (50, 75), (75, 113)]  # (表示する割合, 本数) ＝150本の 25.3%・50.0%・75.3%
SEED = 0


def count_events(entries):
    """このまとまりの技法イベント件数（全部・歌っている区間の中だけ）と技法別。"""
    tot = 0
    inspan = 0
    per = {c: 0 for c in TYPES}
    for d in entries:
        lo, hi = d["span"]
        for c, t in d["events"]:
            if c not in TYPES:
                continue
            tot += 1
            if lo <= t <= hi:
                inspan += 1
                per[c] += 1
    return tot, inspan, per


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    index = json.load(open(os.path.join(ROOT, "index.json"), encoding="utf-8"))
    # 学習台本と同じ条件＝特徴量ファイルがある曲だけを数える（無い曲は台本側で読み飛ばされる）。
    index = [d for d in index if os.path.exists(os.path.join(ROOT, "feats", d["video"] + ".npz"))]
    held = [d for d in index if d["session"] in (DEV, TEST)]
    train = [d for d in index if d["session"] not in (DEV, TEST)]
    train.sort(key=lambda d: d["video"])
    order = list(train)
    random.Random(SEED).shuffle(order)
    print(f"index {len(index)}本（学習 {len(train)}・dev+test {len(held)}）seed={SEED}")

    rank = {d["video"]: i for i, d in enumerate(order)}
    rows = []
    conds = []
    for pct, n in FRACS:
        sub = order[:n]
        path = os.path.join(OUTDIR, f"index_lc{pct}.json")
        # 並びはもとの index.json と同じ順に戻す（読み込み順で結果が揺れないように）。
        keep = set(d["video"] for d in sub)
        out = [d for d in index if d["session"] in (DEV, TEST) or d["video"] in keep]
        json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        tot, inspan, per = count_events(sub)
        conds.append((pct, n, tot, inspan, per, path))
        print(f"  {pct}% → 学習 {n}本（{n / len(train) * 100:.1f}%）"
              f" イベント {tot}件（区間内 {inspan}件） → {os.path.basename(path)}（全 {len(out)}本）")
    tot, inspan, per = count_events(train)
    conds.append((100, len(train), tot, inspan, per, os.path.join(ROOT, "index.json")))
    print(f"  100% → 学習 {len(train)}本 イベント {tot}件（区間内 {inspan}件） → index.json（基準・作り直さない）")

    with open(os.path.join(OUTDIR, "学習量の内訳_2026-10-28.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["条件", "学習曲数", "学習曲の割合", "学習イベント件数", "区間内の件数"] + TYPES + ["index"])
        for pct, n, tot, inspan, per, path in conds:
            w.writerow([f"{pct}%", n, f"{n / len(train) * 100:.1f}%", tot, inspan]
                        + [per[c] for c in TYPES] + [os.path.basename(path)])

    with open(os.path.join(OUTDIR, "学習曲の割り当て_2026-10-28.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["video", "session", "混ぜた後の順番", "25%", "50%", "75%", "100%"])
        for d in sorted(train, key=lambda d: rank[d["video"]]):
            r = rank[d["video"]]
            w.writerow([d["video"], d["session"], r] + [1 if r < n else 0 for _, n in FRACS] + [1])
    print("DONE", OUTDIR)


if __name__ == "__main__":
    main()
