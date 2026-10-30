#!/usr/bin/env python3
"""fix_b_check24.py ＝ 候補B の補足：高さ下限を24にした時、(ii) の正解確定コマがどこで止まるか（2026-10-30）
段（fix_b_boxes.STAGES）と、箱が取れたコマの「いちばん低い箱の最大類似度」の分布を、しゃくり／それ以外で数える。"""
import collections, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_b_boxes as fb, fix_common as fc, glyph_core as gc, unread_sweep as us
st = fc.load_setup(); songs = st["songs"]; rois = fc.load_rois(fc.WORK, songs); tpl = gc.Templates(fc.TPL)
cat = {}
with open(fc.CELLS47, encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        cat[(r["video"], int(r["sample_idx"]), r["counter"])] = r["category"]
P = dict(fb.DEFAULT); P["hmin"] = 24
cnt = collections.Counter()
for k in st["gt"]:
    if not (st["cert"][k] and cat[k].startswith("(ii)")):
        continue
    s, i, c = k; grp = "しゃくり" if c == "shakuri" else "それ以外"
    g = rois[(s, c)][i].max(axis=2)
    res, sl, _ = fb.glyph_boxes_p(g, P, diag=True)
    if res is None:
        cnt[(grp, "箱なし:" + fb.STAGES[max(x[0] for x in sl)])] += 1
        continue
    ndig = 1 if st["gt"][k] < 10 else 2
    if len(res) != ndig:
        cnt[(grp, "箱の数が桁数と違う")] += 1
        continue
    sc_ = []
    labs = []
    for b in res:
        u = fc.unit(gc.norm_glyph(g, b))
        if u is None:
            sc_.append(-1); labs.append(None); continue
        v = tpl.reps_flat @ u; bi = int(np.argmax(v)); sc_.append(float(v[bi])); labs.append(tpl.labels.get(bi))
    m = min(sc_)
    b = "0.75以上" if m >= 0.75 else ("0.60〜0.75" if m >= 0.60 else "0.60未満")
    vv = None if None in labs else (labs[0] if len(labs) == 1 else labs[0] * 10 + labs[1])
    cnt[(grp, f"箱の数は合う・類似度{b}・最も似た見本の値が{'正解' if vv == st['gt'][k] else '正解でない'}")] += 1
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "B_check_hmin24.csv"), "w", encoding="utf-8", newline="") as fo:
    w = csv.writer(fo); w.writerow(["group", "outcome", "certain_cells"])
    for (g_, o), n in sorted(cnt.items()): w.writerow([g_, o, n]); print(g_, o, n)
