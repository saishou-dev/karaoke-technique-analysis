# -*- coding: utf-8 -*-
"""close4（2026-10-30）の正式化＝RUNBOOK 5-c の 3（退避→新しい結果を results_wao_<S> へ写す→md5 で同一性を確認）。
   使い方: python formalize_close4_20261030.py <S> <対応表.csv（列 video,src_dir）>
   退避＝results_wao_<S>/_close4_20261030/pre_close4_backup/<video>/（4ファイル）。台帳＝_close4_20261030/formalize_manifest.csv。
"""
import os, sys, csv, shutil, hashlib
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; S = sys.argv[1]
RES = os.path.join(ROOT, f"results_wao_{S}"); C4 = os.path.join(RES, "_close4_20261030"); BK = os.path.join(C4, "pre_close4_backup")
FILES = ("counter_events.csv", "counter_events.json", "read_log.txt", "roi.json")
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else ""
rows = list(csv.DictReader(open(sys.argv[2], encoding="utf-8-sig"))); man = []
for r in rows:
    v, src = r["video"], r["src_dir"]; dst = os.path.join(RES, v); bk = os.path.join(BK, v)
    os.makedirs(bk, exist_ok=True)
    for f in FILES:
        if os.path.exists(os.path.join(dst, f)) and not os.path.exists(os.path.join(bk, f)): shutil.copy2(os.path.join(dst, f), os.path.join(bk, f))
    for f in FILES:
        a, b = os.path.join(src, v, f), os.path.join(dst, f)
        old = md5(os.path.join(bk, f))
        if os.path.abspath(src) != os.path.abspath(RES) and os.path.exists(a): shutil.copy2(a, b)
        man.append(dict(video=v, file=f, src=src, md5_src=md5(a) if os.path.exists(a) else md5(b), md5_dst=md5(b), md5_before=old, same_as_src=(md5(a) == md5(b)) if os.path.exists(a) else True, changed=(old != md5(b))))
with open(os.path.join(C4, "formalize_manifest.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(man[0].keys())); w.writeheader(); w.writerows(man)
ce = [m for m in man if m["file"] == "counter_events.csv"]
print(f"videos={len(rows)} counter_events 写しの md5 一致={sum(m['same_as_src'] for m in ce)}/{len(ce)} 中身が変わった曲={sum(m['changed'] for m in ce)} 退避={BK}")
