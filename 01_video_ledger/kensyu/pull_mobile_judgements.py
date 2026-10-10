# -*- coding: utf-8 -*-
"""スマホ検収の判定（Artifact db の kensyu/<sess> ドキュメント）をCSVに落とす。
   手順：ArtifactData get（url=検収ページ・path=kensyu/<sess>）の結果JSONを <sess>_db.json に保存 → python pull_mobile_judgements.py --session 20260924 --json 20260924_db.json
   出力＝kensyu_sample<MMDD>_<YYYYMMDDHHMM>.csv（列＝event_id,video,counter,count_value,judgement＝PC版と同じ）"""
import json, sys, argparse, datetime
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "G:/研究アーカイブ/成果物/動画解析"
ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--json", required=True); a = ap.parse_args()
d = json.load(open(a.json, encoding="utf-8")); d = d.get("data", d); J = d.get("j", d)
s = open(f"{ROOT}/review_sample_{a.session}_200.html", encoding="utf-8").read(); i = s.find("const EV=["); j = s.find("];", i); EV = json.loads(s[i + 9:j + 1])
rows = []; V = {"o": "OK", "x": "NG", "k": "HOLD"}
for e in EV:
    v = J.get(e["id"]); 
    if not v: continue
    rows.append(f"{e['id']},{e['video']},{e['c']},{e['n']},{V.get(v.get('v') if isinstance(v, dict) else v, 'HOLD')}")
out = f"{ROOT}/kensyu_sample{a.session[4:]}_{datetime.datetime.now():%Y%m%d%H%M}.csv"
open(out, "w", encoding="utf-8-sig").write("event_id,video,counter,count_value,judgement\n" + "\n".join(rows) + "\n")
c = {k: sum(1 for r in rows if r.endswith(k)) for k in ("OK", "NG", "HOLD")}; print(out, len(rows), "rows", c)
