# -*- coding: utf-8 -*-
"""スマホ検収ページの生成（2026-09-30・本人指示「検収をスマホでも。OK＝左スワイプ／NG＝右スワイプ／その他は画面下部のボタン／CSVはページに残る判定をClaudeが読み取って保存」）。
   入力＝review_sample_<sess>_200.html（EV配列・基準文・ピッチ描画関数を流用）
   出力＝mobile_review/<sess>/index.html ＋ clips/<id>.mp4（640px・30fps・crf28・無音）＋ files.json（Artifact公開用のファイル一覧）
   公開＝Artifact（claude.ai）に index.html＋clips を publish・capabilities={db:{},user:{}}。判定は db の kensyu/<sess> に {j:{id:{v,t}}} で残る。
   使い方: python build_mobile_review.py --session 20260924 [--sample review_sample_20260924_200.html] [--no-encode]"""
import os, re, json, sys, argparse, subprocess
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "G:/研究アーカイブ/成果物/動画解析"
ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--sample", default=""); ap.add_argument("--no-encode", action="store_true")
a = ap.parse_args()
sample = a.sample or f"review_sample_{a.session}_200.html"
s = open(f"{ROOT}/{sample}", encoding="utf-8").read()
i = s.find("const EV=["); j = s.find("];", i); EV = json.loads(s[i + 9:j + 1])
i = s.find('<div id="foot">'); j = s.find("</div>", i); foot = re.sub("<[^>]+>", "", s[i:j]).replace("基準（", "基準（", 1)
foot = foot.split("　J○")[0]  # キー説明は除く
i = s.find("function drawPitch("); j = s.find("\nfunction ", i + 10); draw = s[i:j]
i = s.find("const COL="); j = s.find("\n", i); col = s[i:j]
i = s.find("const NAME="); j = s.find("\n", i); name = s[i:j]
out = f"{ROOT}/mobile_review/{a.session}"; os.makedirs(f"{out}/clips", exist_ok=True)
if not a.no_encode:
    for i, e in enumerate(EV):
        dst = f"{out}/clips/{i:03d}.mp4"; old = f"{out}/clips/{e['id']}.mp4"
        if os.path.exists(old) and not os.path.exists(dst): os.rename(old, dst)
        if os.path.exists(dst) and os.path.getsize(dst) > 0: continue
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{ROOT}/clips3/{e['id']}.mp4", "-vf", "scale=640:-2,fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst])
missing = [e["id"] for i, e in enumerate(EV) if not os.path.exists(f"{out}/clips/{i:03d}.mp4")]
# ピッチ配列は小数1桁のまま・EVの不要キーは落とす
ev_min = [dict(id=e["id"], f=f"{i:03d}", video=e["video"], c=e["c"], n=e["n"], vt=e["vt"], pitch=e.get("pitch", [])) for i, e in enumerate(EV)]
tpl = open(f"{ROOT}/scripts/mobile_review_template.html", encoding="utf-8").read()
html = (tpl.replace("__SESS__", a.session).replace("__FOOT__", foot).replace("__EV__", json.dumps(ev_min, ensure_ascii=False, separators=(",", ":")))
        .replace("__DRAWPITCH__", draw).replace("__COL__", col).replace("__NAME__", name).replace("__TITLE__", f"検収 {a.session[:4]}/{int(a.session[4:6])}/{int(a.session[6:8])}"))
open(f"{out}/index.html", "w", encoding="utf-8").write(html)
files = [{"path": f"clips/{i:03d}.mp4"} for i, e in enumerate(EV) if e["id"] not in missing]
json.dump(files, open(f"{out}/files.json", "w", encoding="utf-8"), ensure_ascii=False)
print("html", len(html) // 1024, "KB; clips", len(files), "missing", len(missing), "; total clips MB", round(sum(os.path.getsize(f"{out}/clips/{i:03d}.mp4") for i, e in enumerate(EV) if e["id"] not in missing) / 1e6, 1))
