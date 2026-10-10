# -*- coding: utf-8 -*-
"""summary.md（summarize.py の出力）を報告書の「5. 結果」節として差し込む。既に節があれば置き換える。"""
import os, re
OUT_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
REPORT = os.path.join(OUT_DIR, "報告_加点マーク時刻_9-24パイロット.md")
body = open(os.path.join(OUT_DIR, "summary.md"), encoding="utf-8").read()
rep = open(REPORT, encoding="utf-8").read()
sec = "## 5. 結果（marks_20260924.csv の集計）\n\n" + body.replace("## ", "### ") + "\n"
if "## 5. 結果" in rep:
    rep = re.sub(r"## 5\. 結果.*?(?=\n## 6\.|\Z)", sec, rep, flags=re.S)
else:
    rep = rep.rstrip() + "\n\n" + sec
open(REPORT, "w", encoding="utf-8").write(rep)
print("appended results section")
