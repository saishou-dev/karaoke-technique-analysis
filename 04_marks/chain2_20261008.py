# -*- coding: utf-8 -*-
"""第2連鎖（10/08）: progress.log に「20260818 完了」が出てから 0915 → 1006 の方式A を run_session.py で通す。
Start-Process の引数で空白・日本語が化けたため、引数を持たない起動用スクリプトにした。"""
import os, sys, time, subprocess
SC = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(SC), "progress.log")
HB = os.path.join(os.path.dirname(SC), "tmp", "heartbeat.log")
def hb(msg):
    with open(HB, "a", encoding="utf-8") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg + "\n")
hb("第2連鎖 待機開始（'20260818 完了' を待つ）")
def done_today():
    # 9/29 の「20260818 完了」（sat70 版・不採用）が残っているので、2026-10-08 以降の行だけを見る
    for line in open(LOG, encoding="utf-8", errors="replace"):
        if "20260818 完了" in line and line[:10] >= "2026-10-08": return True
    return False
while not done_today():
    time.sleep(60)
hb("第2連鎖 開始 0915 → 1006")
subprocess.run([sys.executable, os.path.join(SC, "run_session.py"), "--sessions", "20260915", "20261006", "--workers", "4", "--threads", "2", "--refine_workers", "4"])
hb("第2連鎖 終了")
