# -*- coding: utf-8 -*-
"""音程ガイド表示の読み取り（基準メロディの取得）を、複数の収録セッションへ順番に広げる無人ランチャ（2026-10-23）。

背景。
2026-10-08 に作った読み取り（read_pitchbar.py・run_all_pitchbar.py）は、1セッションずつ手で起動する作りでした。
検出器の入力に「歌った音程 − お手本の音程」の差を足す比較（検証項目C2＝検出法の比較）のためには、
学習に使うセッションの基準メロディも必要なので、セッションを順に流す台本を分けました。

することは1つだけ＝指定したセッションについて run_all_pitchbar.py を順番に呼ぶことです。
読み取り・評価の中身は 2026-10-08 の台本をそのまま使います（この台本は中身を変えません）。
済んだ曲は飛ばすので、途中で止めてもう一度流して構いません。

使い方:
  python run_sessions_pitchbar_20261023.py --dir "G:/研究アーカイブ/成果物/動画解析/e5_refdiff_20261023/scripts" \
      --sessions 20261014,20260911,20261006 --workers 5
  （--dir ＝ run_all_pitchbar.py と read_pitchbar.py を置いた場所。出力はその1つ上の out/ と logs/ に出る）
"""
import os
import sys
import time
import argparse
import subprocess

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run_all_pitchbar.py の置き場所")
    ap.add_argument("--sessions", required=True, help="収録セッション（カンマ区切り・この順に流す）")
    ap.add_argument("--workers", type=int, default=5, help="同時に読む曲数（外付けHDDを読むので控えめに）")
    a = ap.parse_args()
    here = os.path.abspath(a.dir)
    logs = os.path.join(os.path.dirname(here), "logs")
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for s in [x for x in a.sessions.split(",") if x]:
        print(f"=== SESSION {s} start {time.strftime('%H:%M:%S')}", flush=True)
        with open(os.path.join(logs, f"run_{s}.log"), "a", encoding="utf-8") as f:
            p = subprocess.run([sys.executable, os.path.join(here, "run_all_pitchbar.py"), s, "--workers", str(a.workers)],
                               stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"=== SESSION {s} done rc={p.returncode} {time.strftime('%H:%M:%S')}", flush=True)
    print("ALL SESSIONS DONE", flush=True)
