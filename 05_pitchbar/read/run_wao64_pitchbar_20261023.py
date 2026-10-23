# -*- coding: utf-8 -*-
"""お手本音程バーの読み取りを、2026-09-02・08-12・08-18 の3セッション（64曲）へ広げる無人ジョブ（2026-10-23）。

背景。
この3セッションは LIVE DAM WAO! で、画面の形は 2026-09-24 とまったく同じであることを静止画で確認した（2026-10-23 05:17〜05:20）。
ただし画面位置の自動検出の表（dataset/band_probe_v2_<セッション>.csv）が作られていなかったので、読み取りを試せていなかった。
精密採点Ai の2セッション（2026-06-28・08-07）は加点される技法が4つで別の形のため、ここには含めない（本人裁定 2026-10-23）。

すること（順番）。
1. 先に走っている読み取り（10/14→9/11→10/06）の完了を待つ（ログに ALL SESSIONS DONE が出るまで）。
2. セッションごとに、画面位置の自動検出の表が無ければ WSL 側の環境で作る（wao_probe_v2_20261014.py をそのまま使う・無変更）。
3. セッションごとに読み取り（run_all_pitchbar.py）を流す。済んだ曲は飛ばす。

このジョブは優先度を下げて（Idle で）起動する前提である。本人の操作や他の計算を邪魔しないため。

使い方:
  python run_wao64_pitchbar_20261023.py --wait_log "G:/.../e5_refdiff_20261023/logs/launcher.log" \
      --read_dir "G:/.../e5_refdiff_20261023/scripts" --sessions 20260902,20260812,20260818
"""
import os
import sys
import csv
import time
import argparse
import subprocess

ROOT_WIN = "G:/研究アーカイブ/成果物/動画解析"
PROBE_WSL = "/mnt/g/研究アーカイブ/成果物/動画解析/scripts/wao_probe_v2_20261014.py"
VENV_WSL = "~/karaoke-vision/.venv/bin/python"


def probe_done(session):
    """画面位置の自動検出の表があり、行が1行以上あるか。"""
    p = f"{ROOT_WIN}/dataset/band_probe_v2_{session}.csv"
    if not os.path.exists(p):
        return 0
    try:
        return len(list(csv.DictReader(open(p, encoding="utf-8-sig"))))
    except Exception:
        return 0


def run_probe(session, yr):
    """WSL 側の環境で画面位置の自動検出を走らせる（済んだ曲は台本側が飛ばす）。"""
    cmd = (f"cd ~/karaoke-vision && export LANG=C.UTF-8 && "
           f"KV_PROBE_YR={yr} {VENV_WSL} \"{PROBE_WSL}\" {session}")
    print(f"--- 画面位置の自動検出 {session} 開始 {time.strftime('%H:%M:%S')}", flush=True)
    p = subprocess.run(["wsl.exe", "bash", "-lc", cmd], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = (p.stdout or "").splitlines()[-3:] + (p.stderr or "").splitlines()[-3:]
    print(f"--- 画面位置の自動検出 {session} 終了 rc={p.returncode} 行数={probe_done(session)} {time.strftime('%H:%M:%S')}", flush=True)
    for t in tail:
        print("   ", t[:200], flush=True)
    return p.returncode


def run_read(session, read_dir, workers):
    logs = os.path.join(os.path.dirname(os.path.abspath(read_dir)), "logs")
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    print(f"=== 読み取り {session} 開始 {time.strftime('%H:%M:%S')}", flush=True)
    with open(os.path.join(logs, f"run_{session}.log"), "a", encoding="utf-8") as f:
        p = subprocess.run([sys.executable, os.path.join(read_dir, "run_all_pitchbar.py"), session,
                            "--workers", str(workers)], stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f"=== 読み取り {session} 終了 rc={p.returncode} {time.strftime('%H:%M:%S')}", flush=True)
    return p.returncode


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait_log", default="", help="このログに印が出るまで待つ（空なら待たない）")
    ap.add_argument("--wait_mark", default="ALL SESSIONS DONE")
    ap.add_argument("--max_wait_hours", type=float, default=10.0)
    ap.add_argument("--read_dir", required=True, help="run_all_pitchbar.py と read_pitchbar.py の置き場所")
    ap.add_argument("--sessions", default="20260902,20260812,20260818")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--probe_yr", default="0.20,0.80", help="画面位置の粗探索の縦窓（画面の上からの割合）")
    a = ap.parse_args()

    if a.wait_log:
        t0 = time.time()
        print(f"前段待ち 開始 {time.strftime('%H:%M:%S')} 印='{a.wait_mark}' ログ={a.wait_log}", flush=True)
        while time.time() - t0 < a.max_wait_hours * 3600:
            try:
                if a.wait_mark in open(a.wait_log, encoding="utf-8", errors="replace").read():
                    print(f"前段の読み取り完了を確認 {time.strftime('%H:%M:%S')}", flush=True)
                    break
            except FileNotFoundError:
                pass
            print(f"待機中（前段の印 '{a.wait_mark}' を待っています） {time.strftime('%H:%M:%S')}", flush=True)
            time.sleep(120)
        else:
            print("待ち時間の上限に達したので、そのまま進む", flush=True)

    for s in [x for x in a.sessions.split(",") if x]:
        n = probe_done(s)
        print(f"[{s}] 画面位置の表の行数＝{n}", flush=True)
        if n == 0:
            run_probe(s, a.probe_yr)
        if probe_done(s) == 0:
            print(f"[{s}] 画面位置の表が作れなかったので、このセッションは飛ばす", flush=True)
            continue
        run_read(s, a.read_dir, a.workers)
    print("WAO64 DONE", flush=True)
