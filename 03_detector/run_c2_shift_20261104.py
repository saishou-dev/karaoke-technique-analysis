# -*- coding: utf-8 -*-
r"""C2（検出器）を「技法マーク側へ補正した正解」で学習し直す無人ジョブ（2026-11-04・研究 D2 の一歩目）。WSL で動かす。

回すもの（直列・1本ずつ）。
  1. c2_base_wsl_8ep_s0 ＝現行の再現（現行の正解 index.json・8エポック・seed 0・現行の入力）。
     10/23 の c2a_base_8ep は Windows の python（torch 2.14.0）で学習したもの。今回は WSL（torch 2.14.1）で回すので、
     環境の差だけで成績が動くかを先に測る。試験F値 ±0.5秒 が 0.112 から 0.01 以上ずれたら、2 を回さずに止める（指示書の止まる条件）。
  2. c2_shift_8ep_s0 ＝補正した正解（out/c2_shift_20261104/labels/index_shift_20261104.json）で学習・評価。他の条件は 1 と同じ。

そろえる条件＝学習台本 train_crnn_refdiff_20261023.py（書き換えない）／エポック 8／seed 0／スレッド 8（WSL の research cgroup＝8コア相当）／
  --refdiff なし／しきい値は dev（20260911・30曲）／試験＝20260924 の28曲。変えるのは --index だけ。
上限＝1本 90 分・全体 3 時間（超えたらその本を止めて記録する）。学習ログが 30 分のびなければ停滞としてその本を止める。

学習台本は ROOT を "G:/研究アーカイブ/…/g3v2_20260928" と書いているため、WSL では作業場（--shim）に
"G:/研究アーカイブ/成果物/動画解析/g3v2_20260928" → /mnt/g/… の記号リンクを置き、そこを cwd にして相対パスとして解決させる。
（台本本体は1行も変えない。リンクは launch_c2_shift_20261104.sh が作る。）

使い方（WSL・root で launch_c2_shift_20261104.sh から呼ばれる）:
  python -u run_c2_shift_20261104.py --shim /home/toshiaki/c2shift_20261104
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REL = "out/c2_shift_20261104"
OUT = os.path.join(ROOT, REL)
TRAIN = os.path.join(HERE, "train_crnn_refdiff_20261023.py")
BASE_F05 = 0.112
CUR = {"p": None}


def say(s):
    sys.stdout.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}\n")
    sys.stdout.flush()


def on_term(signum, frame):
    say(f"停止の合図（signal {signum}）＝走っている子を止めて終わる")
    p = CUR["p"]
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=60)
        except Exception:
            p.kill()
    say("CHAIN ABORTED")
    sys.exit(143)


def run_stage(tag, extra, shim, a, t_all):
    log_path = os.path.join(OUT, tag + ".log")
    err_path = os.path.join(OUT, tag + ".err")
    cmd = [sys.executable, "-u", TRAIN, "--tag", f"c2_shift_20261104/{tag}",
           "--epochs", str(a.epochs), "--threads", str(a.threads), "--seed", str(a.seed)] + extra
    say(f"=== {tag} 開始 cwd={shim} ログ={log_path}")
    say(f"=== {tag} コマンド＝{' '.join(cmd[2:])}")
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(log_path, "w", encoding="utf-8") as fo, open(err_path, "w", encoding="utf-8") as fe:
        p = subprocess.Popen(cmd, cwd=shim, env=env, stdout=fo, stderr=fe)
        CUR["p"] = p
        say(f"=== {tag} 子の pid={p.pid}")
        last_size, last_grow, beats, why = -1, time.time(), 0, ""
        while p.poll() is None:
            time.sleep(a.poll_sec)
            beats += 1
            size = os.path.getsize(log_path) if os.path.exists(log_path) else -1
            if size > last_size:
                last_size, last_grow = size, time.time()
            run_min = (time.time() - t0) / 60
            all_min = (time.time() - t_all) / 60
            idle_min = (time.time() - last_grow) / 60
            say(f"心拍 {beats} {tag} 経過 {run_min:.1f}分 全体 {all_min:.1f}分 ログ {size}B 無成長 {idle_min:.1f}分")
            if os.path.exists(os.path.join(OUT, "STOP_REQUEST")):
                why = "STOP_REQUEST（綱が停止の合図を受けた）"
            elif idle_min >= a.stall_min:
                why = f"停滞（ログが {idle_min:.0f}分のびない）"
            elif run_min >= a.max_run_min:
                why = f"1本の上限 {a.max_run_min:.0f}分 超過"
            elif all_min >= a.max_all_min:
                why = f"全体の上限 {a.max_all_min:.0f}分 超過"
            if why:
                say(f"!!! {tag} {why}＝この本を止める")
                p.terminate()
                try:
                    p.wait(timeout=120)
                except Exception:
                    p.kill(); p.wait(timeout=60)
                break
    CUR["p"] = None
    sec = time.time() - t0
    say(f"=== {tag} 終了 rc={p.returncode} 所要 {sec:.1f}秒（{sec / 60:.1f}分） .err={os.path.getsize(err_path)}B {why}")
    return p.returncode, sec, why


def test_f05(tag):
    m = os.path.join(OUT, tag, "metrics.json")
    if not os.path.exists(m):
        return None
    return json.load(open(m, encoding="utf-8"))["metrics"]["test_tol0.5"]["ALL"]["f1"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll_sec", type=float, default=60)
    ap.add_argument("--stall_min", type=float, default=30)
    ap.add_argument("--max_run_min", type=float, default=90)
    ap.add_argument("--max_all_min", type=float, default=180)
    ap.add_argument("--only", default="", help="base / shift のどちらかだけ回す")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGHUP, on_term)
    os.makedirs(OUT, exist_ok=True)
    probe = os.path.join(a.shim, "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928/index.json")
    if not os.path.exists(probe):
        say(f"止める＝作業場のリンクが無い（{probe}）"); say("CHAIN DONE 本数=0 rc=0 が 0 本（リンク無し）"); sys.exit(2)
    t_all = time.time()
    say(f"CHAIN START pid={os.getpid()} エポック={a.epochs} スレッド={a.threads} seed={a.seed} 上限 1本 {a.max_run_min}分・全体 {a.max_all_min}分 python={sys.executable}")
    try:
        say(f"cgroup＝{open('/proc/self/cgroup').read().strip()} nice={os.nice(0)}")
    except Exception:
        pass
    results = []
    if a.only in ("", "base"):
        rc, sec, why = run_stage("c2_base_wsl_8ep_s0", [], a.shim, a, t_all)
        results.append(("c2_base_wsl_8ep_s0", rc, sec, why))
        f = test_f05("c2_base_wsl_8ep_s0")
        say(f"再現の確認＝試験F値 ±0.5秒 {f}（基準 {BASE_F05}・許すずれ 0.01 未満）")
        if f is None or abs(f - BASE_F05) >= 0.01:
            say("!!! 止まる条件＝現行の再現が基準から 0.01 以上ずれた（または結果が無い）＝補正の学習は回さない")
            say(f"CHAIN DONE 本数={len(results)} rc=0 が {sum(1 for r in results if r[1] == 0)} 本 再現ずれで停止 合計 {(time.time() - t_all) / 60:.1f}分")
            sys.exit(3)
    if os.path.exists(os.path.join(OUT, "STOP_REQUEST")):
        say("STOP_REQUEST があるので補正の学習は回さない"); a.only = "none"
    if a.only in ("", "shift"):
        rc, sec, why = run_stage("c2_shift_8ep_s0", ["--index", f"{REL}/labels/index_shift_20261104.json"], a.shim, a, t_all)
        results.append(("c2_shift_8ep_s0", rc, sec, why))
    say("結果の一覧（タグ・rc・所要秒・止めた理由）")
    for tag, rc, sec, why in results:
        say(f"  {tag} rc={rc} {sec:.1f}秒 {why}")
    say(f"CHAIN DONE 本数={len(results)} rc=0 が {sum(1 for r in results if r[1] == 0)} 本 合計 {(time.time() - t_all) / 60:.1f}分")
