# -*- coding: utf-8 -*-
r"""C2（検出器）を「D 側へずらした正解」（index_shift2_20261105.json）で学習し直す無人ジョブ（2026-11-05・c2-shift2）。
run_c2_shift_20261104.py と同型（学習台本 train_crnn_refdiff_20261023.py は書き換えない）。違い＝1本だけ（再現の段なし）・タグに seed を入れる。
条件＝エポック 8・seed 0・スレッド 8・--refdiff なし・dev 20260911・試験 20260924。変えるのは --index だけ。
上限＝1本 90 分（超えたら止める）・ログが 30 分のびなければ停滞として止める。STOP_REQUEST を見たら止める。
使い方（WSL・root で launch_c2_shift2_20261105.sh から）: python -u run_c2_shift2_20261105.py --shim /home/toshiaki/c2shift2_20261105
"""
import argparse, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REL = "out/c2_shift2_20261105"
OUT = os.path.join(ROOT, REL)
TRAIN = os.path.join(HERE, "train_crnn_refdiff_20261023.py")
CUR = {"p": None}


def say(s):
    sys.stdout.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}\n"); sys.stdout.flush()


def on_term(signum, frame):
    say(f"停止の合図（signal {signum}）＝走っている子を止めて終わる")
    p = CUR["p"]
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=60)
        except Exception:
            p.kill()
    say("CHAIN ABORTED"); sys.exit(143)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll_sec", type=float, default=60)
    ap.add_argument("--stall_min", type=float, default=30)
    ap.add_argument("--max_run_min", type=float, default=90)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGHUP, on_term)
    os.makedirs(OUT, exist_ok=True)
    probe = os.path.join(a.shim, "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928/index.json")
    if not os.path.exists(probe):
        say(f"止める＝作業場のリンクが無い（{probe}）"); say("CHAIN DONE 本数=0"); sys.exit(2)
    tag = f"c2_shift2_{a.epochs}ep_s{a.seed}"
    log_path, err_path = os.path.join(OUT, tag + ".log"), os.path.join(OUT, tag + ".err")
    cmd = [sys.executable, "-u", TRAIN, "--tag", f"c2_shift2_20261105/{tag}", "--epochs", str(a.epochs), "--threads", str(a.threads),
           "--seed", str(a.seed), "--index", f"{REL}/labels/index_shift2_20261105.json"]
    t0 = time.time()
    say(f"CHAIN START pid={os.getpid()} tag={tag} 上限 {a.max_run_min}分 python={sys.executable}")
    try:
        say(f"cgroup＝{open('/proc/self/cgroup').read().strip()} nice={os.nice(0)}")
    except Exception:
        pass
    say(f"=== {tag} コマンド＝{' '.join(cmd[2:])}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    why = ""
    with open(log_path, "w", encoding="utf-8") as fo, open(err_path, "w", encoding="utf-8") as fe:
        p = subprocess.Popen(cmd, cwd=a.shim, env=env, stdout=fo, stderr=fe)
        CUR["p"] = p
        say(f"=== {tag} 子の pid={p.pid}")
        last_size, last_grow, beats = -1, time.time(), 0
        while p.poll() is None:
            time.sleep(a.poll_sec); beats += 1
            size = os.path.getsize(log_path) if os.path.exists(log_path) else -1
            if size > last_size:
                last_size, last_grow = size, time.time()
            run_min, idle_min = (time.time() - t0) / 60, (time.time() - last_grow) / 60
            say(f"心拍 {beats} {tag} 経過 {run_min:.1f}分 ログ {size}B 無成長 {idle_min:.1f}分")
            if os.path.exists(os.path.join(OUT, "STOP_REQUEST")):
                why = "STOP_REQUEST（停止の合図）"
            elif idle_min >= a.stall_min:
                why = f"停滞（ログが {idle_min:.0f}分のびない）"
            elif run_min >= a.max_run_min:
                why = f"1本の上限 {a.max_run_min:.0f}分 超過"
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
    say(f"CHAIN DONE 本数=1 rc={p.returncode} 合計 {sec / 60:.1f}分")
