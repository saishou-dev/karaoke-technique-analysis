# -*- coding: utf-8 -*-
r"""c4-compare 手順6（モデルの比較）の学習9本を1本ずつ順に回す無人ジョブ（2026-11-05 Opus・追補の指示書 2026-11-05_1820）。
run_c4_compare_20261105.py の写し。違い＝学習台本を train_crnn_arch_c4_20261105.py（--arch・--types を足した写し）・本の一覧・
最初に「手順2〜4 の連結ジョブ（run_c4_compare_20261105.py）」が終わるのを待つ（同時に2本回さない）。
条件は基準と同じ＝全体1つの遅れの正解・エポック 8・スレッド 8・seed 0・dev 20260911・試験 20260924・しきい値＝dev の F 最大。
  （手順7 の終わりの時刻の正解の1本は、run_c4_compare_20261105.py --only c4_endpoint_8ep_s0 --after_pid で別に回す。名前が待ちの型に合う＝この連結ジョブはその本の終わりも待つ）
  6b c4_conv_8ep_s0        畳み込みだけ（--arch conv）
  6c c4_gru_8ep_s0         GRU だけ（--arch gru）
  6d c4_half_8ep_s0        幅を半分（--hid 64）
  6e c4_wide_8ep_s0        幅を2倍（--hid 256・120分を超えたら止めて「時間切れ」）
  6f c4_only_<技法>_8ep_s0 技法ごとに1本ずつ（--types <技法>・5本）
止める条件＝1本 120分・停滞 30分・STOP_REQUEST・締め切り（--deadline＝起点から 18時間）。
使い方（WSL・root で launch_c4_models_20261105.sh から）: python -u run_c4_models_20261105.py --shim /home/toshiaki/c4_compare_20261105 --deadline <epoch秒>
"""
import argparse, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REL = "out/c4_compare_20261105"
OUT = os.path.join(ROOT, REL)
TRAIN = os.path.join(HERE, "train_crnn_arch_c4_20261105.py")
G = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
STAGES = [
    ("c4_conv_8ep_s0", ["--arch", "conv"]),
    ("c4_gru_8ep_s0", ["--arch", "gru"]),
    ("c4_half_8ep_s0", ["--hid", "64"]),
    ("c4_wide_8ep_s0", ["--hid", "256"]),
] + [(f"c4_only_{c}_8ep_s0", ["--types", c]) for c in ("shakuri", "kobushi", "fall", "accent", "hammering")] + [
    # 追補7（11/6 05:55）手順10a＝アクセントだけ 16エポック。--only c4_only_accent_16ep_s0 --epochs 16 --after_pid で列の最後に回す。
    ("c4_only_accent_16ep_s0", ["--types", "accent"]),
]
WAIT_PAT = "run_c4_compare_2026110[5].py"
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


def other_alive():
    r = subprocess.run(["pgrep", "-f", WAIT_PAT], capture_output=True, text=True)
    return [x for x in r.stdout.split() if x.strip() and int(x) != os.getpid()]


def run_one(a, tag, extra):
    log_path, err_path = os.path.join(OUT, tag + ".log"), os.path.join(OUT, tag + ".err")
    cmd = [sys.executable, "-u", TRAIN, "--tag", f"c4_compare_20261105/{tag}", "--epochs", str(a.epochs), "--threads", str(a.threads),
           "--seed", "0", "--index", G] + extra
    t0 = time.time()
    say(f"=== {tag} 開始 コマンド＝{' '.join(cmd[2:])}")
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
                why = f"時間切れ（1本の上限 {a.max_run_min:.0f}分 超過）"
            elif a.deadline and time.time() >= a.deadline:
                why = "全体の締め切り（起点から 18時間）超過"
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--poll_sec", type=float, default=60)
    ap.add_argument("--stall_min", type=float, default=30)
    ap.add_argument("--max_run_min", type=float, default=120)
    ap.add_argument("--deadline", type=float, default=0)
    ap.add_argument("--only", default="")
    # 追補5（11/6 00:4x）＝上限を 24時間へ。走っている連結ジョブの締め切り（12:06）は起動時の値のまま＝締め切りで切れた本を、後から回す連結ジョブで拾う。
    ap.add_argument("--after_pid", type=int, default=0, help="この pid が居なくなるまで待つ（待ちの型 WAIT_PAT の代わり）")
    ap.add_argument("--skip_done", action="store_true", help="metrics.json がある本と、1本の上限（時間切れ）で止めた本は回さない＝締め切りで切れた本だけ回し直す")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGHUP, on_term)
    os.makedirs(OUT, exist_ok=True)
    probe = os.path.join(a.shim, "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928/index.json")
    if not os.path.exists(probe):
        say(f"止める＝作業場のリンクが無い（{probe}）"); say("CHAIN DONE 本数=0"); sys.exit(2)
    stages = [s for s in STAGES if not a.only or s[0] in set(a.only.split(","))]
    say(f"CHAIN START pid={os.getpid()} 本数={len(stages)} 1本の上限 {a.max_run_min}分 締め切り {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(a.deadline)) if a.deadline else 'なし'}")
    try:
        say(f"cgroup＝{open('/proc/self/cgroup').read().strip()} nice={os.nice(0)}")
    except Exception:
        pass
    say("順番＝" + " → ".join(s[0] for s in stages))
    n = 0
    alive = (lambda: os.path.exists(f"/proc/{a.after_pid}")) if a.after_pid else other_alive
    while alive():
        if n % 10 == 0:
            say(f"待つ＝先の連結ジョブ（{('pid=' + str(a.after_pid)) if a.after_pid else WAIT_PAT}）が動いている")
        n += 1
        if os.path.exists(os.path.join(OUT, "STOP_REQUEST")) or (a.deadline and time.time() >= a.deadline):
            say("待つ間に STOP_REQUEST か締め切り＝回さない"); say("CHAIN DONE 本数=0"); sys.exit(0)
        time.sleep(60)
    say("先の連結ジョブが居ない＝始める")
    if a.skip_done:
        logs = ""
        for nm in ("c4_models_chain_20261105.log", "c4_compare_chain_20261105.log"):
            p = os.path.join(OUT, nm)
            if os.path.exists(p):
                logs += open(p, encoding="utf-8", errors="replace").read()
        keep = []
        for s in stages:
            if os.path.exists(os.path.join(OUT, s[0], "metrics.json")):
                say(f"飛ばす＝{s[0]}（metrics.json あり＝終わっている）")
            elif f"!!! {s[0]} 時間切れ" in logs:
                say(f"飛ばす＝{s[0]}（1本の上限で止めた＝時間切れ）")
            else:
                keep.append(s)
        stages = keep
        say("回す本＝" + (" → ".join(s[0] for s in stages) or "なし"))
    t_all = time.time()
    results = []
    for i, (tag, extra) in enumerate(stages, 1):
        if os.path.exists(os.path.join(OUT, "STOP_REQUEST")):
            say(f"--- STOP_REQUEST あり＝{tag} 以降を回さない"); break
        if a.deadline and time.time() >= a.deadline:
            say(f"--- 締め切り超過＝{tag} 以降を回さない"); break
        rc, sec, why = run_one(a, tag, extra)
        results.append((tag, rc, sec, why))
        say(f"--- 進捗 {i}/{len(stages)} 完了 ここまでの合計 {(time.time() - t_all) / 60:.1f}分")
    say("結果の一覧（タグ・rc・所要秒）")
    for tag, rc, sec, why in results:
        say(f"  {tag} rc={rc} {sec:.1f}秒 {why}")
    ok = sum(1 for r in results if r[1] == 0)
    say(f"CHAIN DONE 本数={len(results)} rc=0 が {ok} 本 合計 {(time.time() - t_all) / 60:.1f}分")
