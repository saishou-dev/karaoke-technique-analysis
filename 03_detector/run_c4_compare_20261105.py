# -*- coding: utf-8 -*-
r"""c4-compare の学習8本を1本ずつ順に回す無人ジョブ（2026-11-05 Opus・指示書 2026-11-05_1806）。
run_c3_5tech_20261105.py の写し（1本の回し方は同じ）＋ run_c2_seed_chain_20261024.py の作り（続けて回す・前の本が失敗しても次へ）。
学習台本＝train_crnn_refdiff_c4_20261105.py（c3-5tech の学習台本の写し・中身は同じ）。条件は基準と同じ＝エポック 8・スレッド 8・--refdiff なし・dev 20260911・試験 20260924・しきい値＝dev の F 最大。
変えるのは --index（正解の一覧）と --seed だけ。
  1 c4_shift0_8ep_s0   遅れ 0 の正解（index_c4_shift0_5tech）
  2 c4_pertech_8ep_s0  技法ごとの遅れの正解（index_c4_pertech_5tech）
  3 c4_global_8ep_s1   全体1つの遅れ・seed 1
  4 c4_global_8ep_s2   全体1つの遅れ・seed 2
  5〜8 c4_lc1000/2000/4000/5000_8ep_s0  学習の曲を入れ子で減らした一覧（全体1つの遅れ）
止める条件＝1本 120分（超えたらその本を止めて次へ）・ログが 30分のびない（停滞）・STOP_REQUEST・締め切り（--deadline＝起点から 9時間・超えたら走っている本を止め、残りは回さない）。
使い方（WSL・root で launch_c4_compare_20261105.sh から）: python -u run_c4_compare_20261105.py --shim /home/toshiaki/c4_compare_20261105 --deadline <epoch秒>
"""
import argparse, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REL = "out/c4_compare_20261105"
OUT = os.path.join(ROOT, REL)
TRAIN = os.path.join(HERE, "train_crnn_refdiff_c4_20261105.py")
G = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
L = f"{REL}/labels"
STAGES = [
    ("c4_shift0_8ep_s0", f"{L}/index_c4_shift0_5tech_20261105.json", 0),
    ("c4_pertech_8ep_s0", f"{L}/index_c4_pertech_5tech_20261105.json", 0),
    # 追補2（18:41）手順7＝終わりの時刻の正解。18:14 に起動した連結ジョブ（8本）はこの行の前の版で動いている＝この本は --only c4_endpoint_8ep_s0 --after_pid <その pid> で別に回す。
    ("c4_endpoint_8ep_s0", f"{L}/index_c4_endpoint_5tech_20261105.json", 0),
    ("c4_global_8ep_s1", G, 1),
    ("c4_global_8ep_s2", G, 2),
    ("c4_lc1000_8ep_s0", f"{L}/index_c4_lc1000_20261105.json", 0),
    ("c4_lc2000_8ep_s0", f"{L}/index_c4_lc2000_20261105.json", 0),
    ("c4_lc4000_8ep_s0", f"{L}/index_c4_lc4000_20261105.json", 0),
    ("c4_lc5000_8ep_s0", f"{L}/index_c4_lc5000_20261105.json", 0),
    # 追補4（11/6 00:2x）手順9＝終わりの時刻の正解で seed 1・2。手順6 の連結ジョブ（run_c4_models）の後に --only と --after_pid で回す。
    ("c4_endpoint_8ep_s1", f"{L}/index_c4_endpoint_5tech_20261105.json", 1),
    ("c4_endpoint_8ep_s2", f"{L}/index_c4_endpoint_5tech_20261105.json", 2),
]
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


def run_one(a, tag, index, seed):
    log_path, err_path = os.path.join(OUT, tag + ".log"), os.path.join(OUT, tag + ".err")
    cmd = [sys.executable, "-u", TRAIN, "--tag", f"c4_compare_20261105/{tag}", "--epochs", str(a.epochs), "--threads", str(a.threads),
           "--seed", str(seed), "--index", index]
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
                why = f"1本の上限 {a.max_run_min:.0f}分 超過"
            elif a.deadline and time.time() >= a.deadline:
                why = "全体の締め切り（起点から 9時間）超過"
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
    ap.add_argument("--deadline", type=float, default=0, help="この epoch 秒を過ぎたら止める（0＝なし）")
    ap.add_argument("--only", default="", help="回す本を絞る（タグをカンマ区切り）")
    ap.add_argument("--after_pid", type=int, default=0, help="この pid（先に走る連結ジョブ）が居なくなるまで待ってから始める（同時に2本回さない）")
    ap.add_argument("--skip_done", action="store_true", help="metrics.json がある本と、1本の上限で止めた本は回さない（追補5・締め切りで切れた本だけ回し直す）")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGHUP, on_term)
    os.makedirs(OUT, exist_ok=True)
    probe = os.path.join(a.shim, "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928/index.json")
    if not os.path.exists(probe):
        say(f"止める＝作業場のリンクが無い（{probe}）"); say("CHAIN DONE 本数=0"); sys.exit(2)
    stages = [s for s in STAGES if not a.only or s[0] in set(a.only.split(","))]
    t_all = time.time()
    say(f"CHAIN START pid={os.getpid()} 本数={len(stages)} 1本の上限 {a.max_run_min}分 締め切り {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(a.deadline)) if a.deadline else 'なし'} python={sys.executable}")
    try:
        say(f"cgroup＝{open('/proc/self/cgroup').read().strip()} nice={os.nice(0)}")
    except Exception:
        pass
    say("順番＝" + " → ".join(s[0] for s in stages))
    if a.after_pid:
        n = 0
        while os.path.exists(f"/proc/{a.after_pid}"):
            if n % 10 == 0:
                say(f"待つ＝先の連結ジョブ pid={a.after_pid} が動いている")
            n += 1
            if os.path.exists(os.path.join(OUT, "STOP_REQUEST")) or (a.deadline and time.time() >= a.deadline):
                say("待つ間に STOP_REQUEST か締め切り＝回さない"); say("CHAIN DONE 本数=0"); sys.exit(0)
            time.sleep(30)
        say(f"先の連結ジョブ pid={a.after_pid} が居ない＝始める")
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
            elif f"!!! {s[0]} 1本の上限" in logs:
                say(f"飛ばす＝{s[0]}（1本の上限で止めた）")
            else:
                keep.append(s)
        stages = keep
        say("回す本＝" + (" → ".join(s[0] for s in stages) or "なし"))
    results = []
    for i, (tag, index, seed) in enumerate(stages, 1):
        if os.path.exists(os.path.join(OUT, "STOP_REQUEST")):
            say(f"--- STOP_REQUEST あり＝{tag} 以降を回さない"); break
        if a.deadline and time.time() >= a.deadline:
            say(f"--- 締め切り超過＝{tag} 以降を回さない"); break
        rc, sec, why = run_one(a, tag, index, seed)
        results.append((tag, rc, sec, why))
        say(f"--- 進捗 {i}/{len(stages)} 完了 ここまでの合計 {(time.time() - t_all) / 60:.1f}分")
    say("結果の一覧（タグ・rc・所要秒）")
    for tag, rc, sec, why in results:
        say(f"  {tag} rc={rc} {sec:.1f}秒 {why}")
    ok = sum(1 for r in results if r[1] == 0)
    say(f"CHAIN DONE 本数={len(results)} rc=0 が {ok} 本 合計 {(time.time() - t_all) / 60:.1f}分")
