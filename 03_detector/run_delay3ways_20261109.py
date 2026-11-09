# -*- coding: utf-8 -*-
r"""delay-3ways の無人ジョブ（2026-11-09 Opus・指示書 2026-11-09_1429）。run_c4_compare_20261105.py の写し（1本の回し方・止める条件は同じ）＋前の段の結果を使う段。
順＝B→A→D（指示書の順）。学習は1本ずつ（同時に2本は回さない）。
  0  推論＝基準種0（c3_5tech_8ep_s0）で全208曲の確率の列（案B の候補 (ii) と案D の1回目用）→ 案B の対応付け（候補 (ii)）→ 中間1
  1  dB_rule_8ep_s0   案B＝規則の候補で置き直した正解・種0（点の正解の損失＝元と同じ）→ 中間2
  2〜4 dA_weak_8ep_s0／s1／s2  案A＝弱いラベルの損失（区間 [tc−3, tc]）・全体1つの遅れの正解の時刻から tc を戻す → 中間3〜5
  5〜7 dD_r1／r2／r3_8ep_s0  案D＝置き直し（前の回のモデルの山へ）→ 学習・種0 → 中間6〜8
学習の前に毎回「関所」＝nvidia-smi の記録と、別の学習のプロセス（train_crnn・run_fall4tech・run_c4・run_c3 など自分以外）が無い事を確認。有れば 5分待って再確認。
自分の学習中は out/gpu_busy_delay3ways.txt を置き、終わったら消す。
止める条件＝1本 120分・ログが 30分のびない・STOP_REQUEST（out/delay_3ways_20261109/STOP_REQUEST）・締め切り（--deadline）。
使い方（WSL・root で launch_delay3ways_20261109.sh から）: python -u run_delay3ways_20261109.py --shim /home/toshiaki/delay3ways_20261109 --deadline <epoch秒> [--only 段の名,…]
"""
import argparse, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REL = "out/delay_3ways_20261109"
OUT = os.path.join(ROOT, REL)
BUSY = os.path.join(ROOT, "out", "gpu_busy_delay3ways.txt")
TRAIN = os.path.join(HERE, "train_crnn_delay3ways_20261109.py")
PY = sys.executable
SMI = "/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else "nvidia-smi"
IXG = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
LB = f"{REL}/labels"
BASE = "out/c3_5tech_global_20261105/c3_5tech_8ep_s0"
OTHER = r"train_crnn|train_[a-z0-9_]*.py|run_fall4tech|run_c4_|run_c3_|run_c2_"  # 学習と学習の連結ジョブ（解析だけの物は数えない）
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
    if os.path.exists(BUSY):
        os.remove(BUSY)
    say("CHAIN ABORTED"); sys.exit(143)


def others():
    """自分（この連結ジョブとその子）以外の学習のプロセス。"""
    me = {os.getpid(), os.getppid()}
    try:
        out = subprocess.run(["pgrep", "-af", OTHER], capture_output=True, text=True).stdout.splitlines()
    except Exception as e:
        return [f"pgrep 失敗 {e}"]
    res = []
    for ln in out:
        pid = int(ln.split()[0])
        if pid in me or "run_delay3ways_20261109" in ln or "delay3ways" in ln or "pgrep" in ln:
            continue
        if "python" not in ln:
            continue
        res.append(ln[:160])
    return res


def gate(a, what):
    n = 0
    while True:
        try:
            smi = subprocess.run([SMI, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
            apps = subprocess.run([SMI, "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:
            smi, apps = f"nvidia-smi 失敗 {e}", ""
        oth = others()
        say(f"関所（{what}）nvidia-smi＝{smi}・計算のプロセス＝{apps or 'なし'}・別の学習のプロセス＝{len(oth)}件")
        if not oth and not ("python" in apps):
            return True
        for o in oth[:4]:
            say(f"    {o}")
        if os.path.exists(os.path.join(OUT, "STOP_REQUEST")) or (a.deadline and time.time() >= a.deadline):
            say("待つ間に STOP_REQUEST か締め切り"); return False
        n += 1
        say(f"5分待って再確認（{n}回目）"); time.sleep(300)


def run_cmd(a, name, cmd, cwd, log_name, train=False):
    log_path, err_path = os.path.join(OUT, log_name + ".log"), os.path.join(OUT, log_name + ".err")
    t0 = time.time(); why = ""
    say(f"=== {name} 開始 コマンド＝{' '.join(cmd[1:])}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if train:
        open(BUSY, "w", encoding="utf-8").write(f"delay-3ways 学習中 {name} 開始 {time.strftime('%Y-%m-%d %H:%M:%S')} 連結ジョブ pid {os.getpid()}\n")
    with open(log_path, "w", encoding="utf-8") as fo, open(err_path, "w", encoding="utf-8") as fe:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fo, stderr=fe)
        CUR["p"] = p
        say(f"=== {name} 子の pid={p.pid}")
        last_size, last_grow, beats = -1, time.time(), 0
        while p.poll() is None:
            time.sleep(a.poll_sec); beats += 1
            size = os.path.getsize(log_path) if os.path.exists(log_path) else -1
            if size > last_size:
                last_size, last_grow = size, time.time()
            run_min, idle_min = (time.time() - t0) / 60, (time.time() - last_grow) / 60
            if beats % 5 == 0:
                say(f"心拍 {beats} {name} 経過 {run_min:.1f}分 ログ {size}B 無成長 {idle_min:.1f}分")
            if os.path.exists(os.path.join(OUT, "STOP_REQUEST")):
                why = "STOP_REQUEST（停止の合図）"
            elif idle_min >= a.stall_min:
                why = f"停滞（ログが {idle_min:.0f}分のびない）"
            elif run_min >= a.max_run_min:
                why = f"1本の上限 {a.max_run_min:.0f}分 超過"
            elif a.deadline and time.time() >= a.deadline:
                why = "全体の締め切り超過"
            if why:
                say(f"!!! {name} {why}＝この本を止める")
                p.terminate()
                try:
                    p.wait(timeout=120)
                except Exception:
                    p.kill(); p.wait(timeout=60)
                break
    CUR["p"] = None
    if train and os.path.exists(BUSY):
        os.remove(BUSY)
    sec = time.time() - t0
    say(f"=== {name} 終了 rc={p.returncode} 所要 {sec:.1f}秒（{sec / 60:.1f}分） .err={os.path.getsize(err_path)}B {why}")
    return p.returncode == 0 and not why


def train(a, tag, index, seed, loss, extra=()):
    if not gate(a, tag):
        return False
    cmd = [PY, "-u", TRAIN, "--tag", f"delay_3ways_20261109/{tag}", "--epochs", str(a.epochs), "--threads", str(a.threads),
           "--seed", str(seed), "--index", index, "--loss", loss] + list(extra)
    return run_cmd(a, tag, cmd, a.shim, tag, train=True)


def py(a, name, script, *args):
    return run_cmd(a, name, [PY, "-u", os.path.join(HERE, script)] + list(args), HERE, name)


def interim(a, n, note):
    py(a, f"rescore_interim{n}", "rescore_delay3ways_20261109.py", "--interim", str(n), "--note", note)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--poll_sec", type=float, default=60)
    ap.add_argument("--stall_min", type=float, default=30)
    ap.add_argument("--max_run_min", type=float, default=120)
    ap.add_argument("--deadline", type=float, default=0)
    ap.add_argument("--only", default="", help="段を絞る（infer,B,A0,A1,A2,D1,D2,D3）")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGHUP, on_term)
    os.makedirs(OUT, exist_ok=True)
    probe = os.path.join(a.shim, "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928/index.json")
    if not os.path.exists(probe):
        say(f"止める＝作業場のリンクが無い（{probe}）"); sys.exit(2)
    only = set(x for x in a.only.split(",") if x)
    want = lambda s: not only or s in only
    t_all = time.time()
    say(f"CHAIN START pid={os.getpid()} 締め切り {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(a.deadline)) if a.deadline else 'なし'} python={PY} 段＝{a.only or '全部'}")
    try:
        say(f"cgroup＝{open('/proc/self/cgroup').read().strip()} nice={os.nice(0)}")
    except Exception:
        pass
    stop = lambda: os.path.exists(os.path.join(OUT, "STOP_REQUEST")) or (a.deadline and time.time() >= a.deadline)
    res = {}
    # 試し（7曲・1エポック）＝学習台本の weak と point が最後まで通るか。通らなければ全部止める。
    if want("smoke") and not stop():
        if not gate(a, "試し"):
            say("CHAIN DONE（関所で止めた）"); sys.exit(0)
        for ls in ("weak", "point"):
            cmd = [PY, "-u", TRAIN, "--tag", f"delay_3ways_20261109/smoke_{ls}", "--epochs", "1", "--threads", str(a.threads), "--seed", "0",
                   "--index", f"{LB}/index_smoke_20261109.json", "--loss", ls]
            if not run_cmd(a, f"smoke_{ls}", cmd, a.shim, f"smoke_{ls}", train=True):
                say(f"止める＝試し smoke_{ls} が通らない"); say("CHAIN DONE（試しで止めた）"); sys.exit(3)
    # 0 推論と案B の候補 (ii)
    if want("infer") and not stop():
        if gate(a, "推論 基準種0 全208曲"):
            ok = py(a, "infer_base_all", "infer_all_delay3ways_20261109.py", "--mdir", BASE, "--name", "base", "--out", f"{REL}/probs", "--threads", str(a.threads))
            res["infer"] = ok
            if ok:
                res["Bmodel"] = py(a, "B_align_model", "delay3ways_B_align_20261109.py", "--cand", "model")
            interim(a, 1, "案B の対応付けが終わった（候補 (i) 規則・(ii) 今のモデル種0 の山）。|対応＝順番を崩さず・技法一致・差 0.5〜3.5秒・対応の数が最大・その中で隣の遅れの変化が最小。|次＝案B の正解で種0 を学習。")
    # 1 案B
    if want("B") and not stop():
        res["B"] = train(a, "dB_rule_8ep_s0", f"{LB}/index_B_rule_20261109.json", 0, "point")
        interim(a, 2, f"案B の学習1本が終わった（rc 良し＝{res['B']}）。")
    # 2〜4 案A
    for i, s in enumerate((0, 1, 2)):
        k = f"A{s}"
        if want(k) and not stop():
            res[k] = train(a, f"dA_weak_8ep_s{s}", IXG, s, "weak")
            interim(a, 3 + i, f"案A 種{s} の学習が終わった（rc 良し＝{res[k]}）。")
    # 5〜7 案D
    prev_ix, prev_m, prev_tr, prev_dv = IXG, BASE, f"{REL}/probs/probs_base_all.npz", "out/c4_compare_20261105/probs/probs_base_dev.npz"
    for r in (1, 2, 3):
        k = f"D{r}"
        if not want(k) or stop():
            continue
        if not os.path.exists(os.path.join(ROOT, prev_tr)):
            say(f"--- {k} を飛ばす＝前の回の確率の列が無い（{prev_tr}）"); break
        ok = py(a, f"relabel_D_r{r}", "relabel_D_delay3ways_20261109.py", "--round", str(r), "--mdir", prev_m, "--probs_train", prev_tr, "--probs_dev", prev_dv, "--prev_index", prev_ix)
        if not ok:
            say(f"--- {k} の置き直しが失敗＝案D をここで止める"); break
        tag = f"dD_r{r}_8ep_s0"
        res[k] = train(a, tag, f"{LB}/index_D_r{r}_20261109.json", 0, "point")
        interim(a, 5 + r, f"案D {r}回目の学習が終わった（rc 良し＝{res[k]}）。")
        if not res[k]:
            say(f"--- {k} が失敗＝案D をここで止める"); break
        prev_ix, prev_m = f"{LB}/index_D_r{r}_20261109.json", f"{REL}/{tag}"
        prev_tr, prev_dv = f"{REL}/{tag}/probs_train.npz", f"{REL}/{tag}/probs_dev.npz"
    # 追加の変種（17:2x 追補）＝案A の区間1つの重さを 30（pos_weight だけ）にした種0。--only Ax で別に回す（既定の全部には入れない）。
    if "Ax" in only and not stop():
        res["Ax"] = train(a, "dA_weak_wp30_8ep_s0", IXG, 0, "weak", ("--weak_wp", "30"))
        interim(a, 9, f"案A の変種（区間1つの重さ 30）種0 の学習が終わった（rc 良し＝{res['Ax']}）。")
    say("結果＝" + " ".join(f"{k}:{v}" for k, v in res.items()))
    say(f"CHAIN DONE 合計 {(time.time() - t_all) / 60:.1f}分")
