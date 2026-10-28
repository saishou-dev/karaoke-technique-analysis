# -*- coding: utf-8 -*-
r"""C2（検出器）の学習曲線＝学習曲を 25%・50%・75% に減らした3本を直列で回す無人ジョブ（2026-10-28）。

なぜ必要か。
「目標200曲・1万件は適正か」に客観的に答えるため、学習データ量だけを変えて
試験F値（9/24 の28曲・未見）がまだ伸びているか頭打ちかを測る。
100%（学習150曲）は 10/23 の c2a_base_8ep（8エポック・seed 0）をそのまま基準に使い、再実行しない。

そろえる条件（10/23 の基準と同じ）。
  学習台本＝train_crnn_refdiff_20261023.py／エポック 8／スレッド 12／seed 0／現行の入力（--refdiff なし）。
  しきい値の決め方＝dev（20260911・30曲）／試験＝20260924 の28曲（学習に混ぜない）。
  変えるのは --index で渡す index ファイルだけ＝学習曲の本数（台本は書き換えない）。

1本ずつ直列で回す（GPU/CPU の負荷を重ねない）。前の本が失敗しても次へ進む。
無人ジョブの作り＝ダイアログレス・優先度 Idle・心拍60秒・停滞の自己検知・最後に CHAIN DONE。
停滞したときに止めるのは、この台本が自分で起こした子の python だけ（本人のアプリには触らない）。

使い方（PowerShell・Start-Process で分離起動する）:
  $S = "G:\研究アーカイブ\成果物\動画解析\g3v2_20260928\scripts"
  $O = "G:\研究アーカイブ\成果物\動画解析\g3v2_20260928\out\c2_learning_curve_20261028"
  Start-Process python -ArgumentList "`"$S\run_c2_learning_curve_20261028.py`"" `
    -WorkingDirectory "G:\研究アーカイブ\成果物\動画解析\g3v2_20260928" `
    -RedirectStandardOutput "$O\c2_lc_chain_20261028.log" `
    -RedirectStandardError  "$O\c2_lc_chain_20261028.err" -WindowStyle Hidden
"""
import argparse
import ctypes
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out", "c2_learning_curve_20261028")
TRAIN = os.path.join(HERE, "train_crnn_refdiff_20261023.py")

IDLE_PRIORITY_CLASS = 0x00000040
PROCESS_SET_INFORMATION = 0x0200


def say(s):
    if sys.stdout is None:
        return
    try:
        sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {s}\n")
        sys.stdout.flush()
    except Exception:
        pass


def _kernel32():
    """ハンドルを扱う関数の型を宣言する（宣言しないと 64bit のハンドルが切り詰められ黙って失敗する）。"""
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.OpenProcess.restype = ctypes.c_void_p
    k.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    return k


def lower_self():
    try:
        k = _kernel32()
        ok = k.SetPriorityClass(k.GetCurrentProcess(), IDLE_PRIORITY_CLASS)
        say(f"優先度 自分＝Idle に設定 rc={bool(ok)} pid={os.getpid()}")
    except Exception as e:
        say(f"優先度 自分の設定に失敗（続行）＝{e}")


def lower_pid(pid):
    try:
        k = _kernel32()
        h = k.OpenProcess(PROCESS_SET_INFORMATION, 0, int(pid))
        if not h:
            say(f"優先度 子 pid={pid} のハンドルが取れない（引き継ぎに任せる）")
            return False
        ok = k.SetPriorityClass(h, IDLE_PRIORITY_CLASS)
        k.CloseHandle(h)
        say(f"優先度 子 pid={pid}＝Idle に設定 rc={bool(ok)}")
        return bool(ok)
    except Exception as e:
        say(f"優先度 子 pid={pid} の設定に失敗（続行）＝{e}")
        return False


def stop_own_child(p):
    """この台本が起こした子の python だけを止める（Popen のハンドル経由＝他のプロセスには触らない）。"""
    try:
        p.terminate()
        p.wait(timeout=120)
    except Exception:
        try:
            p.kill()
            p.wait(timeout=60)
        except Exception:
            pass


def run_stage(stage, idx, total, stall_min, poll_sec):
    tag, log_name, extra = stage["tag"], stage["log"], stage["args"]
    log_path = os.path.join(OUT, log_name + ".log")
    err_path = os.path.join(OUT, log_name + ".err")
    cmd = [sys.executable, TRAIN, "--tag", tag] + extra
    say(f"=== [{idx}/{total}] {tag} 開始 ログ={log_path}")
    say(f"=== [{idx}/{total}] {tag} コマンド＝{' '.join(cmd[1:])}")
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(log_path, "w", encoding="utf-8") as fo, open(err_path, "w", encoding="utf-8") as fe:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fo, stderr=fe,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        say(f"=== [{idx}/{total}] {tag} 子の pid={p.pid}")
        lower_pid(p.pid)
        last_size, last_grow, beats = -1, time.time(), 0
        while p.poll() is None:
            time.sleep(poll_sec)
            beats += 1
            try:
                size = os.path.getsize(log_path)
            except OSError:
                size = -1
            if size > last_size:
                last_size, last_grow = size, time.time()
            idle_min = (time.time() - last_grow) / 60
            say(f"心拍 {beats} {tag} 経過 {(time.time() - t0) / 60:.1f}分 "
                f"ログ {size}B 無成長 {idle_min:.1f}分 pid={p.pid}")
            if idle_min >= stall_min:
                say(f"!!! [{idx}/{total}] {tag} 停滞を自己検知（ログが {idle_min:.0f}分のびない）＝この本だけ止めて次へ進む")
                stop_own_child(p)
                break
    rc = p.returncode
    sec = time.time() - t0
    say(f"=== [{idx}/{total}] {tag} 終了 rc={rc} 所要 {sec:.1f}秒（{sec / 60:.1f}分） "
        f".err={os.path.getsize(err_path) if os.path.exists(err_path) else '?'}B")
    return rc, sec


if __name__ == "__main__":
    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stall_min", type=float, default=30,
                    help="走っている本のログがこの分数のびなければ停滞としてその本だけ止める（学習曲が少ないので既定30）")
    ap.add_argument("--poll_sec", type=float, default=60)
    ap.add_argument("--only", default="", help="回す条件を絞る（例 50 ＝50%だけ・カンマ区切り）")
    a = ap.parse_args()

    E, T, S = str(a.epochs), str(a.threads), str(a.seed)
    rel = "out/c2_learning_curve_20261028"
    stages = []
    for pct in (25, 50, 75):
        stages.append(dict(
            tag=f"c2_lc{pct}_8ep", log=f"c2_lc{pct}_8ep",
            args=["--epochs", E, "--threads", T, "--seed", S, "--index", f"{rel}/index_lc{pct}.json"]))
    if a.only:
        want = set(x.strip() for x in a.only.split(",") if x.strip())
        stages = [s for s in stages if s["tag"].replace("c2_lc", "").replace("_8ep", "") in want]

    t_all = time.time()
    say(f"CHAIN START pid={os.getpid()} 本数={len(stages)} エポック={E} スレッド={T} seed={S} "
        f"停滞判定={a.stall_min}分 心拍={a.poll_sec}秒")
    say("100%（学習150曲）は 10/23 の c2a_base_8ep をそのまま基準に使う＝再実行しない")
    lower_self()
    say("順番＝" + " → ".join(s["tag"] for s in stages))

    results = []
    for i, st in enumerate(stages, 1):
        rc, sec = run_stage(st, i, len(stages), a.stall_min, a.poll_sec)
        results.append((st["tag"], rc, sec))
        say(f"--- 進捗 {i}/{len(stages)} 完了 ここまでの合計 {(time.time() - t_all) / 60:.1f}分")

    say("結果の一覧（タグ・rc・所要秒）")
    for tag, rc, sec in results:
        say(f"  {tag} rc={rc} {sec:.1f}秒")
    ok = sum(1 for _, rc, _ in results if rc == 0)
    say(f"CHAIN DONE 本数={len(results)} rc=0 が {ok} 本 合計 {(time.time() - t_all) / 60:.1f}分")
