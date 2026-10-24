# -*- coding: utf-8 -*-
"""C2（入力を変える比較）の「12エポックまで伸ばすと成績が上がるか」を、現行の入力・種0/1/2 の3本で直列に確かめる無人ジョブ（2026-10-24）。

なぜ必要か。
2026-10-24 07:20 の裁定は「8エポックで頭打ちだから 12エポックは不要」だったが、
同日 17:15 の比較表 v7（7条件・種3つ）で、7条件のうち3条件（現行 種1・＋お手本 種1・v6）は
最後の8エポック目が頂点だったことが分かった＝「8エポックで頭打ち」は種0 の2条件だけの性質で、
種を変えると再現しない。
エポックが足りていないなら、ここまでの7条件すべてが「途中で切った値」の比較になり、
入力の軸（お手本との音程差は効くか）の結論そのものが保留になる。
そこでまず現行の入力だけを 12エポックで種0/1/2 の3本回し、採用エポックと dev F値の伸びを見て
8エポックの採用を見直す。
3本を同時に走らせると `--threads 12` を取り合って1本あたりの時間が伸びるため、直列（1本ずつ）で回す。

すること（この順番で3本・前の本が失敗しても次へ進む）。
1. c2a_base_12ep_s0  （現行の入力・seed 0・12エポック）
2. c2a_base_12ep_s1  （現行の入力・seed 1・12エポック）
3. c2a_base_12ep_s2  （現行の入力・seed 2・12エポック）
いずれも `--refdiff` は付けない（お手本との音程差の2列を入れない＝現行の入力）。

見込みの所要時間。
8エポックの実測が約3,900秒（約487秒/エポック）なので、12エポックは約5,900秒＝3本で約4.9時間。

無人ジョブの作り（02_rules「🐕 無人ジョブの番犬」とメモリ unattended-jobs-no-dialogs に従う）。
  ダイアログレス＝標準出力だけに書く（入力待ちをしない）。
  優先度＝自分と子の python を Idle（BelowNormal より下）に下げる。本人が 22:30 ごろ帰宅して PC を使うため必須。
  心拍＝60秒ごとに1行、この台本のログへ書く（止まっていないことを人の目で確かめられるように）。
  停滞の自己検知＝走っている本のログが --stall_min 分のびなければ、その本だけ止めて次へ進む。
  結果はファイルへ＝各本の rc と所要秒をこのログに書き、最後に「CHAIN DONE」を書く（番犬の --done-pattern）。

使い方（PowerShell・Start-Process で分離起動する）:
  $S = "G:\\研究アーカイブ\\成果物\\動画解析\\g3v2_20260928\\scripts"
  $O = "G:\\研究アーカイブ\\成果物\\動画解析\\g3v2_20260928\\out"
  Start-Process python -ArgumentList "`"$S\\run_c2_12ep_chain_20261024.py`"" `
    -WorkingDirectory "G:\\研究アーカイブ\\成果物\\動画解析\\g3v2_20260928" `
    -RedirectStandardOutput "$O\\c2_12ep_chain_20261024.log" `
    -RedirectStandardError  "$O\\c2_12ep_chain_20261024.err" -WindowStyle Hidden
"""
import argparse
import ctypes
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
TRAIN = os.path.join(HERE, "train_crnn_refdiff_20261023.py")

IDLE_PRIORITY_CLASS = 0x00000040
PROCESS_SET_INFORMATION = 0x0200


def say(s):
    """標準出力へ1行書いて即座に流す（pythonw では stdout が None になるのでガードする）。"""
    if sys.stdout is None:
        return
    try:
        sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {s}\n")
        sys.stdout.flush()
    except Exception:
        pass


def _kernel32():
    """kernel32 を取り、ハンドルを扱う関数の型を宣言する。
    宣言しないと 64bit のハンドルが int へ切り詰められ、SetPriorityClass が黙って False を返す
    （2026-10-24 07:21 の初回起動で実際に rc=False になった）。"""
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.OpenProcess.restype = ctypes.c_void_p
    k.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    return k


def lower_self():
    """自分の優先度を Idle に下げる（Windows では子プロセスがこの優先度を引き継ぐ）。"""
    try:
        k = _kernel32()
        ok = k.SetPriorityClass(k.GetCurrentProcess(), IDLE_PRIORITY_CLASS)
        say(f"優先度 自分＝Idle に設定 rc={bool(ok)} pid={os.getpid()}")
    except Exception as e:
        say(f"優先度 自分の設定に失敗（続行）＝{e}")


def lower_pid(pid):
    """子の python の優先度を Idle に下げ直す（引き継ぎに頼らず明示する）。"""
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


def run_stage(stage, idx, total, stall_min, poll_sec):
    """1本を走らせ、心拍を書きながら終わりを待つ。(rc, 所要秒) を返す。"""
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
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                try:
                    p.wait(timeout=120)
                except Exception:
                    pass
                break
    rc = p.returncode
    sec = time.time() - t0
    say(f"=== [{idx}/{total}] {tag} 終了 rc={rc} 所要 {sec:.1f}秒（{sec / 60:.1f}分） .err={os.path.getsize(err_path) if os.path.exists(err_path) else '?'}B")
    return rc, sec


if __name__ == "__main__":
    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--stall_min", type=float, default=45,
                    help="走っている本のログがこの分数のびなければ停滞としてその本だけ止める（1エポック約8分なので既定45）")
    ap.add_argument("--poll_sec", type=float, default=60, help="心拍の間隔（秒）")
    a = ap.parse_args()

    E, T = str(a.epochs), str(a.threads)
    stages = [
        dict(tag="c2a_base_12ep_s0", log="c2_base_12ep_s0",
             args=["--epochs", E, "--threads", T, "--seed", "0"]),
        dict(tag="c2a_base_12ep_s1", log="c2_base_12ep_s1",
             args=["--epochs", E, "--threads", T, "--seed", "1"]),
        dict(tag="c2a_base_12ep_s2", log="c2_base_12ep_s2",
             args=["--epochs", E, "--threads", T, "--seed", "2"]),
    ]

    t_all = time.time()
    say(f"CHAIN START pid={os.getpid()} 本数={len(stages)} エポック={E} スレッド={T} "
        f"停滞判定={a.stall_min}分 心拍={a.poll_sec}秒")
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
