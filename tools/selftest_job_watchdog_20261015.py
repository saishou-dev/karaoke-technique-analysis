# -*- coding: utf-8 -*-
"""番犬 job_watchdog.py の自己テスト（2026-10-15・PID 再利用対策の検証）。

旧実装の弱点＝経過時間が「登録時刻」基準で、再起動をまたぐと PID が再利用された
無関係なプロセスを kill し得た（10/15 09:04 の疑い）。
本テストは実プロセスを1本立て、台帳の「開始時刻」を細工して3通りを確認する。

  T1 登録＝マシン起動時刻と、走っている対象プロセスの開始時刻が台帳に入る。
  T2 開始時刻が不一致（＝PID 再利用を模す）→ **kill しない**・報告に「対象消失」が出る。
  T3 開始時刻が一致 → これまでどおり kill する。
  T4 登録後に再起動した（boot を細工）→ **kill しない**・報告に「再起動」が出る。

本物の watchdog_jobs.json と routine_inbox には触らない（--jobs／--inbox で隔離する）。
使い方: python selftest_job_watchdog_20261015.py
出力  : 標準出力に PASS/FAIL、終了コード 0＝全 PASS。
"""
import json, os, subprocess, sys, tempfile, time, shutil, glob

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
WD = os.path.join(HERE, "job_watchdog.py")
MARK = "kv_selftest_watchdog_20261015"
fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  … " + detail) if detail else ""))
    if not cond:
        fails.append(name)


def alive(pid):
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue) -ne $null"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip().lower().startswith("true")


def run_wd(args):
    return subprocess.run([sys.executable, WD] + args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def inbox_texts(d):
    out = []
    for f in glob.glob(os.path.join(d, "*.md")):
        out.append(open(f, encoding="utf-8").read())
    return out


def clear(d):
    for f in glob.glob(os.path.join(d, "*.md")):
        os.remove(f)


tmp = tempfile.mkdtemp(prefix="wd_selftest_")
jobs = os.path.join(tmp, "watchdog_jobs.json")
inbox = os.path.join(tmp, "inbox"); os.makedirs(inbox)
logf = os.path.join(tmp, "progress.log")
open(logf, "w").write("start\n")
victim = None
try:
    # 対象プロセス＝十分長く眠るだけの python（コマンドラインに MARK を含める）
    victim = subprocess.Popen([sys.executable, "-c", f"import time;_='{MARK}';time.sleep(900)"])
    time.sleep(3)

    print("T1 登録で起動時刻と PID 開始時刻を控える")
    r = run_wd(["--register", "selftest", "--pattern", MARK, "--log", logf,
                "--idle", "999", "--jobs", jobs, "--inbox", inbox])
    j = json.load(open(jobs, encoding="utf-8"))
    check("台帳に1件登録された", len(j) == 1, r.stdout.strip() + r.stderr.strip()[:200])
    job = j[0]
    check("boot（マシン起動時刻）が記録された", bool(job.get("boot")), str(job.get("boot")))
    check("対象 PID の開始時刻が記録された",
          str(victim.pid) in job.get("pids", {}) and job["pids"][str(victim.pid)] > 0,
          "pids=" + json.dumps(job.get("pids", {})))

    print("T2 開始時刻が不一致（PID 再利用）＝停止しない")
    clear(inbox)
    job["pids"][str(victim.pid)] = job["pids"][str(victim.pid)] - 3600  # 1時間前に生まれた別物に見せる
    job["idle"] = 0  # 停滞判定を即座に成立させる
    job["last_progress"] = time.time() - 3600
    json.dump([job], open(jobs, "w", encoding="utf-8"), ensure_ascii=False)
    r = run_wd(["--once", "--jobs", jobs, "--inbox", inbox])
    time.sleep(1)
    check("対象プロセスは生きている（kill されていない）", alive(victim.pid), "pid=%d" % victim.pid)
    t = "\n".join(inbox_texts(inbox))
    check("報告に「対象消失」が出る", "対象消失" in t, t[:200].replace("\n", " / "))
    check("報告に PID 再利用の理由が出る", "PID 再利用" in t, t[:300].replace("\n", " / "))
    check("台帳からジョブが落ちている", json.load(open(jobs, encoding="utf-8")) == [])

    print("T4 登録後に再起動した場合＝停止しない")
    clear(inbox)
    job2 = dict(job)
    job2["pids"] = {str(victim.pid): None}
    ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                         "[int64]([datetimeoffset](Get-CimInstance Win32_Process -Filter "
                         f"\"ProcessId={victim.pid}\").CreationDate).ToUnixTimeSeconds()"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    job2["pids"] = {str(victim.pid): float(ps.stdout.strip())}  # 開始時刻は正しい
    job2["boot"] = job2["boot"] - 86400  # ただし登録時の起動時刻とは別＝再起動した
    job2["idle"] = 0; job2["last_progress"] = time.time() - 3600
    json.dump([job2], open(jobs, "w", encoding="utf-8"), ensure_ascii=False)
    r = run_wd(["--once", "--jobs", jobs, "--inbox", inbox])
    time.sleep(1)
    check("対象プロセスは生きている（kill されていない）", alive(victim.pid), "pid=%d" % victim.pid)
    t = "\n".join(inbox_texts(inbox))
    check("報告に「再起動」の説明が出る", "再起動" in t and "対象消失" in t, t[:300].replace("\n", " / "))

    print("T3 開始時刻が一致＝これまでどおり停止する")
    clear(inbox)
    job3 = dict(job2); job3["boot"] = job["boot"]  # 再起動していない
    job3["idle"] = 0; job3["last_progress"] = time.time() - 3600
    json.dump([job3], open(jobs, "w", encoding="utf-8"), ensure_ascii=False)
    r = run_wd(["--once", "--jobs", jobs, "--inbox", inbox])
    time.sleep(2)
    check("対象プロセスは停止した", not alive(victim.pid), "pid=%d" % victim.pid)
    t = "\n".join(inbox_texts(inbox))
    check("報告が「停滞を検知して停止」になる", "停滞を検知して停止" in t, t[:200].replace("\n", " / "))

    print("T6 点検の合間に生まれた子プロセス（記録が無い PID）は従来どおり停止する")
    clear(inbox)
    victim2 = subprocess.Popen([sys.executable, "-c", f"import time;_='{MARK}';time.sleep(900)"])
    time.sleep(3)
    job6 = dict(job); job6["boot"] = job["boot"]
    job6["pids"] = {}                      # 記録ゼロ＝一度も見ていない状態を作る
    job6["started"] = time.time() - 600    # 監視開始は10分前＝victim2 はその後に生まれた
    job6["idle"] = 0; job6["last_progress"] = time.time() - 3600
    json.dump([job6], open(jobs, "w", encoding="utf-8"), ensure_ascii=False)
    r = run_wd(["--once", "--jobs", jobs, "--inbox", inbox])
    time.sleep(2)
    check("監視開始より後に生まれた子は停止される", not alive(victim2.pid), "pid=%d" % victim2.pid)
    try:
        if victim2.poll() is None: victim2.kill()
    except Exception:
        pass

    print("T5 本物の台帳と受信箱に触れていない")
    real_jobs = os.path.join(HERE, "watchdog_jobs.json")
    check("本物の watchdog_jobs.json は読めるまま",
          (not os.path.exists(real_jobs)) or isinstance(json.load(open(real_jobs, encoding="utf-8")), list))
finally:
    try:
        if victim and victim.poll() is None:
            victim.kill()
    except Exception:
        pass
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("RESULT: " + ("ALL PASS" if not fails else "FAIL %d 件 = %s" % (len(fails), fails)))
sys.exit(0 if not fails else 1)
