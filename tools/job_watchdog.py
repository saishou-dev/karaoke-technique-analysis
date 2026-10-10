# -*- coding: utf-8 -*-
"""無人ジョブの番犬（2026-09-29・本人指示「不必要なタスクの実行は無駄。自律的に確認できるように」）。
   監視対象＝コマンドラインが --pattern に一致するプロセス群と、--log のファイル。
   判定＝--interval 分ごとに「ログのサイズ増加」または「対象プロセスの累積CPU時間の増加（--cpu-min 分以上）」があれば進行中。
        --idle 分のあいだどちらも無ければ停滞＝対象プロセスを全て停止し、<log と同じ場所>/WATCHDOG_STALLED.txt と
        hisho/routine_inbox/<日時>_watchdog.md（<!-- unread -->）を書いて終了する。
   終了＝対象プロセスが 0 になったら「完了」として受信箱に1行書いて終了。--max-hours を超えたら強制停止。
   使い方（分離起動）:
     Start-Process python -ArgumentList '"C:\\Users\\conan\\hisho\\ops\\tools\\job_watchdog.py" --pattern "marks_20260928|ffmpeg.*ビデオ素材" --log "G:\\...\\progress.log" --name マーク抽出 --idle 10' -WindowStyle Hidden
"""
import argparse, os, re, subprocess, sys, time, datetime, json
if sys.stdout is not None: sys.stdout.reconfigure(encoding="utf-8")  # pythonw（タスクスケジューラ）では stdout が None
ap = argparse.ArgumentParser()
ap.add_argument("--pattern", default="", help="対象プロセスのコマンドラインに一致する正規表現")
ap.add_argument("--log", default="", help="進捗ログのパス（サイズ増加で進行中と判定）")
ap.add_argument("--name", default="job", help="受信箱に書くジョブ名")
ap.add_argument("--idle", type=float, default=10, help="この分数、進行がなければ停滞")
ap.add_argument("--interval", type=float, default=2, help="チェック間隔（分）")
ap.add_argument("--cpu-min", type=float, default=0.2, help="この分以上CPU時間が増えれば進行中とみなす")
ap.add_argument("--max-hours", type=float, default=8)
ap.add_argument("--grace", type=float, default=3, help="起動直後にプロセスが見つからなくても待つ分数")
ap.add_argument("--extra-files", default="", help="サイズ/更新で進行判定に使う追加パス（;区切り・フォルダ可）")
ap.add_argument("--once", action="store_true", help="常駐せず watchdog_jobs.json の全ジョブを1回点検して終了（タスクスケジューラ用）")
ap.add_argument("--register", default="", help="watchdog_jobs.json にジョブを登録して終了（--pattern/--log/--name/--idle を使う）")
a = ap.parse_args()
INBOX = r"C:\Users\conan\hisho\routine_inbox"
def procs():
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '" + a.pattern.replace("'", "''") + "' -and $_.ProcessId -ne " + str(os.getpid()) + " } | "
          'ForEach-Object { "{0}`t{1}`t{2}" -f $_.ProcessId, [math]::Round(($_.KernelModeTime + $_.UserModeTime)/1e7/60,3), $_.Name }')
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = []
    for line in r.stdout.splitlines():
        p = line.strip().split("\t")
        if len(p) == 3 and p[0].isdigit(): out.append((int(p[0]), float(p[1]), p[2]))
    return out
def size_of(path):
    tot = 0
    if os.path.isdir(path):
        for root, _, fs in os.walk(path):
            for f in fs:
                try: tot += os.path.getsize(os.path.join(root, f))
                except OSError: pass
    elif os.path.exists(path):
        tot = os.path.getsize(path)
    return tot
def progress_size():
    s = size_of(a.log)
    for p in filter(None, a.extra_files.split(";")): s += size_of(p)
    return s
def inbox(title, body):
    os.makedirs(INBOX, exist_ok=True)
    fn = os.path.join(INBOX, datetime.datetime.now().strftime("%Y-%m-%d_%H%M") + f"_watchdog_{re.sub(r'[^0-9A-Za-z一-龥ぁ-んァ-ン]', '', a.name)[:20]}.md")
    with open(fn, "w", encoding="utf-8") as f: f.write(f"<!-- unread -->\n# 番犬報告：{title}\n\n{body}\n")
    return fn

JOBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchdog_jobs.json")
def load_jobs():
    try: return json.load(open(JOBS, encoding="utf-8"))
    except Exception: return []
def save_jobs(j): json.dump(j, open(JOBS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
if a.register:
    j = load_jobs(); j = [x for x in j if x["name"] != a.register]
    j.append(dict(name=a.register, pattern=a.pattern, log=a.log, extra=a.extra_files, idle=a.idle, started=time.time(), last_size=None, last_cpu=None, last_progress=time.time(), seen=False, max_hours=a.max_hours))
    save_jobs(j); print("registered", a.register); sys.exit(0)
if a.once:
    j = load_jobs(); keep = []
    for job in j:
        a.pattern, a.log, a.extra_files, a.name, a.idle, a.max_hours = job["pattern"], job["log"], job.get("extra", ""), job["name"], job.get("idle", 10), job.get("max_hours", 8)
        ps = procs(); cpu = sum(c for _, c, _ in ps); size = progress_size(); now = time.time()
        if ps: job["seen"] = True
        if job["last_size"] is None or size > job["last_size"] or (job["last_cpu"] is not None and cpu - job["last_cpu"] >= a.cpu_min): job["last_progress"] = now
        job["last_size"], job["last_cpu"] = size, cpu
        elapsed_min = (now - job["started"]) / 60; idle_min = (now - job["last_progress"]) / 60
        if not ps:
            if job["seen"] or elapsed_min > a.grace:
                inbox(f"{a.name}＝プロセス終了（{elapsed_min:.0f}分）", f"対象プロセスが無くなったため監視を終了。ログ最終サイズ {size} B。ログ＝{a.log}")
                continue
            keep.append(job); continue
        if idle_min >= a.idle or elapsed_min >= a.max_hours * 60:
            killed = []
            for pid, _, name in ps:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True); killed.append(f"{name}:{pid}")
            reason = f"{idle_min:.0f}分間、ログ・出力・CPU時間の増加なし" if idle_min >= a.idle else f"上限 {a.max_hours}h 超過"
            msg = f"理由＝{reason}\n停止＝{', '.join(killed)}\n経過＝{elapsed_min:.0f}分\nログ＝{a.log}\n再開は原因を直してから。"
            with open(os.path.join(os.path.dirname(os.path.abspath(a.log)), "WATCHDOG_STALLED.txt"), "w", encoding="utf-8") as f: f.write(datetime.datetime.now().isoformat() + "\n" + msg + "\n")
            inbox(f"{a.name}＝停滞を検知して停止", msg); continue
        keep.append(job)
    save_jobs(keep)
    # 生存記録（デッドマンスイッチ用）
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchdog_heartbeat.txt"), "w", encoding="utf-8") as f: f.write(datetime.datetime.now().isoformat() + f" jobs={len(keep)}\n")
    sys.exit(0)

if not a.pattern or not a.log: sys.exit("--pattern と --log が必要（常駐モード）")
t0 = time.time(); last_size = progress_size(); last_cpu = sum(c for _, c, _ in procs()); last_progress = time.time(); seen = False
log_dir = os.path.dirname(os.path.abspath(a.log))
while True:
    time.sleep(a.interval * 60)
    now = time.time(); ps = procs(); cpu = sum(c for _, c, _ in ps); size = progress_size()
    if ps: seen = True
    if size > last_size or cpu - last_cpu >= a.cpu_min: last_progress = now
    last_size, last_cpu = size, cpu
    elapsed_min = (now - t0) / 60; idle_min = (now - last_progress) / 60
    if not ps:
        if seen or elapsed_min > a.grace:
            inbox(f"{a.name}＝プロセス終了（{elapsed_min:.0f}分）", f"対象プロセスが無くなったため監視を終了。ログ最終サイズ {size} B。停滞停止ではない（正常終了か別要因）。ログ＝{a.log}")
            break
        continue
    stalled = idle_min >= a.idle; over = elapsed_min >= a.max_hours * 60
    if stalled or over:
        killed = []
        for pid, _, name in ps:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True); killed.append(f"{name}:{pid}")
        reason = f"{idle_min:.0f}分間、ログ・出力・CPU時間の増加なし" if stalled else f"上限 {a.max_hours}h 超過"
        msg = f"理由＝{reason}\n停止＝{', '.join(killed)}\n経過＝{elapsed_min:.0f}分\nログ＝{a.log}\n再開は原因を直してから。"
        with open(os.path.join(log_dir, "WATCHDOG_STALLED.txt"), "w", encoding="utf-8") as f: f.write(datetime.datetime.now().isoformat() + "\n" + msg + "\n")
        inbox(f"{a.name}＝停滞を検知して停止", msg)
        break
