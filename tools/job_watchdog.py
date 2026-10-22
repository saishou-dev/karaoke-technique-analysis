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
ap.add_argument("--jobs", default="", help="ジョブ台帳のパス（既定＝このファイルと同じ場所の watchdog_jobs.json・自己テスト用）")
ap.add_argument("--inbox", default="", help="受信箱のパス（既定＝hisho/routine_inbox・自己テスト用）")
a = ap.parse_args()
INBOX = a.inbox or r"C:\Users\conan\hisho\routine_inbox"
# ---- 2026-10-15 追加：PID 再利用対策 ----------------------------------------
# 旧実装は「登録時刻からの経過」だけで停止を決めていたため、再起動をまたぐと
# 同じ PID を割り当てられた無関係なプロセスを kill し得た（10/15 09:04 に実害の疑い）。
# 対策＝(1) 登録時にマシンの起動時刻を記録し、起動時刻が変わっていたら監視対象は消えたものとして
#         報告のみで終了する。(2) 対象プロセスの「開始時刻」を初めて見た時に控え、
#         kill する直前に PID の現在の開始時刻と突き合わせ、一致した PID だけ停止する。
PID_TOL = 2.0  # 開始時刻の一致とみなす秒差（CIM の丸め対策）
def boot_time():
    """このマシンの最終起動時刻（Unix 秒）。取得できなければ None。"""
    ps = "[int64]([datetimeoffset](Get-CimInstance Win32_OperatingSystem).LastBootUpTime).ToUnixTimeSeconds()"
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, encoding="utf-8", errors="replace")
    s = r.stdout.strip()
    return float(s) if s.lstrip("-").isdigit() else None
def procs():
    """[(pid, 累積CPU分, プロセス名, 開始時刻Unix秒)]。開始時刻が取れない行は 0.0。"""
    # 2026-10-23 追加（本人裁定）＝番犬は番犬を監視しない。
    # 番犬のコマンドラインには --pattern の文字列がそのまま載る。そのため素直なパターン（例 train_crnn_refdiff）を書くと
    # 別の番犬の --pattern 欄にも当たり、停滞と判定したときに無関係な番犬まで一緒に止めてしまう（10/23 05:48 に実害）。
    # パターンの書き方で毎回避けるのは注意力に頼る運用なので、対象から job_watchdog.py を含むプロセスを外す。
    #
    # 2026-10-23 追加その2（本人裁定）＝番犬が毎回起動するこの PowerShell 自身も対象から外す。
    # この問い合わせのコマンドラインには --pattern の文字列が丸ごと入るため、番犬は自分の補助プロセスを
    # 「監視対象」として数えていた。そのため①対象が 0 になったら完了して終わる仕組みが働かず（仕事が終わっても番犬が居座る）、
    # ②停滞と判定したときにその補助まで停止対象に入っていた（10/23 05:48 の報告に出ていた powershell.exe がこれ）。
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '" + a.pattern.replace("'", "''") + "' -and $_.CommandLine -notmatch 'job_watchdog\\.py' -and $_.CommandLine -notmatch 'Get-CimInstance Win32_Process' -and $_.ProcessId -ne " + str(os.getpid()) + " } | "
          'ForEach-Object { "{0}`t{1}`t{2}`t{3}" -f $_.ProcessId, [math]::Round(($_.KernelModeTime + $_.UserModeTime)/1e7/60,3), $_.Name, '
          '[int64]([datetimeoffset]$_.CreationDate).ToUnixTimeSeconds() }')
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = []
    for line in r.stdout.splitlines():
        p = line.strip().split("\t")
        if len(p) >= 3 and p[0].isdigit():
            st = float(p[3]) if len(p) >= 4 and p[3].lstrip("-").isdigit() else 0.0
            out.append((int(p[0]), float(p[1]), p[2], st))
    return out
def note_pids(job, ps):
    """初めて見た PID の開始時刻を台帳に控える（既出の PID は上書きしない）。"""
    seen = job.setdefault("pids", {})
    for pid, _, _, st in ps:
        seen.setdefault(str(pid), st)
    return seen
def classify_kill(job, ps):
    """kill してよい PID と、PID 再利用が疑われる PID に分ける。
       戻り値＝(停止してよい[(pid,name)], 対象消失[(pid,name,理由)])"""
    seen = job.get("pids", {})
    born_after = job.get("started", 0) - PID_TOL  # 監視開始より後に生まれたなら正当な子プロセス
    ok, lost = [], []
    for pid, _, name, st in ps:
        rec = seen.get(str(pid))
        if rec is None:
            # 点検と点検の間に生まれた子（ffmpeg 等）は記録が無い。監視開始より後に生まれていれば正当。
            if st > 0 and st >= born_after: ok.append((pid, name))
            else: lost.append((pid, name, "監視開始(%s)より前に生まれた PID＝別プロセス"
                               % datetime.datetime.fromtimestamp(job.get("started", 0)).strftime("%m-%d %H:%M:%S")))
        elif st <= 0 or rec <= 0:
            lost.append((pid, name, "開始時刻を取得できない"))
        elif abs(st - rec) > PID_TOL:
            lost.append((pid, name, "開始時刻が不一致（記録 %s / 現在 %s）＝PID 再利用"
                         % (datetime.datetime.fromtimestamp(rec).strftime("%m-%d %H:%M:%S"),
                            datetime.datetime.fromtimestamp(st).strftime("%m-%d %H:%M:%S"))))
        else:
            ok.append((pid, name))
    return ok, lost
def reboot_gap(job):
    """登録後にマシンが再起動していたら、その旨の文字列を返す（していなければ None）。"""
    rec = job.get("boot")
    if not rec: return None
    now = boot_time()
    if now is None or abs(now - rec) <= PID_TOL: return None
    return ("登録時のマシン起動時刻 %s に対し現在は %s ＝監視開始後に再起動している"
            % (datetime.datetime.fromtimestamp(rec).strftime("%m-%d %H:%M:%S"),
               datetime.datetime.fromtimestamp(now).strftime("%m-%d %H:%M:%S")))
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

JOBS = a.jobs or os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchdog_jobs.json")
def load_jobs():
    try: return json.load(open(JOBS, encoding="utf-8"))
    except Exception: return []
def save_jobs(j): json.dump(j, open(JOBS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
if a.register:
    j = load_jobs(); j = [x for x in j if x["name"] != a.register]
    job = dict(name=a.register, pattern=a.pattern, log=a.log, extra=a.extra_files, idle=a.idle, started=time.time(),
               last_size=None, last_cpu=None, last_progress=time.time(), seen=False, max_hours=a.max_hours,
               boot=boot_time(), pids={})
    note_pids(job, procs())  # 登録時点で既に走っているプロセスの開始時刻を控える
    j.append(job)
    save_jobs(j); print("registered", a.register, "boot=%s pids=%s" % (job["boot"], sorted(job["pids"]))); sys.exit(0)
if a.once:
    j = load_jobs(); keep = []
    for job in j:
        a.pattern, a.log, a.extra_files, a.name, a.idle, a.max_hours = job["pattern"], job["log"], job.get("extra", ""), job["name"], job.get("idle", 10), job.get("max_hours", 8)
        ps = procs(); cpu = sum(c for _, c, _, _ in ps); size = progress_size(); now = time.time()
        if ps: job["seen"] = True
        note_pids(job, ps)
        if job["last_size"] is None or size > job["last_size"] or (job["last_cpu"] is not None and cpu - job["last_cpu"] >= a.cpu_min): job["last_progress"] = now
        job["last_size"], job["last_cpu"] = size, cpu
        elapsed_min = (now - job["started"]) / 60; idle_min = (now - job["last_progress"]) / 60
        if not ps:
            if job["seen"] or elapsed_min > a.grace:
                inbox(f"{a.name}＝プロセス終了（{elapsed_min:.0f}分）", f"対象プロセスが無くなったため監視を終了。ログ最終サイズ {size} B。ログ＝{a.log}")
                continue
            keep.append(job); continue
        if idle_min >= a.idle or elapsed_min >= a.max_hours * 60:
            reason = f"{idle_min:.0f}分間、ログ・出力・CPU時間の増加なし" if idle_min >= a.idle else f"上限 {a.max_hours}h 超過"
            gap = reboot_gap(job)
            if gap:  # 再起動をまたいだ＝一致する PID でも別物の恐れ。停止せず報告だけ。
                inbox(f"{a.name}＝対象消失（停止はしていない）",
                      f"理由＝{reason}\nただし {gap}。\n再起動前のプロセスは既に消えているため、いま一致している "
                      f"{', '.join(f'{n}:{p}' for p, _, n, _ in ps)} は PID 再利用の別プロセスの可能性がある。\n"
                      f"**停止していない**（旧版はここで kill していた）。経過＝{elapsed_min:.0f}分\nログ＝{a.log}")
                continue
            ok, lost = classify_kill(job, ps)
            killed = []
            for pid, name in ok:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True); killed.append(f"{name}:{pid}")
            msg = (f"理由＝{reason}\n停止＝{', '.join(killed) if killed else '（なし）'}\n経過＝{elapsed_min:.0f}分\nログ＝{a.log}\n再開は原因を直してから。")
            if lost:
                msg += "\n\n対象消失＝停止しなかった PID（開始時刻が記録と一致しない）：\n" + "\n".join(f"- {n}:{p} … {w}" for p, n, w in lost)
            title = f"{a.name}＝停滞を検知して停止" if killed else f"{a.name}＝対象消失（停止はしていない）"
            with open(os.path.join(os.path.dirname(os.path.abspath(a.log)), "WATCHDOG_STALLED.txt"), "w", encoding="utf-8") as f: f.write(datetime.datetime.now().isoformat() + "\n" + msg + "\n")
            inbox(title, msg); continue
        keep.append(job)
    save_jobs(keep)
    # 生存記録（デッドマンスイッチ用）
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchdog_heartbeat.txt"), "w", encoding="utf-8") as f: f.write(datetime.datetime.now().isoformat() + f" jobs={len(keep)}\n")
    sys.exit(0)

if not a.pattern or not a.log: sys.exit("--pattern と --log が必要（常駐モード）")
t0 = time.time(); last_size = progress_size()
_job = dict(boot=boot_time(), pids={}); note_pids(_job, procs())
last_cpu = sum(c for _, c, _, _ in procs()); last_progress = time.time(); seen = False
log_dir = os.path.dirname(os.path.abspath(a.log))
while True:
    time.sleep(a.interval * 60)
    now = time.time(); ps = procs(); cpu = sum(c for _, c, _, _ in ps); size = progress_size()
    if ps: seen = True
    note_pids(_job, ps)
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
        reason = f"{idle_min:.0f}分間、ログ・出力・CPU時間の増加なし" if stalled else f"上限 {a.max_hours}h 超過"
        gap = reboot_gap(_job)
        if gap:
            inbox(f"{a.name}＝対象消失（停止はしていない）",
                  f"理由＝{reason}\nただし {gap}。\nいま一致している "
                  f"{', '.join(f'{n}:{p}' for p, _, n, _ in ps)} は PID 再利用の別プロセスの可能性がある。\n"
                  f"**停止していない**。経過＝{elapsed_min:.0f}分\nログ＝{a.log}")
            break
        ok, lost = classify_kill(_job, ps)
        killed = []
        for pid, name in ok:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True); killed.append(f"{name}:{pid}")
        msg = f"理由＝{reason}\n停止＝{', '.join(killed) if killed else '（なし）'}\n経過＝{elapsed_min:.0f}分\nログ＝{a.log}\n再開は原因を直してから。"
        if lost:
            msg += "\n\n対象消失＝停止しなかった PID（開始時刻が記録と一致しない）：\n" + "\n".join(f"- {n}:{p} … {w}" for p, n, w in lost)
        with open(os.path.join(log_dir, "WATCHDOG_STALLED.txt"), "w", encoding="utf-8") as f: f.write(datetime.datetime.now().isoformat() + "\n" + msg + "\n")
        inbox(f"{a.name}＝停滞を検知して停止" if killed else f"{a.name}＝対象消失（停止はしていない）", msg)
        break
