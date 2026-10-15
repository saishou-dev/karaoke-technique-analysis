# -*- coding: utf-8 -*-
r"""10/14 回収の無人オーケストレータ（2026-10-15 02:1x・セッション 0b34fe39 が分離起動）。
   役割＝前セッションの Opus Agent（セッション終了で消滅）の後を継ぎ、
     転送完了待ち → rename dry-run → ゲート（AMBIGUOUS/STOP なら停止）→ --apply → schtasks /run kv_pipeline_20261014
     → 番犬起動 → pipeline.log を監視 → 進捗ファイルを更新 → 終了（FINAL 行を書く）。
   進捗＝G:\hisho\routine_inbox\2026-10-15_kaishu_20261014_進捗.md（メインセッションが監督に使う）。
   判断が要る所（転送欠落・AMBIGUOUS・STOP・停滞）は自分で直さず FINAL status=NEEDS_JUDGMENT で止める。
   起動：powershell Start-Process python -ArgumentList '"<this>"' -WindowStyle Hidden
"""
import os, sys, re, time, subprocess, datetime, glob
S = "20261014"
VID = r"G:\研究アーカイブ\ビデオ素材\20261014"
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
SC = os.path.join(ROOT, "scripts")
PIPE = os.path.join(ROOT, "dataset", "pipeline_" + S)
PLOG = os.path.join(PIPE, "pipeline.log")
PROG = r"G:\hisho\routine_inbox\2026-10-15_kaishu_20261014_進捗.md"
OWN = os.path.join(PIPE, "_orchestrator.log")
PY = r"C:\Users\conan\AppData\Local\Programs\Python\Python312\python.exe"
WD = r"G:\hisho\ops\tools\job_watchdog.py"
TASK = "kv_pipeline_20261014"
os.makedirs(PIPE, exist_ok=True)
state = {"phase": "転送待ち", "transfer": "", "rename": "", "task": "", "watchdog": "", "pipe": "", "notes": []}
t0 = time.time()

def now(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def own(msg):
    with open(OWN, "a", encoding="utf-8") as f: f.write(f"{now()} {msg}\n")
def write_prog(final=None):
    lines = ["<!-- unread -->" if final else "状態：進行中（オーケストレータ稼働中）",
             f"# 10/14 回収 進捗（更新 {now()}・経過 {int((time.time()-t0)/60)} 分）", "",
             f"- 工程：{state['phase']}", f"- 転送：{state['transfer']}", f"- 改名：{state['rename']}",
             f"- タスク：{state['task']}", f"- 番犬：{state['watchdog']}", f"- パイプライン：{state['pipe']}"]
    if state["notes"]: lines += ["", "## メモ"] + [f"- {n}" for n in state["notes"]]
    if os.path.exists(PLOG):
        try:
            tl = open(PLOG, encoding="utf-8", errors="replace").read().splitlines()[-12:]
            lines += ["", "## pipeline.log 末尾", "```"] + tl + ["```"]
        except Exception as e: lines.append(f"(log 読めず {e})")
    if final: lines += ["", f"FINAL status={final} at={now()}"]
    tmp = PROG + ".tmp"
    open(tmp, "w", encoding="utf-8").write("\n".join(lines) + "\n"); os.replace(tmp, PROG)
def finish(status, note=None):
    if note: state["notes"].append(note)
    own(f"FINAL {status} {note or ''}"); write_prog(final=status); sys.exit(0)
def ps(cmd):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
def procs(pattern):
    r = ps(f"[Console]::OutputEncoding=[Text.Encoding]::UTF8; (Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match '{pattern}' }}).Count")
    try: return int(r.stdout.strip() or 0)
    except ValueError: return -1

own("start"); write_prog()
# ---- 1) 転送完了待ち ----
res = os.path.join(VID, "_transfer_result.txt"); tlog = os.path.join(VID, "_transfer.log")
while not os.path.exists(res):
    n_ok = 0
    if os.path.exists(tlog):
        txt = open(tlog, encoding="utf-8", errors="replace").read(); n_ok = txt.count(" OK VID_"); n_fail = txt.count(" FAIL")
        state["transfer"] = f"{n_ok}/30 転送済み（FAIL {n_fail}）"
    if procs("mtp_transfer_20261014") == 0:
        time.sleep(20)
        if not os.path.exists(res): finish("NEEDS_JUDGMENT", f"転送プロセスが消えたが _transfer_result.txt が無い（{state['transfer']}）＝再転送の判断が要る")
    if time.time() - t0 > 120 * 60: finish("NEEDS_JUDGMENT", "転送が120分経っても終わらない")
    write_prog(); time.sleep(30)
rt = open(res, encoding="utf-8", errors="replace").read().strip().lstrip("\ufeff")
state["transfer"] = rt; own("transfer " + rt)
m = re.search(r"DONE ok=(\d+) total=(\d+)", rt)
if not m or m.group(1) != m.group(2): finish("NEEDS_JUDGMENT", f"転送結果が完全ではない：{rt}（RUNBOOK §2 再転送→ --from probe）")
n_mp4 = len([p for p in glob.glob(os.path.join(VID, "*.mp4")) if ".part" not in p]); state["notes"].append(f"mp4 {n_mp4} 本（.part 除く）")
# ---- 2) 改名 ----
state["phase"] = "改名 dry-run"; write_prog()
def rename(apply):
    args = [PY, os.path.join(SC, "rename_videos_20261014.py"), S] + (["--apply"] if apply else [])
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    own(f"rename apply={apply} rc={p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-800:]}"); return p
p = rename(False); out = p.stdout
summ = next((l for l in out.splitlines() if l.startswith("TIME_WINDOW=")), "(集計行なし)"); state["rename"] = "dry-run: " + summ
if p.returncode != 0 or "STOP" in out: finish("NEEDS_JUDGMENT", f"改名 dry-run が STOP/rc={p.returncode}：{out[-600:]}")
amb = re.search(r"AMBIGUOUS=(\d+)", out); nm = re.search(r"NO_MATCH=(\d+)", out); tw = re.search(r"TIME_WINDOW=(\d+)", out)
if amb and int(amb.group(1)) > 0: finish("NEEDS_JUDGMENT", "AMBIGUOUS あり＝手で裁定して改名・rename_mapping.csv に行追加が要る（RUNBOOK §2-3）")
if tw and int(tw.group(1)) != 30: state["notes"].append(f"TIME_WINDOW={tw.group(1)}（30 ではない・NO_MATCH={nm.group(1) if nm else '?'}）")
if tw and int(tw.group(1)) < 28: finish("NEEDS_JUDGMENT", "改名対象が28本未満＝採点行と動画の対応が崩れている可能性")
state["phase"] = "改名 --apply"; write_prog()
p = rename(True); state["rename"] += " / apply: " + next((l for l in p.stdout.splitlines() if l.startswith("RENAMED")), p.stdout[-200:].replace("\n", " "))
if p.returncode != 0 or "RENAMED" not in p.stdout: finish("NEEDS_JUDGMENT", f"改名 --apply 失敗 rc={p.returncode}：{p.stdout[-400:]} {p.stderr[-300:]}")
# ---- 3) タスク起動（60分ルール＝スケジューラ経由）----
state["phase"] = "パイプライン起動"; write_prog()
r = ps(f"[Console]::OutputEncoding=[Text.Encoding]::UTF8; schtasks.exe /query /tn {TASK} /xml | Select-String -Pattern 'Arguments|Command' | ForEach-Object {{ $_.Line.Trim() }}")
state["task"] = "定義: " + r.stdout.strip().replace("\n", " ")[:300]
if "run_pipeline_hidden_20261014" not in r.stdout and "pipeline_20261014" not in r.stdout:
    finish("NEEDS_JUDGMENT", f"タスク {TASK} の定義が 20261014 版を指していない：{r.stdout[:300]} {r.stderr[:200]}")
r = ps(f"[Console]::OutputEncoding=[Text.Encoding]::UTF8; schtasks.exe /run /tn {TASK}"); own("schtasks run: " + r.stdout.strip() + r.stderr.strip())
state["task"] += " / run: " + (r.stdout.strip() or r.stderr.strip())[:120]
time.sleep(20)
if procs("pipeline_20261014.py") == 0:
    time.sleep(40)
    if procs("pipeline_20261014.py") == 0:
        errs = sorted(glob.glob(os.path.join(PIPE, "_task_stderr_*.txt")), key=os.path.getmtime)
        tail = open(errs[-1], encoding="utf-8", errors="replace").read()[-500:] if errs else "(stderr 無し)"
        finish("NEEDS_JUDGMENT", f"タスクを /run したが pipeline_20261014.py のプロセスが見えない：{tail}")
# ---- 4) 番犬 ----
wd_cmd = (f"Start-Process -FilePath '{PY}' -ArgumentList '\"{WD}\" --pattern \"pipeline_20261014|run_pipeline_hidden_20261014\" "
          f"--log \"{PLOG}\" --name 回収10/14パイプライン --idle 25 --interval 2 --max-hours 8 --grace 5' -WindowStyle Hidden")
r = ps(wd_cmd); own("watchdog: " + r.stdout.strip() + r.stderr.strip())
time.sleep(5); state["watchdog"] = f"起動 procs={procs('job_watchdog.py')}（idle 25分・max 8h）"
# ---- 5) 監視 ----
state["phase"] = "パイプライン実行中"; last_sz = -1; last_change = time.time(); t_start = time.time()
while True:
    sz = os.path.getsize(PLOG) if os.path.exists(PLOG) else 0
    if sz != last_sz: last_sz = sz; last_change = time.time()
    tail = ""
    if os.path.exists(PLOG):
        ls = open(PLOG, encoding="utf-8", errors="replace").read().splitlines()
        steps = [l for l in ls if re.search(r"PROBE ok|CROP ok|CALIB ok|READ done=|VERIFY|SYNC|G1 |STOPPED|finished", l)]
        tail = steps[-1][:200] if steps else (ls[-1][:200] if ls else "")
    state["pipe"] = f"log {sz}B・最新: {tail}"
    stopped = os.path.exists(os.path.join(PIPE, "PIPELINE_STOPPED.txt"))
    done = [p for p in glob.glob(os.path.join(PIPE, "*")) if re.search(r"DONE|finished|FINISHED", os.path.basename(p), re.I)]
    alive = procs("pipeline_20261014.py")
    stalled = glob.glob(os.path.join(PIPE, "WATCHDOG_STALLED*"))
    write_prog()
    if stopped: finish("NEEDS_JUDGMENT", "PIPELINE_STOPPED.txt: " + open(os.path.join(PIPE, "PIPELINE_STOPPED.txt"), encoding="utf-8", errors="replace").read()[:500])
    if stalled: finish("NEEDS_JUDGMENT", "番犬が停滞を検知して停止した（WATCHDOG_STALLED）")
    if done or (alive == 0 and time.time() - t_start > 120):
        time.sleep(10)
        if "finished" in (open(PLOG, encoding="utf-8", errors="replace").read()[-2000:] if os.path.exists(PLOG) else "") or done:
            finish("DONE", "パイプライン完走（verify/G1 の検算はメインが行う）")
        finish("NEEDS_JUDGMENT", "pipeline プロセスが消えたが finished も STOPPED も無い（_task_stderr を確認）")
    if time.time() - t_start > 9 * 3600: finish("NEEDS_JUDGMENT", "9時間を超えた")
    time.sleep(60)
