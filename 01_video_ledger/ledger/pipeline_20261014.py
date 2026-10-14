# -*- coding: utf-8 -*-
"""回収パイプライン監督 2026-10-14 **WAO** 版（pipeline_20260915.py の派生・Windows python・タスクスケジューラ kv_pipeline_20261014 で起動）。
   工程：probe → crop（帯pts表＋ゲート）→ calib → read（WSLランナー×4）→ verify → sync → g1（帯ptsゲート）→ DONE（各工程スクリプトは 9/15 版をそのまま呼ぶ）
   10/6 の特殊事情＝同一セッションで機種が途中で変わった（前半 9曲 DAM Ai／後半 23曲 WAO）。9/15 版との差：
     (1) 対象動画＝rename_mapping.csv の machine=wao の動画だけ（kv_config_20261014.Cfg1014.videos()）。n_videos もその数。
     (2) probe は対象動画の一覧ファイル（pipe_dir/probe_videos.txt）を launch_probe_20261014.sh に渡して起動（他機種の動画を拾わない・
         スクリプトファイル経由＝launch_read と同じ型。9/15・9/24 はインライン起動が無反応だった）。起動結果（pgrep）をログに残し、
         90秒進展なし＆プロセス無しなら1回だけ再起動、それでも駄目なら STOPPED（手動起動の手順を理由に書く）。
     (3) PROBE ok の後に「プローブ行の集合＝対象動画の集合」をゲート（余計な行・不足があれば停止）。
     (4) verify＝verify_20261014.py --machine wao（Hearts CSV 固定・対象動画だけ）→ dataset/verify_20261014_wao.csv。
     (5) g1 の出力＝dataset/G1_events_20261014_wao.csv（KV_G1_OUT で差し替え・Ai 版と衝突しない）。G1_events_all への結合は行わない。
   ゲートで止まったら dataset/pipeline_<S>/PIPELINE_STOPPED.txt に理由を書いて終了コード2。再開：--from <step>（済みはスキップ）。
   使い方: python scripts/pipeline_20261014.py 20261014 [--from probe]
"""
import os, sys, time, glob, subprocess, datetime, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20261014 import Cfg1014, to_wsl
from kv_config_20260915 import PYTHON, WSL, WSL_ROOT

STEPS = ["probe", "crop", "calib", "read", "verify", "sync", "g1"]
ap = argparse.ArgumentParser(); ap.add_argument("session", nargs="?", default=None); ap.add_argument("--from", dest="frm", default="probe", choices=STEPS)
ap.add_argument("--until", dest="until", default="g1", choices=STEPS); ap.add_argument("--read-timeout-h", type=float, default=5.0)
a = ap.parse_args(); C = Cfg1014(a.session); S = C.S
os.makedirs(C.pipe_dir, exist_ok=True); SC = os.path.join(C.root, "scripts")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", KV_SESSION=S)

def wsl(cmd):
    return subprocess.run(WSL + [cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
def run_py(script, extra=(), env=None):
    p = subprocess.run([PYTHON, os.path.join(SC, script), S, *extra], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env or ENV)
    C.log(f"### {script} {' '.join(extra)} rc={p.returncode}\n{p.stdout[-4000:]}\n{p.stderr[-1500:]}")
    return p.returncode
def stopped(reason):
    C.log("PIPELINE STOPPED: " + reason)
    open(os.path.join(C.pipe_dir, "PIPELINE_STOPPED.txt"), "w", encoding="utf-8").write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} session={S} machine=wao\n{reason}\n対処＝RUNBOOK_20260915.md「ゲートで止まった時」\n")
    raise SystemExit(2)
def n_done():
    return len([p for p in glob.glob(os.path.join(C.results_dir, "*", "counter_events.csv")) if ".mp4.part" not in p])
def readers():
    r = wsl("pgrep -fa 'read_counters_wa[o]' | wc -l")
    try: return int(r.stdout.strip() or 0)
    except ValueError: return -1
def probe_procs():
    r = wsl(f"pgrep -f '{C.probe_pgrep}' | wc -l")
    try: return int(r.stdout.strip() or 0)
    except ValueError: return -1
def launch_probe():
    lst = C.write_probe_list()
    r = wsl(f"KV_SESSION={S} KV_PROBE_SCRIPT={C.probe_script} KV_PROBE_LOG='{to_wsl(C.probe_log)}' bash '{WSL_ROOT}/scripts/launch_probe_20261014.sh' '{lst}'")
    C.log("PROBE launched: " + r.stdout.strip().replace("\n", " | ")[:300] + (" ERR " + r.stderr.strip()[:300] if r.stderr.strip() else ""))
def want(step): return STEPS.index(a.frm) <= STEPS.index(step) <= STEPS.index(a.until)

C.log(f"PIPELINE(wao) start session={S} pid={os.getpid()} from={a.frm} until={a.until}")
sp = os.path.join(C.pipe_dir, "PIPELINE_STOPPED.txt")
if os.path.exists(sp): os.replace(sp, sp + f".{datetime.datetime.now():%H%M%S}.old")
targets = C.videos(); n_videos = len(targets)
if n_videos == 0: stopped(f"machine=wao の改名済み動画が {C.vid_dir} に無い（rename_videos_20261014.py --apply が先・rename_mapping.csv の machine 列を確認）")
C.log(f"TARGET videos(wao)={n_videos}: " + "；".join(targets)[:600])

# ---- 1) probe（WSL・cv2）----
if want("probe"):
    rows = C.probe_rows()
    if len(rows) < n_videos:
        C.log(f"PROBE launch: rows={len(rows)} < videos={n_videos}")
        launch_probe()
        t0 = time.time(); last = -1; last_change = time.time(); relaunched = 0
        while time.time() - t0 < 2 * 3600:
            n = len(C.probe_rows())
            if n != last: C.log(f"PROBE rows={n}/{n_videos}"); last = n; last_change = time.time()
            if n >= n_videos and os.path.exists(C.probe_log) and "WROTE" in open(C.probe_log, encoding="utf-8", errors="replace").read(): break
            if time.time() - last_change > 90 and probe_procs() == 0:
                if relaunched >= 1:
                    stopped(f"プローブが起動しない／途中で消えた rows={n}/{n_videos}（{C.probe_log} を確認。手動起動＝WSL で "
                            f"KV_SESSION={S} KV_PROBE_SCRIPT={C.probe_script} KV_PROBE_LOG='{to_wsl(C.probe_log)}' bash '{WSL_ROOT}/scripts/launch_probe_20261014.sh' '{to_wsl(C.probe_list)}' → --from probe で再開）")
                C.log("PROBE WARN: 90秒進展なし・プロセス無し → 再起動"); launch_probe(); relaunched += 1; last_change = time.time()
            time.sleep(30)
        if len(C.probe_rows()) < n_videos: stopped(f"プローブが {len(C.probe_rows())}/{n_videos} 本で止まった（{C.probe_log} を確認）")
    pr = {r["video"] for r in C.probe_rows()}; tv = set(targets)
    if pr != tv: stopped(f"プローブ行と対象動画(wao)が一致しない：余計={sorted(pr - tv)[:5]} 不足={sorted(tv - pr)[:5]}（{C.probe_csv} を確認）")
    C.log(f"PROBE ok rows={len(C.probe_rows())}")

# ---- 2) crop＋帯pts表＋ゲート ----
if want("crop"):
    rc = run_py("wincrop_20260915.py")
    fails = C.gate_failures()
    if rc != 0 or fails:
        stopped("帯クロップのゲートで停止：" + "；".join(f"{f['video']}: {f['reason']}" for f in fails) + (f"（rc={rc}）" if rc not in (0, 2) else ""))
    n_bands = len([f for f in glob.glob(os.path.join(C.bands_dir, f"*_{S}_*.mp4")) if ".part" not in f])
    n_pts = len([f for f in os.listdir(C.pts_band_dir) if f.endswith(".csv") and not f.startswith("_")])
    if n_bands != len(C.probe_rows()) or n_pts != n_bands: stopped(f"帯動画 {n_bands}・帯pts表 {n_pts}・プローブ行 {len(C.probe_rows())} が一致しない")
    C.log(f"CROP ok bands={n_bands} pts_tables={n_pts}")

# ---- 2b) calib（帯内 ROI の自動較正・WSL cv2）----
if want("calib"):
    calib_dir = os.path.join(C.root, "dataset", f"calib_{S}"); os.makedirs(calib_dir, exist_ok=True)
    r = wsl(f"cd ~/karaoke-vision && KV_SESSION={S} .venv/bin/python '{WSL_ROOT}/scripts/calib_roi_20260915.py' {S} >> '{WSL_ROOT}/dataset/calib_{S}/calib.log' 2>&1; tail -n 1 '{WSL_ROOT}/dataset/calib_{S}/calib.log'")
    C.log("CALIB " + r.stdout.strip()[-300:] + (" ERR " + r.stderr.strip()[:200] if r.stderr.strip() else ""))
    bands = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(C.bands_dir, f"*_{S}_*.mp4")) if ".part" not in f)
    missing = [b for b in bands if not os.path.exists(os.path.join(C.results_dir, b, "roi.json"))]
    if missing: C.log(f"CALIB WARN: roi.json が無い {len(missing)}/{len(bands)} 曲は既定 ROI（roi_wao.json）で読む: " + "；".join(missing)[:400])
    else: C.log(f"CALIB ok roi.json={len(bands)}/{len(bands)}")
    try:
        import csv as _csv
        low = [f"{x['video'][:24]}/{x['counter']}={x['score_med']}" for x in _csv.DictReader(open(os.path.join(calib_dir, "calib_summary.csv"), encoding="utf-8-sig"))
               if x.get("score_med") not in ("", None) and float(x["score_med"]) < 0.25]
        if low: C.log(f"CALIB WARN: 照合スコア低 {len(low)} 件（既定 ROI との差が大きければ目視）: " + "；".join(low)[:400])
    except Exception as e: C.log(f"CALIB WARN: summary 読めず {e}")

# ---- 3) read（WSL・4ランナー）----
target = len(C.probe_rows())
if want("read"):
    if n_done() < target:
        r = wsl(f"rm -rf /tmp/claims_{S}; KV_SESSION={S} KV_TARGET={target} bash '{WSL_ROOT}/scripts/launch_read_20260915.sh'")
        C.log("READ launched: " + r.stdout.strip().replace("\n", " | ")[:300] + (" ERR " + r.stderr.strip()[:200] if r.stderr.strip() else ""))
        t1 = time.time(); last = n_done(); stall = 0
        while time.time() - t1 < a.read_timeout_h * 3600:
            n = n_done(); rd = readers(); C.log(f"READ done={n}/{target} readers={rd}")
            if n >= target: break
            stall = stall + 1 if (n == last and rd == 0) else 0; last = n
            if stall >= 3:
                for d in glob.glob(os.path.join(C.results_dir, "*")):
                    if os.path.isdir(d) and not os.path.exists(os.path.join(d, "counter_events.csv")):
                        for f in glob.glob(os.path.join(d, "*")):
                            if os.path.basename(f) != "roi.json": os.remove(f)
                        if not os.listdir(d): os.rmdir(d)
                r = wsl(f"rm -rf /tmp/claims_{S}; KV_SESSION={S} KV_TARGET={target} bash '{WSL_ROOT}/scripts/launch_read_20260915.sh'")
                C.log("READ relaunched: " + r.stdout.strip().replace("\n", " | ")[:200]); stall = 0
            time.sleep(60)
        if n_done() < target: stopped(f"読取が {n_done()}/{target} 本で時間切れ（{a.read_timeout_h}h）。status_read_20260915.sh で確認し --from read で再開")
    C.log(f"READ ok done={n_done()}/{target}")

# ---- 4) verify（Hearts CSV 突合・機種別）----
if want("verify"):
    rc = run_py("verify_20261014.py", ["--machine", "wao"])
    if rc != 0: C.log("VERIFY WARN: 突合できず（Hearts CSV か rename_mapping.csv が無い）→ 後で --from verify --until verify")

# ---- 5) sync（録音 mapping.csv が要る・無ければ WARN）----
if want("sync"):
    if os.path.exists(C.sync_csv): C.log(f"SYNC skip（既存 {C.sync_csv}）")
    elif os.path.exists(os.path.join(C.wav_dir, "mapping.csv")):
        rc = run_py("sync_20260915.py")
        if rc != 0: C.log("SYNC WARN: 失敗（ログ参照）→ G1 は audio_time 空欄で生成される")
    else: C.log(f"SYNC WARN: 録音 mapping.csv が無い {C.wav_dir} → G1 は audio_time 空欄。録音配置後に --from sync")

# ---- 6) G1（帯ptsのみ・ゲート）＋照合 ----
if want("g1"):
    rc = run_py("g1_and_match_20260915.py", env=dict(ENV, KV_G1_OUT=C.g1_csv))
    if rc != 0: stopped("G1生成のゲートで停止（pipeline.log の『G1 STOP』行を参照）")
    if not os.path.exists(C.g1_csv): stopped(f"G1 CSV が生成されていない {C.g1_csv}")

open(os.path.join(C.pipe_dir, "PIPELINE_DONE.txt"), "w", encoding="utf-8").write(
    f"finished {datetime.datetime.now():%Y-%m-%d %H:%M:%S} session={S} machine=wao from={a.frm} until={a.until} read_done={n_done()}/{target}\n"
    f"G1={C.g1_csv if os.path.exists(C.g1_csv) else '(未生成)'}\nsync={'あり' if os.path.exists(C.sync_csv) else '無し'}\nverify={'あり' if os.path.exists(C.verify_csv) else '無し'}\n")
C.log("PIPELINE ALL DONE")
