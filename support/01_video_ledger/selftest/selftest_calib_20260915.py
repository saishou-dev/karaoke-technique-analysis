# -*- coding: utf-8 -*-
"""calib 工程（calib_roi_20260915.py＋run_read_20260915.sh の roi.json 対応）の自己テスト（2026-09-15・Windows python・本番ファイルは触らない）。
   (a) 9/15 の帯 27曲：calib の出力 roi.json が「17:42 の再較正（dataset/rescan_20260915/roi/*.json＝verify 21/27 を出した ROI）」と一致 → verify 21/27 が再現される。
       不感帯（10px 以下は既定 ROI）で既定に戻るカウンターを持つ曲は、較正 ROI で読み直して Hearts CSV と突合（旧 verify と同じか良いこと）。
   (b) 9/11 の帯 3曲（CHEERS・Answer・lovin'＝verify PERFECT）：calib → 較正 ROI で読取（read_counters_wao.py v0.8d）→ g1_and_match_20260915.py（帯pts）→
       既存 G1（G1_events_20260911_bandpts_20260914.csv）と (counter,count_value) の集合が一致し video_time_sec の差 ≤0.1s（安定読取のジッタ 6 フレーム）→ PASS。
   出力：dataset/_selftest_calib_20260915/SELFTEST_RESULT.txt。使い方：python scripts/selftest_calib_20260915.py"""
import os, sys, csv, json, glob, subprocess, datetime, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import ROOT, WSL, WSL_ROOT, PYTHON, DAM_DIR, Cfg
T = os.path.join(ROOT, "dataset", "_selftest_calib_20260915"); os.makedirs(T, exist_ok=True); WT = f"{WSL_ROOT}/dataset/_selftest_calib_20260915"
DEADBAND = int(os.environ.get("KV_CALIB_DEADBAND", "10")); TOL = 0.1
DEF = json.load(open(os.path.join(ROOT, "templates/wao_20260816/roi_wao.json"), encoding="utf-8"))
M = {"shakuri": "shakuriCount", "kobushi": "kobushiCount", "fall": "fallCount", "vibrato": "vibratoCount", "accent": "accentCount", "hammering": "hammeringOnCount"}
lines = []; ok_all = True
def log(m): lines.append(m); print(m, flush=True)
def wsl(cmd, timeout=1800): return subprocess.run(WSL + [cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
def run_read(res_wsl, res_win, names):
    """較正 ROI で読取（並列・完了済みはスキップ）"""
    todo = [v for v in names if not os.path.exists(os.path.join(res_win, v, "counter_events.csv"))]
    if not todo: return "skip"
    vs = " ".join(f'"{v}"' for v in todo)
    sh = f"""export LANG=C.UTF-8; cd ~/karaoke-vision
for v in {vs}; do out='{res_wsl}/'"$v"; mkdir -p "$out"
  ( .venv/bin/python read_counters_wao.py '{WSL_ROOT}/bands_wao/'"$v".mp4 "$out" '{WSL_ROOT}/templates/wao_20260816/cluster_reps_wao_ext.npy' "$out/roi.json" > "$out/read_log.txt" 2>&1 ) &
done; wait; ls '{res_wsl}'/*/counter_events.csv | wc -l"""
    p = os.path.join(T, "run_read.sh"); open(p, "w", encoding="utf-8", newline="\n").write(sh)
    r = wsl(f"bash '{WT}/run_read.sh'", timeout=2400); return r.stdout.strip().replace("\n", " | ")[-200:] + (" ERR " + r.stderr.strip()[-200:] if r.stderr.strip() else "")
def hearts_final(session, names, res_win):
    """Hearts CSV の最終値と読取の最終値を突合（verify_rescan_20260915.py と同じ規則）"""
    hearts = sorted(glob.glob(os.path.join(DAM_DIR, "dam_ai-hearts_full_*.csv")))[-1]
    d = pd.read_csv(hearts, encoding="utf-8-sig"); d["dt"] = pd.to_datetime(d["scoringDateTime"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    C = Cfg(session); ren = {r["renamed"][:-4]: r["played_at"] for r in csv.DictReader(open(C.rename_csv, encoding="utf-8-sig")) if r.get("played_at")}
    out = {}
    for v in names:
        hit = d[d["dt"] == pd.Timestamp(ren[v])]; ev = pd.read_csv(os.path.join(res_win, v, "counter_events.csv"))
        fin = {c: int(ev[ev["counter"] == c]["value"].max()) if (ev["counter"] == c).any() else 0 for c in M}
        diff = {c: fin[c] - int(hit.iloc[0][M[c]]) for c in M}; out[v] = ";".join(f"{c}{x:+d}" for c, x in diff.items() if x) or "PERFECT"
    return out
# ---- (a) 9/15 ----
ref_dir = os.path.join(ROOT, "dataset", "rescan_20260915", "roi"); res15 = os.path.join(T, "results_0915")
if len(glob.glob(os.path.join(res15, "*", "roi.json"))) < 27:
    r = wsl(f"cd ~/karaoke-vision && KV_FORCE=1 KV_CALIB_DEADBAND={DEADBAND} KV_RESULTS_DIR='{WT}/results_0915' KV_CALIB_DIR='{WT}/calib_0915' .venv/bin/python '{WSL_ROOT}/scripts/calib_roi_20260915.py' 20260915 >> '{WT}/calib_0915.log' 2>&1; tail -n 1 '{WT}/calib_0915.log'")
    log("(a) calib: " + r.stdout.strip()[-200:])
same = diff = 0; details = []; reverted = {}
for p in sorted(glob.glob(os.path.join(ref_dir, "*.json"))):
    name = os.path.basename(p)[:-5]; q = os.path.join(res15, name, "roi.json")
    if not os.path.exists(q): diff += 1; details.append(f"{name}: roi.json 無し"); continue
    a = json.load(open(p, encoding="utf-8")); b = json.load(open(q, encoding="utf-8")); bad = []
    for c in a:
        sh = max(abs(a[c][0] - DEF[c][0]), abs(a[c][1] - DEF[c][1]))
        exp = a[c] if sh > DEADBAND else DEF[c]
        if sh <= DEADBAND: reverted.setdefault(name, []).append(f"{c}({a[c][0]-DEF[c][0]:+d},{a[c][1]-DEF[c][1]:+d})")
        if list(b[c]) != list(exp): bad.append(f"{c} {exp}->{b[c]}")
    if bad: diff += 1; details.append(f"{name}: " + "; ".join(bad))
    else: same += 1
pa = (same == 27 and diff == 0)
log(f"(a) 9/15 ROI 一致 {same}/27 不一致 {diff}（不感帯 {DEADBAND}px で既定に戻したカウンター {sum(len(v) for v in reverted.values())} 件・{len(reverted)} 曲）→ {'PASS' if pa else 'FAIL'}")
for d in details: log("    " + d)
if reverted:
    names = sorted(reverted); log("(a) 既定に戻した曲を較正 ROI で読み直し（Hearts 突合）: " + "；".join(f"{n[:20]} {','.join(reverted[n])}" for n in names))
    log("(a) read: " + run_read(f"{WT}/results_0915", res15, names))
    old = pd.read_csv(os.path.join(ROOT, "dataset", "verify_20260915_rescan.csv"), encoding="utf-8-sig").set_index("video")
    try:
        new = hearts_final("20260915", names, res15)
        for n in names:
            o = old.loc[n, "diff"] if old.loc[n, "status"] == "DIFF" else "PERFECT"; o = "PERFECT" if pd.isna(o) else o
            okv = (new[n] == "PERFECT") or (new[n] == o); pa &= bool(okv)
            log(f"(a) {n[:26]:28s} 旧 verify={o}  新={new[n]} → {'PASS' if okv else 'FAIL'}")
    except Exception as e: pa = False; log(f"(a) 突合できず {e}")
ok_all &= pa; log(f"(a) → {'PASS' if pa else 'FAIL'}")
# ---- (b) 9/11 ----
V = ["CHEERS_20260911_105621", "Answer_20260911_130234", "lovin'_20260911_113934"]; res11 = os.path.join(T, "results_0911")
if any(not os.path.exists(os.path.join(res11, v, "roi.json")) for v in V):
    vs = " ".join(f'"{v}"' for v in V)
    r = wsl(f"cd ~/karaoke-vision && KV_FORCE=1 KV_CALIB_DEADBAND={DEADBAND} KV_RESULTS_DIR='{WT}/results_0911' KV_CALIB_DIR='{WT}/calib_0911' .venv/bin/python '{WSL_ROOT}/scripts/calib_roi_20260915.py' 20260911 {vs} >> '{WT}/calib_0911.log' 2>&1; tail -n 1 '{WT}/calib_0911.log'")
    log("(b) calib: " + r.stdout.strip()[-200:])
for v in V:
    q = os.path.join(res11, v, "roi.json")
    if os.path.exists(q): b = json.load(open(q)); log(f"(b) {v[:20]} 適用ずれ: " + json.dumps({c: (b[c][0] - DEF[c][0], b[c][1] - DEF[c][1]) for c in b}))
log("(b) read: " + run_read(f"{WT}/results_0911", res11, V))
env = dict(os.environ, KV_SESSION="20260911", KV_RESULTS_DIR=res11, KV_PTS_BAND_DIR=os.path.join(ROOT, "dataset", "pts_bands_20260911"), KV_G1_OUT=os.path.join(T, "G1_0911_selftest.csv"),
           KV_PIPE_DIR=T, KV_BLIND_DIR=os.path.join(T, "blind"), PYTHONIOENCODING="utf-8")
p = subprocess.run([PYTHON, os.path.join(ROOT, "scripts", "g1_and_match_20260915.py"), "20260911"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
log(f"(b) g1 rc={p.returncode} " + (p.stdout.strip().splitlines() or [""])[-1][:200])
pb = p.returncode == 0 and os.path.exists(env["KV_G1_OUT"])
if pb:
    N = pd.read_csv(env["KV_G1_OUT"], encoding="utf-8-sig"); O = pd.read_csv(os.path.join(ROOT, "dataset", "G1_events_20260911_bandpts_20260914.csv"), encoding="utf-8-sig")
    for v in V:
        # 9/12 に rescan_missing_20260911 の目視追記が counter_events.csv に注入された曲（*.pre_inject_20260912 が残る）は、注入分を除いて比較する
        ce = os.path.join(ROOT, "results_wao_20260911", v, "counter_events.csv"); pre = ce + ".pre_inject_20260912"
        cur = pd.read_csv(ce); ref = pd.read_csv(pre) if os.path.exists(pre) else cur
        inj = set(zip(cur["counter"], cur["value"])) - set(zip(ref["counter"], ref["value"]))
        new_ce = pd.read_csv(os.path.join(res11, v, "counter_events.csv"))
        ident = sorted(map(tuple, ref[ref["delta"] > 0][["counter", "value", "frame"]].values.tolist())) == sorted(map(tuple, new_ce[new_ce["delta"] > 0][["counter", "value", "frame"]].values.tolist()))
        n = N[N["video"] == v]; o = O[O["video"] == v].drop_duplicates(["counter", "count_value"])
        o = o[[(c, x) not in inj for c, x in zip(o["counter"], o["count_value"])]]
        key = lambda d: sorted(zip(d["counter"], d["count_value"].astype(int)))
        eq = key(n) == key(o); m = n.merge(o, on=["counter", "count_value"], suffixes=("_new", "_old"))
        dt = (m["video_time_sec_new"] - m["video_time_sec_old"]).abs().max() if len(m) else float("nan")
        okv = eq and len(m) and dt <= TOL; pb &= bool(okv)
        log(f"(b) {v[:24]:26s} events new={len(n)} old={len(o)}（注入分 {len(inj)} 件を除外） 集合一致={eq} 読取器出力と同一={ident} max|dt|={dt:.3f}s(許容{TOL}) → {'PASS' if okv else 'FAIL'}")
ok_all &= pb
log(f"(b) 9/11 3曲 既存 G1 と一致 → {'PASS' if pb else 'FAIL'}")
log(f"RESULT {'PASS' if ok_all else 'FAIL'} deadband={DEADBAND}px {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
open(os.path.join(T, "SELFTEST_RESULT.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
raise SystemExit(0 if ok_all else 1)
