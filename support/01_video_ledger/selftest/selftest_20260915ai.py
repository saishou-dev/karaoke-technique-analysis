# -*- coding: utf-8 -*-
"""DAM Ai 版パイプラインのセルフテスト（2026-09-15）。0807（Ai）の元動画2本（虹 12:08・ダーリン 14:07＝午後の手動座標曲）で
   プローブ→クロップ→帯pts表→読取（v0.8）→G1 を通し、0807 の既存 G1（dataset/G1_events_all.csv session=20260807・video=VID_20260807_HHMMSS）と
   カウント値・時刻を突合する。出力はすべて dataset/_selftest_20260915ai/（KV_AI_BASE）。本番ファイルは触らない。
   各工程は済みをスキップ（プローブ・クロップ・読取を先に手で流していてもよい）。
   使い方: python scripts/selftest_20260915ai.py
"""
import os, sys, subprocess, time, glob
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; T = os.path.join(ROOT, "dataset", "_selftest_20260915ai")
os.environ["KV_AI_BASE"] = T; os.environ["PYTHONIOENCODING"] = "utf-8"
from kv_config_20260915ai import CfgAi, PYTHON, WSL, WSL_ROOT
S = "20260807"; C = CfgAi(S)
VIDS = ["虹_20260807_120836", "ダーリン_20260807_140751"]
SC = os.path.join(ROOT, "scripts"); res = []
def ok(name, cond, detail=""):
    res.append((name, bool(cond), detail)); print(("PASS " if cond else "FAIL ") + name + (" — " + detail if detail else ""), flush=True)
def wsl(cmd): return subprocess.run(WSL + [cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")

# ① プローブ
if len(C.probe_rows()) < len(VIDS):
    vids = " ".join(f"'/mnt/g/研究アーカイブ/ビデオ素材/{S}/{v}.mp4'" for v in VIDS)
    r = wsl(f"cd ~/karaoke-vision && KV_AI_BASE='{C.wsl_base}' .venv/bin/python '{WSL_ROOT}/scripts/ai_probe_20260915ai.py' {S} {vids}")
    print(r.stdout[-1500:], r.stderr[-500:])
rows = {r["video"]: r for r in C.probe_rows()}
ok("①プローブ 2本", all(v in rows for v in VIDS), "; ".join(f"{v}: ({rows[v]['xL']},{rows[v]['yL']},{rows[v]['w']},{rows[v]['h']}) s={rows[v]['scale']} a={rows[v]['angle_deg']}" for v in VIDS if v in rows))

# ② クロップ＋帯pts表＋ゲート
need = [v for v in VIDS if not (os.path.exists(os.path.join(C.bands_dir, v + ".mp4")) and os.path.exists(os.path.join(C.pts_band_dir, v + ".csv")))]
if need:
    p = subprocess.run([PYTHON, os.path.join(SC, "wincrop_20260915ai.py"), S], capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ))
    print(p.stdout[-800:], p.stderr[-500:])
gate = {r["video"]: r for r in (pd.read_csv(C.gate_csv, encoding="utf-8").to_dict("records") if os.path.exists(C.gate_csv) else [])}
ok("②クロップ＋帯pts表 ゲート PASS（落ち0）", all(v in gate and gate[v]["status"] == "PASS" and int(gate[v]["dropped"]) == 0 for v in VIDS),
   "; ".join(f"{v}: {gate[v]['status']} band={gate[v]['n_band']} src={gate[v]['n_src']} dropped={gate[v]['dropped']} last={gate[v]['last_band']}/{gate[v]['last_src']}" for v in VIDS if v in gate))

# ③ 読取（WSL・ランナー2本・完了まで待つ）
def n_done(): return sum(os.path.exists(os.path.join(C.results_dir, v, "counter_events.csv")) for v in VIDS)
if n_done() < len(VIDS):
    r = wsl(f"rm -rf /tmp/claims_ai_{S}; KV_AI_BASE='{C.wsl_base}' KV_SESSION={S} KV_TARGET={len(VIDS)} KV_RUNNERS=2 bash '{WSL_ROOT}/scripts/launch_read_20260915ai.sh'")
    print("READ launched:", r.stdout.strip()[-300:], r.stderr.strip()[-200:])
    t0 = time.time()
    while n_done() < len(VIDS) and time.time() - t0 < 50 * 60: time.sleep(30)
ok("③読取 2本完了", n_done() == len(VIDS), f"done={n_done()}")
for v in VIDS:
    lg = os.path.join(C.results_dir, v, "read_log.txt")
    if os.path.exists(lg): print(f"--- read_log {v}\n" + open(lg, encoding="utf-8", errors="replace").read()[-700:])

# ④ G1（帯ptsのみ）
p = subprocess.run([PYTHON, os.path.join(SC, "g1_and_match_20260915ai.py"), S], capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ))
print(p.stdout[-600:], p.stderr[-600:])
ok("④G1 生成（rc=0・pts_source=pts_bands_ai）", p.returncode == 0 and os.path.exists(C.g1_csv), f"rc={p.returncode} {C.g1_csv}")

# ⑤ 0807 既存 G1 との突合（VID_ ID ↔ 曲名 ID は時刻で対応）
if os.path.exists(C.g1_csv):
    G = pd.read_csv(C.g1_csv, encoding="utf-8-sig"); R = pd.read_csv(os.path.join(ROOT, "dataset", "G1_events_all.csv"), encoding="utf-8-sig")
    R = R[R["session"] == int(S)].copy(); R["video"] = R["video"].str.replace("VID_", "", regex=False)
    G["vid"] = G["video"].str.extract(r"_(\d{8}_\d{6})$")[0]
    lines = []; allmatch = True; n_all = 0; n_close = 0; outliers = []
    for v in VIDS:
        key = v.split("_", 1)[1]; g = G[G["vid"] == key]; r = R[R["video"] == key]
        for c in C.counters:
            gc = g[g["counter"] == c].sort_values("count_value"); rc = r[r["counter"] == c].sort_values("count_value")
            same_n = len(gc) == len(rc); fin_g = int(gc["count_value"].max()) if len(gc) else 0; fin_r = int(rc["count_value"].max()) if len(rc) else 0
            M = gc.merge(rc, on="count_value", suffixes=("_new", "_ref"))
            dv = (M["video_time_sec_new"] - M["video_time_sec_ref"]) if len(M) else pd.Series(dtype=float)
            di = (M["video_time_sec_new"] - M["video_time_idx_sec"]) if len(M) and "video_time_idx_sec" in M else pd.Series(dtype=float)
            lines.append(f"{v} {c}: events new={len(gc)} ref={len(rc)} final new={fin_g} ref={fin_r} | Δt(new−ref pts) n={len(M)} med={dv.median():+.3f} max|Δ|={dv.abs().max() if len(dv) else 0:.3f} | Δ(new−ref idx) med={di.median():+.3f} max={di.abs().max() if len(di) else 0:.3f}")
            if not same_n or fin_g != fin_r or (len(dv) and abs(dv.median()) > 0.1): allmatch = False
            n_all += len(M); n_close += int((dv.abs() <= 0.3).sum())
            for _, m in M[dv.abs() > 0.3].iterrows():
                # 0.3s 超の差は「遮蔽区間」（安定読取−最初の変化 ≧1s が新旧どちらかにある）でのみ許容＝フラッシュアンカーが可読フレームに依存する既知の性質（D4）
                gap_new = m["stable_read_time_sec_new"] - m["video_time_sec_new"]; gap_ref = m["stable_read_time_sec_ref"] - m["video_time_sec_ref"]
                occl = max(gap_new, gap_ref) >= 1.0
                outliers.append(f"{v} {c} #{int(m['count_value'])}: Δ={m['video_time_sec_new'] - m['video_time_sec_ref']:+.3f}s gap(new/ref)={gap_new:.2f}/{gap_ref:.2f} {'遮蔽区間' if occl else '※要確認'}")
                if not occl: allmatch = False
    print("\n".join(lines)); print("\n".join(outliers))
    ok("⑤0807 既存 G1 と一致（件数・最終値が全一致・時刻差の中央値≦0.1s・0.3s超は遮蔽区間のみ）", allmatch,
       f"events={n_all} |Δ|≦0.3s={n_close} 0.3s超={len(outliers)}（{'; '.join(outliers) if outliers else 'なし'}）")
    lines += outliers
    G.drop(columns=["vid"]).to_csv(C.g1_csv, index=False, encoding="utf-8-sig")
    with open(os.path.join(T, "compare_0807.txt"), "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")

with open(os.path.join(T, "SELFTEST_RESULT.txt"), "w", encoding="utf-8") as f:
    for n, c, dtl in res: f.write(("PASS " if c else "FAIL ") + n + (" — " + dtl if dtl else "") + "\n")
    f.write(f"ALL {'PASS' if all(c for _, c, _ in res) else 'FAIL'} ({sum(c for _, c, _ in res)}/{len(res)})\n")
print("ALL", "PASS" if all(c for _, c, _ in res) else "FAIL")
