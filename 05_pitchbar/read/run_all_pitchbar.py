# -*- coding: utf-8 -*-
"""E4 音程バー読取のセッション一括実行（2026-10-08）。read（約 25 分/曲・5 並列）→ eval を全曲に回し、結果表 out/summary_<S>.csv を書く。
   済みの曲（out/<video>_frames.npz があれば read を、_eval.json があれば eval をスキップ）。失敗は logs/<video>.log に残して次へ進む（無人運転）。
   使い方: python run_all_pitchbar.py <session> [--workers 5] [--only name1,name2] [--eval-only]
"""
import os, sys, csv, json, glob, argparse, subprocess, time, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE); OUT = os.path.join(BASE, "out"); LOGS = os.path.join(BASE, "logs")
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
PY = sys.executable

def videos(S):
    p = os.path.join(ROOT, "dataset", f"band_probe_v2_{S}.csv")
    return [r["video"] for r in csv.DictReader(open(p, encoding="utf-8-sig"))]

def run_one(S, name, eval_only):
    log = os.path.join(LOGS, name + ".log"); env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time(); st = []
    with open(log, "a", encoding="utf-8") as f:
        if not eval_only and not os.path.exists(os.path.join(OUT, name + "_frames.npz")):
            p = subprocess.run([PY, os.path.join(HERE, "read_pitchbar.py"), S, name, "--stage", "read"], stdout=f, stderr=subprocess.STDOUT, env=env)
            st.append(f"read rc={p.returncode}")
        if os.path.exists(os.path.join(OUT, name + "_frames.npz")):
            p = subprocess.run([PY, os.path.join(HERE, "read_pitchbar.py"), S, name, "--stage", "eval"], stdout=f, stderr=subprocess.STDOUT, env=env)
            st.append(f"eval rc={p.returncode}")
    return name, " ".join(st), time.time() - t0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("session"); ap.add_argument("--workers", type=int, default=5); ap.add_argument("--only", default=""); ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args(); S = a.session; os.makedirs(OUT, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    names = videos(S)
    if a.only: names = [n for n in names if n in a.only.split(",")]
    print(f"START session={S} videos={len(names)} workers={a.workers}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for name, st, dt in ex.map(lambda n: run_one(S, n, a.eval_only), names):
            print(f"{name}: {st} {dt:.0f}s", flush=True)
    rows = []
    for name in names:
        p = os.path.join(OUT, name + "_eval.json")
        if not os.path.exists(p): rows.append(dict(video=name, status="FAIL")); continue
        e = json.load(open(p, encoding="utf-8"))
        rows.append(dict(video=name, status="ok", agree50=round(e["agree50"], 3), agree50_single_offset=round(e["agree50_single_offset"], 3), agree100=round(e["agree100"], 3),
                         coverage=round(e["coverage_of_voiced"], 3), n_compared=e["n_compared"], n_notes=e["n_notes_compared"], note_agree50=round(e["note_agree50"], 3), note_agree100=round(e["note_agree100"], 3), ref_sec=round(e["ref_seconds"]), pages=f"{e['pages_fitted']}/{e['pages_total']}",
                         pages_cursor=e["n_pages_cursor"], pages_fallback=e["n_pages_fallback"], k=e["k_semi_per_line"], m0=e["midi_of_top_line"], dt=e["display_dt_sec"],
                         line_sp=round(e["line_spacing_px"], 1), tilt=round(e["tilt_px_per_col"], 4)))
    keys = ["video", "status", "agree50", "agree50_single_offset", "agree100", "coverage", "n_compared", "n_notes", "note_agree50", "note_agree100", "ref_sec", "pages", "pages_cursor", "pages_fallback", "k", "m0", "dt", "line_sp", "tilt"]
    with open(os.path.join(OUT, f"summary_{S}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]
    print("DONE", os.path.join(OUT, f"summary_{S}.csv"), flush=True)
