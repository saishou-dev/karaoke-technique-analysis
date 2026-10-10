# -*- coding: utf-8 -*-
"""1 セッションの方式A を通しで実行する：detect → rematch → refine → summarize → progress.log に 1 行。
使い方: python run_session.py --sessions 20260812 20260818 20260902 20260911 [--workers 6 --threads 2]
各段は独立プロセスで実行（失敗しても次のセッションへ）。途中結果は partial_<session>/ に曲ごとに保存される。"""
import os, sys, time, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_marks as d
import pandas as pd
SC = os.path.dirname(os.path.abspath(__file__))

def run(args, env=None):
    t = time.time(); p = subprocess.run([sys.executable] + args, env=env); return p.returncode, time.time() - t

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--wait_for", default=None, help="progress.log にこの文字列が現れるまで待ってから開始（例: '20260812 完了'）")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--refine_workers", type=int, default=4)
    a = ap.parse_args()
    env = dict(os.environ); env["MARKS_FFMPEG_THREADS"] = str(a.threads)
    if a.wait_for:
        logp = os.path.join(d.OUT_DIR, "progress.log")
        while a.wait_for not in open(logp, encoding="utf-8", errors="replace").read():
            time.sleep(60)
    for s in a.sessions:
        d.configure(s)
        if not os.path.exists(d.BAND_CSV):
            d.log(f"[{s}] 帯位置ファイルが無いのでスキップ: {d.BAND_CSV}"); continue
        d.log(f"[{s}] ===== 方式A 展開開始（detect 並列{a.workers}×ffmpeg{a.threads}スレッド → rematch → refine 並列{a.refine_workers} → summarize）")
        rc, dt = run([os.path.join(SC, "detect_marks.py"), "--session", s, "--workers", str(a.workers), "--threads", str(a.threads), "--resume"], env)
        d.log(f"[{s}] detect 終了 rc={rc} {dt/60:.0f}分")
        rc, dt = run([os.path.join(SC, "rematch.py"), "--session", s]); d.log(f"[{s}] rematch 終了 rc={rc}")
        rc, dt = run([os.path.join(SC, "refine_onsets.py"), "--session", s, "--workers", str(a.refine_workers)], env); d.log(f"[{s}] refine 終了 rc={rc} {dt/60:.0f}分")
        rc, dt = run([os.path.join(SC, "summarize.py"), "--session", s]); d.log(f"[{s}] summarize 終了 rc={rc}")
        try:
            df = pd.read_csv(d.MARKS_CSV); ok = df[df.mark_time_sec.notna()]
            med = ok.groupby("counter").lead_sec.median().round(2).to_dict()
            d.log(f"{s} 完了 抽出率{ok.shape[0]/len(df):.1%}（{ok.shape[0]}/{len(df)}） lead中央値 {med}")
        except Exception as e:
            d.log(f"{s} 完了（集計失敗 {e}）")

if __name__ == "__main__":
    main()
