# -*- coding: utf-8 -*-
"""C2（入力を変える比較）v3＝学習の回数を 8 回に増やして2条件をもう一度比べる無人ジョブ（2026-10-23）。

なぜ必要か。
2026-10-23 の比較 v1・v2 は学習 4 回で止めた暫定の値で、どちらの条件もまだ成績が上がり続けている途中だった。
学習の回数をそろえて 8 回まで回し、「回数が足りないせいで差が出なかった」可能性を潰すのが v3 の目的である。

すること（順番）。
1. 前段のジョブ（run_c2_followup_20261023.py）のログに FOLLOWUP DONE が出るまで待つ（待ち時間の上限つき・空なら待たない）。
2. build_refdiff_feats_20261023.py を走らせ、その時点までに読み取れたお手本をすべて特徴量へ反映する。
   ここで1回だけ作り直し、以降の2条件は同じ特徴量の状態を使う（2条件の土俵をそろえるため）。
3. 現行の入力だけの条件を 8 回学習する。
4. 現行＋お手本との音程差の条件を 8 回学習する（曲の分割・乱数の種・バッチは 3 と同じ）。
5. make_c2_table_20261023.py で比較表 v3 を書く。

使い方:
  python run_c2_v3_20261023.py --wait_log "G:/.../out/c2b2_refdiff_4ep_followup.log" \
      --base_tag c2a_base_8ep --ref_tag c2b_refdiff_8ep --epochs 8 --max_wait_hours 6
"""
import os
import sys
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args, label):
    print(f"--- {label} 開始 {time.strftime('%H:%M:%S')}", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable] + args, env=env)
    print(f"--- {label} 終了 rc={p.returncode} {time.strftime('%H:%M:%S')}", flush=True)
    return p.returncode


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait_log", default="", help="このログに指定の印が出るまで待つ")
    ap.add_argument("--wait_mark", default="FOLLOWUP DONE")
    ap.add_argument("--max_wait_hours", type=float, default=6.0)
    ap.add_argument("--base_tag", default="c2a_base_8ep")
    ap.add_argument("--ref_tag", default="c2b_refdiff_8ep")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_build", action="store_true", help="特徴量の作り直しを飛ばす")
    a = ap.parse_args()

    if a.wait_log:
        t0 = time.time()
        print(f"前段待ち 開始 {time.strftime('%H:%M:%S')} 印='{a.wait_mark}' ログ={a.wait_log}", flush=True)
        while time.time() - t0 < a.max_wait_hours * 3600:
            try:
                if a.wait_mark in open(a.wait_log, encoding="utf-8", errors="replace").read():
                    print(f"前段の完了を確認 {time.strftime('%H:%M:%S')}", flush=True)
                    break
            except FileNotFoundError:
                pass
            print(f"待機中（前段の印 '{a.wait_mark}' を待っています） {time.strftime('%H:%M:%S')}", flush=True)
            time.sleep(60)
        else:
            print("待ち時間の上限に達したので、今ある状態のまま進む", flush=True)

    if not a.skip_build:
        run([os.path.join(HERE, "build_refdiff_feats_20261023.py")], "特徴量の作り直し（2条件で1回だけ）")
    common = ["--epochs", str(a.epochs), "--threads", str(a.threads), "--seed", str(a.seed)]
    run([os.path.join(HERE, "train_crnn_refdiff_20261023.py"), "--tag", a.base_tag] + common, f"学習（{a.base_tag}・現行の入力）")
    run([os.path.join(HERE, "train_crnn_refdiff_20261023.py"), "--tag", a.ref_tag] + common + ["--refdiff"], f"学習（{a.ref_tag}・現行＋お手本との音程差）")
    run([os.path.join(HERE, "make_c2_table_20261023.py"),
         "--rows", f"現行の入力（学習{a.epochs}回）={a.base_tag},現行＋お手本との音程差（学習{a.epochs}回）={a.ref_tag}",
         "--out_name", f"C2比較表_v3_{time.strftime('%Y-%m-%d')}"], "比較表 v3")
    print("V3 DONE", flush=True)
