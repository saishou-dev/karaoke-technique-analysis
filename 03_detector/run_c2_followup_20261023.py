# -*- coding: utf-8 -*-
"""C2（入力を変える比較）の追い込み＝お手本の読み取りが増えたら、同じ条件で学習をやり直す無人ジョブ（2026-10-23）。

なぜ必要か。
2026-10-23 の比較 v1 は、お手本との音程差が作れていた学習曲が 150 本中 8 本しかない状態で測った。
読み取りは1曲 20〜35 分かかるので、全曲が揃うのを待つと朝までに結果が出ない。
そこで「読み取りが終わったら自動で特徴量を作り直し、同じ条件でもう一度学習する」ところまでを1本の台本にした。

すること（順番）。
1. 指定した読み取りログに DONE が出るまで待つ（待ち時間の上限つき）。
2. build_refdiff_feats_20261023.py を走らせ、お手本との音程差の列を作り直す（既存のものは飛ばす）。
3. train_crnn_refdiff_20261023.py を --refdiff 付きで走らせる（曲の分割・乱数の種・エポック数は引数でそろえる）。
4. make_c2_table_20261023.py で比較表を書き直す。

使い方:
  python run_c2_followup_20261023.py --wait_log "G:/.../e5_refdiff_20261023/logs/run_20261014.log" \
      --tag c2b2_refdiff_4ep --epochs 4 --base_tag c2a_base_4ep --max_wait_hours 4
"""
import os
import sys
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(args, label):
    print(f"--- {label} 開始 {time.strftime('%H:%M:%S')}", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable] + args, env=env)
    print(f"--- {label} 終了 rc={p.returncode} {time.strftime('%H:%M:%S')}", flush=True)
    return p.returncode


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait_log", default="", help="このログに DONE が出るまで待つ（空なら待たない）")
    ap.add_argument("--max_wait_hours", type=float, default=4.0)
    ap.add_argument("--sessions", default="", help="特徴量を作り直す収録セッション（空なら全部）")
    ap.add_argument("--tag", default="c2b2_refdiff_4ep")
    ap.add_argument("--base_tag", default="c2a_base_4ep", help="比較表に並べる現行入力の条件")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.wait_log:
        t0 = time.time()
        print(f"前段待ち 開始 {time.strftime('%H:%M:%S')} 印='{a.wait_mark}' ログ={a.wait_log}", flush=True)
        while time.time() - t0 < a.max_wait_hours * 3600:
            try:
                if "DONE" in open(a.wait_log, encoding="utf-8", errors="replace").read():
                    print(f"読み取り完了を確認 {time.strftime('%H:%M:%S')}", flush=True)
                    break
            except FileNotFoundError:
                pass
            print(f"待機中（前段の印 '{a.wait_mark}' を待っています） {time.strftime('%H:%M:%S')}", flush=True)
            time.sleep(60)
        else:
            print("待ち時間の上限に達したので、今ある読み取り結果だけで進む", flush=True)

    cmd = [os.path.join(HERE, "build_refdiff_feats_20261023.py")]
    if a.sessions:
        cmd += ["--sessions", a.sessions]
    run(cmd, "特徴量の作り直し")
    run([os.path.join(HERE, "train_crnn_refdiff_20261023.py"), "--tag", a.tag, "--epochs", str(a.epochs),
         "--threads", str(a.threads), "--seed", str(a.seed), "--refdiff"], f"学習（{a.tag}）")
    run([os.path.join(HERE, "make_c2_table_20261023.py"),
         "--rows", f"現行の入力={a.base_tag},現行＋お手本との音程差（読み取り拡大後）={a.tag}",
         "--out_name", f"C2比較表_v2_{time.strftime('%Y-%m-%d')}"], "比較表の書き直し")
    print("FOLLOWUP DONE", flush=True)
