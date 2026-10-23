# -*- coding: utf-8 -*-
"""別軸の確認＝しきい値の決め方を「件数の差が最小」に変えて測り直す（2026-10-23・本人裁定）。

なぜ必要か。
比較表 v2 で、しゃくりの F値だけが 0.094 → 0.024 に落ちた。
原因は入力の違いではなく、しきい値決定の 30 本で選ばれた境目が高すぎて、試験でほとんど予測を出さなくなったこと（463 件 → 25 件）。
しきい値の決め方は「入力を変える軸（検証項目C2）」とは別の軸なので、比較表には混ぜない。
ここでは **学習し直さず**、v3 で学習済みのモデルをそのまま使い、しきい値の選び方だけを変えて測り直す。

すること（順番）。
1. v3 のジョブのログに V3 DONE が出るまで待つ。
2. v3 の2条件のモデル（out/<タグ>/model.pt）を、別の名前のフォルダへ複製する（v3 の結果ファイルを上書きしないため）。
3. そのフォルダに対して --eval_only --thr_mode count で測り直す（学習はしない・しきい値の選び方だけが違う）。
4. 2条件を並べた小さい表を書く（比較表 v1〜v3 とは別のファイル）。

使い方:
  python run_c2_thrmode_count_20261023.py --wait_log "G:/.../out/c2_v3_8ep.log" \
      --base_tag c2a_base_8ep --ref_tag c2b_refdiff_8ep --epochs 8
"""
import os
import sys
import time
import shutil
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


def clone_model(src_tag, dst_tag):
    """学習済みモデルだけを別名のフォルダへ複製する（元のフォルダには触らない）。"""
    src = os.path.join(ROOT, "out", src_tag)
    dst = os.path.join(ROOT, "out", dst_tag)
    os.makedirs(dst, exist_ok=True)
    ok = False
    for name in ("model.pt", "norm.npy"):
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, name))
            if name == "model.pt":
                ok = True
    print(f"複製 {src_tag} -> {dst_tag} model.pt={'あり' if ok else 'なし'}", flush=True)
    return ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait_log", default="")
    ap.add_argument("--wait_mark", default="V3 DONE")
    ap.add_argument("--max_wait_hours", type=float, default=10.0)
    ap.add_argument("--base_tag", default="c2a_base_8ep")
    ap.add_argument("--ref_tag", default="c2b_refdiff_8ep")
    ap.add_argument("--epochs", type=int, default=8, help="見出しに書くための学習回数（測り直しでは学習しない）")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="", help="表のファイル名に付ける見分け（同日に2回走らせる時の衝突回避）")
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
            time.sleep(120)
        else:
            print("待ち時間の上限に達した。前段が終わっていないので何もせずに終わる", flush=True)
            sys.exit(0)

    pairs = [(a.base_tag, a.base_tag + "_count", False), (a.ref_tag, a.ref_tag + "_count", True)]
    done = []
    for src, dst, is_ref in pairs:
        if not clone_model(src, dst):
            print(f"!! {src} に model.pt が無いので飛ばす", flush=True)
            continue
        cmd = [os.path.join(HERE, "train_crnn_refdiff_20261023.py"), "--tag", dst, "--eval_only",
               "--thr_mode", "count", "--threads", str(a.threads), "--seed", str(a.seed)]
        if is_ref:
            cmd.append("--refdiff")
        run(cmd, f"測り直し（{dst}・しきい値＝件数の差が最小）")
        done.append(dst)
    if len(done) == 2:
        run([os.path.join(HERE, "make_c2_table_20261023.py"),
             "--rows", f"現行の入力（しきい値＝件数最小）={done[0]},現行＋お手本との音程差（しきい値＝件数最小）={done[1]}",
             "--out_name", f"C2別軸_しきい値の決め方_{time.strftime('%Y-%m-%d')}{a.label}"], "小さい表")
    print("THRMODE COUNT DONE", flush=True)
