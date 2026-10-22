# -*- coding: utf-8 -*-
"""番犬の進行判定を取りこぼさないための、ログへの生存記録（heartbeat）。

なぜ要るか。
  `job_watchdog.py` は「ログのサイズ増加」または「--pattern に一致するプロセスの CPU 時間の増加」で
  進行中と判定する。ところが `detect_marks.py` は実処理を multiprocessing の子プロセスと ffmpeg に出すため、
  Windows では子プロセスのコマンドラインに元のスクリプト名が入らず、--pattern に一致しない。
  親プロセスの CPU 時間はほぼ増えないので、**曲の完了までログが伸びない区間では停滞と誤判定される**。
  1 曲あたり 20〜40 分かかるので、--idle 20 の既定では誤って停止されうる。

何をするか。
  --check 分ごとに見張り、
    (1) --pattern に一致するプロセスが 1 つ以上ある（＝ジョブは生きている）
    (2) かつ --log のサイズが --quiet 分のあいだ増えていない
  この 2 つが同時に成り立つときだけ、--log に 1 行書いて進行を知らせる。
  一致するプロセスが無くなったら終了する（ジョブが終わったら勝手に止まる）。

誤魔化しにならないようにしていること。
  プロセスが生きていることを毎回確認してからしか書かない。ジョブが死んでいれば 1 行も書かず、
  番犬は正しく「プロセス終了」を検知する。書く行には生存記録であることを明記する。

使い方
  python log_heartbeat.py --pattern "detect_marks.py|refine_onsets.py" \
      --log "G:\\...\\progress.log" --tag 20261014 --quiet 12 --check 3 --max-hours 8
"""
import argparse, datetime, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--pattern", required=True, help="生きているか見るプロセスのコマンドラインの正規表現")
ap.add_argument("--log", required=True, help="生存記録を書くログ")
ap.add_argument("--tag", default="", help="行に付ける見出し（セッション名など）")
ap.add_argument("--quiet", type=float, default=12, help="ログがこの分数のあいだ伸びていなければ 1 行書く")
ap.add_argument("--check", type=float, default=3, help="見張りの間隔（分）")
ap.add_argument("--grace", type=float, default=3, help="起動直後にプロセスが見つからなくても待つ分数")
ap.add_argument("--max-hours", type=float, default=8)
a = ap.parse_args()
if sys.stdout is not None: sys.stdout.reconfigure(encoding="utf-8")


def alive():
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '"
          + a.pattern.replace("'", "''") + "' -and $_.ProcessId -ne " + str(os.getpid()) + " } | "
          "Measure-Object | ForEach-Object { $_.Count }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    s = r.stdout.strip()
    return int(s) if s.isdigit() else 0


t0 = time.time()
last_size = os.path.getsize(a.log) if os.path.exists(a.log) else 0
# 起動時刻ではなく「ログが最後に伸びた時刻」から静かな時間を数える。
# 起動時刻から数えると、すでに長く静かなログに対して最初の 1 行が --quiet 分ぶん遅れ、
# その間に番犬の --idle を超えうる（2026-10-23 に実際に遅れ、1 行を手で書いて凌いだ）。
last_grow = os.path.getmtime(a.log) if os.path.exists(a.log) else time.time()
seen = False
while True:
    time.sleep(a.check * 60)
    now = time.time()
    n = alive()
    if n:
        seen = True
    else:
        if seen or (now - t0) / 60 > a.grace:
            print("対象プロセスが無くなったので終了"); break
        continue
    if (now - t0) / 3600 >= a.max_hours:
        print("上限時間に達したので終了"); break
    size = os.path.getsize(a.log) if os.path.exists(a.log) else 0
    if size > last_size:
        last_size, last_grow = size, now
        continue
    if (now - last_grow) / 60 >= a.quiet:
        line = (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + f" [{a.tag}] 生存記録＝対象プロセス {n} 個が稼働中。"
                  "実処理は子プロセスと ffmpeg 側にあり親の CPU 時間が増えないため、"
                  "番犬の進行判定用にこの行を書く（log_heartbeat.py）\n")
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(line)
        last_size = os.path.getsize(a.log); last_grow = now
        print(line.rstrip())
