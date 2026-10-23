# -*- coding: utf-8 -*-
"""E5 お手本音程バー読取の再開ラッパ（2026-10-23 09:1x）。

経緯＝番犬が「上限 6.0h 超過」で 09:05 に読取を停止した（20260911 は16/30曲で中断・20261006 は未着手）。
やること＝run_sessions_e5.py を 20260911,20261006 で呼び直すだけ。済みの曲は run_all_pitchbar.py 側が
          out/<曲>_frames.npz の有無で飛ばすので、重複計算は起きない（読取は飛ばし・eval だけ再計算）。

なぜ PowerShell の .ps1 ではなくこの Python ラッパか（2026-10-23 の失敗から）＝
  Windows PowerShell 5.1 は BOM の無い UTF-8 の .ps1 を ANSI として読むため、
  台本に書いた日本語パス（G:\\研究アーカイブ\\…）が文字化けして Out-File が失敗した。
  Python は既定で UTF-8 で読むので、日本語パスを含む無人ジョブはこちらで組む。

ログ＝launcher.log へ「追記」する（上書き禁止）。
      連結ジョブ run_wao64_pitchbar_20261023.py が launcher.log の中の
      「ALL SESSIONS DONE」という印を待っているため、このラッパが最後まで通れば WAO64 の64曲へ自然につながる。

優先度＝自分を Idle に落としてから子（python・ffmpeg）を起こすので、子も Idle を継ぐ（本人の操作を邪魔しない）。
"""
import os
import sys
import ctypes
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
LOG = os.path.join(BASE, "logs", "launcher.log")
IDLE_PRIORITY_CLASS = 0x00000040

if __name__ == "__main__":
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),
                                                IDLE_PRIORITY_CLASS)
    except Exception as e:
        print("優先度の変更に失敗:", e, file=sys.stderr)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(LOG, "a", encoding="utf-8") as f:
        p = subprocess.run([sys.executable, os.path.join(HERE, "run_sessions_e5.py"),
                            "--sessions", "20260911,20261006", "--workers", "5"],
                           stdout=f, stderr=subprocess.STDOUT, env=env)
    sys.exit(p.returncode)
