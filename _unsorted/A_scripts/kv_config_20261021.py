# -*- coding: utf-8 -*-
"""2026-10-14 回収用の共通設定（kv_config_20261006.Cfg1006 の派生・既存ファイルは無変更）。

10/14 のセッション＝**全曲 DAM WAO（Hearts）の1機種のみ**（30曲・採点 10:28:24〜12:53:06・カラオケBanBan蒲田西口）。
10/6 は1セッションで機種が途中で変わったため machine 列と MACHINE_SPLIT で分けていたが、10/14 は ai 側が0行なので
MACHINE_SPLIT="00:00:00"（＝当日の全採点行が wao）とし、構造だけ 10/6 版を踏襲する（rename_mapping.csv の machine 列も残す）。
出力＝dataset/G1_events_20261021_wao.csv・dataset/verify_20261021_wao.csv（10/6 版と同じ命名）。

使い方：`from kv_config_20261021 import Cfg1014`（セッション＝引数 → KV_SESSION → 既定 20261021）。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, ROOT, VIDEO_ROOT, DAM_DIR, WSL_ROOT  # noqa: F401
from kv_config_20261006 import _Mixin1006, to_wsl  # noqa: F401

SESSION = "20261021"
MACHINE_SPLIT = "00:00:00"  # 10/14 は全曲 wao（Ai CSV の当日行＝0）。採点時刻がこれより前＝ai／以後＝wao
# 12:55 取得（当日30行・10:28:24〜12:53:06）。01:22 再取得（…T16-22-34.csv）も当日30行で同一＝既存を使う（RUNBOOK §1）
HEARTS_CSV = os.path.join(DAM_DIR, "dam_ai-hearts_full_2026-10-21T10-50-11.csv")
AI_CSV = os.path.join(DAM_DIR, "dam_ai_full_2026-10-21T10-50-34.csv")  # 当日0行（参照のみ・突合には出てこない）
PROBE_SCRIPTS = {"wao": "wao_probe_v2_20261021.py"}
PROBE_PGREP = {"wao": "wao_probe_v2_2026102[1].py"}


class Cfg1014(_Mixin1006, Cfg):
    def __init__(self, session=None):
        Cfg.__init__(self, session or os.environ.get("KV_SESSION") or SESSION)
        self._init1014("wao")

    def _init1014(self, machine):
        self._init1006(machine)
        S = self.S
        self.dam_csv = HEARTS_CSV if machine == "wao" else AI_CSV
        self.probe_script = PROBE_SCRIPTS[machine]
        self.probe_pgrep = PROBE_PGREP[machine]
