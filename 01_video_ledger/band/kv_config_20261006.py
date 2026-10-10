# -*- coding: utf-8 -*-
"""2026-10-06 回収用の共通設定（kv_config_20260915.Cfg／kv_config_20260915ai.CfgAi の派生・既存ファイルは無変更）。

10/6 の特殊事情＝**1セッションで機種が途中で変わった**（前半 9曲＝DAM Ai 10:00〜10:48／後半 23曲＝DAM WAO(Hearts) 11:02〜13:00）。
動画フォルダ（ビデオ素材/20261006）と rename_mapping.csv は両機種で共有し、rename_mapping.csv の `machine` 列（ai|wao）で
各監督（pipeline_20261006.py＝WAO／pipeline_20261006ai.py＝Ai）が自分の動画だけを処理する。
機種で分ける出力＝dataset/G1_events_<S>_<machine>.csv・dataset/verify_<S>_<machine>.csv（従来名 G1_events_<S>.csv は使わない＝衝突回避）。
帯動画・帯pts表・プローブ表・results・pipeline ログは 9/15 版の設計どおり元から機種別（bands_wao／bands_ai 等）。
同期表 dataset/sync/sync_offsets_<S>.csv は共有（動画名で引くので1表でよい・録音WAV受領後に --from sync）。

使い方：`from kv_config_20261006 import Cfg1006, CfgAi1006`（セッション＝引数 → KV_SESSION → 既定 20261006）。
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, ROOT, VIDEO_ROOT, DAM_DIR, WSL_ROOT  # noqa: F401
from kv_config_20260915ai import CfgAi

SESSION = "20261006"
MACHINE_SPLIT = "10:55:00"  # 採点時刻がこれより前＝ai／以後＝wao（当日の機種切替。Ai 最終 10:48:24・Hearts 最初 11:02:24）
HEARTS_CSV = os.path.join(DAM_DIR, "dam_ai-hearts_full_2026-10-06T04-05-48.csv")  # WAO（当日 23 行・最終 13:00:14 Love me,Love you）
AI_CSV = os.path.join(DAM_DIR, "dam_ai_full_2026-10-06T04-06-12.csv")             # Ai（当日 9 行・10:00:03〜10:48:24）
PROBE_SCRIPTS = {"wao": "wao_probe_v2_20261006.py", "ai": "ai_probe_20261006ai.py"}  # 10/6 版＝縦窓拡大（WAO）／左 pad（Ai）
PROBE_PGREP = {"wao": "wao_probe_v2_2026100[6].py", "ai": "ai_probe_20261006a[i].py"}


def to_wsl(p):
    p = os.path.abspath(p); return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")


class _Mixin1006:
    def _init1006(self, machine):
        S = self.S
        self.machine = machine
        self.g1_csv = os.path.join(ROOT, "dataset", f"G1_events_{S}_{machine}.csv")
        self.verify_csv = os.path.join(ROOT, "dataset", f"verify_{S}_{machine}.csv")
        self.dam_csv = HEARTS_CSV if machine == "wao" else AI_CSV
        self.probe_script = PROBE_SCRIPTS[machine]
        self.probe_pgrep = PROBE_PGREP[machine]
        self.probe_list = os.path.join(self.pipe_dir, "probe_videos.txt")  # プローブに渡す対象動画の一覧（WSLパス・1行1本）

    def mapping_rows(self):
        if not os.path.exists(self.rename_csv): return []
        return list(csv.DictReader(open(self.rename_csv, encoding="utf-8-sig")))

    def videos(self):
        """この機種の改名済み動画（拡張子なし・実在するもの）。rename_mapping.csv の how=TIME_WINDOW かつ machine 一致。"""
        names = [r["renamed"][:-4] for r in self.mapping_rows()
                 if r.get("how") == "TIME_WINDOW" and r.get("machine") == self.machine and r.get("renamed", "").lower().endswith(".mp4")]
        return sorted(n for n in set(names) if os.path.exists(os.path.join(self.vid_dir, n + ".mp4")))

    def write_probe_list(self):
        """対象動画の WSL パス一覧を pipe_dir/probe_videos.txt に書き、その WSL パスを返す（launch_probe_20261006.sh が読む）。"""
        os.makedirs(self.pipe_dir, exist_ok=True)
        with open(self.probe_list, "w", encoding="utf-8", newline="\n") as f:
            for n in self.videos(): f.write(f"{to_wsl(self.vid_dir)}/{n}.mp4\n")
        return to_wsl(self.probe_list)


class Cfg1006(_Mixin1006, Cfg):
    def __init__(self, session=None):
        Cfg.__init__(self, session or os.environ.get("KV_SESSION") or SESSION)
        self._init1006("wao")


class CfgAi1006(_Mixin1006, CfgAi):
    def __init__(self, session=None):
        CfgAi.__init__(self, session or os.environ.get("KV_SESSION") or SESSION)
        self._init1006("ai")
