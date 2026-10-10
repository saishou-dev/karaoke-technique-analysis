# -*- coding: utf-8 -*-
"""回収パイプライン共通設定 DAM Ai 版（2026-09-15・WAO版 kv_config_20260915.py の派生）。

本日 2026-09-15 の研究セッションは予約エラーで DAM Ai（WAO無し）で実施（10:18確定）。WAO版の恒久修正（帯動画自身の pts 表・G1 はその表だけを参照・
2段ゲート）は kv_config_20260915.Cfg をそのまま継承し、**帯の定義と読取器だけ Ai 用に差し替える**。

Ai の帯定義（根拠＝0628・0807 の処理実体）：
  - 帯位置特定＝WSL ~/karaoke-vision/locate_band.py v2（基準帯画像 templates/sample_counterband_t70.png（金枠スキン）＋ dark_band_ref.png（暗スキン）・多スケール）
  - 帯クロップ＝ffmpeg crop=<w>:<h>:<x>:<y>,scale=1500:180（process_videos.ps1 2026-08-08）→ bands/<video>.mp4（1500x180）
  - 読取器＝WSL ~/karaoke-vision/read_counters.py v0.8（2026-08-08 08:44・md5 89c6912793780240239b4577fa7a1dfd）
      DIGIT_ROIS（1500x180 帯上の数字 ROI）＝shakuri (245,85,350,148)／kobushi (565,85,690,148)／fall (925,85,1020,148)／vibrato (1230,85,1460,148)
      数字テンプレ＝~/karaoke-vision/out/cluster_reps.npy（40クラスタ・md5 52a117ad8cf53038ff8a96227ef080e5・LABELS は読取器内蔵）
      技法カウンター＝4つ（shakuri・kobushi・fall・vibrato）＝0807 の G1（dataset/G1_events_all.csv session=20260807）の counter 列と一致
  - 0807 の実績＝18曲すべて4カウンター完全一致（10曲＝自動プローブ・午後7曲＝手動座標 (245,571,1530,208)）
  ※ 8/8 の暗スキン再試行（roi_dark_skin.json）は既定 ROI と同一値のため無効（no-op）→ Ai 版では省く。

使い方：各 *_20260915ai.* は `from kv_config_20260915ai import CfgAi` → `C=CfgAi()`（セッション＝引数 → 環境変数 KV_SESSION → 既定 20260915）。
セルフテスト：環境変数 KV_AI_BASE=<dir> を与えると機種固有の出力（bands_ai／results_ai_<S>／pts_bands_ai_<S>／band_probe_ai_<S>.csv／pipeline_ai_<S>／
G1・verify・blind）をすべてそのディレクトリ配下に置く（本番ファイルに触れない）。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import (Cfg, ROOT, VIDEO_ROOT, WAV_ROOT, DAM_DIR, FFMPEG, FFPROBE, PYTHON, WSL, WSL_ROOT,  # noqa: F401
                                GATE_LAST_PTS_TOL, GATE_MIN_KEEP_RATIO, session_from_args)

MACHINE = "ai"
DAM_GLOB = "dam_ai_full_*.csv"                     # DAM★とも CSV（Ai 用・npm run -s export:csv:ai:full）
COUNTERS = ["shakuri", "kobushi", "fall", "vibrato"]  # Ai の技法カウンター（0807 G1 の counter 列で確認）
CSV_MAP = {"shakuri": "shakuriCount", "kobushi": "kobushiCount", "fall": "fallCount", "vibrato": "vibratoCount"}
BAND_W, BAND_H = 1500, 180                          # 帯クロップの正規化サイズ（0628/0807 と同じ）
DIGIT_ROIS = {"shakuri": (245, 85, 350, 148), "kobushi": (565, 85, 690, 148), "fall": (925, 85, 1020, 148), "vibrato": (1230, 85, 1460, 148)}
TEMPLATE_DIR = os.path.join(ROOT, "templates")
BAND_REFS = [os.path.join(TEMPLATE_DIR, "sample_counterband_t70.png"), os.path.join(TEMPLATE_DIR, "dark_band_ref.png")]  # 金枠／暗スキン
REPS_NPY = os.path.join(TEMPLATE_DIR, "ai_20260915ai", "cluster_reps.npy")  # WSL out/cluster_reps.npy の複製（md5 同一）
READER = "read_counters_ai_20260915ai.py"           # WSL read_counters.py v0.8 の複製（md5 同一・ラベル内蔵・4ROI）
READER_PGREP = "read_counters_a[i]"                # ランナー監視用
PROBE_SCRIPT = "ai_probe_20260915ai.py"


class CfgAi(Cfg):
    def __init__(self, session=None):
        super().__init__(session)
        S = self.S
        base = os.environ.get("KV_AI_BASE") or ROOT
        self.machine = MACHINE
        self.bands_dir = os.path.join(base, "bands_ai")
        self.gatefail_dir = os.path.join(self.bands_dir, "_gatefail_" + S)
        self.results_dir = os.path.join(base, "results_ai_" + S)
        self.pts_band_dir = os.path.join(base, "dataset", "pts_bands_ai_" + S) if base == ROOT else os.path.join(base, "pts_bands_ai_" + S)
        self.gate_csv = os.path.join(self.pts_band_dir, "_gate.csv")
        self.gate_log = os.path.join(self.pts_band_dir, "_gate.log")
        ds = os.path.join(base, "dataset") if base == ROOT else base
        self.probe_csv = os.path.join(ds, f"band_probe_ai_{S}.csv")
        self.probe_log = os.path.join(ds, f"band_probe_ai_{S}.log")
        self.pipe_dir = os.path.join(ds, "pipeline_ai_" + S)
        self.pipe_log = os.path.join(self.pipe_dir, "pipeline.log")
        if base != ROOT:  # セルフテスト＝G1・verify・blind・sync もテスト側へ
            self.g1_csv = os.path.join(ds, f"G1_events_{S}.csv")
            self.verify_csv = os.path.join(ds, f"verify_{S}.csv")
            self.blind_dir = os.path.join(ds, "blind_" + S)
            self.sync_csv = os.path.join(ds, "sync", f"sync_offsets_{S}.csv")
        if os.environ.get("KV_PIPE_DIR"): self.pipe_dir = os.environ["KV_PIPE_DIR"]; self.pipe_log = os.path.join(self.pipe_dir, "pipeline.log")
        if os.environ.get("KV_BLIND_DIR"): self.blind_dir = os.environ["KV_BLIND_DIR"]
        self.pts_source = f"pts_bands_ai_{S}"
        self.dam_glob = DAM_GLOB; self.counters = COUNTERS; self.csv_map = CSV_MAP
        self.wsl_base = self._to_wsl(base)

    @staticmethod
    def _to_wsl(p):
        p = os.path.abspath(p); return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")

    def latest_dam_csv(self):
        import glob
        if os.environ.get("KV_DAM_CSV"): return os.environ["KV_DAM_CSV"]
        c = sorted(glob.glob(os.path.join(DAM_DIR, DAM_GLOB)))
        return c[-1] if c else None
