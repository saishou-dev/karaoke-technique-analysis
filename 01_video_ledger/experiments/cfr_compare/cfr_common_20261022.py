# -*- coding: utf-8 -*-
r"""固定フレームレート（CFR）変換との比較・共通定義（2026-10-22）。

背景＝2026-10-22 ゼミ④で先生から「スマホの可変フレームレートを固定にしてみては」という指摘があった。
現行は「切り出しは1フレームも落とさない（-fps_mode passthrough）＋切り出し映像自身の pts 表で時刻を引く」という方式
（RUNBOOK_20260915.md §「今回の恒久修正」）。これを固定フレームレートに変えた場合に加点の時刻がどれだけ動くかを実測する。

比較する3方式（同じ元録画・同じ帯クロップのフィルタ・同じ読取器）:
  A = 現行     -fps_mode passthrough -enc_time_base 1/90000 -video_track_timescale 90000  時刻＝帯pts表[frame]
  B = 固定60   -fps_mode cfr -r 60                                                        時刻＝frame/60
  C = 固定59.94 -fps_mode cfr -r 60000/1001                                               時刻＝frame*1001/60000

既存の成果物（bands_wao\・dataset\pts_bands_20261014\・results_wao_20261014\・G1_events_20261014_wao.csv）は
読むだけで書き換えない。出力は全て OUT 配下に作る。
"""
import os
import sys

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
SRC_DIR = r"G:\研究アーカイブ\ビデオ素材\20261014"
OUT = os.path.join(ROOT, "dataset", "cfr_compare_2026-10-22")

FFMPEG = r"C:\Users\conan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
FFPROBE = os.path.join(os.path.dirname(FFMPEG), "ffprobe.exe")

SESSION = "20261014"
# 既存（現行方式）の成果物＝突合の基準
BANDS_EXIST = os.path.join(ROOT, "bands_wao")
PTS_EXIST = os.path.join(ROOT, "dataset", "pts_bands_" + SESSION)
RESULTS_EXIST = os.path.join(ROOT, "results_wao_" + SESSION)
G1_EXIST = os.path.join(ROOT, "dataset", "G1_events_%s_wao.csv" % SESSION)
PROBE_CSV = os.path.join(ROOT, "dataset", "band_probe_v2_%s.csv" % SESSION)
REPS = os.path.join(ROOT, "templates", "wao_20260816", "cluster_reps_wao_ext.npy")
READER = os.path.join(ROOT, "scripts", "read_counters_wao_single_20261015.py")

# 対象3曲（10/14 の収録・指示どおり 怪獣の花唄・RPG・共犯。共犯は4テイクあるので最初のテイク）
VIDEOS = [
    "怪獣の花唄_20261014_104344",
    "RPG(クレヨンしんちゃんアニメバージョン)_20261014_114902",
    "共犯_20261014_102428",
]

# 変換の定義。rate=None はフレームレートを指定しない（passthrough）。
# vf_suffix があれば帯クロップのフィルタ列の末尾に足す（＝fps フィルタで固定化する書き方）。
#
# 2026-10-22 実測で分かった注意（B・C の結果がこれ）＝ffmpeg 9.0.1 でこの録画に
#   `-fps_mode cfr -r <R>` と書くと、入力フレームの約74%を捨てて同じ画を約4回ずつ並べる
#   （30秒の試験で入力1,780枚→残った中身535枚・dup=1350 drop=1314。スレッド数・rotate の有無・
#    -ss の有無を変えても同じ）。固定化は `-vf ...,fps=<R>` と書けば正しく動く（B2・C2・D2）。
VARIANTS = {
    "A_passthrough": dict(
        args=["-fps_mode", "passthrough", "-enc_time_base", "1/90000", "-video_track_timescale", "90000"],
        rate=None, time_mode="pts", vf_suffix="",
        label="A 現行（passthrough＋表示時刻の一覧）"),
    "B2_fps60_filter": dict(
        args=["-fps_mode", "passthrough"], vf_suffix=",fps=60",
        rate=60.0, time_mode="index",
        label="B2 固定60fps・fpsフィルタ（frame/60）"),
    "C2_fps5994_filter": dict(
        args=["-fps_mode", "passthrough"], vf_suffix=",fps=60000/1001",
        rate=60000.0 / 1001.0, time_mode="index",
        label="C2 固定59.94fps・fpsフィルタ（frame*1001/60000）"),
    "D2_fps5933_filter": dict(
        args=["-fps_mode", "passthrough"], vf_suffix=",fps=450000/7584",
        rate=450000.0 / 7584.0, time_mode="index",
        label="D2 固定59.3354fps＝録画の実測レート・fpsフィルタ（frame/59.3354）"),
    "B_cfr60": dict(
        args=["-fps_mode", "cfr", "-r", "60"], vf_suffix="",
        rate=60.0, time_mode="index",
        label="B 固定60fps・-r 指定（frame/60）※中身が間引かれる書き方"),
    "C_cfr5994": dict(
        args=["-fps_mode", "cfr", "-r", "60000/1001"], vf_suffix="",
        rate=60000.0 / 1001.0, time_mode="index",
        label="C 固定59.94fps・-r 指定（frame*1001/60000）※中身が間引かれる書き方"),
}

sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    from mp4_pts import video_pts  # noqa: F401  （moov だけ読む高速 pts 取得・既存スクリプト・無変更）
except Exception as _e:  # pragma: no cover
    video_pts = None
    _IMPORT_ERR = _e


def ensure_out():
    for sub in ("", "bands", "pts", "results", "report"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    return OUT


def band_path(variant, video):
    return os.path.join(OUT, "bands", "%s__%s.mp4" % (variant, video))


def pts_path(variant, video):
    return os.path.join(OUT, "pts", "%s__%s.csv" % (variant, video))


def result_dir(variant, video):
    return os.path.join(OUT, "results", "%s__%s" % (variant, video))


def log(msg, path=None):
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
