# -*- coding: utf-8 -*-
r"""「60fps だからこその読み取り方法」の比較・共通定義（2026-10-28）。

背景＝10/22 の比較（材料38）で「固定フレームレートに変換しても加点の時刻は良くならず、
間隔をそろえるために挿入される複製フレームが『同じ値が6フレーム続いたら確定』という判定を変えて
RPG のしゃくりを 26→23 に倒す」ことが分かった。
今回は「固定60fps に変換した映像に合わせた読み取り規則」を作り、現行（方式A＝可変のまま＋表示時刻の表）と比べる。

設計の軸＝固定fps では「フレーム番号 ÷ fps ＝ 時刻」が厳密に成り立つ。
その利点（頭出しが必ず当たる・pts 表が不要）を残したまま、複製フレームの害を受けない規則にする。

実装の形＝1フレームごとの読み（どのフレームでどの値が読めたか）と、直前フレームとの画素差を一度だけ書き出し、
そのあとは「安定の数え方」「複製の扱い」だけを変えた規則を後処理として何通りも当てる。
こうすると OCR のやり直しが要らないので、候補を全部実行できる（材料38 は候補ごとに映像を作って読み直していた）。

元データ・既存スクリプトは読むだけで書き換えない。
"""
import os
import sys

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
OLD = os.path.join(ROOT, "dataset", "cfr_compare_2026-10-22")   # 10/22 の中間ファイル（再利用）
OUT = os.path.join(ROOT, "cfr_native_20261028")                  # 今回の作業場所

SESSION = "20261014"
RESULTS_EXIST = os.path.join(ROOT, "results_wao_" + SESSION)      # 曲ごとの較正 ROI
REPS = os.path.join(ROOT, "templates", "wao_20260816", "cluster_reps_wao_ext.npy")
G1_EXIST = os.path.join(ROOT, "dataset", "G1_events_%s_wao.csv" % SESSION)
VERIFY_EXIST = os.path.join(ROOT, "dataset", "verify_%s_wao.csv" % SESSION)

VIDEOS = [
    "怪獣の花唄_20261014_104344",
    "RPG(クレヨンしんちゃんアニメバージョン)_20261014_114902",
    "共犯_20261014_102428",
]

# 1フレームずつ読む対象（SOURCE）＝3種類。いずれも 10/22 に作った帯動画を入力にする。
#   g60  … B2_fps60_filter（fps フィルタで固定60fps・足りない格子は直前フレームの複製で埋まっている）
#   g5933… D2_fps5933_filter（録画の実測レート 59.3354 で固定・複製はごく少ない）
#   blend60 … A_passthrough（可変のまま）＋その pts 表から、60fps の格子の各時刻で前後フレームを
#             時間の重みで混ぜた映像を「その場で」作って読む（＝複製を作らない固定60fps・候補(4)）
SOURCES = {
    "g60": dict(
        band=os.path.join(OLD, "bands", "B2_fps60_filter__%s.mp4"),
        rate=60.0, kind="file",
        label="固定60fps（fps フィルタ・複製で埋めた映像）"),
    "g5933": dict(
        band=os.path.join(OLD, "bands", "D2_fps5933_filter__%s.mp4"),
        rate=450000.0 / 7584.0, kind="file",
        label="固定59.3354fps（録画の実測レート）"),
    "blend60": dict(
        band=os.path.join(OLD, "bands", "A_passthrough__%s.mp4"),
        pts=os.path.join(OLD, "pts", "A_passthrough__%s.csv"),
        rate=60.0, kind="blend",
        label="固定60fps（前後フレームを時間の重みで混ぜて作る・複製なし）"),
    # 2026-10-28 追加（理由＝P4 の「全フレームを混ぜる」が効いた理由を切り分けるため）
    #   gapblend60 … 60fps の格子で、近い元フレームがある所はそのまま使い（＝複製は残る）、
    #                抜けフレームの所だけ前後の中間で埋める。指示(4) の字面どおりの版。
    #                「害は複製そのものか、抜けフレームの埋め方か」を分ける。
    #   blend5933  … 混ぜ方は P4 と同じで、格子を録画の実測レートにしたもの。
    #                「混ぜる効果」と「60 という数字」を分ける。
    "gapblend60": dict(
        band=os.path.join(OLD, "bands", "A_passthrough__%s.mp4"),
        pts=os.path.join(OLD, "pts", "A_passthrough__%s.csv"),
        rate=60.0, kind="gapblend",
        label="固定60fps（抜けフレームだけ前後の中間で埋める・それ以外は複製）"),
    "blend5933": dict(
        band=os.path.join(OLD, "bands", "A_passthrough__%s.mp4"),
        pts=os.path.join(OLD, "pts", "A_passthrough__%s.csv"),
        rate=450000.0 / 7584.0, kind="blend",
        label="固定59.3354fps（前後の中間で混ぜて作る・複製なし）"),
    # 2026-10-28 追加（理由＝候補(4) をエンコーダに通したあとでも同じ結果になるかを確かめる。
    #   その場で作った映像は x264 を一度も通っていないので、実運用の条件と違う）。
    "blend60file": dict(
        band=os.path.join(OUT, "bands", "blend60__%s.mp4"),
        rate=60.0, kind="file",
        label="固定60fps（混ぜて作ってから x264 で書き出したファイル）"),
    # 2026-10-28 追加（理由＝一番素直な実運用＝「現行の帯動画から固定60fps の帯動画を作る」を再現する。
    #   足りない格子は複製で埋め、抜けフレームだけ中間で埋めたものを x264 で書き出したファイル）。
    "gapblend60file": dict(
        band=os.path.join(OUT, "bands", "gapblend60__%s.mp4"),
        rate=60.0, kind="file",
        label="固定60fps（現行の帯動画から作り直して x264 で書き出したファイル）"),
    # 現行（方式A）を同じ土俵で測り直すための入力。時刻は pts 表から引く（rate は使わない）。
    # 1フレームごとの読みが 10/22 の既存結果と一致するかの検算にも使う。
    "vfr": dict(
        band=os.path.join(OLD, "bands", "A_passthrough__%s.mp4"),
        pts=os.path.join(OLD, "pts", "A_passthrough__%s.csv"),
        rate=None, kind="file",
        label="A 現行（可変のまま・表示時刻の表で引く）"),
}

COUNTERS = ["shakuri", "kobushi", "fall", "vibrato", "accent", "hammering"]


def dump_path(source, video):
    return os.path.join(OUT, "dumps", "%s__%s.npz" % (source, video))


def roi_path(video):
    return os.path.join(RESULTS_EXIST, video, "roi.json")


def ensure_out():
    for sub in ("", "dumps", "results", "report"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    return OUT


def log(msg, path=None):
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
