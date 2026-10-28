# -*- coding: utf-8 -*-
r"""手順4＝候補(4)（前後の中間で混ぜた固定60fps）を実ファイルとして書き出す（2026-10-28）。

なぜ必要か＝候補(4) は dump_reads が映像を「その場で」作って読むので、エンコーダを一度も通っていない。
実際の運用では帯動画をファイルとして持つから、x264 を通したあとでも同じ結果になるかを確かめないと使えない。
（10/22 の固定60fps は、同じ4K元から1回のエンコードで作ったのに現行と違う読みになった。
  複製フレームが入るとエンコーダの割り当てが変わり、どのフレームも微妙に変わるため。）

書き出しの条件は 10/22 の帯動画と同じ（libx264・crf 15・preset fast・音なし）。
フレームは dump_reads_20261028.gen_blend と同一の作り方で、ffmpeg の標準入力へ生の画素を流す。

出力＝cfr_native_20261028\bands\blend60__<video>.mp4
使い方＝python make_blend_video_20261028.py [--videos 0,1,2]
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_reads_20261028 import gen_blend
from native_common_20261028 import OUT, SOURCES, VIDEOS, ensure_out, log

FFMPEG = os.environ.get("KV_FFMPEG") or (
    r"C:\Users\conan\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")


def one(video, source="blend60"):
    S = SOURCES[source]
    band = S["band"] % video
    out = os.path.join(OUT, "bands", "%s__%s.mp4" % (source, video))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        return "skip（既にある） %s" % video
    gen = gen_blend(band, S["pts"] % video, S["rate"], gap_only=(S["kind"] == "gapblend"))
    kind, rate, nout = next(gen)
    t0 = time.time()
    first = next(gen)[1]
    h, w = first.shape[:2]
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (w, h), "-r", "%.6f" % S["rate"], "-i", "-",
           "-c:v", "libx264", "-crf", "15", "-preset", "fast", "-an",
           "-fps_mode", "passthrough", out + ".part.mp4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    try:
        p.stdin.write(first.tobytes())
        n += 1
        for kind, img in gen:
            p.stdin.write(img.tobytes())
            n += 1
    finally:
        p.stdin.close()
    rc = p.wait()
    if rc != 0:
        return "ERR ffmpeg rc=%d %s" % (rc, video)
    os.replace(out + ".part.mp4", out)
    return "ok %s frames=%d（格子 %d）%.0f 秒 %.0f MB" % (video[:14], n, nout, time.time() - t0,
                                                      os.path.getsize(out) / 1e6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="")
    ap.add_argument("--source", default="blend60")
    A = ap.parse_args()
    ensure_out()
    LOG = os.path.join(OUT, "report", "make_blend_video.log")
    vids = VIDEOS if not A.videos else [VIDEOS[int(i)] for i in A.videos.split(",")]
    for v in vids:
        log("[make] " + one(v, A.source), LOG)
