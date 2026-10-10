# -*- coding: utf-8 -*-
"""MP4のmoovだけを読んで映像トラックの各フレームの提示時刻(pts秒)を出す（デコードなし・高速）。
   用途: 解析パイプライン(read_counters.py)は時刻を frame_index/59.94 で計算しているが、
   スマホ録画はフレーム落ちがあり、実時刻は index/59.94 より遅れる。index→実pts の対応表を作る。
   出力: dataset/pts/<video_stem>.csv (index,pts_sec)
"""
import os, struct, sys, csv
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; SRC = r"G:\研究アーカイブ\ビデオ素材"
OUT = os.path.join(ROOT, "dataset", "pts"); os.makedirs(OUT, exist_ok=True)

def boxes(f, start, end):
    pos = start
    while pos + 8 <= end:
        f.seek(pos); hdr = f.read(8)
        if len(hdr) < 8: return
        size, typ = struct.unpack(">I4s", hdr); hsz = 8
        if size == 1: size = struct.unpack(">Q", f.read(8))[0]; hsz = 16
        elif size == 0: size = end - pos
        yield typ.decode("latin1"), pos + hsz, pos + size
        pos += size

def find(f, start, end, path):
    typ = path[0]
    for t, s, e in boxes(f, start, end):
        if t == typ:
            if len(path) == 1: yield s, e
            else: yield from find(f, s, e, path[1:])

def read_full(f, s, e): f.seek(s); return f.read(e - s)

def video_pts(path):
    with open(path, "rb") as f:
        f.seek(0, 2); fsize = f.tell()
        moov = list(find(f, 0, fsize, ["moov"]))
        if not moov: raise RuntimeError("no moov")
        ms, me = moov[0]
        for ts, te in find(f, ms, me, ["trak"]):
            hd = list(find(f, ts, te, ["mdia", "hdlr"]))
            if not hd: continue
            h = read_full(f, *hd[0])
            if h[8:12] != b"vide": continue
            mdhd = read_full(f, *list(find(f, ts, te, ["mdia", "mdhd"]))[0])
            ver = mdhd[0]; timescale = struct.unpack(">I", mdhd[20:24])[0] if ver == 1 else struct.unpack(">I", mdhd[12:16])[0]
            stbl_s, stbl_e = list(find(f, ts, te, ["mdia", "minf", "stbl"]))[0]
            stts = read_full(f, *list(find(f, stbl_s, stbl_e, ["stts"]))[0])
            n = struct.unpack(">I", stts[4:8])[0]; durs = []
            for i in range(n):
                c, d = struct.unpack(">II", stts[8 + 8 * i: 16 + 8 * i]); durs += [d] * c
            dts = [0]
            for d in durs[:-1]: dts.append(dts[-1] + d)
            cts = list(dts)
            ct = list(find(f, stbl_s, stbl_e, ["ctts"]))
            if ct:
                b = read_full(f, *ct[0]); ver = b[0]; m = struct.unpack(">I", b[4:8])[0]; k = 0
                for i in range(m):
                    c, off = struct.unpack(">Ii" if ver == 1 else ">II", b[8 + 8 * i: 16 + 8 * i])
                    if ver == 0 and off >= 2**31: off -= 2**32
                    for _ in range(c):
                        if k < len(cts): cts[k] = dts[k] + off; k += 1
            # edit list: 先頭の media_time 分だけ提示時刻を引く（ffmpegの start_time 正規化と同等）
            media_start = 0
            el = list(find(f, ts, te, ["edts", "elst"]))
            if el:
                b = read_full(f, *el[0]); ver = b[0]; cnt = struct.unpack(">I", b[4:8])[0]
                if cnt:
                    if ver == 1: seg_dur, mt = struct.unpack(">Qq", b[8:24])
                    else: seg_dur, mt = struct.unpack(">Ii", b[8:16])
                    if mt > 0: media_start = mt
            pts = sorted(cts)
            return [(p - media_start) / timescale for p in pts], timescale
    raise RuntimeError("no video track")

if __name__ == "__main__":
    sessions = sys.argv[1:] or ["20260628", "20260807", "20260812", "20260818", "20260902"]
    for s in sessions:
        for fn in sorted(os.listdir(os.path.join(SRC, s))):
            if not fn.lower().endswith(".mp4"): continue
            stem = os.path.splitext(fn)[0]; out = os.path.join(OUT, stem + ".csv")
            if os.path.exists(out): continue
            try:
                pts, tsc = video_pts(os.path.join(SRC, s, fn))
            except Exception as e:
                print("ERR", s, fn, e, flush=True); continue
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["index", "pts_sec"])
                for i, p in enumerate(pts): w.writerow([i, f"{p:.4f}"])
            n = len(pts); idx_end = (n - 1) / (60000 / 1001)
            print(f"{s} {stem[:36]:38s} frames={n} last_pts={pts[-1]:.2f} idx_time={idx_end:.2f} drift_end={pts[-1]-idx_end:+.2f}s", flush=True)
