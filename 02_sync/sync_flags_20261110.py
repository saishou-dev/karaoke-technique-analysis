# -*- coding: utf-8 -*-
r"""11/10（session=20261110）の録音との時刻合わせ（sync）に旗を立てる別表を作る（2026-11-10・kaishu20261110・新規）。
   sync_flags_20261109.py（11/4）と同じ列・同じ扱い（統合台帳には列を足さない・旗は別表・in_index で索引から外す・書き換えて戻せる）。

   11/4 版との違い＝見込みの offset を表の中で計算する（11/4 は報告の手書きの値）。
     見込み offset＝動画の開始時刻（ファイル名 HHMMSS）−WAV の開始時刻（ファイル名の終了時刻 HHMMSS − WAV の長さ）。
     audio_time＝video_time＋offset の向き（WAV が動画より後に始まると負）。
     スマホと録音 PC の時計の差のぶん、見込み−実測 はセッション内でほぼ一定になる（11/4 の 27 曲で −6.4〜−4.4 秒・中央値 −5.4 秒＝Opus 11/10 実測）。
   判定＝(見込み−実測) とセッションの中央値との差の絶対値＝dev。
     dev > 30 秒＝sync_suspect（sync 不採用・索引から外す）／ 2 秒 < dev ≤ 30 秒＝suspect_keep（結合はする・索引から外す）／ それ以外＝ok。
     r_main は列に残す（11/4 では r 0.128〜0.194 の曲も時計の見込みと 1 秒以内で一致＝r だけでは外さない）。
   使い方: python sync_flags_20261110.py      （既にあれば上書きしない）
"""
import os, sys, csv, wave, datetime, statistics
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
S = "20261110"
SRC = os.path.join(ROOT, "dataset", "sync", f"sync_offsets_{S}.csv")
OUT = os.path.join(ROOT, "dataset", "sync", f"sync_flags_{S}.csv")
WAVD = os.path.join(r"G:\研究アーカイブ\録音", f"songs_{S}_raw")
DEV_KEEP, DEV_DROP = 2.0, 30.0


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"既にある＝上書きしない: {OUT}")
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    tmp = []
    for r in rows:
        v = r["video"]; w = r["wav"]
        vs = datetime.datetime.strptime(S + v[-6:], "%Y%m%d%H%M%S")
        we = datetime.datetime.strptime(S + os.path.splitext(w)[0].split("_")[-1], "%Y%m%d%H%M%S")
        with wave.open(os.path.join(WAVD, w)) as f:
            dur = f.getnframes() / float(f.getframerate())
        exp = (vs - (we - datetime.timedelta(seconds=dur))).total_seconds()
        tmp.append((r, exp, exp - float(r["offset_sec"]), dur))
    med = statistics.median(d for _, _, d, _ in tmp)
    out = []
    for r, exp, d, dur in tmp:
        dev = d - med
        if abs(dev) > DEV_DROP: flag, st, inx = "suspect", "sync_suspect", 0
        elif abs(dev) > DEV_KEEP: flag, st, inx = "suspect", "suspect_keep", 0
        else: flag, st, inx = "ok", "ok", 1
        out.append(dict(session=int(S), video=r["video"], wav=r["wav"], offset_sec=r["offset_sec"], method=r["method"],
                        r_main=r["r_main"], expected_offset_sec=round(exp, 2), expected_minus_offset=round(d, 2),
                        sync_flag=flag, status=st, in_index=inx,
                        ruling="Opus 2026-11-10（指示書④8＝11/9 の Fable 裁定と同じ扱い・判定式は本表の docstring）" if flag != "ok" else "",
                        note=f"dev={dev:+.2f}s（セッション中央値 {med:+.2f}s）wav_dur={dur:.1f}s"))
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    n = {k: sum(1 for o in out if o["status"] == k) for k in ("ok", "suspect_keep", "sync_suspect")}
    print(f"WROTE {OUT} rows={len(out)} {n} in_index={sum(o['in_index'] for o in out)} median(exp-offset)={med:+.2f}")
    for o in out:
        print(o["video"], o["offset_sec"], o["expected_offset_sec"], o["expected_minus_offset"], o["r_main"], o["method"], o["status"])


if __name__ == "__main__":
    main()
