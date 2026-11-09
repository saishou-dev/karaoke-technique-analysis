# -*- coding: utf-8 -*-
r"""11/4（session=20261104）の録音との時刻合わせ（sync）に旗を立てる別表を作る（2026-11-09・kaishu20261104-merge・新規）。

なぜ別表か
  * 統合台帳 G1_events_all.csv に列を足すと既存 13,499 行のバイトが変わる（既存行は 1 バイトも変えない約束）。
  * sync_offsets_20261104.csv も dataset の既存ファイルなので変えない。
  → 旗は dataset\sync\sync_flags_20261104.csv に持たせ、索引を作るスクリプト（make_g1_shifted_global_20261109.py）が読む。
    本人の裁定が出たら、この表の sync_flag／status／in_index を書き換えて索引を作り直せば戻せる。

裁定（Fable 2026-11-09 13:59 の指示書 ④2）
  * 共犯_20261104_102733（10:27）＝offset −31.51・見込み +36.5（差 68 秒・r_main 0.194）＝sync 不採用
      → sync_flag=suspect・status=sync_suspect・in_index=0（結合はする・学習の索引から外す）
  * 私は最強・きらり・Midnight・輪廻転生・僕のこと（差 6〜15 秒）
      → sync_flag=suspect・status=suspect_keep・in_index=0（結合はする・学習の索引から外す）
  * 残り 21 曲＝sync_flag=ok・status=ok・in_index=1
  見込みの offset は 11/5 報告（2026-11-05_1219_kaishu20261104_2.md ④1）の値（WAV の終了時刻・長さと動画の開始時刻から出した値）。

使い方: python sync_flags_20261109.py      （既にあれば上書きしない＝退避なしの上書き禁止。作り直すときは手で退避してから）
"""
import os, sys, csv
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
SRC = os.path.join(ROOT, "dataset", "sync", "sync_offsets_20261104.csv")
OUT = os.path.join(ROOT, "dataset", "sync", "sync_flags_20261104.csv")
# video の先頭（曲名_日付_時刻）→ (見込み offset, status)
SUSPECT = {
    "共犯_20261104_102733": (36.5, "sync_suspect", "差 68 秒・r_main 0.194・band-arbitrated＝sync 不採用（Fable 11/9）"),
    "私は最強 (ウタ from ONE PIECE FILM RED)_20261104_121704": (9.9, "suspect_keep", "差 15 秒（Fable 11/9＝結合はする・索引から外す）"),
    "きらり_20261104_111229": (8.0, "suspect_keep", "差 10 秒（Fable 11/9＝結合はする・索引から外す）"),
    "Midnight_20261104_122341": (6.9, "suspect_keep", "差 9 秒・band-arbitrated（Fable 11/9＝結合はする・索引から外す）"),
    "輪廻転生_20261104_121201": (6.4, "suspect_keep", "差 7 秒（Fable 11/9＝結合はする・索引から外す）"),
    "僕のこと(生音)_20261104_130450": (6.1, "suspect_keep", "差 6 秒（Fable 11/9＝結合はする・索引から外す）"),
}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"既にある＝上書きしない: {OUT}")
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    vids = {r["video"] for r in rows}
    miss = [v for v in SUSPECT if v not in vids]
    assert not miss, f"同期表に無い: {miss}"
    out = []
    for r in rows:
        v = r["video"]
        if v in SUSPECT:
            exp, st, note = SUSPECT[v]
            flag, inx = "suspect", 0
            diff = round(exp - float(r["offset_sec"]), 2)
        else:
            exp, st, note, flag, inx, diff = "", "ok", "", "ok", 1, ""
        out.append(dict(session=20261104, video=v, wav=r["wav"], offset_sec=r["offset_sec"], method=r["method"],
                        r_main=r["r_main"], expected_offset_sec=exp, expected_minus_offset=diff,
                        sync_flag=flag, status=st, in_index=inx, ruling="Fable 2026-11-09 指示書 ④2" if flag != "ok" else "",
                        note=note))
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    n = {k: sum(1 for o in out if o["status"] == k) for k in ("ok", "suspect_keep", "sync_suspect")}
    print(f"WROTE {OUT} rows={len(out)} {n} in_index={sum(o['in_index'] for o in out)}")


if __name__ == "__main__":
    main()
