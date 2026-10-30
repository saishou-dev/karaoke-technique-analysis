# -*- coding: utf-8 -*-
"""観測区間の表 dataset/song_valid_range.csv に 10/21（WAO 30 曲）を足す（2026-10-30）。

作り方は 10/14 を足したとき（2026-10-20）と同じ＝song_valid_range_20261015.py の SRC に
  "20261021": (["results_wao_20261021"], ["pts_bands_20261021", "pts"])
を 1 行足して build() を流す。ただし本体は今回の編集範囲の外なので、本体は変えずにここで SRC と
帯の消失表（BANDGONE）だけ差し込んで build() を呼ぶ（本体の処理・列・並びはそのまま使う）。

帯の消失＝band_gone_20261020.py を 10/21 の 30 曲で流した結果（thr 0.25・10/20 と同じ・帯は 10/30 の差し替え後の bands_wao）
  dataset/close4_20261030/song_band_gone_20261021.csv
を既存の dataset/song_band_gone_20261020.csv と縦に足した一時ファイルを BANDGONE として読ませる。

検算＝既存行（10/21 以外）が退避と全セル一致すること・10/21 が 30 行入ること。

使い方
  python song_valid_range_add_20261021.py            # 退避（無ければ）→作り直し→検算
  python song_valid_range_add_20261021.py --check    # 作らずに内訳だけ
"""
import os, sys, csv, shutil, tempfile
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import song_valid_range_20261015 as svr

SES = "20261021"
BG_NEW = os.path.join(svr.DS, "close4_20261030", "song_band_gone_20261021.csv")
BAK = svr.OUT + ".pre_20261021_20261030.bak"


def merged_bandgone():
    rows = []; hdr = None
    for p in (svr.BANDGONE, BG_NEW):
        with open(p, encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            if hdr is None:
                hdr = r.fieldnames
            elif r.fieldnames != hdr:
                raise SystemExit("帯の消失表の列が違う: " + p)
            rows += [x for x in r if not (p == svr.BANDGONE and x["session"] == SES)]
    fd, tmp = tempfile.mkstemp(suffix="_bandgone.csv"); os.close(fd)
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)
    return tmp


def main():
    check = "--check" in sys.argv
    svr.SRC[SES] = (["results_wao_20261021"], ["pts_bands_20261021", "pts"])
    tmp = merged_bandgone()
    svr.BANDGONE = tmp
    try:
        if not check and not os.path.exists(BAK):
            shutil.copy2(svr.OUT, BAK)
            print("退避", BAK)
        rows = svr.build(check=check)
    finally:
        os.remove(tmp)
    if check:
        return
    old = {(r["session"], r["video"]): r for r in csv.DictReader(open(BAK, encoding="utf-8-sig"))}
    new = {(r["session"], r["video"]): r for r in csv.DictReader(open(svr.OUT, encoding="utf-8-sig"))}
    changed = [k for k in old if k[0] != SES and (k not in new or new[k] != old[k])]
    n_ses = sum(1 for k in new if k[0] == SES)
    print(f"既存行 {sum(1 for k in old if k[0] != SES)}・変わった/消えた {len(changed)}・{SES} の行 {n_ses}")
    if changed or n_ses != 30:
        print("検算 NG", changed[:5]); sys.exit(2)
    print("検算 OK")


if __name__ == "__main__":
    main()
