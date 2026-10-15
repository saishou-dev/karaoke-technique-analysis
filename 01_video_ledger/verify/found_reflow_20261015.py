# -*- coding: utf-8 -*-
"""目視用の帯状写真（found_exact_20261015.py / found_pin_20261015.py / scan_runs_20261015.py の出力）を
**折り返して格子に組み直す**（2026-10-15）。

なぜ必要か＝横一列のまま（19タイル＝幅 7,000〜8,400px）だと、表示時に縮小されて数字が 10px 程度になり読めない。
1 行あたりのタイル数を決めて折り返すと、同じ情報が縦に積まれて 1 タイル 300px 以上で見られる。
元の PNG は読むだけで、`_grid.png` を別名で書く（元画像は残す＝根拠画像のパスは変わらない）。

タイルは等幅で並んでいるので、ヘッダ（上端の帯）を除いた部分を N 等分して切り出す。N は `--tiles`（既定19＝found_exact の
ev-14..ev+4）。ヘッダの高さは `--header`（既定30）。

使い方:
  python found_reflow_20261015.py <png またはディレクトリ> [--per-row=5] [--tiles=19] [--header=30] [--pattern=exact_*.png]
出力: 同じディレクトリの <元の名前>_grid.png
"""
import os, sys, glob
import cv2, numpy as np

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")


def imread(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, img):
    cv2.imencode(".png", img)[1].tofile(p)


def reflow(path, per_row=5, tiles=19, header=30, in_per_row=0):
    """in_per_row＝元画像が既に何枚で折り返されているか（found_strip.strip は 16・found_exact は折り返しなし＝0）。"""
    a = imread(path)
    assert a is not None, path
    H, W = a.shape[:2]
    head = a[:header]
    body = a[header:]
    if in_per_row and tiles > in_per_row:
        nrows = (tiles + in_per_row - 1) // in_per_row
        rh = (H - header) // nrows
        w = W // in_per_row
        cols = []
        for r in range(nrows):
            for i in range(in_per_row):
                if len(cols) >= tiles:
                    break
                cols.append(body[r * rh:(r + 1) * rh, i * w:(i + 1) * w])
    else:
        w = W // tiles
        cols = [body[:, i * w:(i + 1) * w] for i in range(tiles)]
    rows = []
    for i in range(0, len(cols), per_row):
        chunk = cols[i:i + per_row]
        line = np.hstack(chunk)
        if line.shape[1] < per_row * w:
            line = cv2.copyMakeBorder(line, 0, 0, 0, per_row * w - line.shape[1],
                                      cv2.BORDER_CONSTANT, value=(20, 20, 20))
        rows.append(line)
    h = max(r.shape[0] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)) if r.shape[0] != h else r for r in rows]
    grid = np.vstack(rows)
    gw = grid.shape[1]
    head = head[:, :gw] if head.shape[1] >= gw else cv2.copyMakeBorder(
        head, 0, 0, 0, gw - head.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
    out = np.vstack([head, grid])
    p = os.path.splitext(path)[0] + "_grid.png"
    imwrite(p, out)
    print(f"{out.shape[1]}x{out.shape[0]}  {os.path.basename(p)}")
    return p


if __name__ == "__main__":
    tgt = sys.argv[1]
    per_row = tiles = header = None
    in_per_row = 0
    pat = "exact_*.png"
    for a in sys.argv[2:]:
        if a.startswith("--per-row="):
            per_row = int(a.split("=")[1])
        elif a.startswith("--tiles="):
            tiles = int(a.split("=")[1])
        elif a.startswith("--header="):
            header = int(a.split("=")[1])
        elif a.startswith("--in-per-row="):
            in_per_row = int(a.split("=")[1])
        elif a.startswith("--pattern="):
            pat = a.split("=", 1)[1]
    kw = dict(per_row=per_row or 5, tiles=tiles or 19, header=header or 30, in_per_row=in_per_row)
    files = sorted(glob.glob(os.path.join(tgt, pat))) if os.path.isdir(tgt) else [tgt]
    files = [f for f in files if not f.endswith("_grid.png")]
    for f in files:
        reflow(f, **kw)
    print(len(files), "files")
