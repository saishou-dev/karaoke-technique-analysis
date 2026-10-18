# 数字の読み取り手法の比較（既製OCR との対照）2026-10-18

録画の加点回数表示（画面下部の加点表示領域にある技法別の回数）を「数字として読み取る」工程について、
現行の自作手法（見本照合＝正規化相互相関 NCC・読取器 v0.8e）と既製の文字認識（OCR）エンジンを
**同じ入力・同じ後処理・同じ尺度**で比べる実験の一式。

結果の読み物＝`C:\Users\conan\hisho\research\比較_数字読み取り手法_2026-10-18.md`

## ファイル

| ファイル | 役目 |
|---|---|
| `glyph_core.py` | 読取器 v0.8e の「数字の読み取り」部分を関数として切り出した参照実装（正本は書き換えない） |
| `identity_check.py` | 参照実装が正本 v0.8e と同じ答えを出すことを1曲の全コマ走査で実測 |
| `extract_rois.py` | 対象曲の帯動画から等間隔でコマを採り、6つの関心領域を保存＋正解値表を作る（入力を1回だけ作る） |
| `run_candidates.py` | 候補を1つ指定して同じ入力を読ませ、予測と処理時間を書き出す |
| `run_all.sh` | 候補を3レーンに分けて専用4コアで走らせる（処理時間を同条件で測るため） |
| `run_shard.sh` | 1候補を分割実行する（壁時計を縮めるため・1プロセスは専用4コアのまま） |
| `merge_shards.py` | 分割実行した予測と処理時間を1つにまとめる |
| `tess_timing.py` | Tesseract の処理時間を「落ちない一括実行」と「1枚ずつ」で測り直す補足測定 |
| `score_compare.py` | 予測を同じ尺度で採点して比較表（compare_table.csv・per_counter.csv・finals.csv）を作る |
| `make_figure.py` | 比較表から図1枚（PNG）を作る |
| `watchdog_lanes.sh` | WSL 内で走るジョブの番犬（CPU時間が増えなければ停止して受信箱へ記録） |
| `setup_env.sh` | 既製OCR の導入（sudo を使わない・ダイアログを出さない手順） |

## 再現手順

```bash
# 0) 既製OCR の用意（sudo 不要・ダイアログレス）
bash setup_env.sh

# 1) 参照実装が正本と同じであることの確認
source ~/ocr_compare_20261018/env.sh
~/ocr_compare_20261018/.venv312/bin/python identity_check.py "恋人ごっこ(生音)_20261014_124950"
#  → RESULT: IDENTICAL

# 2) 入力（同じコマ）と正解値を作る
~/ocr_compare_20261018/.venv312/bin/python extract_rois.py \
    --out /mnt/g/研究アーカイブ/成果物/動画解析/ocr_compare_20261018/work --stride 8

# 3) 候補を走らせる（3レーン並行・各レーン専用4コア）
bash run_all.sh A &   # PaddleOCR
bash run_all.sh B &   # EasyOCR 3種
bash run_all.sh C &   # 現行NCC・対照・Tesseract 3種
#   重い候補は分割して壁時計を縮める（1プロセスは専用4コアのままなので処理時間の比較は保たれる）
#     bash run_shard.sh paddle_rec 0 4 0-3 &  … 0〜3 の4分割
#     python merge_shards.py --pred .../pred --cand paddle_rec --nshards 4
#   PaddleOCR の一式は重いので間引いて実行する（採点側がその部分集合だけで集計する）
#     bash run_shard.sh paddle_full 0 11 8-11
#     cp pred_paddle_full.shard0of11.csv pred_paddle_full.csv（time_ も同様）
#   長時間になるので番犬を同時に分離起動する
#     setsid nohup bash watchdog_lanes.sh 10 > /dev/null 2>&1 &

# 4) 採点して比較表を作る
~/ocr_compare_20261018/.venv312/bin/python score_compare.py \
    --work .../work --pred .../pred --out .../result

# 5) 図（Windows の python3.12＝日本語フォントがある側で実行）
python make_figure.py --result .../result --out .../result/fig_compare.png
```

## 注意

- 正本（`read_counters.py`・`read_counters_wao_v08e_20261015.py`）と正式結果
  （`results_wao_20261014`・`dataset/G1_events_20261014_wao.csv`）は**読み取り専用**で参照する。
  この実験の出力はすべて `ocr_compare_20261018/` の下に出す。
- 帯動画は可変フレームレートで cv2 のシークがずれるため、コマの採取は必ず**逐次デコード**で行う
  （2026-10-15 に確認済みの癖）。
- 既製OCR は pip、Tesseract 本体は Ubuntu の公式アーカイブから .deb を取得して
  ホーム配下へ展開する（`apt-get download` ＋ `dpkg-deb -x`）。sudo もダイアログも使わない。
