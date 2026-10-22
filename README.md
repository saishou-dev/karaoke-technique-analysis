# karaoke-technique-analysis

カラオケ採点（DAM 精密採点）の加点技法を、家庭用機材の録音・録画から時刻単位で分析する卒業研究のプログラム一式。

## 目的

> 個人が家庭用機材だけで、初心者でもボイストレーナーなしで、カラオケ採点の加点技法を「どこで・どう」取ったかまでの分析レポートを提供する

「家庭用機材」＝個人が市販で買える機材（スマホ・USB オーディオインターフェース・ダイナミックマイク・ノート PC）。

## 何をするプログラムか（流れ）

```
入力  スマホ録画 mp4（採点画面）／オーディオインターフェース録音 wav／採点結果 CSV
  │
  ├ 01 回収・台帳   録画を採点結果と突合して改名 → 画面下の技法カウンター帯を切り出し（帯 pts 表＋ゲート）
  │                → 数字グリフをテンプレ NCC で読取 → 曲末の値を採点結果と照合（verify）
  │                → カウンター増分の時刻を帯 pts 表で秒にして台帳 G1（どの技法が・何秒に・加点されたか）
  │                → 検収（2 秒クリップを人が○×）
  ├ 02 同期         録画音声 × 録音 wav の RMS 包絡・オンセットの相互相関で offset を 1 本決める
  ├ 03 検出器       録音 wav → 特徴量（log-mel・RMS・pYIN f0）→ 予測器（CRNN／ノート分類／DTW 転写／規則）
  │                → 評価（±tol 秒の事象 F1・件数整合）
  ├ 04 マーク抽出   録画の加点マーク（色付きの印）の出現時刻を拾い、正解 v2 の候補にする
  ├ 05 音程バー     録画のお手本音程バーを読んで基準メロディ列にする／録音 1 本をカラオケ風音程バー画像に描く
  └ 06 レポート     録音 1 本 → 加点候補の時刻・技法・確率と小図を 1 枚の HTML（1 録音レポート）に・合否付き台帳
```

時刻はフレーム番号ではなく pts（提示時刻）で持つ。録画と録音は相互相関の offset 1 本で結ぶ。工程の受け渡しはすべて CSV と npz。

「01 回収・台帳」のカウンター読取器について＝本体は `01_video_ledger/read/read_counters.py`（v0.8・194 行）で、`read_counters_wao.py`（v0.8d）・`read_counters_wao_v08e_20261015.py`（v0.8e）・`read_counters_wao_v09_20261015.py` はこの本体を文字列置換で改変して実行する薄い皮（本体が無いと動かない）。
通して読む用に、本体＋v0.8d＋v0.8e を置換の仕掛けなしで 1 本に書き下ろした `01_video_ledger/read/read_counters_wao_single_20261015.py` を置いている（出力は v0.8e と md5 一致・関数は `glyph_boxes` → `norm_glyph` → `classify` → `detect_increments` の順に読む）。

## フォルダ構成

| フォルダ | 中身（各フォルダ直下＝その系列の最新版） |
|---|---|
| `01_video_ledger/` | 回収パイプライン。`rename/` 改名と転送／`band/` 共通設定 `kv_config_*`・帯プローブ・帯クロップ・ROI 較正・`mp4_pts.py`／`read/` カウンター読取器とランナー／`verify/`／`ledger/` 監督 `pipeline_*`・G1 生成 `g1_and_match_*`・結合 `merge_g1_all_*`／`kensyu/` 検収ページとクリップ。手順書 `RUNBOOK_20260915.md` |
| `02_sync/` | 録画↔録音の同期 `sync_*` |
| `03_detector/` | `features/` pYIN・索引・特徴量・吸着ラベル・過去演奏ペア／`models/` CRNN `train_crnn.py`・規則・ノート分類・DTW 転写・件数混合／`eval/` 事象 F1・件数整合／`vibrato_depth/` ビブラートの深さ・速さの測定 |
| `04_marks/` | 加点マーク抽出 `detect_marks.py` → `rematch.py` → `refine_onsets.py` → `summarize.py`（駆動 `run_session.py`・別案 `flash_detect.py`） |
| `05_pitchbar/` | `read/` 音程バー読取 `read_pitchbar.py`（設計メモ同梱）／`render/` 音程バー描画 `render_pitchbar.py` |
| `06_report/` | 1 録音レポート `evaluate_recording.py`（使い方 `README_evaluate_recording.md`）・合否付き台帳 `build_ledgers_*`・歌声の数値 `song_metrics.py`・C3 整合表 `c3_consistency_table.py`（下ごしらえ＝100 fps ピッチ曲線 `c3_pitch_feats.py`） |
| `tools/` | 無人ジョブの番犬 `job_watchdog.py` |
| `legacy/` | 同系の旧版（日付サフィックス違い）。系列ごとに同じ構成 |
| `support/` | 一回限りの修正・調査・セルフテスト・実験用 |

WAO（6 カウンター・帯 2480×120）と Ai（4 カウンター・帯 1500×180）の二系統は `kv_config_20260915.py`（基底）→ `kv_config_20260915ai.py` → `kv_config_20261006.py` の継承で差し替える。

## 規模（2026-10-10 集計・空行とコメント込み）

| 種類 | 本数 | 行数 |
|---|---|---|
| Python | 146 | 12,449 |
| PowerShell | 7 | 246 |
| sh（WSL 用） | 21 | 304 |
| Markdown（手順書・設計メモ） | 3 | 369 |
| HTML（検収ページの台紙） | 1 | 131 |

## 実行環境

- Windows 10 ＋ Python 3.12（監督・改名・同期・特徴量・学習・描画）／WSL Ubuntu の venv（カウンター読取器の並列ランナー・sh）
- 主要ライブラリ＝numpy・scipy・pandas・matplotlib・librosa・soundfile・opencv-python（cv2）・torch・scikit-learn・Pillow・joblib
- 外部ツール＝ffmpeg／ffprobe（帯の切り出し・復号・クリップ生成）
- パスは各スクリプト先頭または `kv_config_*.py` の定数（`G:\研究アーカイブ\...`）で指定しているので、使う環境に合わせて書き換える

## データは含まない

このリポジトリにはプログラムだけを置く。録音・録画・画像・採点結果 CSV・台帳 G1・索引・学習済みモデル・検収ページ・ログは含まない（`.gitignore` で除外）。
採点結果のデータは株式会社第一興商のサービス（DAM★とも）から本人が取得したもので、再配布しない。

## 著者

Inoue Toshiaki（[saishou-dev](https://github.com/saishou-dev)）
