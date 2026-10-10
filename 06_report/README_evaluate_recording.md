# evaluate_recording.py の使い方（録音1本を1コマンドで評価）

作成 2026-10-08（E1）。置き場＝`G:\研究アーカイブ\成果物\動画解析\g3v2_20260928\scripts\evaluate_recording.py`。新しい学習はしない（学習済みモデルの推論だけ）。

## 1コマンド

```
cd G:\研究アーカイブ\成果物\動画解析\g3v2_20260928
python scripts\evaluate_recording.py --wav "G:\研究アーカイブ\録音\songs_20260924_raw\T02_20260924_101915.wav" --out out\evaluate\lulu_20260924
```

| 引数 | 必須 | 意味 |
|---|---|---|
| `--wav <path>` | 必須 | 評価する録音WAV（SSL2 の T番号ファイルでも、曲名ファイルでも可） |
| `--video <名前>` | 任意 | index.json のエントリ名（例 `lulu._20260924_101433`）。指定すると正解G1と照合する。省略時は WAV のパスが index と一致すれば自動で正解を付ける。index に無い WAV は「予測のみ」 |
| `--dam-csv <path>` | 任意 | DAM の採点CSV（`G:\研究アーカイブ\成果物\dam_api\dam_ai-hearts_full_*.csv`）。WAV名の時刻（`_YYYYMMDD_HHMMSS`）±15分の playDate の行を自動で選び、技法回数と件数比較する |
| `--out <dir>` | 任意 | 出力先（既定 `out\evaluate\`） |
| `--tags` `--lag` `--tol` `--threads` `--no-fig` | 任意 | モデル（既定 v2d,v2d2,v2e）・ラベル遅れ補正（既定 0.3s）・照合許容（既定 ±1.0s）・スレッド数（既定 8）・図を省く |
| `--html` | 任意 | **1録音レポート**（録音1本につき1枚・`report_<録音ID>.html`・録音ID＝WAV の stem）を追加出力する。録音だけから検出器で得た独自分析の出力で、正解や DAM の採点は使わない（下記）。既存の出力は変えない |

## 何をするか（順番）

1. 特徴量＝`feats\<名前>.npz`（extract_feats.py と同一手順＝log-mel 80＋RMS＋pYIN。既にあれば再利用。pYIN は `dataset\pitch_pyin*\<WAV名>_f0.csv` があれば読む・無ければその場で計算）
2. 学習済みCRNN 3本（v2d・v2d2・v2e）の確率平均 → 技法別のしきい値・最短間隔（`out\pitchbar_thr_v2d+v2d2+v2e_lag0.3.json`＝dev 9/11 で ±1s F1 最大。キャッシュが無ければ同じ手順で作る）で局所最大を拾い、lag +0.3s を足して予測時刻にする
3. 正解があれば ±1s の貪欲1対1照合（`dataset\ledgers_20261008\build_ledgers_20261008.py` と同じ規則＝予測を時刻順に、未使用の正解のうち最も近いものを取る。採点区間 span 内だけ数える）
4. DAM CSV があれば技法回数（shakuriCount 等）と予測件数・正解件数を並べる
5. 音程バー図＝`render_pitchbar.py` を同じ引数で呼ぶ（図の「一致／誤検知／見逃し」が本ツールの値と同じことを確認して summary に書く）

## 出力（`--out` 直下・`<名前>`＝index の video 名、無ければ WAV のファイル名）

| ファイル | 列 | 意味 |
|---|---|---|
| `<名前>_detections.csv` | technique, technique_ja, time_sec, score, in_span | 予測した加点イベント。time_sec＝録音WAVの秒（lag 補正済み）。score＝3モデル平均の確率。in_span＝採点区間内なら1 |
| `<名前>_matching.csv` | session, song, video, technique, technique_ja, kind, pred_time_sec, truth_time_sec, diff_sec, score, truth_video_time_sec | 正解あり時のみ。kind＝TP（一致）／FP（誤検知＝予測だけ）／FN（見逃し＝正解だけ）。diff_sec＝予測−正解。truth_video_time_sec＝録画の秒（G1台帳の video_time_sec と対応） |
| `<名前>_by_technique.csv` | technique, technique_ja, n_truth, n_pred, TP, FP, FN, precision, recall, f1 | 技法別集計＋ALL。precision=TP/(TP+FP)・recall=TP/(TP+FN) |
| `<名前>_dam_compare.csv` | technique, technique_ja, dam_count, pred_count, g1_count, pred_minus_dam, g1_minus_dam | DAM CSV あり時のみ。dam_count＝DAMの技法回数、pred_count＝採点区間内の予測件数、g1_count＝正解件数（index の events＝multi_increment==0 の行だけなので DAM 回数より少ないことがある） |
| `<名前>_pitchbar.png` | — | 音程バー図（render_pitchbar.py の出力） |
| `<名前>_summary.md` | — | 上の全部の要約と所要時間（画面に出たものと同じ） |
| `report_<録音ID>.html` | — | `--html` 時のみ。1録音レポート（自己完結 HTML・画像は base64 埋め込み・スマホ幅でも読める）。中身は下記 |

## 1録音レポート（`--html`・2026-10-08 追加）

**位置づけ（本人裁定 10/08）**＝「録音1本から検出器で得た独自の分析結果」を、DAM の採点 CSV を形式の手本にして HTML 1枚にしたもの。入力は録音だけ（DAM の採点が無い録音・index に無い録音でも出せる）。
正解（G1）との一致／誤検知／見逃し・DAM 件数比較・測れる範囲表などの **評価は「補助レポート」の仕事**＝このツールの既存出力（`_matching.csv`・`_by_technique.csv`・`_dam_compare.csv`・`_summary.md`）であり、report.html には入れない。

構成（上から）：

| 節 | 中身 | 出どころ |
|---|---|---|
| 1. 録音の情報 | 録音ID（WAV の stem）・曲名（index.json にあれば）・録音日時（ファイル名の `_YYYYMMDD_HHMMSS`）・長さ・サンプリング・機材（録音のみ）・検出器としきい値・生成日時 | WAV ヘッダ・ファイル名 |
| 2. DAM 風の集計 | 技法別の検出回数（6技法）・ビブラートの合計秒／平均深さ／平均速さ（**測れる範囲内の候補だけで計算**し範囲外は件数のみ）・歌唱音域（持続音＝pYIN 有声 0.15 s 以上・確信度 0.5 以上 の最低〜最高・音名と Hz）・有声時間の割合 | 検出表＋feats（pYIN） |
| 3. 音程バー図 | 録音全体・10 秒×N 段・歌った音程（半音に吸着）＋上レーンに検出マーク（render_pitchbar_icons.py と同じ図形・色・濃さ＝確率）。**正解レーンは描かない**。縦に長いので ≤2800 px の帯に分けて埋め込み（スマホでも復号できる） | 本スクリプト内で描画 |
| 4. 加点候補の一覧 | 時刻順に #・技法・時刻・確率・中身（ビブラート＝深さ／速さ／持続、他技法は現時点「—」）。# を押すと小図へ飛ぶ | 検出表 |
| 5. 候補ごとの小図 | 候補前後 −2.5〜+0.5 s の録音波形（上）と pYIN 音程（下）・技法の図形アイコン・縦線＝候補時刻。候補が 50 件を超えると **確率の高い順に上位 30 件を展開**し残りは details で折りたたみ | WAV＋feats |

ビブラートの中身＝`vibrato_depth_20261008\vibrato_depth.py` の measure() と同一定義（窓 [−1.5, +0.3] s・0.3 s 移動中央値の残差・山谷差 ≥15 cent・3〜9 Hz の連鎖）。「測れる範囲」＝`vibrato_depth_20261008\calib\報告_測定の信頼性_2026-10-08.md` §4＝深さは 速さ 4〜5 Hz・40 cent 以上・持続 0.6 s 以上／速さは連鎖あり・4〜6 Hz／持続は 速さ 6 Hz 以下・0.6 s 以上。範囲外は「—」。

確認済み（lulu.・T02_20260924_101915・正解と DAM を使わず生成）＝6.7 MB・候補カード 145 件（しゃくり 23／こぶし 12／ビブラート 53／フォール 0／アクセント 33／ハンマリング 24）・図 152 枚（音程バー 1・アイコン 6・小図 145）・HTML 部の所要 約 50 s（全体 約 2 分）。

## 所要時間（DESKTOP-TGOL9I2・CPU 16スレッド・4分の曲）

| 条件 | 合計 |
|---|---|
| 特徴量キャッシュあり（index 登録曲） | 約 45〜60 秒（推論 16〜26s・図 25〜28s） |
| キャッシュ無し・pYIN の CSV あり | 約 30〜65 秒（特徴量 +5s） |
| キャッシュ無し・pYIN も無い（初めてのWAV） | 上に加えて pYIN 1〜3 分 |
| しきい値キャッシュが無い（初回のみ） | 上に加えて dev 30曲の推論 数分 |

`--no-fig` で図を省くと約 25〜30 秒短くなる。`--html` を付けると約 50 秒増える（小図 145 枚＋音程バー図）。

## 確認済み（2026-10-08）

- lulu.（`lulu._20260924_101433`・T02_20260924_101915.wav）＝一致 24／誤検知 119／見逃し 27・F1 0.247。技法別 F1＝しゃくり 0.410／こぶし 0.111／ビブラート 0.222／フォール 0／アクセント 0.327／ハンマリング 0 ＝ #14 の図および `dataset\ledgers_20261008\検出照合台帳_test20260924.csv`（170行）と行単位で完全一致。
- index に無い WAV（T23 共犯・10/6）＝予測のみ＋DAM 比較で動作（29秒）。
- 10/6 の曲（Lemon）＝正解あり・DAM 10/6 で動作（46秒）。

## 注意

- 予測の精度そのものは G3v2 報告のとおり低い（±1s F1 ≈ 0.2）。このツールは「同じ規則で毎回同じ数字を出す」ための道具で、精度を上げる道具ではない。
- WAV が曲の途中から始まっている場合（10/6 の 阿修羅ちゃん・うっせぇわ）、WAV の前にある正解イベントは照合できず見逃し（FN）に数えられる。
