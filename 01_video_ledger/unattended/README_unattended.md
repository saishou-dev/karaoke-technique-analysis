# 回収パイプラインの無人実行（2026-10-22 新規・指南書）

目的＝録画の回収（転送→改名→監督→仕上げ）を、Claude のセッションに頼らずメインPCのタスクスケジューラと WSL だけで通せる形にする。
正本の手順書は `01_video_ledger/RUNBOOK_20260915.md`。
この指南書はその手順を1本のランチャに落とした形だけを書く（手順そのものの正本は RUNBOOK のまま）。

ランチャは既存スクリプトを**呼ぶだけ**で、1文字も変更しない。

## 1. ファイル

| ファイル | 役割 |
|---|---|
| `kaishu_unattended.ps1` | 本体。detect → prepare → dam → transfer → rename → supervise → finish を1プロセスで通す |
| `start_kaishu.cmd` | 1回押す用。`schtasks /run /tn kv_kaishu_unattended` を叩くだけ（ASCII のみ） |
| `register_task_kaishu.ps1` | タスクを**無効状態**で登録する。`-Shortcut` でデスクトップに「回収開始」も作る |
| `README_unattended.md` | この指南書 |

ログ＝`G:\研究アーカイブ\成果物\動画解析\logs\kaishu_unattended_<日付>_<時刻>.log`。
報告書＝`C:\Users\conan\hisho\routine_inbox\<日付>_<時刻>_kaishu_unattended_<session>.md`（先頭 `<!-- unread -->`・5項目）。

## 2. 工程表

所要は 10/21（30本・64.36GB・29曲）と 9/11・9/15 の実測。

| # | 工程 | 入力 | 出力 | 所要 | 失敗時の挙動 |
|---|---|---|---|---|---|
| 1 | detect | MTP のデバイス一覧（`Shell.Application` NameSpace 17） | セッション（日付タグ）・スマホ側の本数と容量 | 10秒 | スマホ未接続なら日付タグをビデオ素材の最新フォルダから推定し、転送工程で停止して報告 |
| 2 | prepare | 直近世代のセッション別スクリプト8本 | `*_<session>.py` / `.ps1` / `.sh` 8本 | 10秒 | ひな型が揃っていなければ停止。当日の採点行を含む Hearts CSV が無ければ停止（改名も突合もできないため） |
| 3 | dam | DAM★とも（WSL `~/karaoke-api`） | `dam_api\dam_ai-hearts_full_*.csv`・`dam_ai_full_*.csv` | 2分 | 当日行が0なら停止（退室前ログアウト忘れ・取得失敗）。既に当日行を含む CSV があれば飛ばす |
| 4 | transfer | スマホ `DCIM\Camera\ビデオ\VID_<session>_*.mp4` | `ビデオ素材\<session>\*.mp4`・`_transfer.log`・`_transfer_result.txt` | 30〜65分（10/21＝64分22秒・1本92〜174秒） | `ok≠total` か `PHONE_NOT_FOUND` で停止。3時間（`-TransferHours`）で打ち切り。再実行は済みを SKIP するので続きから |
| 5 | rename | VID_ 動画＋Hearts CSV（時刻窓＝動画開始 < 採点時刻 ≦ 次の動画開始） | 曲名付き mp4・`rename_mapping.csv` | 1分未満 | `STOP:` か TIME_WINDOW=0 か「8割未満」で停止。AMBIGUOUS・UNUSED_ROW は停止せず報告書の「判断待ち」に積む |
| 6 | supervise | 改名済み mp4（machine=wao）・プローブ | `band_probe_v2_<S>.csv`・`bands_wao\*`・`pts_bands_<S>\*`・`results_wao_<S>\*`・`verify_<S>_wao.csv`・`G1_events_<S>_wao.csv` | 約2時間（probe 3分・crop 45分・calib 3分・read 50分・verify/g1 各10秒） | 監督が `PIPELINE_STOPPED.txt` を書いたらその理由を報告書に転記して停止。6時間（`-PipelineHours`）で見届けを打ち切るが**プロセスは殺さない**（番犬が停滞を見ている） |
| 7 | finish | `verify_<S>_wao.csv`・`G1_events_<S>_wao.csv`・`calib_<S>\calib_summary.csv`・`results_wao_<S>` | 報告書（受信箱・未読） | 10秒 | 無いファイルは「失敗と対処」に書くだけで止めない |

**仕上げに含めないこと**＝統合台帳 `G1_events_all.csv` への結合、目視追記、検収サンプル、D: ミラー。
いずれも裁定と目視が要る工程（RUNBOOK §6・D-2）なので、報告書の「判断待ち」に出すだけにした。

## 3. 停滞の自己検知（番犬）

常設の `hisho-watchdog`（5分ごとに `ops/tools/job_watchdog.py --once`）が台帳 `watchdog_jobs.json` の全ジョブを点検する。
ランチャは工程の直前に自分で2件登録する。

- `回収転送<session>`＝pattern `mtp_transfer_<session>`／log `_transfer.log`／idle 15分／上限4時間
- `回収監督<session>`＝pattern `pipeline_<session>|wincrop|calib_roi|wao_probe`／log `pipeline.log`／idle 25分／上限8時間

pattern にランチャ自身（`kaishu_unattended`）は入れていない。
停滞で番犬が子プロセスを止めても、ランチャは生きていて報告書を書けるようにするため。

## 4. 起動の合図

**採用＝デスクトップの「回収開始」を1回押す**（`start_kaishu.cmd` → `schtasks /run`）。

採った理由。
誤発火がゼロである（スマホは充電や写真の取り出しでも日に何度も繋ぐ。接続だけを合図にすると、研究と無関係な接続で 64GB の転送が始まる）。
本人の運用と一致する（「回収よろしく」＝接続済みの合図、という既存の合図の形をそのまま押しボタンにした）。
タイミングを本人が握れる（帰宅直後に風呂へ入る日は、押さなければ始まらない）。

**採らなかった案＝MTP 接続の自動検知**。
MTP はドライブレターを取らないので、`Get-PnpDevice` のポーリング（5分ごと）か WMI の `__InstanceCreationEvent` 常駐のどちらかが必要になる。
前者は上の誤発火の問題がそのまま残り、後者は「番犬の番犬が要る常駐プロセス」になって常設の形（OS のスケジューラを根にする）から外れる。
自動検知を足すなら、`ビデオ素材\_ARM_KAISHU.txt` のような武装ファイルがある時だけ反応する形にするのが筋。

## 5. 使い方

```
powershell -NoProfile -ExecutionPolicy Bypass -File "...\unattended\register_task_kaishu.ps1" -Shortcut   # 登録（無効状態）
schtasks /change /tn kv_kaishu_unattended /enable                                                          # 有効化（本人の○のあと1回だけ）
```

以後は帰宅後にスマホを USB（MTP・ファイル転送）で繋いで「回収開始」を1回押す。
報告書が受信箱に未読で届く。

ドライラン（実データを動かさない・存在確認と引数の組み立てだけ）＝
```
powershell -NoProfile -ExecutionPolicy Bypass -File "...\unattended\kaishu_unattended.ps1" -DryRun
```

途中から＝`-From prepare|dam|transfer|rename|supervise|finish`（`-Session YYYYMMDD` を併せて指定する）。

## 6. 前提

- 本人がログオン中であること。MTP のデバイス一覧は対話セッションからしか見えない。
- G: がマウントされていること。
- WSL `Ubuntu-26.04` が起動できること（dam 工程とプローブ・読取で使う）。
- 再起動保留がないこと。監督側の PREFLIGHT が WARN を出す（10/15 03:26 に Windows Update の計画再起動で ffmpeg 4本が全滅した）。
