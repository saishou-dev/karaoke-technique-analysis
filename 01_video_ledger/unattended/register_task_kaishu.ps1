# 無人ランチャのタスク登録（2026-10-22 新規）。
# 既定＝**無効状態**で登録する（有効化は本人の○のあと・1行コマンドは README に書いた）。
# BOM 付き UTF-8 で保存すること（10/6 の教訓）。
#
# 使い方:
#   powershell -NoProfile -ExecutionPolicy Bypass -File "<this>"              # 無効状態で登録（既定）
#   ... -File "<this>" -Shortcut                                             # デスクトップに「回収開始」のショートカットも作る
#   ... -File "<this>" -Remove                                               # タスクを削除
#
# 注意＝/rl HIGHEST は昇格が要るので付けない（10/21 に「アクセスが拒否されました」で失敗した）。
# 注意＝MTP（スマホ）は本人がログオン中のセッションからしか見えないので LogonType は対話トークンのまま。

param([switch]$Shortcut, [switch]$Remove, [string]$TaskName = "kv_kaishu_unattended")

$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$Here     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Launcher = Join-Path $Here "kaishu_unattended.ps1"
$Cmd      = Join-Path $Here "start_kaishu.cmd"

if ($Remove) {
    & schtasks /delete /tn $TaskName /f
    "削除しました: $TaskName"
    exit 0
}
if (-not (Test-Path $Launcher)) { "ランチャが無い: $Launcher"; exit 2 }

$tr = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $Launcher + '"'
# 過去日の /sc once ＝自動では走らない（schtasks /run か start_kaishu.cmd で手動起動する形）
& schtasks /create /tn $TaskName /tr $tr /sc once /st 00:01 /sd (Get-Date -Format "yyyy/MM/dd") /f
if ($LASTEXITCODE -ne 0) { "登録に失敗（exit $LASTEXITCODE）"; exit 2 }

# 長時間ジョブ用の設定（10/14・10/21 の再開時と同じ形）
try {
    $t = Get-ScheduledTask -TaskName $TaskName
    $t.Settings.ExecutionTimeLimit = "PT12H"
    $t.Settings.StopIfGoingOnBatteries = $false
    $t.Settings.DisallowStartIfOnBatteries = $false
    $t.Settings.StartWhenAvailable = $true
    $t.Settings.IdleSettings.StopOnIdleEnd = $false
    $t.Settings.MultipleInstances = "IgnoreNew"
    Set-ScheduledTask -InputObject $t | Out-Null
    "設定＝実行時間上限 12h・バッテリ条件なし・アイドルで止めない・多重起動は無視"
} catch { "設定の変更で例外: $($_.Exception.Message)" }

# 既定は無効（本人の○まで走らせない）
& schtasks /change /tn $TaskName /disable | Out-Null
$st = (& schtasks /query /tn $TaskName /fo LIST | Select-String -Pattern "^(Status|状態)" | ForEach-Object { $_.Line.Trim() })
"登録しました: $TaskName ／ $st"

if ($Shortcut) {
    $desk = [Environment]::GetFolderPath("Desktop")
    $lnk = Join-Path $desk "回収開始.lnk"
    $ws = New-Object -ComObject WScript.Shell
    $s = $ws.CreateShortcut($lnk)
    $s.TargetPath = $Cmd
    $s.WorkingDirectory = $Here
    $s.WindowStyle = 7          # 最小化（ダイアログレス）
    $s.Description = "回収パイプラインの無人実行を1回起動する"
    $s.Save()
    "ショートカット＝$lnk"
}
exit 0
