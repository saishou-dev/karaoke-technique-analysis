# 2026-10-14 回収の自動連鎖（新規・Agent 停止後も走り続ける分離プロセス用）。
# 役割＝MTP転送の完了（_transfer_result.txt）を待ち → 改名（dry-run ログ → --apply）→ 監督タスク kv_pipeline_20261014 を /run で起動。
# タスクスケジューラから起動すること（60分ルール＝Claude のツールから直接起動したプロセスは約60分で消える）。
# 進捗は $chain（dataset\pipeline_20261014\chain.log）に1行ずつ書く（番犬 job_watchdog.py がサイズ増加を見る）。
# BOM 付きで保存すること（10/6 の教訓＝BOM 無しだと日本語コメントで構文エラー＝タスク exit 1）。
# 使い方（タスク /tr）: powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<this>"
param([string]$Session = "20261014", [double]$WaitHours = 3.0)
$ErrorActionPreference = 'Continue'
$py = "C:\Users\conan\AppData\Local\Programs\Python\Python312\python.exe"
$root = "G:\研究アーカイブ\成果物\動画解析"
$vid = "G:\研究アーカイブ\ビデオ素材\$Session"
$pipeDir = Join-Path $root "dataset\pipeline_$Session"
New-Item -ItemType Directory -Force $pipeDir | Out-Null
$chain = Join-Path $pipeDir "chain.log"
function W($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $chain -Append -Encoding utf8 }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:KV_SESSION = $Session
W "CHAIN start session=$Session pid=$PID"

$res = Join-Path $vid "_transfer_result.txt"
$deadline = (Get-Date).AddHours($WaitHours)
while (-not (Test-Path $res)) {
    if ((Get-Date) -gt $deadline) { W "CHAIN STOP 転送が $WaitHours 時間で終わらない（$res が無い）。対処＝_transfer.log を見て schtasks /run /tn kv_transfer_$Session で再開（済みは SKIP）"; exit 2 }
    Start-Sleep -Seconds 60
    $n = @(Select-String -Path (Join-Path $vid "_transfer.log") -Pattern " OK | SKIP" -AllMatches -ErrorAction SilentlyContinue).Count
    W "CHAIN waiting transfer copied=$n"
}
$r = (Get-Content $res -Raw).Trim()
W "CHAIN transfer result = $r"
if ($r -notmatch 'ok=(\d+) total=(\d+)') { W "CHAIN STOP 転送結果が読めない: $r"; exit 2 }
if ($Matches[1] -ne $Matches[2]) { W "CHAIN STOP 転送が不完全（$r）。_transfer.log の SIZE_MISMATCH を直して schtasks /run /tn kv_transfer_$Session"; exit 2 }

W "RENAME dry-run"
& $py "$root\scripts\rename_videos_$Session.py" $Session *>&1 | Out-File (Join-Path $pipeDir "rename_dryrun.txt") -Encoding utf8
W "RENAME apply"
& $py "$root\scripts\rename_videos_$Session.py" $Session --apply *>&1 | Out-File (Join-Path $pipeDir "rename_apply.txt") -Encoding utf8
$mp = Join-Path $vid "rename_mapping.csv"
if (-not (Test-Path $mp)) { W "CHAIN STOP 改名できなかった（rename_mapping.csv が無い）。dataset\pipeline_$Session\rename_apply.txt を見る"; exit 2 }
$nren = @(Get-ChildItem $vid -Filter *.mp4 | Where-Object { $_.Name -notlike 'VID_*' }).Count
W "RENAME ok renamed_mp4=$nren mapping=$mp"

W "PIPELINE launch via schtasks /run /tn kv_pipeline_$Session"
$out = & schtasks /run /tn "kv_pipeline_$Session" 2>&1
W "PIPELINE schtasks: $out"
W "CHAIN end（以降は dataset\pipeline_$Session\pipeline.log と PIPELINE_DONE.txt／PIPELINE_STOPPED.txt を見る）"
