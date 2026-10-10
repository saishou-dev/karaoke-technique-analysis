# 2026-09-25 版：監督スクリプト名（-Script・既定 pipeline_20260915.py）とセッション（-Session）を分離。run_pipeline_hidden_20260915.ps1 は「pipeline_<Session>.py」を仮定していたため 9/24 セッションで起動失敗（exit=2）した。
# 監督 pipeline_20260915.py をコンソール非表示で起動する launcher（2026-09-15 16:45 作成）。
# 背景：15:37 に schtasks で起動した監督（可視コンソール）が 15:57:04 に ffmpeg 3 本もろとも同時停止（LastTaskResult=1・ログ無し）。
#       Ctrl+C／ウィンドウ閉じの疑いがあるため、タスクからは powershell -WindowStyle Hidden 経由で起動し、標準出力・エラーをファイルに残す。
# 使い方（タスク /tr）: powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<this>" -From crop
param([string]$From = "probe", [string]$Session = "20260924", [string]$Script = "pipeline_20260915.py")
$py = "C:\Users\conan\AppData\Local\Programs\Python\Python312\python.exe"
$root = "G:\研究アーカイブ\成果物\動画解析"
$dir = Join-Path $root "dataset\pipeline_$Session"
New-Item -ItemType Directory -Force $dir | Out-Null
$stamp = Get-Date -Format "HHmmss"
$out = Join-Path $dir "_task_stdout_${From}_$stamp.txt"
$err = Join-Path $dir "_task_stderr_${From}_$stamp.txt"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$p = Start-Process -FilePath $py -ArgumentList @("`"$root\scripts\$Script`"", $Session, "--from", $From) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
"exit=$($p.ExitCode) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $err -Append -Encoding utf8
