# 2026-10-14 版 launcher（run_pipeline_hidden_20261006.ps1 の派生・既存は無変更）。
# 監督をコンソール非表示で起動し stdout/stderr をファイルに残す。10/14 は WAO の1機種のみ（-Machine wao 固定）。
# BOM 付きで保存すること（10/6 の教訓＝BOM 無し UTF-8 を Windows PowerShell が ANSI として読み、日本語コメントで構文エラー＝タスク exit 1）。
# 使い方（タスク /tr）: powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<this>" -From probe
param([string]$From = "probe", [string]$Session = "20261014")
$py = "C:\Users\conan\AppData\Local\Programs\Python\Python312\python.exe"
$root = "G:\研究アーカイブ\成果物\動画解析"
$script = "pipeline_20261014.py"
$dir = Join-Path $root "dataset\pipeline_$Session"
New-Item -ItemType Directory -Force $dir | Out-Null
$stamp = Get-Date -Format "HHmmss"
$out = Join-Path $dir "_task_stdout_${From}_$stamp.txt"
$err = Join-Path $dir "_task_stderr_${From}_$stamp.txt"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:KV_SESSION = $Session
$env:KV_PROBE_YR = "0.20,0.80"
"start machine=wao script=$script from=$From $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $err -Encoding utf8
$p = Start-Process -FilePath $py -ArgumentList @("`"$root\scripts\$script`"", $Session, "--from", $From) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
"exit=$($p.ExitCode) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $err -Append -Encoding utf8
