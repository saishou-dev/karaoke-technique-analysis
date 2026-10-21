# 2026-10-22 版 launcher（run_pipeline_hidden_20261021.ps1 の派生・scripts は無変更）。
# 違い＝(1) KV_PROBE_YR を 0.36,0.55 にする (2) WSLENV で WSL 側へ渡す。
# WSLENV が無いと wsl.exe は Windows の環境変数を Linux に渡さないため、
# 旧ランチャの $env:KV_PROBE_YR は帯プローブに届いていなかった（10/22 02:47 実測＝YR 空）。
# BOM 付きで保存すること（10/6 の教訓）。
param([string]$From = "probe", [string]$Session = "20261021")
$py = "C:\Users\conan\AppData\Local\Programs\Python\Python312\python.exe"
$root = "G:\研究アーカイブ\成果物\動画解析"
$script = "pipeline_20261021.py"
$dir = Join-Path $root "dataset\pipeline_$Session"
New-Item -ItemType Directory -Force $dir | Out-Null
$stamp = Get-Date -Format "HHmmss"
$out = Join-Path $dir "_task_stdout_yr_${From}_$stamp.txt"
$err = Join-Path $dir "_task_stderr_yr_${From}_$stamp.txt"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:KV_SESSION = $Session
$env:KV_PROBE_YR = "0.36,0.55"
$env:WSLENV = "KV_PROBE_YR"
"start machine=wao script=$script from=$From yr=$($env:KV_PROBE_YR) wslenv=$($env:WSLENV) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $err -Encoding utf8
$p2 = Start-Process -FilePath $py -ArgumentList @("`"$root\scripts\$script`"", $Session, "--from", $From) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
"exit=$($p2.ExitCode) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $err -Append -Encoding utf8
