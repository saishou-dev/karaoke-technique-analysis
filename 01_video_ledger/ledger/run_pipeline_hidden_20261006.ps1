# 18:55 BOM 付与（BOM 無し UTF-8 を Windows PowerShell が ANSI として読み、日本語コメントで構文エラー＝タスク exit 1 の原因）
# 2026-10-06 版 launcher：監督をコンソール非表示で起動し stdout/stderr をファイルに残す（run_pipeline_hidden_20260924.ps1 の派生）。
# 10/6 は機種が2つ（-Machine wao → pipeline_20261006.py・ログ dataset\pipeline_20261006／-Machine ai → pipeline_20261006ai.py・ログ dataset\pipeline_ai_20261006）。
# -Next <タスク名> を与えると監督の終了後（成否を問わず）に schtasks /run で次のタスクを起動する（WAO → Ai の直列実行・CPU競合回避）。
# 使い方（タスク /tr）: powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<this>" -Machine wao -From probe -Next kv_pipeline_20261006ai
param([string]$Machine = "wao", [string]$From = "probe", [string]$Session = "20261006", [string]$Next = "")
$py = "C:\Users\conan\AppData\Local\Programs\Python\Python312\python.exe"
$root = "G:\研究アーカイブ\成果物\動画解析"
if ($Machine -eq "ai") { $script = "pipeline_20261006ai.py"; $dir = Join-Path $root "dataset\pipeline_ai_$Session" }
else { $script = "pipeline_20261006.py"; $dir = Join-Path $root "dataset\pipeline_$Session" }
New-Item -ItemType Directory -Force $dir | Out-Null
$stamp = Get-Date -Format "HHmmss"
$out = Join-Path $dir "_task_stdout_${From}_$stamp.txt"
$err = Join-Path $dir "_task_stderr_${From}_$stamp.txt"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:KV_SESSION = $Session
"start machine=$Machine script=$script from=$From $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $err -Encoding utf8
$p = Start-Process -FilePath $py -ArgumentList @("`"$root\scripts\$script`"", $Session, "--from", $From) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
"exit=$($p.ExitCode) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $err -Append -Encoding utf8
if ($Next -ne "") {
  Start-Sleep -Seconds 5
  $r = & schtasks /run /tn $Next
  "next=$Next run=$r $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $err -Append -Encoding utf8
}
