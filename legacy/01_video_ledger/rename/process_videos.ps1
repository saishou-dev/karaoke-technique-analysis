# 動画一括処理ドライバ v1 (2026-08-08)
# ビデオ素材/inbox の未処理mp4を: 帯位置特定 -> 帯切り出し -> カウンター読み取り -> results/ 保存
# 台帳 processed.json で冪等(処理済みはスキップ)
param(
    [string]$SourceDir = "C:\大学\授業\041 4年\043 卒研\卒業研究_カラオケ採点\ビデオ素材",
    [string]$WorkDir = "C:\大学\授業\041 4年\043 卒研\卒業研究_カラオケ採点\動画解析"
)
$ffm = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin\ffmpeg.exe"
$ledgerPath = Join-Path $WorkDir "processed.json"
$ledger = @{}
if (Test-Path $ledgerPath) {
    (Get-Content $ledgerPath -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $ledger[$_.Name] = $_.Value }
}
New-Item -ItemType Directory -Force (Join-Path $WorkDir "bands") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $WorkDir "results") | Out-Null

function ToWsl($p) { "/mnt/" + $p.Substring(0,1).ToLower() + ($p.Substring(2) -replace "\\", "/") }

$videos = Get-ChildItem $SourceDir -Filter *.mp4
foreach ($v in $videos) {
    $name = [IO.Path]::GetFileNameWithoutExtension($v.Name)
    if ($ledger.ContainsKey($name)) { Write-Output "SKIP(done): $name"; continue }
    Write-Output "=== $name ==="
    $wslVideo = ToWsl $v.FullName
    # 1) 帯位置特定
    $loc = wsl -d Ubuntu-26.04 -- bash -lc "~/karaoke-vision/.venv/bin/python ~/karaoke-vision/locate_band.py '$wslVideo' ~/karaoke-vision/sample_counterband_t70.png ~/karaoke-vision/dark_band_ref.png" | Select-String "^BAND"
    if (-not $loc) { Write-Output "LOCATE_FAIL: $name"; continue }
    $p = ($loc.ToString() -split " ")
    $bx = $p[1]; $by = $p[2]; $bw = $p[3]; $bh = $p[4]
    Write-Output "band: x=$bx y=$by w=$bw h=$bh scale=$($p[5]) score=$($p[6])"
    # 2) 帯切り出し
    $band = Join-Path $WorkDir "bands\$name.mp4"
    & $ffm -y -v error -i $v.FullName -vf "crop=${bw}:${bh}:${bx}:${by},scale=1500:180" -c:v libx264 -crf 15 -preset fast -an $band
    if ($LASTEXITCODE -ne 0) { Write-Output "CROP_FAIL: $name"; continue }
    # 3) 読み取り(既定ROI -> 読取率低ければ暗スキンROIで再試行し良い方を採用)
    $resDir = Join-Path $WorkDir "results\$name"
    New-Item -ItemType Directory -Force $resDir | Out-Null
    $wslBand = ToWsl $band
    $wslRes = ToWsl $resDir
    $log1 = wsl -d Ubuntu-26.04 -- bash -lc "~/karaoke-vision/.venv/bin/python ~/karaoke-vision/read_counters.py '$wslBand' '$wslRes' ~/karaoke-vision/out/cluster_reps.npy"
    $log2 = $null
    if (($log1 | Select-String "kobushi: readable" ).ToString() -match "\((\d+)%\)" -and [int]$Matches[1] -lt 60) {
        Write-Output "retry with dark-skin ROI (kobushi low)"
        $log2 = wsl -d Ubuntu-26.04 -- bash -lc "~/karaoke-vision/.venv/bin/python ~/karaoke-vision/read_counters.py '$wslBand' '$wslRes' ~/karaoke-vision/out/cluster_reps.npy ~/karaoke-vision/roi_dark_skin.json"
    }
    $finalLog = if ($log2) { $log2 } else { $log1 }
    $finalLog | Out-File (Join-Path $resDir "read_log.txt") -Encoding utf8
    $finalLog | Where-Object { $_ -match "initial=" } | ForEach-Object { Write-Output $_ }
    # 4) 台帳更新
    $ledger[$name] = @{ processed = (Get-Date -Format "yyyy-MM-dd HH:mm"); band = "$bx,$by,$bw,$bh"; source = $v.FullName }
    $ledger | ConvertTo-Json -Depth 4 | Out-File $ledgerPath -Encoding utf8
}
Write-Output "ALL DONE. ledger: $ledgerPath"
