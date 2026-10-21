# MTP転送 2026-10-14 版（mtp_transfer_20260911.ps1 の派生・既存は無変更）。
# スマホ REDMAGIC 10 Pro の DCIM\Camera\ビデオ → G:\研究アーカイブ\ビデオ素材\<DateTag>\ へ1本ごとにサイズ照合しながら転送。
# 9/11 版との差＝(1) 進捗を1行ずつ進捗ログ（_transfer.log）に書く（番犬 job_watchdog.py が増加を見る）
#               (2) 既に同サイズで存在するファイルは SKIP（再実行で続きから）
#               (3) 完了時に _transfer_result.txt に DONE ok=N total=N を書く
# BOM 付きで保存すること（10/6 の教訓＝BOM 無しだと日本語コメントで構文エラー）。
# 使い方: powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<this>" -DateTag 20261021
param([string]$DateTag = "20261021", [string]$Dest = "")
$ErrorActionPreference = 'Continue'
if ($Dest -eq "") { $Dest = "G:\研究アーカイブ\ビデオ素材\$DateTag" }
New-Item -ItemType Directory -Force $Dest | Out-Null
$log = Join-Path $Dest "_transfer.log"
function W($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
W "START DateTag=$DateTag Dest=$Dest"
$sh = New-Object -ComObject Shell.Application
$pc = $sh.NameSpace(17)
$phone = $pc.Items() | Where-Object { $_.Name -eq 'REDMAGIC 10 Pro' }
if (-not $phone) { W "PHONE_NOT_FOUND"; "PHONE_NOT_FOUND" | Out-File (Join-Path $Dest "_transfer_result.txt") -Encoding utf8; exit 1 }
$storage = $phone.GetFolder.Items() | Select-Object -First 1
$vid = $storage.GetFolder.ParseName('DCIM').GetFolder.ParseName('Camera').GetFolder.ParseName('ビデオ').GetFolder
$destNs = $sh.NameSpace($Dest)
$items = @($vid.Items() | Where-Object { $_.Name -like "VID_${DateTag}_*.mp4" } | Sort-Object Name)
W "TOTAL $($items.Count)"
$ok = 0
foreach ($it in $items) {
    $name = $it.Name
    $target = Join-Path $Dest $name
    $srcSize = [int64]$it.ExtendedProperty("System.Size")
    if ((Test-Path $target) -and ((Get-Item $target).Length -eq $srcSize)) { W "SKIP(done) $name $srcSize"; $ok++; continue }
    if (Test-Path $target) { W "RETRY(size mismatch) $name"; Remove-Item $target -Force }
    $t0 = Get-Date
    $destNs.CopyHere($it, 16 + 4 + 1024)
    $tries = 0
    while ($true) {
        Start-Sleep -Seconds 5
        if (Test-Path $target) {
            $len = (Get-Item $target).Length
            if ($len -eq $srcSize) { break }
        }
        $tries++
        if (($tries % 12) -eq 0) {
            $cur = if (Test-Path $target) { (Get-Item $target).Length } else { 0 }
            W "PROGRESS $name $cur/$srcSize"
        }
        if ($tries -gt 720) { break }
    }
    $len = if (Test-Path $target) { (Get-Item $target).Length } else { -1 }
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    if ($len -eq $srcSize) { $ok++; W "OK $name $srcSize ${sec}s" } else { W "SIZE_MISMATCH $name src=$srcSize dst=$len" }
}
W "DONE ok=$ok total=$($items.Count)"
"DONE ok=$ok total=$($items.Count)" | Out-File (Join-Path $Dest "_transfer_result.txt") -Encoding utf8
