# MTP転送 v2 (2026-09-11) スマホ DCIM\Camera\ビデオ → G:\研究アーカイブ\ビデオ素材\<date>\ 1本ごとにサイズ照合
param([string]$DateTag = "20260911", [string]$Dest = "G:\研究アーカイブ\ビデオ素材\20260911")
New-Item -ItemType Directory -Force $Dest | Out-Null
$sh = New-Object -ComObject Shell.Application
$pc = $sh.NameSpace(17)
$phone = $pc.Items() | Where-Object { $_.Name -eq 'REDMAGIC 10 Pro' }
if (-not $phone) { "PHONE_NOT_FOUND"; exit 1 }
$storage = $phone.GetFolder.Items() | Select-Object -First 1
$vid = $storage.GetFolder.ParseName('DCIM').GetFolder.ParseName('Camera').GetFolder.ParseName('ビデオ').GetFolder
$destNs = $sh.NameSpace($Dest)
$items = @($vid.Items() | Where-Object { $_.Name -like "VID_${DateTag}_*.mp4" } | Sort-Object Name)
"TOTAL " + $items.Count
$ok = 0
foreach ($it in $items) {
    $name = $it.Name
    $target = Join-Path $Dest $name
    $srcSize = [int64]$it.ExtendedProperty("System.Size")
    if ((Test-Path $target) -and ((Get-Item $target).Length -eq $srcSize)) { "SKIP(done) $name $srcSize"; $ok++; continue }
    $t0 = Get-Date
    $destNs.CopyHere($it, 16 + 4 + 1024)
    # wait until file exists and size stable == srcSize
    $tries = 0
    while ($true) {
        Start-Sleep -Seconds 5
        if (Test-Path $target) {
            $len = (Get-Item $target).Length
            if ($len -eq $srcSize) { break }
        }
        $tries++
        if ($tries -gt 720) { break }
    }
    $len = if (Test-Path $target) { (Get-Item $target).Length } else { -1 }
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    if ($len -eq $srcSize) { $ok++; "OK $name $srcSize ${sec}s" } else { "SIZE_MISMATCH $name src=$srcSize dst=$len" }
}
"DONE ok=$ok total=" + $items.Count
"DONE ok=$ok total=" + $items.Count | Out-File (Join-Path $Dest "_transfer_result.txt") -Encoding utf8
