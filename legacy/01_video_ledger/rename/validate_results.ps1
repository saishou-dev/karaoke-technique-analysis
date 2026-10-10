# 読取結果 vs DAM採点CSV(正解値) の自動突合 v1
param(
    [string]$WorkDir = "C:\大学\授業\041 4年\043 卒研\卒業研究_カラオケ採点\動画解析",
    [string]$CsvPath = "\\wsl.localhost\Ubuntu-26.04\home\toshiaki\karaoke-api\output\dam_ai_full_2026-08-07T04-15-55.csv"
)
$csv = Import-Csv $CsvPath
function ParseLog($path) {
    $r = @{}
    Get-Content $path -Encoding utf8 | ForEach-Object {
        if ($_ -match "^(shakuri|kobushi|fall|vibrato): initial=(\S+) final=(\S+) increments=(\d+)") {
            $r[$Matches[1]] = @{ initial = $Matches[2]; final = $Matches[3]; inc = $Matches[4] }
        }
    }
    return $r
}
$rows = @()
Get-ChildItem (Join-Path $WorkDir "results") -Directory | ForEach-Object {
    $name = $_.Name
    if ($name -notmatch "_(\d{8})_(\d{6})$") { Write-Output "SKIP(名前不一致): $name"; return }
    $vidTime = [datetime]::ParseExact($Matches[1] + $Matches[2], "yyyyMMddHHmmss", $null)
    $best = $null; $bestDiff = [double]::MaxValue
    foreach ($row in $csv) {
        $sdt = $row.scoringDateTime
        if ($sdt.Length -ne 14) { continue }
        $t = [datetime]::ParseExact($sdt, "yyyyMMddHHmmss", $null)
        $diff = [math]::Abs(($t - $vidTime).TotalSeconds)
        if ($diff -lt $bestDiff) { $bestDiff = $diff; $best = $row }
    }
    if ($bestDiff -gt 300) { Write-Output "NO_CSV_MATCH: $name"; return }
    $log = ParseLog (Join-Path $_.FullName "read_log.txt")
    $ok = 0; $ng = @()
    $pairs = @(@("shakuri", "shakuriCount"), @("kobushi", "kobushiCount"), @("fall", "fallCount"), @("vibrato", "vibratoCount"))
    foreach ($p in $pairs) {
        $mine = $log[$p[0]].final
        $truth = $best.($p[1])
        if ("$mine" -eq "$truth") { $ok++ } else { $ng += "$($p[0])=$mine(正$truth)" }
    }
    $rows += [pscustomobject]@{
        video = $name; csv曲 = $best.contentsName; 一致 = "$ok/4"
        乖離 = ($ng -join " "); 時刻差s = [int]$bestDiff
    }
}
$rows | Sort-Object 一致 | Format-Table -AutoSize -Wrap
$perfect = ($rows | Where-Object { $_.一致 -eq "4/4" }).Count
Write-Output "PERFECT: $perfect / $($rows.Count)"
