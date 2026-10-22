# 回収パイプラインの無人ランチャ（2026-10-22 新規・既存スクリプトは一切変更しない＝呼ぶだけ）。
# 役割＝転送 → 改名 → 監督 → 仕上げ（数値集計）→ 受信箱へ報告書 を1本のプロセスで通す。
# 設計の根拠＝無人ジョブの5原則（ダイアログレス・セッションから分離・停滞の自己検知・結果はファイルへ・完了時だけ戻す）。
# 起動はタスクスケジューラ（kv_kaishu_unattended）から。60分ルール（Claude のツールから起動したプロセスは約60分で消える）を避けるため。
# BOM 付き UTF-8 で保存すること（10/6 の教訓＝BOM 無しだと日本語コメントで構文エラー＝タスク exit 1）。
#
# 使い方:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<this>"                 # 全工程（自動でセッション判定）
#   ... -File "<this>" -DryRun                                                                           # ドライラン（実データを動かさない）
#   ... -File "<this>" -Session 20261028 -From rename                                                     # 途中から
#
# ダイアログは一切出さない（Read-Host・Out-GridView・-Confirm を使わない）。
# 本人のウィンドウを持つプロセス（ブラウザ・ゲーム・エミュレータ）は起動も終了もしない。
# 外向きの送信（push・メール・公開）は行わない。

param(
    [string]$Session = "",
    [ValidateSet("detect", "prepare", "dam", "transfer", "rename", "supervise", "finish")]
    [string]$From = "detect",
    [switch]$DryRun,
    [double]$TransferHours = 3.0,
    [double]$PipelineHours = 6.0,
    [switch]$NoReport
)

$ErrorActionPreference = 'Continue'
$ConfirmPreference = 'None'
$ProgressPreference = 'SilentlyContinue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

# ---------- 定数 ----------
$PY       = "C:\Users\conan\AppData\Local\Programs\Python\Python312\python.exe"
$ROOT     = "G:\研究アーカイブ\成果物\動画解析"
$SC       = Join-Path $ROOT "scripts"
$DS       = Join-Path $ROOT "dataset"
$LOGDIR   = Join-Path $ROOT "logs"
$DAMDIR   = "G:\研究アーカイブ\成果物\dam_api"
$VIDROOT  = "G:\研究アーカイブ\ビデオ素材"
$WD       = "C:\Users\conan\hisho\ops\tools\job_watchdog.py"
$INBOX    = "C:\Users\conan\hisho\routine_inbox"
$PHONE    = "REDMAGIC 10 Pro"
$DISTRO   = "Ubuntu-26.04"
$WSLUSER  = "toshiaki"
$STAGES   = @("detect", "prepare", "dam", "transfer", "rename", "supervise", "finish")
# セッション別に作られる8本（prepare がこの一式を揃える）
$GEN_FILES = @("kv_config_{0}.py", "mtp_transfer_{0}.ps1", "rename_videos_{0}.py", "pipeline_{0}.py",
               "run_pipeline_hidden_{0}.ps1", "verify_{0}.py", "wao_probe_v2_{0}.py", "launch_probe_{0}.sh")

New-Item -ItemType Directory -Force $LOGDIR | Out-Null
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Log = Join-Path $LOGDIR ("kaishu_unattended_" + (Get-Date -Format "yyyy-MM-dd") + "_" + $RunStamp + ".log")

$Facts  = New-Object System.Collections.Generic.List[string]   # 報告書の②数値
$Trouble = New-Object System.Collections.Generic.List[string]  # 報告書の③失敗と対処
$Judge  = New-Object System.Collections.Generic.List[string]   # 報告書の④判断待ち
$Script:Stage = "detect"
$Script:Next  = ""
$T0 = Get-Date

function W([string]$m) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Script:Stage] $m"
    $line | Out-File $Log -Append -Encoding utf8
}
function Fact([string]$m) { $Facts.Add($m); W "FACT $m" }
function Want([string]$s) { return ($STAGES.IndexOf($s) -ge $STAGES.IndexOf($From)) }
function Ps1([string]$file, [string[]]$extra) {
    # 既存 .ps1 を子プロセスで実行（このプロセス自体がタスク配下なので60分ルールの対象外）
    $pargs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", $file) + $extra
    return Start-Process -FilePath "powershell.exe" -ArgumentList $pargs -WindowStyle Hidden -PassThru
}
function PyRun([string]$script, [string[]]$extra, [string]$outFile) {
    $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
    $pargs = @("`"$script`"") + $extra
    $err = $outFile + ".err.txt"
    $p = Start-Process -FilePath $PY -ArgumentList $pargs -NoNewWindow -Wait -PassThru `
         -RedirectStandardOutput $outFile -RedirectStandardError $err
    $txt = ""
    if (Test-Path $outFile) { $txt = (Get-Content $outFile -Raw -Encoding UTF8) }
    return [pscustomobject]@{ Code = $p.ExitCode; Out = $txt; ErrFile = $err }
}
function Wsl([string]$cmd) {
    # 注意＝wsl -lc に渡す文字列内のドル変数はこの環境で消える。変数を使う処理はスクリプトファイル側に置く。
    $r = & wsl.exe -d $DISTRO -u $WSLUSER -- bash -lc $cmd
    return ($r -join "`n")
}
function Report([string]$status) {
    if ($NoReport) { W "報告書は -NoReport のため書かない"; return }
    New-Item -ItemType Directory -Force $INBOX | Out-Null
    $f = Join-Path $INBOX ((Get-Date -Format "yyyy-MM-dd_HHmm") + "_kaishu_unattended_" + $Session + ".md")
    $min = [int]((Get-Date) - $T0).TotalMinutes
    $head = if ($DryRun) { "ドライラン" } else { "実行" }
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add("<!-- unread -->")
    $L.Add("# 報告書：回収パイプライン無人実行（$head ・session=$Session ・machine=wao）")
    $L.Add("")
    $L.Add("作成＝$(Get-Date -Format 'yyyy-MM-dd HH:mm')／担当＝無人ランチャ kaishu_unattended.ps1／経過＝$min 分")
    $L.Add("")
    $L.Add("## 1. 結果")
    $L.Add("")
    $L.Add("$status（工程＝$Script:Stage で終了）")
    $L.Add("")
    $L.Add("## 2. 数値")
    $L.Add("")
    if ($Facts.Count -eq 0) { $L.Add("（なし）") } else { foreach ($x in $Facts) { $L.Add("- $x") } }
    $L.Add("")
    $L.Add("## 3. 失敗と対処")
    $L.Add("")
    if ($Trouble.Count -eq 0) { $L.Add("失敗はありません。") } else { foreach ($x in $Trouble) { $L.Add("- $x") } }
    $L.Add("")
    $L.Add("## 4. 判断待ち")
    $L.Add("")
    if ($Judge.Count -eq 0) { $L.Add("ありません。") } else { foreach ($x in $Judge) { $L.Add("- $x") } }
    $L.Add("")
    $L.Add("## 5. 次")
    $L.Add("")
    if ($Script:Next -eq "") { $L.Add("（なし）") } else { $L.Add($Script:Next) }
    $L.Add("")
    $L.Add("ログ＝$Log")
    $L -join "`n" | Out-File $f -Encoding utf8
    W "報告書＝$f"
    return $f
}
function Guard([bool]$bad, [string]$reason, [string]$next) {
    # 前提が無い＝本番は停止、ドライランは「そこで止まる」と記録して先の工程の組み立てを続ける
    if (-not $bad) { return $false }
    if ($DryRun) { W "DRYRUN NG: $reason"; $Trouble.Add("ドライラン：$reason"); return $true }
    Die $reason $next
}
function Die([string]$reason, [string]$next) {
    W "STOP $reason"
    $Trouble.Add($reason)
    $Script:Next = $next
    Report "停止しました。"
    exit 2
}

W "START dryrun=$DryRun from=$From pid=$PID host=$env:COMPUTERNAME"

# ============================================================ 1. detect
$Script:Stage = "detect"
$phoneFound = $false
$phoneCount = 0
$phoneBytes = [int64]0
if (Want "detect") {
    try {
        $sh = New-Object -ComObject Shell.Application
        $dev = $sh.NameSpace(17).Items() | Where-Object { $_.Name -eq $PHONE }
        if ($dev) {
            $phoneFound = $true
            $store = $dev.GetFolder.Items() | Select-Object -First 1
            $vidf = $store.GetFolder.ParseName('DCIM').GetFolder.ParseName('Camera').GetFolder.ParseName('ビデオ').GetFolder
            $tags = @{}
            foreach ($it in $vidf.Items()) {
                if ($it.Name -match '^VID_(\d{8})_\d+\.mp4$') {
                    $t = $Matches[1]
                    if (-not $tags.ContainsKey($t)) { $tags[$t] = @{ n = 0; b = [int64]0 } }
                    $tags[$t].n++
                    $tags[$t].b += [int64]$it.ExtendedProperty("System.Size")
                }
            }
            if ($tags.Count -gt 0) {
                $newest = ($tags.Keys | Sort-Object -Descending | Select-Object -First 1)
                if ($Session -eq "") { $Session = $newest }
                if ($tags.ContainsKey($Session)) { $phoneCount = $tags[$Session].n; $phoneBytes = $tags[$Session].b }
                Fact ("スマホ検知＝$PHONE ・日付タグ " + (($tags.Keys | Sort-Object -Descending) -join "／") + "・採用 $Session（$phoneCount 本・$([math]::Round($phoneBytes/1GB,2)) GB）")
            } else { Fact "スマホ検知＝$PHONE （VID_ 動画は0本）" }
        }
    } catch { W "スマホ検知で例外: $($_.Exception.Message)" }
    if (-not $phoneFound) {
        W "スマホ未接続（MTP に $PHONE が見えない）"
        $how = "引数 -Session で指定"
        if ($Session -eq "") {
            $how = "ビデオ素材の最新フォルダから推定"
            $d = Get-ChildItem $VIDROOT -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^\d{8}$' } | Sort-Object Name -Descending | Select-Object -First 1
            if ($d) { $Session = $d.Name }
        }
        Fact "スマホ未接続＝転送は実行できない（セッション $Session ＝$how）"
    }
}
if ($Session -eq "") { Die "セッション（日付タグ）が決まらない。スマホが未接続で、ビデオ素材にも日付フォルダが無い。" "USB（MTP・ファイル転送）で接続してから引数なしで再実行、または -Session YYYYMMDD を指定。" }
$env:KV_SESSION = $Session
$VID = Join-Path $VIDROOT $Session
$PIPE = Join-Path $DS ("pipeline_" + $Session)
$PLOG = Join-Path $PIPE "pipeline.log"
$TLOG = Join-Path $VID "_transfer.log"
$TRES = Join-Path $VID "_transfer_result.txt"
W "session=$Session vid=$VID pipe=$PIPE"

# ============================================================ 2. prepare（セッション別スクリプト8本を揃える）
$Script:Stage = "prepare"
if (Want "prepare") {
    $missing = @()
    foreach ($f in $GEN_FILES) { $n = ($f -f $Session); if (-not (Test-Path (Join-Path $SC $n))) { $missing += $n } }
    if ($missing.Count -eq 0) {
        Fact "セッション別スクリプト＝8本すべて既存（生成不要）"
    } else {
        # 直近の完全な世代をひな型にする
        $gens = @()
        foreach ($c in (Get-ChildItem $SC -Filter "kv_config_*.py" -ErrorAction SilentlyContinue)) {
            if ($c.Name -match '^kv_config_(\d{8})\.py$') {
                $g = $Matches[1]
                if ($g -eq $Session) { continue }
                $full = $true
                foreach ($f in $GEN_FILES) { if (-not (Test-Path (Join-Path $SC ($f -f $g)))) { $full = $false } }
                if ($full) { $gens += $g }
            }
        }
        if ($gens.Count -eq 0) { Die "ひな型になる世代が無い（kv_config_YYYYMMDD.py の一式が揃っていない）" "scripts フォルダを確認する。" }
        $G = ($gens | Sort-Object -Descending | Select-Object -First 1)
        Fact "スクリプト生成＝ひな型 $G → $Session（不足 $($missing.Count) 本：$($missing -join '／')）"
        # 当日の採点行を含む DAM CSV を先に決める（kv_config の HEARTS_CSV／AI_CSV は日付書式が違い機械置換では直らない）
        $d = [datetime]::ParseExact($Session, "yyyyMMdd", $null).ToString("yyyy-MM-dd")
        function NewestDam([string]$pat) {
            $cands = Get-ChildItem $DAMDIR -Filter $pat -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
            foreach ($c in $cands) {
                $n = @(Select-String -LiteralPath $c.FullName -Pattern $d -AllMatches -Encoding UTF8 -ErrorAction SilentlyContinue).Count
                if ($n -gt 0) { return [pscustomobject]@{ Name = $c.Name; Rows = $n } }
            }
            return $null
        }
        $hearts = NewestDam "dam_ai-hearts_full_*.csv"
        $aics   = NewestDam "dam_ai_full_*.csv"
        if (-not $hearts) {
            if ($DryRun) { W "DRYRUN: 当日 $d の行を含む Hearts CSV が無い（本番では dam 工程で取得してから prepare をやり直す）" }
            else { Die "当日 $d の採点行を含む Hearts CSV が dam_api に無い。改名も突合もできない。" ("CSV を取得してから再実行＝powershell -ExecutionPolicy Bypass -File " + $PSCommandPath + " -Session " + $Session + " -From dam") }
        }
        if ($hearts) { Fact "Hearts CSV＝$($hearts.Name)（当日 $d の行 $($hearts.Rows)）" }
        if ($aics) { Fact "Ai CSV＝$($aics.Name)（当日 $d の行 $($aics.Rows)）" } else { W "Ai CSV に当日行なし（WAO 単独セッションでは正常）" }
        if ($DryRun) {
            W "DRYRUN: 生成はしない。置換＝'$G'→'$Session'・HEARTS_CSV／AI_CSV の実名差し替え・PROBE_PGREP の角括弧（wao_probe_v2_$($Session.Substring(0,7))[$($Session.Substring(7,1))].py）"
        } else {
            $enc = New-Object System.Text.UTF8Encoding($true)  # BOM 付き（.ps1 は BOM が無いと日本語コメントで構文エラー）
            foreach ($f in $GEN_FILES) {
                $src = Join-Path $SC ($f -f $G); $dst = Join-Path $SC ($f -f $Session)
                if (Test-Path $dst) { continue }
                $txt = [System.IO.File]::ReadAllText($src)
                $txt = $txt.Replace($G, $Session)
                if ($f -like "kv_config_*") {
                    if ($hearts) { $txt = [regex]::Replace($txt, 'dam_ai-hearts_full_[^"]*\.csv', $hearts.Name) }
                    $aiName = if ($aics) { $aics.Name } else { "" }
                    if ($aiName -ne "") { $txt = [regex]::Replace($txt, '(?<!-hearts)dam_ai_full_[^"]*\.csv', $aiName) }
                    # pgrep の自己一致回避のための角括弧（置換から漏れる＝監督がプローブを二重起動する）
                    $txt = [regex]::Replace($txt, 'wao_probe_v2_\d{7}\[\d\]\.py', ("wao_probe_v2_" + $Session.Substring(0, 7) + "[" + $Session.Substring(7, 1) + "].py"))
                }
                if ($f -like "*.ps1") { [System.IO.File]::WriteAllText($dst, $txt, $enc) }
                else { [System.IO.File]::WriteAllText($dst, $txt, (New-Object System.Text.UTF8Encoding($false))) }
                W "生成 $dst"
            }
            $still = @()
            foreach ($f in $GEN_FILES) { $n = ($f -f $Session); if (-not (Test-Path (Join-Path $SC $n))) { $still += $n } }
            if ($still.Count -gt 0) { Die "スクリプト生成に失敗（$($still -join '／') が無い）" "scripts フォルダを確認する。" }
            Fact "スクリプト生成＝8本そろった"
            $Judge.Add("生成したセッション別スクリプト8本は機械置換です。HEARTS_CSV・PROBE_PGREP 以外に日付書式の違う参照が増えていないか、次回の整備で目を通してください。")
        }
    }
}

# ============================================================ 3. dam（DAM★とも CSV 取得）
$Script:Stage = "dam"
if (Want "dam") {
    $d = [datetime]::ParseExact($Session, "yyyyMMdd", $null).ToString("yyyy-MM-dd")
    $has = $false
    foreach ($c in (Get-ChildItem $DAMDIR -Filter "dam_ai-hearts_full_*.csv" -ErrorAction SilentlyContinue)) {
        if (@(Select-String -LiteralPath $c.FullName -Pattern $d -AllMatches -Encoding UTF8 -ErrorAction SilentlyContinue).Count -gt 0) { $has = $true }
    }
    if ($has) { Fact "DAM CSV＝当日 $d の行を含む Hearts CSV が既にある（取得を飛ばす）" }
    elseif ($DryRun) { W "DRYRUN: WSL で npm run -s export:csv:hearts:full と export:csv:ai:full を実行し、新しい CSV を dam_api へコピーする工程" }
    else {
        $o = Wsl "cd ~/karaoke-api && npm run -s export:csv:hearts:full && npm run -s export:csv:ai:full && ls -t output | head -3"
        W "WSL export: $($o -replace "`n", ' | ')"
        $o2 = Wsl "cp -n ~/karaoke-api/output/dam_ai-hearts_full_*.csv ~/karaoke-api/output/dam_ai_full_*.csv '/mnt/g/研究アーカイブ/成果物/dam_api/' && ls -t '/mnt/g/研究アーカイブ/成果物/dam_api' | head -3"
        W "WSL copy: $($o2 -replace "`n", ' | ')"
        $has2 = $false
        foreach ($c in (Get-ChildItem $DAMDIR -Filter "dam_ai-hearts_full_*.csv" -ErrorAction SilentlyContinue)) {
            if (@(Select-String -LiteralPath $c.FullName -Pattern $d -AllMatches -Encoding UTF8 -ErrorAction SilentlyContinue).Count -gt 0) { $has2 = $true }
        }
        if (-not $has2) { Die "DAM CSV を取得したが当日 $d の採点行が無い（退室前ログアウト忘れ・取得失敗の疑い）" "RUNBOOK §1 の代替（node ts-node で exportCsv）を手で試す。" }
        Fact "DAM CSV＝取得して dam_api へコピーした（当日 $d の行あり）"
        if (-not (Want "prepare")) { $Judge.Add("CSV を新しく取得したので kv_config_$Session.py の HEARTS_CSV が古い実名を指していないか確認が要ります。") }
    }
}

# ============================================================ 4. transfer（MTP 転送）
$Script:Stage = "transfer"
if (Want "transfer") {
    $done = $false
    if (Test-Path $TRES) {
        $r = (Get-Content $TRES -Raw -Encoding UTF8).Trim().TrimStart([char]0xFEFF)
        if ($r -match 'ok=(\d+) total=(\d+)' -and $Matches[1] -eq $Matches[2] -and [int]$Matches[1] -gt 0) { $done = $true; Fact "転送＝既に完了（$r）＝飛ばす" }
    }
    if (-not $done) {
        $ps = Join-Path $SC ("mtp_transfer_" + $Session + ".ps1")
        $ng = Guard (-not (Test-Path $ps)) "転送スクリプトが無い（$ps）" "prepare 工程から実行＝ -From prepare"
        $ng = (Guard (-not $phoneFound) "スマホ（$PHONE）が MTP に見えないので転送できない" "USB をファイル転送（MTP）で接続し直してから再実行。") -or $ng
        $wdArgs = @("`"$WD`"", "--register", ("回収転送" + $Session), "--pattern", "`"mtp_transfer_$Session`"",
                    "--log", "`"$TLOG`"", "--extra-files", "`"$VID`"", "--idle", "15", "--max-hours", "4")
        if ($DryRun) {
            W "DRYRUN: 番犬登録＝$PY $($wdArgs -join ' ')"
            W "DRYRUN: 転送＝powershell -File `"$ps`" -DateTag $Session -Dest `"$VID`"（スマホ側 $phoneCount 本・$([math]::Round($phoneBytes/1GB,2)) GB・実績 1本90〜175秒）"
        } else {
            New-Item -ItemType Directory -Force $VID | Out-Null
            Start-Process -FilePath $PY -ArgumentList $wdArgs -WindowStyle Hidden -Wait
            Fact "番犬登録＝回収転送$Session（idle 15分・上限4h）"
            $p = Ps1 $ps @("-DateTag", $Session, "-Dest", $VID)
            W "転送起動 pid=$($p.Id)"
            $dead = (Get-Date).AddHours($TransferHours)
            while (-not $p.HasExited) {
                Start-Sleep -Seconds 60
                $n = 0
                if (Test-Path $TLOG) { $n = @(Select-String -LiteralPath $TLOG -Pattern ' OK VID_| SKIP' -AllMatches -ErrorAction SilentlyContinue).Count }
                W "転送中 copied=$n"
                if ((Get-Date) -gt $dead) {
                    # 止めるのは自分が起動した転送プロセスだけ（本人のウィンドウを持つプロセスには触れない）
                    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
                    Die "転送が $TransferHours 時間で終わらないので打ち切った（copied=$n）" "`_transfer.log` の SIZE_MISMATCH を直して `-From transfer` で再実行（済みは SKIP）。"
                }
            }
            if (-not (Test-Path $TRES)) { Die "転送プロセスが終わったが _transfer_result.txt が無い" "`_transfer.log` の末尾を見てから `-From transfer` で再実行。" }
            $r = (Get-Content $TRES -Raw -Encoding UTF8).Trim().TrimStart([char]0xFEFF)
            if ($r -eq "PHONE_NOT_FOUND") { Die "転送＝PHONE_NOT_FOUND（転送中にスマホが外れたか MTP が落ちた）" "接続し直して `-From transfer` で再実行。" }
            if ($r -notmatch 'ok=(\d+) total=(\d+)') { Die "転送結果が読めない（$r）" "`_transfer.log` を見る。" }
            if ($Matches[1] -ne $Matches[2]) { Die "転送が不完全（$r）" "`_transfer.log` の SIZE_MISMATCH の動画を消して `-From transfer` で再実行。" }
            $bytes = (Get-ChildItem $VID -Filter *.mp4 | Measure-Object Length -Sum).Sum
            Fact "転送＝$r ・$([math]::Round($bytes/1GB,2)) GB ・所要 $([int]((Get-Date)-$T0).TotalMinutes) 分（起動からの通算）"
        }
    }
}

# ============================================================ 5. rename（改名＝dry-run → ゲート → apply）
$Script:Stage = "rename"
if (Want "rename") {
    $rn = Join-Path $SC ("rename_videos_" + $Session + ".py")
    $map = Join-Path $VID "rename_mapping.csv"
    if (-not $DryRun) { New-Item -ItemType Directory -Force $PIPE | Out-Null }
    $ng = Guard (-not (Test-Path $rn)) "改名スクリプトが無い（$rn）" "-From prepare で生成する。"
    if (Test-Path $map) {
        $n = @(Get-ChildItem $VID -Filter *.mp4 | Where-Object { $_.Name -notlike 'VID_*' }).Count
        Fact "改名＝既に済み（rename_mapping.csv あり・曲名付き mp4 $n 本）＝飛ばす"
    } elseif ($DryRun) {
        $nvid = @(Get-ChildItem $VID -Filter "VID_${Session}_*.mp4" -ErrorAction SilentlyContinue).Count
        W "DRYRUN: 改名＝$PY `"$rn`" $Session → ゲート（STOP／TIME_WINDOW=0／AMBIGUOUS）→ $PY `"$rn`" $Session --apply（対象 VID_ $nvid 本）"
    } else {
        $dry = PyRun $rn @($Session) (Join-Path $PIPE "rename_dryrun.txt")
        W "改名 dry-run rc=$($dry.Code)"
        if ($dry.Out -match 'STOP:') { Die "改名 dry-run が STOP：$($Matches[0])" "rename_dryrun.txt を見て採点行と動画の対応を直す。" }
        if ($dry.Code -ne 0) { Die "改名 dry-run が異常終了（rc=$($dry.Code)）" "rename_dryrun.txt と .err.txt を見る。" }
        $tw = 0; $nm = 0; $amb = 0; $un = 0
        if ($dry.Out -match 'TIME_WINDOW=(\d+)') { $tw = [int]$Matches[1] }
        if ($dry.Out -match 'NO_MATCH=(\d+)') { $nm = [int]$Matches[1] }
        if ($dry.Out -match 'AMBIGUOUS=(\d+)') { $amb = [int]$Matches[1] }
        if ($dry.Out -match 'UNUSED_ROW=(\d+)') { $un = [int]$Matches[1] }
        $nvid = @(Get-ChildItem $VID -Filter "VID_${Session}_*.mp4" -ErrorAction SilentlyContinue).Count
        Fact "改名 dry-run＝TIME_WINDOW $tw ／NO_MATCH $nm ／AMBIGUOUS $amb ／UNUSED_ROW $un（VID_ 動画 $nvid 本）"
        if ($tw -eq 0) { Die "改名できる動画が0本（採点行と動画の時刻窓が合っていない）" "rename_dryrun.txt と Hearts CSV の当日行を突き合わせる。" }
        if ($nvid -gt 0 -and $tw -lt [int]($nvid * 0.8)) { Die "改名できる動画が $tw/$nvid 本（8割未満）＝対応が崩れている疑い" "rename_dryrun.txt を見てから手で裁定する。" }
        if ($amb -gt 0) { $Judge.Add("AMBIGUOUS $amb 本＝1動画に採点行が2つ。自動では改名せず解析対象外にしました。手で裁定して改名し rename_mapping.csv に行を足すかはご判断ください（RUNBOOK §2-3）。") }
        if ($un -gt 0) { $Judge.Add("UNUSED_ROW $un 件＝動画が対応しない採点行（録画開始前の曲の可能性・RUNBOOK D-3）。") }
        $ap = PyRun $rn @($Session, "--apply") (Join-Path $PIPE "rename_apply.txt")
        if ($ap.Code -ne 0 -or $ap.Out -notmatch 'RENAMED') { Die "改名 --apply 失敗（rc=$($ap.Code)）" "rename_apply.txt を見る。" }
        if (-not (Test-Path $map)) { Die "rename_mapping.csv ができていない" "rename_apply.txt を見る。" }
        $n = @(Get-ChildItem $VID -Filter *.mp4 | Where-Object { $_.Name -notlike 'VID_*' }).Count
        Fact "改名＝曲名付き mp4 $n 本・rename_mapping.csv 作成"
    }
}

# ============================================================ 6. supervise（監督＝probe〜g1・約2時間）
$Script:Stage = "supervise"
if (Want "supervise") {
    $runner = Join-Path $SC ("run_pipeline_hidden_" + $Session + ".ps1")
    $ng = Guard (-not (Test-Path $runner)) "監督ランチャが無い（$runner）" "-From prepare で生成する。"
    $wdArgs = @("`"$WD`"", "--register", ("回収監督" + $Session), "--pattern", "`"pipeline_$Session|wincrop|calib_roi|wao_probe`"",
                "--log", "`"$PLOG`"", "--idle", "25", "--max-hours", "8")
    if ($DryRun) {
        W "DRYRUN: 番犬登録＝$PY $($wdArgs -join ' ')"
        W "DRYRUN: 監督＝powershell -File `"$runner`" -From probe -Session $Session（probe→crop→calib→read→verify→sync→g1・実績 約2時間）"
        W "DRYRUN: 完了＝$PIPE\PIPELINE_DONE.txt ／停止＝PIPELINE_STOPPED.txt"
    } else {
        New-Item -ItemType Directory -Force $PIPE | Out-Null
        foreach ($f in @("PIPELINE_DONE.txt")) { $p0 = Join-Path $PIPE $f; if (Test-Path $p0) { Move-Item $p0 ($p0 + "." + $RunStamp + ".old") -Force } }
        Start-Process -FilePath $PY -ArgumentList $wdArgs -WindowStyle Hidden -Wait
        Fact "番犬登録＝回収監督$Session（idle 25分・上限8h）"
        $p = Ps1 $runner @("-From", "probe", "-Session", $Session)
        W "監督起動 pid=$($p.Id)"
        $dead = (Get-Date).AddHours($PipelineHours)
        $lastTail = ""
        while (-not $p.HasExited) {
            Start-Sleep -Seconds 120
            if (Test-Path $PLOG) {
                $ls = Get-Content $PLOG -Encoding UTF8 -ErrorAction SilentlyContinue
                $t = ""
                if ($ls) { $t = ($ls | Select-Object -Last 1) }
                if ($t -ne $lastTail) { W "監督: $t"; $lastTail = $t }
            }
            if (Test-Path (Join-Path $PIPE "PIPELINE_STOPPED.txt")) { W "PIPELINE_STOPPED.txt を検知" }
            if ((Get-Date) -gt $dead) {
                W "監督が $PipelineHours 時間を超えた（打ち切らず報告だけ＝番犬が停滞を見ている）"
                break
            }
        }
        $stopFile = Join-Path $PIPE "PIPELINE_STOPPED.txt"
        $doneFile = Join-Path $PIPE "PIPELINE_DONE.txt"
        if (Test-Path $stopFile) {
            $why = (Get-Content $stopFile -Raw -Encoding UTF8).Trim()
            $Trouble.Add("監督が停止＝" + ($why -replace "`n", " / "))
            $Script:Next = "原因を直してから再開＝``powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Session $Session -From supervise``（工程は済みをスキップ）"
            Fact "監督＝PIPELINE_STOPPED"
        } elseif (Test-Path $doneFile) {
            Fact ("監督＝PIPELINE_DONE（" + ((Get-Content $doneFile -Encoding UTF8 | Select-Object -First 1)) + "）")
        } else {
            $Trouble.Add("監督が DONE も STOPPED も残さずに終わった（タスクの実行時間上限・再起動・番犬の停止を疑う）")
            Fact "監督＝不明（DONE/STOPPED なし）"
        }
    }
}

# ============================================================ 7. finish（仕上げ＝数値の集計・台帳の結合はしない）
$Script:Stage = "finish"
if (Want "finish") {
    $verify = Join-Path $DS ("verify_" + $Session + "_wao.csv")
    $g1 = Join-Path $DS ("G1_events_" + $Session + "_wao.csv")
    $calib = Join-Path $DS ("calib_" + $Session + "\calib_summary.csv")
    $results = Join-Path $ROOT ("results_wao_" + $Session)
    if ($DryRun) {
        W "DRYRUN: 仕上げ＝verify($verify)・G1($g1)・calib($calib)・results($results) を読んで数値を集計し報告書にする"
        W "DRYRUN: 統合台帳 G1_events_all.csv への結合と目視追記は行わない（裁定が要るため）"
    } else {
        if (Test-Path $verify) {
            $v = Import-Csv $verify -Encoding UTF8
            $perf = @($v | Where-Object { $_.status -eq "PERFECT" }).Count
            Fact "verify＝PERFECT $perf/$($v.Count)"
        } else { $Trouble.Add("verify CSV が無い（$verify）→ ``--from verify --until verify`` で作れる") }
        if (Test-Path $g1) { Fact "G1＝$((@(Get-Content $g1 -Encoding UTF8).Count - 1)) 行（$g1）" }
        else { $Trouble.Add("G1 CSV が無い（$g1）") }
        if (Test-Path $results) { Fact "読取結果＝$(@(Get-ChildItem $results -Directory -ErrorAction SilentlyContinue).Count) 曲ぶんのフォルダ" }
        if (Test-Path $calib) {
            $c = Import-Csv $calib -Encoding UTF8
            $low = @($c | Where-Object { [double]$_.score_med -lt 0.25 }).Count
            Fact "calib＝$($c.Count) 行・score_med 0.25 未満 $low 件（多いときは帯の位置ずれを疑う）"
            if ($low -gt ([int]($c.Count * 0.2))) { $Judge.Add("calib の score_med が 0.25 未満の割合が高いです（$low/$($c.Count)）。帯プローブの座標を見直すかの裁定をお願いします（10/21 の全滅2本と同じ兆候）。") }
        }
        $Judge.Add("統合台帳 G1_events_all.csv への結合と目視追記は行っていません（裁定が要る工程・RUNBOOK §6）。")
        if ($Script:Next -eq "") {
            $Script:Next = "検収と結合に進む＝RUNBOOK §6（``G1_events_$Session`_wao.csv`` の pts_source を確認してから結合）"
        }
    }
}

$Script:Stage = "finish"
$st = if ($DryRun) { "ドライランを通しました（実データは動かしていません）。" } else { "指定の工程を通しました。" }
if ($Script:Next -eq "") {
    $Script:Next = "有効化＝``schtasks /change /tn kv_kaishu_unattended /enable``（そのあとデスクトップの「回収開始」を1回押す）"
}
$f = Report $st
W "END $([int]((Get-Date)-$T0).TotalMinutes) 分"
exit 0
