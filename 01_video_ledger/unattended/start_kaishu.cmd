@echo off
rem One-press entry for the unattended collection pipeline (2026-10-22).
rem It only kicks the scheduled task "kv_kaishu_unattended" so that the real work
rem runs under Task Scheduler (not under this console) and survives the 60-minute limit.
rem ASCII only on purpose: a .cmd is read with the OEM codepage, so non-ASCII text breaks it.
schtasks /run /tn kv_kaishu_unattended >nul 2>&1
if errorlevel 1 (
  rem The task is registered DISABLED on purpose. Enable it once, after approval:
  rem   schtasks /change /tn kv_kaishu_unattended /enable
  exit /b 1
)
exit /b 0
