#!/usr/bin/env python3
"""9/2欠落増分: rescan_tail(機械)+帯ROIタイルモンタージュ(目視)で実フレーム根拠を確定した取りこぼしをcounter_eventsへ追記。
D22原則: 補間なし・全件が実読取フレームまたは目視確定フレーム(モンタージュ証拠保存済み evidence_20260902/m_*.png)。
規約はinject_20260818.pyと同一: event_frame_est = 旧値最終フレーム+1。既存行はSKIP(冪等)。
usage: inject_20260903.py [jobset]  (jobset: A=CHEERS/Soranji/心/輪廻, B=lulu.)"""
import csv, json, os, cv2, sys
JOBS_A = [
    ("CHEERS_20260902_131319", "shakuri", 21, 1, 11232, 11262, "visual m_cheers_sh_186_194: 20 last f11232(t188.0), 21 from f11262(t188.5; t188.25 transitional); machine 15 hits t188.9-192.4 nccmax0.85"),
    ("Soranji_20260902_134521", "shakuri", 35, 1, 18993, 19007, "visual m_soranji_sh_320_330: 34 last f18993(t323.0), 35 from f19007(t323.25) to t328.5; machine 13 hits t323.1-328.6 nccmax0.82"),
    ("心という名の不可解_20260902_133141", "kobushi", 11, 1, 14894, 14909, "visual m_kokoro_kob_172_268 + m_kokoro_kob_248_250: 10 last f14894(t249.25), 11 from f14909(t249.5) to t262; machine 16 hits t249.6-253.8 nccmax0.54 (2nd group t206.8-212.9 rejected: visual shows 10 there)"),
    ("輪廻転生_20260902_133635", "accent", 11, 1, 12836, 12842, "visual m_rinne_ac_210_222 + m_rinne_ac_2149_2156: 10 last f12836(t215.45), 11 from f12842(t215.55; f12839 transitional) to t216.5 (band ends t217); machine 1 hit f12898 ncc0.41"),
]
JOBS_B = [
    ("lulu._20260902_111702", "shakuri", 12, 1, 9258, 9273, "visual m_lulu_sh_150_266 + m_lulu_sh_153.5_156.5: 11 last f9258(t155.25), 12 from f9273(t155.5). Reader had locked on misread 17->11 at t254 (off_path 155 runs); all 12..18 clearly displayed in band"),
    ("lulu._20260902_111702", "shakuri", 13, 1, 9556, 9571, "visual m_lulu_sh_160_162.5: 12 last f9556(t160.25), 13 from f9571(t160.5) to t234"),
    ("lulu._20260902_111702", "shakuri", 14, 1, 13969, 13984, "visual m_lulu_sh_234_236.5: 13 last f13969(t234.25), 14 from f13984(t234.5)"),
    ("lulu._20260902_111702", "shakuri", 15, 1, 14282, 14297, "visual m_lulu_sh_238_240.5: 14 last f14282(t239.5), 15 from f14297(t239.75)"),
    ("lulu._20260902_111702", "shakuri", 16, 1, 14908, 14923, "visual m_lulu_sh_250_252.5: 15 last f14908(t250.0), 16 from f14923(t250.25)"),
    ("lulu._20260902_111702", "shakuri", 17, 1, 15027, 15042, "visual m_lulu_sh_250_252.5 + m_lulu_sh_252.5_254.5: 16 last f15027(t252.0), 17 from f15042(t252.25); machine 5 hits t257.4-260.6 ncc<=0.46 (late)"),
    ("lulu._20260902_111702", "shakuri", 18, 1, 15311, 15325, "visual m_lulu_sh_256_258.5: 17 last f15311(t256.75), 18 from f15325(t257.0) to end(t260)"),
]
JOBS_C = [
    ("ケセラセラ_20260902_123805", "kobushi", 7, 1, 16171, 16200, "visual m_kesera_kob_267_283: 6 last f16171(t276.5), 7 from f16200(t277.0) to end(t283, band fading); machine rescan from f15990"),
]
BANDS = "/mnt/d/研究アーカイブ/成果物/動画解析/bands_wao"
OUT = os.path.expanduser("~/karaoke-vision/out_wao")
jobs = {"A": JOBS_A, "B": JOBS_B, "C": JOBS_C}[sys.argv[1] if len(sys.argv) > 1 else "A"]
n = 0
for video, counter, value, delta, last_old, stable, evidence in jobs:
    d = os.path.join(OUT, video)
    cap = cv2.VideoCapture(os.path.join(BANDS, video + ".mp4")); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    ev_fr = last_old + 1
    rows = list(csv.reader(open(os.path.join(d, "counter_events.csv"))))
    if any(r[0] == counter and r[3] == str(value) for r in rows[1:]):
        print(f"SKIP {video} {counter}={value} (already present)"); continue
    rows.append([counter, str(stable), f"{stable/fps:.3f}", str(value), str(delta), str(ev_fr), f"{ev_fr/fps:.3f}"])
    with open(os.path.join(d, "counter_events.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    jj = json.load(open(os.path.join(d, "counter_events.json"))); jj[counter].append([stable, value, delta])
    json.dump(jj, open(os.path.join(d, "counter_events.json"), "w"))
    with open(os.path.join(d, "read_log.txt"), "a") as f:
        f.write(f"\n[manual-rescan 2026-09-03] {counter} +{delta} -> {value} event_frame_est={ev_fr} stable={stable} :: {evidence}\n")
    print(f"INJECTED {video} {counter}={value} ev_fr={ev_fr} t={ev_fr/fps:.3f} fps={fps:.2f}"); n += 1
print(f"done injected={n}")
