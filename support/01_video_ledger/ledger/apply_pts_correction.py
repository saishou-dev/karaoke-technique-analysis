# -*- coding: utf-8 -*-
"""G1の時刻をフレーム番号基準（fidx/CAP_PROP_FPS）から実提示時刻（pts）に補正する（2026-09-09）。
   背景: スマホ録画はフレーム落ちがあり、cv2の平均fpsで割った時刻は曲中で最大±2.6秒ずれる（5,476件中567件が0.5秒超）。
   処理: dataset/pts/<video>.csv を使い video_time_sec・stable_read_time_sec を補正。元の値は *_idx_sec 列に保存。
         G1_events_all.csv / G1_events_synced.csv を更新（バックアップ *_pre_ptsfix_20260909.bak.csv）。
         検収ページ2枚（0807_821・0628_561）の vt を補正し、|shift|>0.1s のクリップを clips3 から削除（再生成対象）。
"""
import csv, os, re, shutil
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; SRC = r"G:\研究アーカイブ\ビデオ素材"; FPS_NOMINAL = 60000 / 1001
pts = {}
for fn in os.listdir(os.path.join(ROOT, "dataset", "pts")):
    if fn.endswith(".csv"): pts[fn[:-4]] = [float(r["pts_sec"]) for r in csv.DictReader(open(os.path.join(ROOT, "dataset", "pts", fn), encoding="utf-8"))]
ren = {}
for s in os.listdir(SRC):
    p = os.path.join(SRC, s, "rename_mapping.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            o = r.get("orig") or r.get("original"); ren[os.path.splitext(o)[0]] = os.path.splitext(r["renamed"])[0]

def fix(video, t):
    p = pts[ren.get(video, video)]; fps_cv2 = len(p) / (p[-1] + 1 / FPS_NOMINAL)
    i = min(int(round(t * fps_cv2)), len(p) - 1); return p[i]

def backup(path):
    b = path.replace(".csv", "_pre_ptsfix_20260909.bak.csv")
    if not os.path.exists(b): shutil.copy2(path, b)

# 1) G1_events_all
pa = os.path.join(ROOT, "dataset", "G1_events_all.csv"); backup(pa)
rows = list(csv.DictReader(open(pa, encoding="utf-8-sig"))); fields = list(rows[0].keys())
if "video_time_idx_sec" not in fields: fields += ["video_time_idx_sec", "stable_read_time_idx_sec"]
shift = {}
for r in rows:
    if r.get("video_time_idx_sec"):  # 既に補正済み
        continue
    r["video_time_idx_sec"] = r["video_time_sec"]; r["stable_read_time_idx_sec"] = r["stable_read_time_sec"]
    t0 = float(r["video_time_sec"]); t1 = fix(r["video"], t0); r["video_time_sec"] = f"{t1:.3f}"
    if r["stable_read_time_sec"]: r["stable_read_time_sec"] = f"{fix(r['video'], float(r['stable_read_time_sec'])):.3f}"
    if r.get("audio_time_sec") and r.get("take"):
        # audio_time = video_time + offset（差分を保つ）
        off = float(r["audio_time_sec"]) - t0; r["audio_time_sec"] = f"{t1 + off:.3f}"
    shift[f"{r['video']}_{r['counter']}_{r['count_value']}"] = t1 - t0
with open(pa, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print("G1_events_all updated", len(rows), "shifted>0.5s", sum(abs(v) > 0.5 for v in shift.values()))

# 2) G1_events_synced
ps = os.path.join(ROOT, "dataset", "G1_events_synced.csv"); backup(ps)
rows = list(csv.DictReader(open(ps, encoding="utf-8-sig"))); fields = list(rows[0].keys())
if "video_time_idx_sec" not in fields: fields += ["video_time_idx_sec"]
n = 0
for r in rows:
    if r.get("video_time_idx_sec"): continue
    t0 = float(r["video_time_sec"]); t1 = fix(r["video"], t0); r["video_time_idx_sec"] = r["video_time_sec"]; r["video_time_sec"] = f"{t1:.3f}"
    if r.get("stable_read_time_sec"): r["stable_read_time_sec"] = f"{fix(r['video'], float(r['stable_read_time_sec'])):.3f}"
    for c in ("audio_time_sec", "pitch_time_sec"):
        if r.get(c): r[c] = f"{float(r[c]) + (t1 - t0):.3f}"
    n += 1
with open(ps, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print("G1_events_synced updated", n)

# 3) 検収ページの vt 補正 と 再生成対象クリップの削除
todo = 0
for page in ["review_sample_20260807_821.html", "review_sample_20260628_561.html"]:
    pp = os.path.join(ROOT, page); backup_html = pp.replace(".html", "_pre_ptsfix_20260909.bak.html")
    if not os.path.exists(backup_html): shutil.copy2(pp, backup_html)
    h = open(pp, encoding="utf-8", errors="replace").read()
    def rep(m):
        global todo
        eid, vt = m.group(1), float(m.group(2)); sh = shift.get(eid)
        if sh is None: return m.group(0)
        if abs(sh) > 0.1:
            c = os.path.join(ROOT, "clips3", eid + ".mp4")
            if os.path.exists(c): os.remove(c)
            todo += 1
        return m.group(0).replace(f'"vt":{m.group(2)}', f'"vt":{vt + sh:.3f}')
    h2 = re.sub(r'\{"id":"([^"]+)","video":"[^"]+","c":"[^"]+","n":\d+,"d":\d+,"vt":([0-9.]+)', rep, h)
    h2 = h2.replace(">v1.5 clips3<", ">v1.6 clips3 pts<")
    open(pp, "w", encoding="utf-8", newline="\n").write(h2)
print("clips to regenerate", todo)
with open(os.path.join(ROOT, "dataset", "pts_shift_by_event_20260909.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["event_id", "shift_sec"]); [w.writerow([k, f"{v:.3f}"]) for k, v in shift.items()]
