# -*- coding: utf-8 -*-
"""回収パイプライン共通設定＋帯pts表ゲート（2026-09-15版・2026-09-14作成）。

背景（9/11の時刻ずれ・第2種）：counter_events.csv の frame 番号は帯クロップ動画（bands_wao/<video>.mp4・ffmpeg で再エンコード）の
フレーム番号だが、G1生成は元動画の pts 表（dataset/pts/）を同じ番号で引いていた。再エンコードで ffmpeg が時刻丸めの衝突フレームを
落とすため（9/11＝全30曲・48〜104枚）、曲の後半ほど時刻が早くなった（最大1.985秒）。

恒久対策（方式a）：帯クロップの直後に「帯動画自身の moov から読んだ pts 表」を dataset/pts_bands_<session>/<video>.csv に必ず作り、
G1生成はその表だけを参照する（元動画の pts 表は参照しない）。表が無い／不整合ならゲートで工程を止める。

使い方：各スクリプトは `from kv_config_20260915 import Cfg` → `C=Cfg()`（セッションは 引数 → 環境変数 KV_SESSION → 既定 20260915）。
"""
import os, sys, csv, json, subprocess, datetime

DEFAULT_SESSION = "20260915"
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
VIDEO_ROOT = r"G:\研究アーカイブ\ビデオ素材"
WAV_ROOT = r"G:\研究アーカイブ\録音"
DAM_DIR = r"G:\研究アーカイブ\成果物\dam_api"
FFBIN = os.path.join(os.environ.get("LOCALAPPDATA", r"C:\Users\conan\AppData\Local"), "Microsoft", "WinGet", "Packages",
                     "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe", "ffmpeg-9.0.1-full_build", "bin")
FFMPEG = os.path.join(FFBIN, "ffmpeg.exe"); FFPROBE = os.path.join(FFBIN, "ffprobe.exe")
PYTHON = sys.executable or r"C:\Users\conan\AppData\Local\Programs\Python\Python312\python.exe"
WSL = ["wsl", "-d", "Ubuntu-26.04", "-u", "toshiaki", "--", "bash", "-lc"]
WSL_ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"

# ゲートのしきい値
GATE_LAST_PTS_TOL = 0.10   # 帯動画と元動画の最終ptsの差（秒）がこれを超えたら FAIL（時刻が保たれていない）
GATE_MIN_KEEP_RATIO = 0.50 # 帯動画のフレーム数が元動画の50%未満なら FAIL（fpsが半減するような再エンコード）

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mp4_pts import video_pts  # noqa: E402  moovから映像ptsを読む（デコードなし）


def session_from_args(argv=None):
    a = [x for x in (argv if argv is not None else sys.argv[1:]) if x.isdigit() and len(x) == 8]
    return a[0] if a else os.environ.get("KV_SESSION", DEFAULT_SESSION)


class Cfg:
    def __init__(self, session=None):
        self.S = session or session_from_args()
        S = self.S
        self.root = ROOT
        self.vid_dir = os.path.join(VIDEO_ROOT, S)
        self.bands_dir = os.path.join(ROOT, "bands_wao")
        self.gatefail_dir = os.path.join(self.bands_dir, "_gatefail_" + S)
        self.results_dir = os.path.join(ROOT, "results_wao_" + S)
        self.pts_src_dir = os.path.join(ROOT, "dataset", "pts")                 # 元動画のpts（参考・G1では使わない）
        self.pts_band_dir = os.path.join(ROOT, "dataset", "pts_bands_" + S)     # 帯動画のpts（G1はこれだけを使う）
        self.gate_csv = os.path.join(self.pts_band_dir, "_gate.csv")
        self.gate_log = os.path.join(self.pts_band_dir, "_gate.log")
        self.probe_csv = os.path.join(ROOT, "dataset", f"band_probe_v2_{S}.csv")
        self.probe_log = os.path.join(ROOT, "dataset", f"band_probe_v2_{S}.log")
        self.pipe_dir = os.path.join(ROOT, "dataset", "pipeline_" + S)
        self.pipe_log = os.path.join(self.pipe_dir, "pipeline.log")
        self.blind_dir = os.path.join(ROOT, "dataset", "blind_" + S)
        self.sync_csv = os.path.join(ROOT, "dataset", "sync", f"sync_offsets_{S}.csv")
        self.wav_dir = os.path.join(WAV_ROOT, f"songs_{S}_raw")
        self.rename_csv = os.path.join(self.vid_dir, "rename_mapping.csv")
        self.g1_csv = os.path.join(ROOT, "dataset", f"G1_events_{S}.csv")
        self.verify_csv = os.path.join(ROOT, "dataset", f"verify_{S}.csv")
        # セルフテスト用の差し替え（本番では未設定）
        if os.environ.get("KV_PIPE_DIR"): self.pipe_dir = os.environ["KV_PIPE_DIR"]; self.pipe_log = os.path.join(self.pipe_dir, "pipeline.log")
        if os.environ.get("KV_BLIND_DIR"): self.blind_dir = os.environ["KV_BLIND_DIR"]

    # ---------- ログ ----------
    def log(self, msg, path=None):
        p = path or self.pipe_log
        os.makedirs(os.path.dirname(p), exist_ok=True)
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        with open(p, "a", encoding="utf-8") as f: f.write(line + "\n")
        try: print(line, flush=True)
        except Exception: pass

    # ---------- 補助 ----------
    def probe_rows(self):
        if not os.path.exists(self.probe_csv): return []
        return list(csv.DictReader(open(self.probe_csv, encoding="utf-8")))

    def ffprobe_nb_frames(self, path):
        """mp4 の stsz（サンプル数）由来の nb_frames を ffprobe で取る。取れなければ None。"""
        try:
            p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                                "-of", "json", path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
            return int(json.loads(p.stdout)["streams"][0]["nb_frames"])
        except Exception:
            return None

    def read_pts_csv(self, path):
        return [float(r["pts_sec"]) for r in csv.DictReader(open(path, encoding="utf-8"))]

    def write_pts_csv(self, path, pts):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["index", "pts_sec"])
            for i, x in enumerate(pts): w.writerow([i, f"{x:.4f}"])
        os.replace(tmp, path)

    def ensure_src_pts(self, name):
        """元動画の pts 表（dataset/pts/<name>.csv）が無ければ作る（比較専用）。返り値＝pts list or None。"""
        p = os.path.join(self.pts_src_dir, name + ".csv")
        if not os.path.exists(p):
            src = os.path.join(self.vid_dir, name + ".mp4")
            if not os.path.exists(src): return None
            try:
                pts, _ = video_pts(src)
            except Exception:
                return None
            self.write_pts_csv(p, pts)
        return self.read_pts_csv(p)

    # ---------- 帯pts表の生成＋ゲート ----------
    def band_pts_gate(self, name, band_path=None, out_dir=None, gate_csv=None, gate_log=None):
        """帯動画 <name>.mp4 から pts 表を作り、整合を検証する。
        返り値: dict(video, status=PASS|WARN|FAIL, n_band, n_src, dropped, last_band, last_src, reason)
        PASS/WARN なら <out_dir>/<name>.csv が存在する。FAIL のときは表を残さない（G1側のゲートが確実に止まるように）。
        """
        band_path = band_path or os.path.join(self.bands_dir, name + ".mp4")
        out_dir = out_dir or self.pts_band_dir; gate_csv = gate_csv or self.gate_csv; gate_log = gate_log or self.gate_log
        out = os.path.join(out_dir, name + ".csv")
        r = dict(video=name, status="FAIL", n_band=None, n_src=None, dropped=None, last_band=None, last_src=None, reason="")
        try:
            pts, _ = video_pts(band_path)
        except Exception as e:
            r["reason"] = f"帯動画のmoovからptsを読めない: {e}"; return self._gate_finish(r, out, gate_csv, gate_log)
        if not pts or any(pts[i + 1] < pts[i] for i in range(len(pts) - 1)):
            r["reason"] = "帯ptsが空または非単調"; return self._gate_finish(r, out, gate_csv, gate_log)
        r["n_band"] = len(pts); r["last_band"] = round(pts[-1], 4)
        nb = self.ffprobe_nb_frames(band_path)
        if nb is not None and nb != len(pts):
            r["reason"] = f"帯動画の stts由来フレーム数 {len(pts)} と ffprobe nb_frames {nb} が不一致"; return self._gate_finish(r, out, gate_csv, gate_log)
        src = self.ensure_src_pts(name)
        if src is None:
            r["reason"] = "元動画のpts表が作れない（元動画が見つからない）"; return self._gate_finish(r, out, gate_csv, gate_log)
        r["n_src"] = len(src); r["last_src"] = round(src[-1], 4); r["dropped"] = len(src) - len(pts)
        if len(pts) > len(src):
            r["reason"] = f"帯動画のフレーム数 {len(pts)} が元動画 {len(src)} より多い（重複フレーム）"; return self._gate_finish(r, out, gate_csv, gate_log)
        if len(pts) < GATE_MIN_KEEP_RATIO * len(src):
            r["reason"] = f"帯動画のフレーム数 {len(pts)} が元動画 {len(src)} の{GATE_MIN_KEEP_RATIO:.0%}未満（fps半減など）"; return self._gate_finish(r, out, gate_csv, gate_log)
        if abs(pts[-1] - src[-1]) > GATE_LAST_PTS_TOL:
            r["reason"] = f"最終ptsの差 {pts[-1] - src[-1]:+.3f}s が許容 {GATE_LAST_PTS_TOL}s を超える（時刻が保たれていない）"; return self._gate_finish(r, out, gate_csv, gate_log)
        self.write_pts_csv(out, pts)
        r["status"] = "PASS" if r["dropped"] == 0 else "WARN"
        r["reason"] = "" if r["dropped"] == 0 else f"再エンコードで {r['dropped']} 枚落ち（帯pts表で吸収・G1はこの表を参照）"
        return self._gate_finish(r, out, gate_csv, gate_log)

    def _gate_finish(self, r, out, gate_csv, gate_log):
        if r["status"] == "FAIL" and os.path.exists(out):
            os.remove(out)
        os.makedirs(os.path.dirname(gate_csv), exist_ok=True)
        rows = []
        if os.path.exists(gate_csv):
            rows = [x for x in csv.DictReader(open(gate_csv, encoding="utf-8")) if x["video"] != r["video"]]
        rows.append({k: ("" if v is None else v) for k, v in r.items()})
        with open(gate_csv + ".part", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(r.keys())); w.writeheader(); w.writerows(rows)
        os.replace(gate_csv + ".part", gate_csv)
        self.log(f"GATE {r['status']} {r['video']} band={r['n_band']} src={r['n_src']} dropped={r['dropped']} "
                 f"last={r['last_band']}/{r['last_src']} {r['reason']}", gate_log)
        return r

    def gate_failures(self, gate_csv=None):
        p = gate_csv or self.gate_csv
        if not os.path.exists(p): return []
        return [x for x in csv.DictReader(open(p, encoding="utf-8")) if x["status"] == "FAIL"]
