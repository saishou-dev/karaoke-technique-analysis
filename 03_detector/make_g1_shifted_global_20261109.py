# -*- coding: utf-8 -*-
r"""正解データ（G1）の時刻を全技法共通の1つの遅れでずらした写しを、11/4 結合後の主台帳から作る（2026-11-09・kaishu20261104-merge・新規1本）。
make_g1_shifted_global_20261105.py と同型（遅れ M の出し方・ずらし方・検算は同じ）。既存の 20261105 版は変えない・出力も上書きしない。

遅れ M＝a_counter_minus_D の中央値（ビブラートを除く5技法・D がある件）＝1.6722 秒（11/5 と同じ値になることを assert）。6技法とも同じ M を引く。

11/5 版からの変更
  1. 入力の主台帳＝11/4 を結合した後の dataset\G1_events_all.csv（14,700 行）。写しには sync_flag・sync_status 列を足す
     （11/4 の行だけ dataset\sync\sync_flags_20261104.csv から引く・他のセッションは空）。主台帳そのものには列を足さない。
  2. 索引＝11/5 の 208 曲（g3v2_20260928\index.json・md5 6c9dc6fc…）の後ろに 11/4 の曲を足す。
     11/4 の曲の作り方＝build_index.py の EXTRA（後ろに足すセッション）と同じ規則
       （multi_increment==0 の行・events＝(counter, audio_time_sec)・span＝[min−5, max+2]・wav＝録音\songs_20261104_raw\<wav>・
        pitch＝dataset\pitch_pyin_20261104\<stem>_f0.csv があればそれ・無ければ None・machine＝wao）。
     **sync の旗が ok の曲だけ**入れる（sync_flags_20261104.csv の in_index=1）。旗を書き換えて流し直せば戻せる。
     build_index.py・index.json は変えない（11/4 は EXTRA に無い＝本体を変えずにここで足す）。
  3. 検算に「0 秒より前」と「WAV の長さを超える」を足す（11/5 版は span の外だけ）。
     11/4 の加点で、ずらした後の時刻が録音の外（0 秒より前／WAV の長さより後）になるものは索引に入れない（台帳には残る・件数を check に書く）。
     実測＝天国_20261104_124950 の fall 1（audio 451.6s・WAV 324.0s＝読み過ぎ fall+1 の偽イベント）の 1 件。
     11/5 の 208 曲の中の 0 秒より前（77 件・9/11 Midnight 66・10/6 阿修羅ちゃん 10・うっせぇわ 1＝録音が曲の途中から）は
     既存の事情なので触らない（11/5 版と同数であることだけ確かめる）。

出力（out\c2_shift3_global_20261109\labels\ だけ・既にあれば止める＝上書きしない）
  shift_table_20261109.csv / G1_events_all_shift3_global_20261109.csv / index_base_20261109.json（ずらす前）/
  index_shift3_global_20261109.json / check_20261109.txt
使い方: python make_g1_shifted_global_20261109.py
"""
import csv, json, os, statistics, sys, collections, hashlib
import soundfile as sf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.dirname(ROOT)
REC = r"G:\研究アーカイブ\録音"
SES = "20261104"
OUT = os.path.join(ROOT, "out", "c2_shift3_global_20261109", "labels")
DT = os.path.join(BASE, "marks_20260928", "out", "mark_pos_time_20261105", "D_times_20261105.csv")
G1 = os.path.join(BASE, "dataset", "G1_events_all.csv")
INDEX = os.path.join(ROOT, "index.json")
INDEX_MD5 = "6c9dc6fc55978d7dc1dafda537717648"   # 11/5 の 208 曲版（RUNBOOK・close4-finish 報告の値）
SYNC = os.path.join(BASE, "dataset", "sync", f"sync_offsets_{SES}.csv")
FLAGS = os.path.join(BASE, "dataset", "sync", f"sync_flags_{SES}.csv")
PITCH = os.path.join(BASE, "dataset", f"pitch_pyin_{SES}")
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
REF = dict(shakuri=1.64, kobushi=1.94, fall=1.90, accent=1.62, hammering=1.62)
TIME_COLS = ["video_time_sec", "audio_time_sec", "video_time_idx_sec"]
EXPECT_M = 1.6722


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    names = ["shift_table_20261109.csv", "G1_events_all_shift3_global_20261109.csv", "index_base_20261109.json",
             "index_shift3_global_20261109.json", "check_20261109.txt"]
    if any(os.path.exists(os.path.join(OUT, n)) for n in names):
        raise SystemExit(f"止める＝出力が既にある（上書きしない）: {OUT}")
    os.makedirs(OUT, exist_ok=True)
    log = []
    say = lambda s: (print(s), log.append(s))

    # ---- 遅れ M（11/5 と同じ出し方）----
    v = collections.defaultdict(list); ns = collections.Counter()
    for r in csv.DictReader(open(DT, encoding="utf-8-sig")):
        if r["D"] == "" or r["a_counter_minus_D"] == "":
            continue
        v[r["counter"]].append(float(r["a_counter_minus_D"])); ns[(r["session"], r["counter"])] += 1
    per, rows = {}, []
    for c in TYPES:
        per[c] = statistics.median(v[c]); ref = REF.get(c)
        rows.append([c, ns[("20260924", c)], ns[("20261014", c)], len(v[c]), round(per[c], 4), "" if ref is None else ref,
                     "" if ref is None else round(per[c] - ref, 4)])
    gv = [x for c in TYPES if c != "vibrato" for x in v[c]]
    GM = statistics.median(gv)
    assert round(GM, 4) == EXPECT_M, f"止める＝M が 11/5 と違う: {GM}"
    shift = {c: GM for c in TYPES}
    rows.append(["GLOBAL_5技法（使う値）", "", "", len(gv), round(GM, 4), "", "ビブラートを除く5技法の中央値"])
    with open(os.path.join(OUT, names[0]), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["技法", "件数_9/24", "件数_10/14", "件数_合計", "M_秒（前へ）", "参考値_1010報告", "差"]); w.writerows(rows)
    say(f"使う遅れ（全技法共通・秒・前へ）＝{GM:.4f}（5技法 {len(gv)}件の中央値・11/5 と同じ値）")

    # ---- sync の旗 ----
    flags = {r["video"]: r for r in csv.DictReader(open(FLAGS, encoding="utf-8-sig"))}
    say("sync の旗（11/4）＝" + "・".join(f"{k} {n}" for k, n in collections.Counter(r["status"] for r in flags.values()).items())
        + f"・索引に入れる {sum(r['in_index'] == '1' for r in flags.values())} 曲")

    # ---- G1 の写し ----
    with open(G1, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f); fields = rd.fieldnames; g = list(rd)
    extra = ["shift_sec"] + [f"orig_{c}" for c in TIME_COLS] + ["sync_flag", "sync_status"]
    out_rows, unknown = [], collections.Counter()
    for r in g:
        r2 = dict(r); s = shift.get(r["counter"])
        if s is None:
            unknown[r["counter"]] += 1; s = 0.0
        for c in TIME_COLS:
            r2[f"orig_{c}"] = r[c]
            if r[c] not in ("", None):
                r2[c] = f"{float(r[c]) - s:.4f}"
        r2["shift_sec"] = f"{s:.4f}"
        fl = flags.get(r["video"]) if r["session"] == SES else None
        r2["sync_flag"] = fl["sync_flag"] if fl else ""
        r2["sync_status"] = fl["status"] if fl else ""
        out_rows.append(r2)
    gpath = os.path.join(OUT, names[1])
    with open(gpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields + extra); w.writeheader(); w.writerows(out_rows)
    g2 = list(csv.DictReader(open(gpath, encoding="utf-8-sig")))
    c1 = collections.Counter((r["session"], r["counter"]) for r in g); c2 = collections.Counter((r["session"], r["counter"]) for r in g2)
    ok_g = len(g) == len(g2) and c1 == c2
    n_ses = sum(1 for r in g if r["session"] == SES)
    say(f"G1 写し＝元 {len(g)} 行（うち {SES} {n_ses} 行）・写し {len(g2)} 行・収録×技法の件数 {'一致' if ok_g else '不一致'}・技法外の行 {dict(unknown) or 0}・"
        f"sync_flag suspect の行 {sum(1 for r in g2 if r['sync_flag'] == 'suspect')}")

    # ---- 索引（ずらす前）＝11/5 の 208 曲 ＋ 11/4 ----
    assert md5f(INDEX) == INDEX_MD5, "止める＝index.json が 11/5 の版と違う"
    base = json.load(open(INDEX, encoding="utf-8"))
    smap = {r["video"]: r for r in csv.DictReader(open(SYNC, encoding="utf-8-sig"))}
    gs = collections.defaultdict(list)
    for r in g:
        if r["session"] == SES and int(r["multi_increment"]) == 0:
            gs[r["video"]].append(r)
    new, skipped, warn, dur, dropped = [], [], [], {}, []
    for video in sorted(gs):
        fl = flags[video]
        if fl["in_index"] != "1":
            skipped.append((video, fl["status"], len(gs[video]))); continue
        s = smap[video]; wav = os.path.join(REC, f"songs_{SES}_raw", s["wav"]); off = float(s["offset_sec"])
        assert os.path.exists(wav), f"wav なし: {wav}"
        stem = os.path.splitext(s["wav"])[0]; pf = os.path.join(PITCH, f"{stem}_f0.csv")
        pf = pf if os.path.exists(pf) else None
        ev = gs[video]
        dev = max(abs(float(r["video_time_sec"]) + off - float(r["audio_time_sec"])) for r in ev)
        if dev > 0.002:
            warn.append(f"audio_time_sec と video_time+offset の差 {dev:.3f}s: {video}")
        assert all(r["take"] + ".wav" == s["wav"] for r in ev), f"G1 の take と同期表の wav が違う: {video}"
        dur[video] = sf.info(wav).duration
        events = [(r["counter"], float(r["audio_time_sec"])) for r in ev]
        # ずらした後の時刻（t − M）が録音の外（0 秒より前・WAV の長さより後）になる加点は索引に入れない（学習できない）。台帳には残る。
        inside = lambda c, t: 0 <= t - shift[c] <= dur[video]
        out_ev = [(c, t) for c, t in events if not inside(c, t)]
        if out_ev:
            dropped.append((video, out_ev))
            events = [(c, t) for c, t in events if inside(c, t)]
        ts = [t for _, t in events]
        new.append(dict(video=video, session=SES, wav=wav.replace("\\", "/"), offset=off, pitch=pf, events=events,
                        span=[min(ts) - 5.0, max(ts) + 2.0], n=len(events), machine="wao"))
    ix = base + new
    json.dump(ix, open(os.path.join(OUT, names[2]), "w", encoding="utf-8"), ensure_ascii=False)
    say(f"索引（ずらす前）＝11/5 の {len(base)} 曲 {sum(d['n'] for d in base)} 件 ＋ {SES} の {len(new)} 曲 {sum(d['n'] for d in new)} 件 ＝ {len(ix)} 曲 {sum(d['n'] for d in ix)} 件")
    say(f"{SES} で索引から外した曲＝{len(skipped)} 曲 {sum(n for *_, n in skipped)} 件（" + "・".join(f"{v_.split('_')[0]} {st} {n}件" for v_, st, n in skipped) + "）")
    say(f"{SES} の multi_increment=1 で索引に入らない行＝{sum(1 for r in g if r['session'] == SES and int(r['multi_increment']) == 1)}（build_index.py と同じ規則）")
    say(f"{SES} の pitch あり＝{sum(1 for d in new if d['pitch'])}/{len(new)}（無い曲は pitch=None＝extract_feats.py がその場で pYIN）")
    say(f"{SES} でずらした後に録音の外になるため索引に入れなかった加点＝{sum(len(e) for _, e in dropped)} 件（"
        + "・".join(f"{v_.split('_')[0]} {e} WAV {dur[v_]:.1f}s" for v_, e in dropped) + "）")
    for w_ in warn:
        say("注意: " + w_)

    # ---- ずらす ----
    ix2, bad, bad_t, neg, over = [], 0, collections.Counter(), 0, 0
    neg0 = sum(1 for d in ix for _, t in d["events"] if t < 0)
    negb = 0   # 11/5 の 208 曲の中の 0 秒より前（既存の事情・11/5 版と同数であることだけ確かめる）
    for i, d in enumerate(ix):
        d2 = dict(d)
        d2["events"] = [[c, round(t - shift.get(c, 0.0), 4)] for c, t in d["events"]]
        lo, hi = d["span"]
        for c, t in d2["events"]:
            if not (lo <= t <= hi):
                bad += 1; bad_t[c] += 1
            if t < 0:
                if i < len(base):
                    negb += 1
                else:
                    neg += 1
            if d["video"] in dur and t > dur[d["video"]]:
                over += 1
        d2["shift_note"] = "c2_shift3_global_20261109: events = 元の時刻 − M（カウンター−D の中央値・全技法共通の1つ・span は元のまま）"
        ix2.append(d2)
    ipath = os.path.join(OUT, names[3])
    json.dump(ix2, open(ipath, "w", encoding="utf-8"), ensure_ascii=False)
    ix3 = json.load(open(ipath, encoding="utf-8"))
    e1 = collections.Counter(c for d in ix for c, _ in d["events"]); e2 = collections.Counter(c for d in ix3 for c, _ in d["events"])
    pair_ok = all(len(d["events"]) == len(d3["events"]) and all(c == c3 and abs((t - t3) - shift[c]) < 1e-3
                  for (c, t), (c3, t3) in zip(d["events"], d3["events"])) for d, d3 in zip(ix, ix3))
    ok_i = len(ix) == len(ix3) and e1 == e2 and pair_ok
    n_ev = sum(e1.values()); frac = bad / n_ev if n_ev else 0
    say(f"index 写し＝曲 {len(ix)}→{len(ix3)}・イベント {n_ev}→{sum(e2.values())}・技法別 {'一致' if e1 == e2 else '不一致'}・"
        f"1対1の M {'一致' if pair_ok else '不一致'}・span の外へ出た件 {bad}（{frac * 100:.2f}%・技法別 {dict(bad_t) or 0}）・"
        f"{SES} の 0 秒より前 {neg}・{SES} の WAV の長さを超える件 {over}・"
        f"（参考）11/5 の {len(base)} 曲の中の 0 秒より前 {negb}（ずらす前 {neg0}）")
    say("index 技法別＝" + "・".join(f"{c} {e2[c]}" for c in TYPES))
    old = os.path.join(ROOT, "out", "c2_shift3_global_20261105", "labels", "index_shift3_global_20261105.json")
    same = True
    if os.path.exists(old):
        o = json.load(open(old, encoding="utf-8"))
        same = len(o) == len(base) and all(a["video"] == b["video"] and a["events"] == b["events"] and a["span"] == b["span"]
                                           for a, b in zip(o, ix3[:len(base)]))
        nego = sum(1 for d in o for _, t in d["events"] if t < 0)
        same = same and nego == negb
        say(f"11/5 版との突合＝先頭 {len(base)} 曲の events・span {'一致' if same else '不一致'}・0 秒より前 11/5 版 {nego}（11/5 のファイルは読むだけ）")
    open(os.path.join(OUT, names[4]), "w", encoding="utf-8").write("\n".join(log) + "\n")
    if not (ok_g and ok_i and same):
        raise SystemExit("止める＝件数の一致が取れない")
    if frac > 0.01 or neg or over:
        raise SystemExit(f"止める＝span の外 {frac * 100:.2f}%・0 秒より前 {neg}・WAV 超え {over}")
    print("OK")


if __name__ == "__main__":
    main()
