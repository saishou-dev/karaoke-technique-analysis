# -*- coding: utf-8 -*-
r"""正解データ（G1）の時刻を全技法共通の1つの遅れでずらした写しを、11/10 結合後の主台帳から作る（2026-11-10・kaishu20261110・新規1本）。
make_g1_shifted_global_20261109.py と同型（遅れ M の出し方・ずらし方・検算は同じ）。既存の 20261105／20261109 版は変えない・出力も上書きしない。

遅れ M＝a_counter_minus_D の中央値（ビブラートを除く5技法・D がある件）＝1.6722 秒（11/5・11/9 と同じ値になることを assert）。6技法とも同じ M を引く。

11/9 版からの変更
  1. 入力の主台帳＝11/10 を結合した後の dataset\G1_events_all.csv。写しの sync_flag・sync_status 列は
     11/4 の行＝sync_flags_20261104.csv・11/10 の行＝sync_flags_20261110.csv から引く（他のセッションは空）。
  2. 索引（ずらす前）＝11/9 の 229 曲（out\c2_shift3_global_20261109\labels\index_base_20261109.json・md5 f2c9be84…）の後ろに 11/10 の曲を足す。
     11/10 の曲の作り方は 11/9 版の 11/4 と同じ（multi_increment==0 の行・events＝(counter, audio_time_sec)・span＝[min−5, max+2]・
     wav＝録音\songs_20261110_raw\<wav>・pitch＝dataset\pitch_pyin_20261110\<stem>_f0.csv があれば・無ければ None・machine＝wao）。
     sync の旗が ok の曲だけ（sync_flags_20261110.csv の in_index=1）。
  3. 検算＝先頭 229 曲の events・span が 11/9 の index_shift3_global_20261109.json と一致・229 曲の中の 0 秒より前が 11/9 版と同数・
     11/10 の 0 秒より前 0・11/10 の WAV 超え 0・span の外 1% 以下。

出力（out\c2_shift3_global_20261110\labels\ だけ・既にあれば止める＝上書きしない）
  shift_table_20261110.csv / G1_events_all_shift3_global_20261110.csv / index_base_20261110.json（ずらす前）/
  index_shift3_global_20261110.json / check_20261110.txt
使い方: python make_g1_shifted_global_20261110.py
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
SES = "20261110"
TAG = "20261110"
OUT = os.path.join(ROOT, "out", f"c2_shift3_global_{TAG}", "labels")
DT = os.path.join(BASE, "marks_20260928", "out", "mark_pos_time_20261105", "D_times_20261105.csv")
G1 = os.path.join(BASE, "dataset", "G1_events_all.csv")
PREV = os.path.join(ROOT, "out", "c2_shift3_global_20261109", "labels")
INDEX = os.path.join(PREV, "index_base_20261109.json")
INDEX_MD5 = "f2c9be84281bf1f370c3332e44f817d9"   # 11/9 の 229 曲版（ずらす前）
PREV_SHIFT = os.path.join(PREV, "index_shift3_global_20261109.json")
SYNC = os.path.join(BASE, "dataset", "sync", f"sync_offsets_{SES}.csv")
FLAGS = {s: os.path.join(BASE, "dataset", "sync", f"sync_flags_{s}.csv") for s in ("20261104", SES)}
PITCH = os.path.join(BASE, "dataset", f"pitch_pyin_{SES}")
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
REF = dict(shakuri=1.64, kobushi=1.94, fall=1.90, accent=1.62, hammering=1.62)
TIME_COLS = ["video_time_sec", "audio_time_sec", "video_time_idx_sec"]
EXPECT_M = 1.6722


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    names = [f"shift_table_{TAG}.csv", f"G1_events_all_shift3_global_{TAG}.csv", f"index_base_{TAG}.json",
             f"index_shift3_global_{TAG}.json", f"check_{TAG}.txt"]
    if any(os.path.exists(os.path.join(OUT, n)) for n in names):
        raise SystemExit(f"止める＝出力が既にある（上書きしない）: {OUT}")
    os.makedirs(OUT, exist_ok=True)
    log = []
    say = lambda s: (print(s), log.append(s))
    prev_md5 = {n: md5f(os.path.join(PREV, n)) for n in os.listdir(PREV) if os.path.isfile(os.path.join(PREV, n))}

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
    say(f"使う遅れ（全技法共通・秒・前へ）＝{GM:.4f}（5技法 {len(gv)}件の中央値・11/5・11/9 と同じ値）")

    # ---- sync の旗 ----
    flags = {s: {r["video"]: r for r in csv.DictReader(open(p, encoding="utf-8-sig"))} for s, p in FLAGS.items()}
    fl10 = flags[SES]
    say(f"sync の旗（{SES}）＝" + "・".join(f"{k} {n}" for k, n in collections.Counter(r["status"] for r in fl10.values()).items())
        + f"・索引に入れる {sum(r['in_index'] == '1' for r in fl10.values())} 曲")

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
        fl = flags[r["session"]].get(r["video"]) if r["session"] in flags else None
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
        f"sync_flag suspect の行 {sum(1 for r in g2 if r['sync_flag'] == 'suspect')}（うち {SES} {sum(1 for r in g2 if r['sync_flag'] == 'suspect' and r['session'] == SES)}）")

    # ---- 索引（ずらす前）＝11/9 の 229 曲 ＋ 11/10 ----
    assert md5f(INDEX) == INDEX_MD5, "止める＝index_base_20261109.json が 11/9 の版と違う"
    base = json.load(open(INDEX, encoding="utf-8"))
    smap = {r["video"]: r for r in csv.DictReader(open(SYNC, encoding="utf-8-sig"))}
    gs = collections.defaultdict(list)
    for r in g:
        if r["session"] == SES and int(r["multi_increment"]) == 0:
            gs[r["video"]].append(r)
    new, skipped, warn, dur, dropped = [], [], [], {}, []
    for video in sorted(gs):
        fl = fl10[video]
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
        inside = lambda c, t: 0 <= t - shift[c] <= dur[video]
        out_ev = [(c, t) for c, t in events if not inside(c, t)]
        if out_ev:
            dropped.append((video, out_ev))
            events = [(c, t) for c, t in events if inside(c, t)]
        if not events:
            skipped.append((video, "録音の内側の加点 0", 0)); continue
        ts = [t for _, t in events]
        new.append(dict(video=video, session=SES, wav=wav.replace("\\", "/"), offset=off, pitch=pf, events=events,
                        span=[min(ts) - 5.0, max(ts) + 2.0], n=len(events), machine="wao"))
    ix = base + new
    json.dump(ix, open(os.path.join(OUT, names[2]), "w", encoding="utf-8"), ensure_ascii=False)
    say(f"索引（ずらす前）＝11/9 の {len(base)} 曲 {sum(d['n'] for d in base)} 件 ＋ {SES} の {len(new)} 曲 {sum(d['n'] for d in new)} 件 ＝ {len(ix)} 曲 {sum(d['n'] for d in ix)} 件")
    say(f"{SES} で索引に入れた分の技法別＝" + "・".join(f"{c} {sum(1 for d in new for cc, _ in d['events'] if cc == c)}" for c in TYPES))
    say(f"{SES} で索引から外した曲＝{len(skipped)} 曲 {sum(n for *_, n in skipped)} 件（" + "・".join(f"{v_.rsplit('_', 2)[0]}_{v_.rsplit('_', 1)[1]} {st} {n}件" for v_, st, n in skipped) + "）")
    say(f"{SES} の multi_increment=1 で索引に入らない行＝{sum(1 for r in g if r['session'] == SES and int(r['multi_increment']) == 1)}（build_index.py と同じ規則）")
    say(f"{SES} の pitch あり＝{sum(1 for d in new if d['pitch'])}/{len(new)}（無い曲は pitch=None＝extract_feats.py がその場で pYIN）")
    say(f"{SES} でずらした後に録音の外になるため索引に入れなかった加点＝{sum(len(e) for _, e in dropped)} 件（"
        + "・".join(f"{v_.rsplit('_', 2)[0]} {e} WAV {dur[v_]:.1f}s" for v_, e in dropped) + "）")
    for w_ in warn:
        say("注意: " + w_)

    # ---- ずらす ----
    ix2, bad, bad_t, neg, over = [], 0, collections.Counter(), 0, 0
    neg0 = sum(1 for d in ix for _, t in d["events"] if t < 0)
    negb = 0
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
        d2["shift_note"] = f"c2_shift3_global_{TAG}: events = 元の時刻 − M（カウンター−D の中央値・全技法共通の1つ・span は元のまま）"
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
        f"（参考）11/9 の {len(base)} 曲の中の 0 秒より前 {negb}（ずらす前 {neg0}）")
    say("index 技法別＝" + "・".join(f"{c} {e2[c]}" for c in TYPES))
    o = json.load(open(PREV_SHIFT, encoding="utf-8"))
    same = len(o) == len(base) and all(a["video"] == b["video"] and a["events"] == b["events"] and a["span"] == b["span"]
                                       for a, b in zip(o, ix3[:len(base)]))
    nego = sum(1 for d in o for _, t in d["events"] if t < 0)
    same = same and nego == negb
    say(f"11/9 版との突合＝先頭 {len(base)} 曲の events・span {'一致' if same else '不一致'}・0 秒より前 11/9 版 {nego}（11/9 のファイルは読むだけ）")
    after = {n: md5f(os.path.join(PREV, n)) for n in prev_md5}
    say(f"11/9 の labels の md5＝実行前後で {'不変' if after == prev_md5 else '変化あり'}（{len(prev_md5)} ファイル）")
    open(os.path.join(OUT, names[4]), "w", encoding="utf-8").write("\n".join(log) + "\n")
    if not (ok_g and ok_i and same and after == prev_md5):
        raise SystemExit("止める＝件数の一致が取れない")
    if frac > 0.01 or neg or over:
        raise SystemExit(f"止める＝span の外 {frac * 100:.2f}%・0 秒より前 {neg}・WAV 超え {over}")
    print("OK")


if __name__ == "__main__":
    main()
