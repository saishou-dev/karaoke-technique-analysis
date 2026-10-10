# -*- coding: utf-8 -*-
"""検収サンプルページ生成 v3・9/24専用（2026-09-25）。gen_sample_page_20260911.py（v3）のコピーを 20260911 向けに直したもの。元スクリプトは不変。
   v2との違い：
   ① EVの元＝review_20260902.html（5,476件・9/11は未収録）ではなく G1 の CSV を直接読む。
      9/14: dataset/G1_events_20260911.csv は帯クロップ動画のフレーム番号で元動画の pts 表を引いていて最大2.0秒早い（クロップ再エンコードでフレームが48〜104枚落ちている）ため、
      帯動画自身の pts で引き直した dataset/G1_events_20260911_bandpts_20260914.csv（scripts/fix_g1_bandpts_20260911.py）を読む。shift表は不要（本スクリプト内で pts と再照合して確認）。
   ② Midnight は録音WAVが2本（中断テイク T04・本テイク T05）で G1 に同一 event_id が2行ある → audio_time_sec がWAV長の範囲内の行を採用して1件にする。
   ③ 除外＝台帳（dataset/kensyu_reviewed_registry.csv）・合算イベント（delta>1＝multi_increment=1）・凍結動画（D8）。ページに「+n」は出さない。
   ④ クリップ窓＝[vt-0.5, vt+1.5]＝2秒・赤帯＝最後の0.5秒（04_research「次版でクリップを短縮」）。ページのJS（ピッチ曲線のx軸・加点マーカー位置・脚注）も2秒窓に合わせて書き換える。
   ⑤ ピッチ曲線＝dataset/pitch_pyin_20260924/<wav stem>_f0.csv（scripts/pitch_pyin_20260924.py の出力・WAV時間軸）があれば audio_time_sec を中心に [-0.5,+1.5] を切って付ける。無ければ「未対応」表示。
   ⑥ 並び＝時系列（動画名末尾の録画時刻 → vt）。
   使い方: python scripts/gen_sample_page_20260924.py [20260924] [--n 200] [--seed 20260924] [--absclips 出力パス]
   出力: review_sample_20260924_200.html・dataset/kensyu_sample_20260924_200_manifest.csv（--absclips は G:/本人確認待ち 用の複製＝クリップ参照を file:/// 絶対パスにしたもの）
"""
import re, json, csv, os, sys, random, argparse, datetime
from collections import Counter
ROOT = r'G:\研究アーカイブ\成果物\動画解析'; WAVDIR = r'G:\研究アーカイブ\録音\songs_20260924_raw'
ap = argparse.ArgumentParser(); ap.add_argument('session', nargs='?', default='20260924'); ap.add_argument('--n', type=int, default=200); ap.add_argument('--seed', type=int, default=None)
ap.add_argument('--absclips', default=None, help='複製の出力先。クリップ参照を file:/// 絶対パスにする')
a = ap.parse_args(); sess = a.session; seed = a.seed or int(sess)
assert sess == '20260924', 'このスクリプトは 20260924 専用（G1_events_20260924.csv を直接読む・帯pts時刻）'
os.chdir(ROOT)
# 9/14 追記: G1_events_20260911.csv の時刻は「帯クロップ動画のフレーム番号」で元動画の pts 表を引いており最大2.0秒早い（scripts/fix_g1_bandpts_20260911.py の docstring 参照）。
#            ここでは帯動画自身の pts で引き直した作業用CSV（同スクリプトの出力）を読む。元CSVは不変。
G1 = 'dataset/G1_events_20260924.csv'; PTSDIR = 'dataset/pts_bands_20260924'
rows = list(csv.DictReader(open(G1, encoding='utf-8-sig')))
print('G1 rows', len(rows), 'from', G1)
# --- pts 照合（video_time_sec が帯動画 pts の event_frame_est 位置と一致することを全件で確認） ---
pts = {}; mism = 0
for r in rows:
    v = r['video']
    if v not in pts: pts[v] = [float(x['pts_sec']) for x in csv.DictReader(open(f'{PTSDIR}/{v}.csv', encoding='utf-8'))]
    p = pts[v]; fi = min(int(r['event_frame_est']), len(p) - 1)
    if abs(p[fi] - float(r['video_time_sec'])) > 0.002: mism += 1
print('pts check: mismatch', mism, '(0 なら補正済み)')
assert mism == 0, 'video_time_sec が pts と一致しない行がある。apply_pts_correction 相当の処理が必要'
# --- WAV長（Midnight の2本テイクの切り分け用） ---
wavdur = {r['renamed']: float(r['duration_s']) for r in csv.DictReader(open(os.path.join(WAVDIR, 'mapping.csv'), encoding='utf-8-sig')) if r.get('duration_s')}
by_id = {}
for r in rows:
    eid = f"{r['video']}_{r['counter']}_{r['count_value']}"; by_id.setdefault(eid, []).append(r)
def choose(cands):
    if len(cands) == 1: return cands[0]
    ok = [r for r in cands if r['wav'] in wavdur and 0 <= float(r['audio_time_sec']) <= wavdur[r['wav']]]
    return (ok or cands)[0]
dedup = [choose(c) for c in by_id.values()]
print('unique events', len(dedup), 'dup rows removed', len(rows) - len(dedup))
EV = [dict(id=f"{r['video']}_{r['counter']}_{r['count_value']}", video=r['video'], c=r['counter'], n=int(r['count_value']), d=int(r['delta']),
           vt=round(float(r['video_time_sec']), 3), sess=sess, dur=2.0, pitch=None, wav=r['wav'], at=round(float(r['audio_time_sec']), 3)) for r in dedup]
reg = set(r['event_id'] for r in csv.DictReader(open('dataset/kensyu_reviewed_registry.csv', encoding='utf-8-sig')))
FROZEN = {'虹_20260628_112135'}  # D8: 演出遮蔽で読取11%・凍結（9/11には該当なし・v2から踏襲）
n_multi = sum(1 for e in EV if e['d'] > 1); n_frozen = sum(1 for e in EV if e['video'] in FROZEN); n_reg = sum(1 for e in EV if e['id'] in reg)
pool = [e for e in EV if e['id'] not in reg and e['d'] == 1 and e['video'] not in FROZEN]
print(f'excluded: registry {n_reg} / multi_increment(d>1) {n_multi} / frozen {n_frozen} -> pool {len(pool)}')
rnd = random.Random(seed)
pick = rnd.sample(pool, min(a.n, len(pool)))
def tkey(e):
    m = re.search(r'_(\d{8})_(\d{6})$', e['video']); return (m.group(1) + m.group(2) if m else e['video'], e['vt'])
pick.sort(key=tkey)
# --- ピッチ曲線（あれば） ---
PD = 'dataset/pitch_pyin_20260924'; f0cache = {}; n_pitch = 0
def load_f0(stem):
    if stem in f0cache: return f0cache[stem]
    p = os.path.join(PD, stem + '_f0.csv'); f0cache[stem] = None
    if os.path.exists(p):
        f0cache[stem] = [(float(r['audio_time_sec']), float(r['f0_hz'])) for r in csv.DictReader(open(p, encoding='utf-8-sig'))]
    return f0cache[stem]
for e in pick:
    f = load_f0(e['wav'][:-4]) if e['wav'] else None
    if f:
        lo, hi = e['at'] - 0.5, e['at'] + 1.5
        seg = [round(v, 1) for t, v in f if lo <= t < hi]
        if len(seg) >= 100: e['pitch'] = seg; n_pitch += 1
print('pitch attached', n_pitch, '/', len(pick), '(0 なら未対応表示)')
# --- 台紙＝review_sample_20260812_200.html（v1.6 clips3 pts）を2秒窓に合わせて書き換え ---
tmpl = open('review_sample_20260812_200.html', encoding='utf-8').read()
ti = tmpl.find('const EV='); tj = tmpl.find('];', ti)
ev_json = json.dumps([{k: v for k, v in e.items() if k not in ('wav', 'at')} for e in pick], ensure_ascii=False, separators=(',', ':'))
out = tmpl[:ti] + 'const EV=' + ev_json + tmpl[tj + 1:]
def rep(old, new):
    global out
    assert old in out, 'template mismatch: ' + old[:60]
    out = out.replace(old, new)
rep('kensyu-sample-0812', 'kensyu-sample-0924'); rep('kensyu_sample0812_', 'kensyu_sample0924_')
rep('<title>技法イベント検収ツール v1.3</title>', '<title>検収サンプル 0924（200件）</title>')
rep('>v1.6 clips3 pts<', '>v1.7 clips3 pts 2秒窓（0924）<')
rep('基準（v1.5・9/9）：<b>赤帯（最後の1秒）が出る前に', '基準（v1.5・9/9）：<b>赤帯（最後の0.5秒）が出る前に')
rep('（クリップ＝許容窓・赤点線は参考の推定加点時刻）', '（クリップ＝推定加点時刻の−0.5〜+1.5秒・赤線は推定加点時刻＝0.5秒位置・ピッチ曲線は参考）')
rep('${e.d>1?"（+"+e.d+"）":""}', '')  # 合算イベントは除外済み・「+n」は出さない
rep('const x=i=>i/ns.length*W*((e.dur||2.0)/((e.dur||2.0)+1.0));', 'const x=i=>i/ns.length*W;  // ピッチ配列＝クリップ全体[vt-0.5,vt+1.5]と同じ2秒')
rep('const evX=W*(1.0/((e.dur||2.0)+1.0));', 'const evX=W*0.25;  // 加点推定時刻＝クリップの0.5秒位置\n  ctx.fillStyle="rgba(200,40,40,0.18)";ctx.fillRect(W*0.75,0,W*0.25,H);  // 赤帯区間（最後の0.5秒）')
rep('// --- 加点時刻マーカー(クリップ窓[T-1, T+dur-1]と同比率・参考表示) ---', '// --- 加点時刻マーカー(クリップ窓[T-0.5, T+1.5]と同比率・参考表示) ---')
rep('ctx.fillText("（この曲のピッチ曲線は未生成）",10*dpr,20*dpr);return}', 'ctx.fillText("ピッチ曲線：未対応（この曲の音声ピッチ未生成・判定には不要）",10*dpr,20*dpr);return}')
name = f'review_sample_{sess}_{len(pick)}.html'
open(name, 'w', encoding='utf-8', newline='\n').write(out)
if a.absclips:
    base = 'file:///G:/研究アーカイブ/成果物/動画解析/clips3/'
    o2 = out.replace('vid.src="clips3/"+e.id+".mp4?v="+Date.now();', 'vid.src="' + base + '"+encodeURIComponent(e.id)+".mp4?v="+Date.now();')
    assert o2 != out; open(a.absclips, 'w', encoding='utf-8', newline='\n').write(o2); print('WROTE abs copy', a.absclips)
man = f'dataset/kensyu_sample_{sess}_{len(pick)}_manifest.csv'
with open(man, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['event_id', 'video', 'counter', 'count_value', 'vt', 'seed', 'generated', 'wav', 'audio_time_sec', 'has_pitch'])
    for e in pick: w.writerow([e['id'], e['video'], e['c'], e['n'], e['vt'], seed, datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), e['wav'], e['at'], int(bool(e['pitch']))])
print('WROTE', name, 'items', len(pick), 'manifest', man, 'seed', seed)
missing = [e['id'] for e in pick if not os.path.exists(os.path.join('clips3', e['id'] + '.mp4'))]
print('clips3 missing', len(missing), '-> python scripts/gen_clips_v4_2s_20260924.py', man)
print('by counter', Counter(e['c'] for e in pick)); print('by video', Counter(e['video'] for e in pick))
