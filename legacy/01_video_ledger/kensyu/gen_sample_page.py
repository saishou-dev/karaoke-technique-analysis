# -*- coding: utf-8 -*-
"""検収サンプルページ生成 v2（2026-09-10）。除外＝台帳済み・合算イベント（d>1）・凍結動画。vtはpts補正（dataset/pts_shift_by_event_20260909.csv）。クリップ＝clips3（不足分は gen_clips_v3_redband.py で生成）。台紙＝review_sample_20260807_821.html（基準v1.5・赤帯・キャッシュ対策）。
セッションを指定→既検収（dataset/kensyu_reviewed_registry.csv）を除外→固定シードで無作為200件→時系列順→review_sample_<sess>_200.html を出力。
使い方: python gen_sample_page.py 20260812 [--n 200] [--seed 20260812]
台紙＝review_sample_0807_200.html（KEY・書き出し名・見出しの0807を置換）。EVの元＝review_20260902.html（全5,476件・pitch配列付き）。
"""
import re, json, csv, os, sys, random, argparse, datetime
ROOT = r'G:\研究アーカイブ\成果物\動画解析'
ap = argparse.ArgumentParser(); ap.add_argument('session'); ap.add_argument('--n', type=int, default=200); ap.add_argument('--seed', type=int, default=None)
a = ap.parse_args(); sess = a.session; seed = a.seed or int(sess)
os.chdir(ROOT)
src = open('review_20260902.html', encoding='utf-8', errors='replace').read()
i = src.find('const EV='); j = src.find('];', i)
EV = json.loads(src[i + len('const EV='):j + 1])
print('EV total', len(EV))
reg = set(r['event_id'] for r in csv.DictReader(open('dataset/kensyu_reviewed_registry.csv', encoding='utf-8-sig')))
FROZEN = {'虹_20260628_112135'}  # D8: 演出遮蔽で読取11%・凍結
shift = {r['event_id']: float(r['shift_sec']) for r in csv.DictReader(open('dataset/pts_shift_by_event_20260909.csv', encoding='utf-8-sig'))}
for e in EV:
    if e['id'] in shift: e['vt'] = round(e['vt'] + shift[e['id']], 3)
pool = [e for e in EV if e['sess'] == sess and e['id'] not in reg and int(e.get('d', 1)) == 1 and e['video'] not in FROZEN]
print('excluded d>1', sum(1 for e in EV if e['sess'] == sess and int(e.get('d', 1)) > 1), 'frozen', sum(1 for e in EV if e['sess'] == sess and e['video'] in FROZEN))
print('session', sess, 'pool(unreviewed, clip exists)', len(pool), 'reviewed in session', sum(1 for e in EV if e['sess'] == sess and e['id'] in reg))
rnd = random.Random(seed)
pick = rnd.sample(pool, min(a.n, len(pool)))
pick.sort(key=lambda e: (e['video'], e['vt']))
tmpl = open('review_sample_20260807_821.html', encoding='utf-8').read()
ti = tmpl.find('const EV='); tj = tmpl.find('];', ti)
out = tmpl[:ti] + 'const EV=' + json.dumps(pick, ensure_ascii=False, separators=(',', ':')) + tmpl[tj + 1:]
out = out.replace('kensyu-sample-0807', f'kensyu-sample-{sess[4:]}').replace('kensyu_sample0807_', f'kensyu_sample{sess[4:]}_')
out = re.sub(r'0807', sess[4:], out) if sess[4:] != '0807' else out
name = f'review_sample_{sess}_{len(pick)}.html'
open(name, 'w', encoding='utf-8').write(out)
man = f'dataset/kensyu_sample_{sess}_{len(pick)}_manifest.csv'
with open(man, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['event_id', 'video', 'counter', 'count_value', 'vt', 'seed', 'generated'])
    for e in pick: w.writerow([e['id'], e['video'], e['c'], e['n'], e['vt'], seed, datetime.datetime.now().strftime('%Y-%m-%d %H:%M')])
print('WROTE', name, 'items', len(pick), 'manifest', man, 'seed', seed)
missing = [e['id'] for e in pick if not os.path.exists(os.path.join('clips3', e['id'] + '.mp4'))]
print('clips3 missing', len(missing), '→ python scripts/gen_clips_v3_redband.py', name)
from collections import Counter
print('by counter', Counter(e['c'] for e in pick)); print('by video', Counter(e['video'] for e in pick))
