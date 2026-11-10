"""Paired-seed evaluation and prediction-audit summary for models2 v4.1.

Never interprets shadow counterfactual proposals as verified successful rescues.
"""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path


def load(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not rows or 'seed' not in rows[0] or 'survival_s' not in rows[0]:
        raise ValueError('Expected episodes.csv with seed and survival_s: '+str(path))
    out={}
    for r in rows:
        seed=int(r['seed'])
        if seed in out: raise ValueError('Duplicate seed: '+str(seed))
        out[seed]=r
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1',required=True)
    p.add_argument('--v4',required=False)
    p.add_argument('--v41',required=True)
    p.add_argument('--out',default=None)
    a=p.parse_args()
    v1=load(a.v1); v41=load(a.v41)
    v4=load(a.v4) if a.v4 else {}
    common=sorted(set(v1).intersection(v41))
    if not common: p.error('No shared seeds between v1 and v4.1')
    if set(v1) != set(v41):
        print('NOTE: non-overlapping seeds omitted (must match for paired comparisons)')
    rows=[]
    for seed in common:
        base=float(v1[seed]['survival_s']); new=float(v41[seed]['survival_s'])
        row={'seed':seed,'v1_s':base,'v41_s':new,'delta_s':new-base,
             'v41_overrides':int(float(v41[seed].get('v41_overrides',0))),
             'v41_proposals':int(float(v41[seed].get('v41_proposals',0)))}
        if seed in v4: row['v4_s']=float(v4[seed]['survival_s'])
        rows.append(row)
    diffs=[r['delta_s'] for r in rows]
    survivors=[r['v41_s'] for r in rows]
    result={
        'shared_seed_count':len(common),
        'mean_v1_s':statistics.mean(r['v1_s'] for r in rows),
        'mean_v41_s':statistics.mean(survivors),
        'median_v41_s':statistics.median(survivors),
        'paired_delta_s':statistics.mean(diffs),
        'wins':sum(d>0 for d in diffs),
        'losses':sum(d<0 for d in diffs),
        'ties':sum(d==0 for d in diffs),
        'reached_120s':sum(s>=120 for s in survivors),
        'reached_150s':sum(s>=150 for s in survivors),
        'reached_180s':sum(s>=180 for s in survivors),
        'all_rows':rows,
        'note':'Compare predeclared holdout seeds for claims of generalization; 5 seeds cannot establish 150s mean.'
    }
    if v4 and all(x in v4 for x in common):
        result['mean_v4_s']=statistics.mean(float(v4[x]['survival_s']) for x in common)
    print('SEED      V1(s)    V4.1(s)     DELTA     OVERRIDES')
    for r in rows:
        print(f"{r['seed']}  {r['v1_s']:8.3f} {r['v41_s']:9.3f} {r['delta_s']:+9.3f} {r['v41_overrides']:8d}")
    print(json.dumps({k:v for k,v in result.items() if k!='all_rows'},indent=2,ensure_ascii=False))
    if a.out:
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
