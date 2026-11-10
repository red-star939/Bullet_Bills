"""Reproducible v4 experiment matrix with separate tuning and holdout seeds.

Runs never modify checkpoints. Scores are real game outcomes, not predicted scores.
Use --quick to perform the 5-seed smoke test; default includes holdout.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, median


def run_one(args, label, seed, n, options):
    out = Path(args.out) / f'{label}_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, 'watch_models2_v10_v4.py', '--model',args.model,
           '--altitude', 'v1' if label == 'baseline_v1' else 'v4',
           '--render','none','--episodes',str(n),'--steps',str(args.steps),
           '--seed',str(seed),'--profile-every','0','--out',str(out),
           '--v4-mode', 'shadow' if label=='baseline_v1' else label.split('_')[0]] + options
    print('RUN', label, 'seed',seed,'episodes',n,flush=True)
    subprocess.run(cmd,check=True)
    with (out/'episodes.csv').open(encoding='utf-8',newline='') as f:
        rows = list(csv.DictReader(f))
    return {int(r['seed']): float(r['survival_s']) for r in rows}


def summarize(base, experiment, label):
    shared = sorted(set(base) & set(experiment))
    if not shared:
        raise RuntimeError('No matching seeds')
    b = [base[i] for i in shared]; e = [experiment[i] for i in shared]
    differences=[x-y for x,y in zip(e,b)]
    return {'variant':label,'n':len(shared),'mean_s':mean(e),'median_s':median(e),
            'baseline_mean_s':mean(b),'paired_mean_delta_s':mean(differences),
            'wins':sum(x>0 for x in differences),'reached_120':sum(x>=120 for x in e),
            'reached_150':sum(x>=150 for x in e),'reached_180':sum(x>=180 for x in e),
            'target_150_met':mean(e)>=150}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='models/bullet_bill_ppo_v10_best.pt')
    p.add_argument('--out',default='logs/models2_v4_benchmark')
    p.add_argument('--seed',type=int,default=20261110)
    p.add_argument('--steps',type=int,default=9000)
    p.add_argument('--train-seeds',type=int,default=5)
    p.add_argument('--holdout-seeds',type=int,default=15)
    p.add_argument('--quick',action='store_true')
    a=p.parse_args()
    if a.train_seeds<1 or (not a.quick and a.holdout_seeds<1):p.error('At least one seed required')
    root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
    profiles={'active_default':['--v4-replan-every','3'],
              'explore_conservative':['--v4-replan-every','3','--v4-min-gain','27'],
              'explore_proactive':['--v4-replan-every','2','--v4-min-gain','15',
                                    '--v4-lower-min','9','--v4-lower-advantage','3']}
    train_base=run_one(a,'baseline_v1',a.seed,a.train_seeds,[])
    results=[]
    records={}
    for label, options in profiles.items():
        # label prefixes map directly to v4 modes.
        e=run_one(a,label,a.seed,a.train_seeds,options)
        scores=summarize(train_base,e,label)
        results.append(scores);records[label]=e
        print('TUNE',json.dumps(scores,ensure_ascii=False),flush=True)
    # Choose only on tuning seeds; never tune by peeking at holdout.
    ranked=sorted(results,key=lambda x:x['paired_mean_delta_s'],reverse=True)
    selected=ranked[0]['variant'] if ranked[0]['paired_mean_delta_s']>0 else 'baseline_v1'
    report={'target_mean_s':150,'train':results,'selected':selected,
            'selection_rule':'highest positive paired mean; otherwise keep v1'}
    if not a.quick:
        start=a.seed+10000
        held_base=run_one(a,'baseline_v1',start,a.holdout_seeds,[])
        if selected=='baseline_v1':held_exp=held_base
        else:held_exp=run_one(a,selected,start,a.holdout_seeds,profiles[selected])
        report['holdout']=summarize(held_base,held_exp,selected)
        report['holdout']['validated_improvement']=report['holdout']['paired_mean_delta_s']>0
        print('HOLDOUT',json.dumps(report['holdout'],ensure_ascii=False),flush=True)
    (root/'benchmark_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('SELECTED',selected,'RESULT',root/'benchmark_summary.json')

if __name__=='__main__':main()
