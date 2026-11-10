"""Held-out evaluation of PPO policy; no training, no policy tuning on holdout."""
from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from models.ppo_policy import ACTION_TABLE, StateEncoder, load_platforms, load_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--steps',type=int,default=900)
    p.add_argument('--seed',type=int,default=20261111)
    p.add_argument('--map',default='map_data/object_tiles.csv')
    p.add_argument('--out',default='logs/ppo_eval')
    p.add_argument('--stochastic',action='store_true')
    args=p.parse_args()
    if args.episodes<1 or args.steps<1: p.error('episodes and steps must be positive')
    net,ck=load_policy(args.model)
    if not ck.get('trained'):
        raise SystemExit('REFUSING TO EVALUATE AS TRAINED: this .pt contains only initialized weights.')
    from agent_env import create_env
    enc=StateEncoder(load_platforms(args.map));rng=random.Random(args.seed)
    rows=[]
    for epi in range(args.episodes):
        env=create_env(render_mode='none')
        seed=rng.randrange(2**32)
        compute_s=0.; info={};terminated=truncated=False
        try:
            obs,info=env.reset(seed=seed)
            for step in range(args.steps):
                st=enc.encode(obs,float(info.get('time_elapsed',0.)))
                t0=time.perf_counter()
                with torch.no_grad():
                    a,_,_=net.act(torch.as_tensor(st).unsqueeze(0),deterministic=not args.stochastic)
                compute_s+=time.perf_counter()-t0
                obs,_,terminated,truncated,info=env.step(ACTION_TABLE[int(a.item())].copy())
                if terminated or truncated:break
            rows.append({'episode':epi,'env_seed':seed,'time_s':float(info['time_elapsed']),
                         'game_state':info.get('game_state','unknown'),
                         'terminated':bool(terminated),'truncated':bool(truncated),
                         'decisions':step+1,'compute_s':compute_s})
        finally:env.close()
        if (epi+1)%10==0:print('PROGRESS',epi+1,'/',args.episodes,flush=True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'episode_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'episodes':len(rows),'model':args.model,'total_env_steps':ck['total_env_steps'],
             'mean_capped_survival_s':float(np.mean([r['time_s'] for r in rows])),
             'survivors_at_60s':sum(r['time_s']>=60.-1e-5 for r in rows) if args.steps>=900 else None,
             'mean_policy_compute_ms_per_decision':1000*sum(r['compute_s'] for r in rows)/sum(r['decisions'] for r in rows),
             'evaluation_seed':args.seed,'stochastic':args.stochastic,
             'warning':'Time limit censoring; use independent validation and holdout seeds.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('SUMMARY',summary)

if __name__=='__main__':main()
