"""Matched-seed PPO-v10 / known-world Shield / Spawn-Shield evaluation.

All modes use the SAME collaborator checkpoint and v10 FeatureExtractor.
Each row is flushed to CSV immediately. Requires real BulletBill environment.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import random
import statistics
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import gymnasium as gym
import kymnasium
import torch
import agent_ppo_v10 as v10
from models.v10_spawn_guidance import V10SpawnGuidance

FIELDS = ('seed','mode','survival_s','decisions','result','overrides','spawn_overrides',
          'known_world_overrides','wall_step_fraction','runtime_s')
MODES = ('ppo','shield','spawn_shield')


def episode(agent, guide, *, seed, mode, steps, caution_px):
    env = gym.make(id=v10.ENV_ID, render_mode='none', bgm=False)
    agent.fe.reset()
    previous = None
    changed = spawned = known = wall = 0
    last_time = 0.0
    started = time.perf_counter()
    result = 'step_limit'
    try:
        obs, info = env.reset(seed=seed)
        for count in range(1, steps+1):
            original = agent.act(obs, info)
            if mode == 'ppo':
                action = original
                meta = {'overridden':False,'spawn_overridden':False,'known_world_overridden':False}
            else:
                action, meta = guide.decide(obs, original, previous_action=previous, mode=mode)
            prev_x0, _, prev_x1, _ = obs['mario'][:4]
            wall += int(min(float(prev_x0), 1296-float(prev_x1)) < caution_px)
            changed += int(meta['overridden'])
            spawned += int(meta['spawn_overridden'])
            known += int(meta['known_world_overridden'])
            previous = tuple(float(x) for x in action)
            obs, _, terminated, truncated, info = env.step(action)
            last_time = float(info.get('time_elapsed', count*v10.DT))
            if terminated or truncated:
                result = 'terminated' if terminated else 'truncated'
                break
        return dict(seed=seed,mode=mode,survival_s=round(last_time,4),decisions=count,
                    result=result,overrides=changed,spawn_overrides=spawned,
                    known_world_overrides=known,wall_step_fraction=round(wall/count,5),
                    runtime_s=round(time.perf_counter()-started,2))
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='models/bullet_bill_ppo_v10_best.pt')
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--steps',type=int,default=900)
    p.add_argument('--seed',type=int,default=20261110)
    p.add_argument('--map',default='map_data/object_tiles.csv')
    p.add_argument('--caution-px',type=int,default=180)
    p.add_argument('--critical-px',type=int,default=96)
    p.add_argument('--out',default='logs/v10_spawn_comparison')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if args.episodes < 1 or args.steps < 1: p.error('episodes/steps must be positive')
    if not Path(args.model).is_file(): p.error(f'Checkpoint not found: {args.model}')
    torch.set_num_threads(1)
    agent = v10.BulletBillAgent.load(args.model)
    agent.deterministic = True
    guide = V10SpawnGuidance(map_path=args.map,caution_px=args.caution_px,
                             critical_px=args.critical_px)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    csv_path = out/'episodes.csv'
    if csv_path.exists() and not args.resume:
        raise FileExistsError(f'{csv_path} already exists; use a fresh --out or --resume')
    previous_rows = []
    if csv_path.exists():
        with csv_path.open(encoding='utf-8',newline='') as f:
            previous_rows = list(csv.DictReader(f))
    seeds = random.Random(args.seed).sample(range(2**32),args.episodes)
    requested = {(str(s),m) for s in seeds for m in MODES}
    done = {(str(r['seed']),r['mode']) for r in previous_rows}
    if not done.issubset(requested):
        raise ValueError('Resume CSV includes seeds/modes inconsistent with this configuration')
    results = previous_rows.copy()
    for i, seed in enumerate(seeds,1):
        for mode in MODES:
            if (str(seed),mode) in done: continue
            row = episode(agent,guide,seed=seed,mode=mode,steps=args.steps,caution_px=args.caution_px)
            results.append(row)
            with csv_path.open('a',newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f,fieldnames=FIELDS)
                if f.tell()==0:w.writeheader()
                w.writerow(row)
            print(f'RESULT {i}/{len(seeds)} {mode} seed={seed} survival={row["survival_s"]:.3f}s '
                  f'override={row["overrides"]} spawn={row["spawn_overrides"]}',flush=True)
        means = {m:statistics.mean(float(r['survival_s']) for r in results if r['mode']==m)
                 for m in MODES if any(r['mode']==m for r in results)}
        print('PROGRESS',i,'/',len(seeds), ' '.join(f'{m}={v:.3f}' for m,v in means.items()),flush=True)
    summary = {m:dict(n=sum(r['mode']==m for r in results),
                      mean_survival_s=statistics.mean(float(r['survival_s']) for r in results if r['mode']==m),
                      mean_wall_step_fraction=statistics.mean(float(r['wall_step_fraction']) for r in results if r['mode']==m),
                      total_spawn_overrides=sum(int(r['spawn_overrides']) for r in results if r['mode']==m))
               for m in MODES}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
