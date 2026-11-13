"""Evaluate a standalone PPO-v10-compatible .pt with no v4.1 runtime overlay."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import agent_ppo_v10 as v10


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--seed-file',required=True)
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--steps',type=int,default=9000)
    p.add_argument('--render',choices=('none','human'),default='none')
    p.add_argument('--fps',type=float,default=0.0)
    p.add_argument('--out',default='logs/models2_distilled_eval')
    a=p.parse_args()
    if a.episodes<1 or a.steps<1:p.error('episodes and steps must be positive')
    seeds=[int(x) for x in json.loads(Path(a.seed_file).read_text(encoding='utf-8'))]
    if len(seeds)<a.episodes:raise ValueError('Insufficient seeds in --seed-file')
    torch.set_num_threads(1)
    agent=v10.BulletBillAgent.load(a.model)
    agent.deterministic=True
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    env=gym.make(id=v10.ENV_ID,render_mode=a.render,bgm=False)
    clock=None
    if a.render=='human':
        import pygame
        clock=pygame.time.Clock()
    rows=[]
    try:
        for i,seed in enumerate(seeds[:a.episodes]):
            obs,info=env.reset(seed=seed)
            agent.fe.reset()
            termination='step_limit'
            for step in range(a.steps):
                if a.render=='human':
                    import pygame
                    for event in pygame.event.get():
                        if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
                            termination='user_quit';break
                    if termination=='user_quit':break
                action=agent.act(obs,info)
                obs,rew,terminated,truncated,info=env.step(action)
                if clock is not None and a.fps>0:clock.tick(a.fps)
                if terminated or truncated:
                    termination='terminated' if terminated else 'truncated'
                    break
            elapsed=float(info.get('time_elapsed',(step+1)*v10.DT))
            rows.append(dict(seed=seed,survival_s=elapsed,steps=step+1,status=termination))
            print(f'EPISODE {i+1} seed={seed} survival={elapsed:.3f}s status={termination}',flush=True)
            if termination=='user_quit':break
    finally:
        env.close()
        if a.render=='human':
            import pygame
            pygame.quit()
    with (out/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['seed','survival_s','steps','status']);w.writeheader();w.writerows(rows)
    summary=dict(model=a.model,seeds=seeds[:len(rows)],mean_s=float(np.mean([r['survival_s'] for r in rows])),
      median_s=float(np.median([r['survival_s'] for r in rows])),
      reached_120=sum(r['survival_s']>=120 for r in rows),
      reached_150=sum(r['survival_s']>=150 for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
