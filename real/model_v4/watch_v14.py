"""Model V14 visual/headless simulation with distinct seeds."""
import argparse, csv, json, secrets
from pathlib import Path
import numpy as np
from agent_ppo_v14_model_v4 import BulletBillAgent, ENV_ID

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--render',choices=['human','none'],default='human')
    p.add_argument('--stochastic',action='store_true')
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'eval_results'/'v14')
    a=p.parse_args()
    if a.episodes<1: raise ValueError('--episodes must be positive')
    import gymnasium as gym
    import kymnasium
    agent=BulletBillAgent.load(str(a.model))
    agent.deterministic=not a.stochastic
    env=gym.make(id=ENV_ID,render_mode=a.render,bgm=False)
    a.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(secrets.randbelow(2**63) if a.seed is None else a.seed)
    seeds=rng.choice(2_147_483_000,size=a.episodes,replace=False)
    rows=[]
    try:
        for i,sd in enumerate(seeds):
            obs,info=env.reset(seed=int(sd))
            agent.fe.reset()
            steps=0
            while True:
                obs,_,term,trunc,info=env.step(agent.act(obs,info))
                steps+=1
                if term or trunc:break
            row={'episode':i+1,'seed':int(sd),'steps':steps,'survival_s':float(info.get('time_elapsed',0)),
                 'game_state':str(info.get('game_state',''))}
            print('EPISODE',json.dumps(row,ensure_ascii=False),flush=True)
            rows.append(row)
    finally:
        env.close()
    with (a.out/'episodes.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary={'model':str(a.model),'episodes':len(rows),'deterministic':not a.stochastic,
             'mean_survival_s':float(np.mean([r['survival_s'] for r in rows])),
             'reached120':sum(r['survival_s']>=120 for r in rows),
             'reached150':sum(r['survival_s']>=150 for r in rows)}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
