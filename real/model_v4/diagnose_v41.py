"""Action-to-motion audit on the actual BulletBill environment, paired seeds."""
from __future__ import annotations
import argparse,csv,json,random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from eval_v41 import load_model,make_env,get_time
from model_v41_net import obs_vector,env_action

def run(args, env_factory=make_env):
    torch.set_num_threads(1)
    model,_=load_model(args.model,torch.device('cpu'))
    seeds=random.Random(args.seed).sample(range(1,2**31-1),args.episodes)
    rows=[];episodes=[]
    for mode in ('deterministic','stochastic'):
        env=env_factory('none')
        try:
            for ep,seed in enumerate(seeds,1):
                obs,info=env.reset(seed=seed)
                torch.manual_seed(seed)
                tally=Counter();dxs=[]
                for step in range(args.max_steps):
                    with torch.inference_mode():
                        features=torch.from_numpy(obs_vector(obs)).unsqueeze(0)
                        j,m,_,_=model.act(features,deterministic=(mode=='deterministic'))
                        ji,mi=int(j.item()),int(m.item())
                    action=env_action(mi,ji)
                    old=np.asarray(obs['mario'],np.float64).ravel()
                    next_obs,_,terminated,truncated,info=env.step(action)
                    new=np.asarray(next_obs['mario'],np.float64).ravel()
                    dx=float((new[0]+new[2]-old[0]-old[2])/2)
                    tally[mi]+=1;dxs.append(dx)
                    rows.append({'mode':mode,'episode':ep,'seed':seed,'step':step+1,'move_index':mi,
                                 'sent_move':float(action[0]),'jump_index':ji,'sent_jump':float(action[1]),'actual_dx':dx})
                    obs=next_obs
                    if terminated or truncated:break
                seconds,_=get_time(info,len(dxs),args.game_fps)
                record={'mode':mode,'episode':ep,'seed':seed,'survival_s':seconds,'steps':len(dxs),
                        'move_left':tally[0],'move_neutral':tally[1],'move_right':tally[2],
                        'mean_abs_dx':float(np.mean(np.abs(dxs))) if dxs else 0.,
                        'zero_dx_pct':float(np.mean(np.abs(dxs)<.01)*100) if dxs else 0.}
                episodes.append(record)
                print('EPISODE '+json.dumps(record,ensure_ascii=False),flush=True)
        finally:env.close()
    summary={mode:{'episodes':len([r for r in episodes if r['mode']==mode]),
                   'mean_survival_s':float(np.mean([r['survival_s'] for r in episodes if r['mode']==mode])),
                   'move_counts':{str(k):sum(r[['move_left','move_neutral','move_right'][k]] for r in episodes if r['mode']==mode) for k in range(3)},
                   'mean_abs_dx':float(np.mean([r['mean_abs_dx'] for r in episodes if r['mode']==mode]))}
             for mode in ('deterministic','stochastic')}
    args.out.mkdir(parents=True,exist_ok=True)
    for name,items in (('actions.csv',rows),('episodes.csv',episodes)):
        with (args.out/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('SUMMARY '+json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parent/'model_v41_untrained.pt')
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--max-steps',type=int,default=9000)
    p.add_argument('--game-fps',type=float,default=50)
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'eval_results'/'v41_diagnostic')
    return p.parse_args(argv)

if __name__=='__main__':run(parse_args())
