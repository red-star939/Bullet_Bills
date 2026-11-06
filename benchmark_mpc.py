"""Paired seed baseline ActionEvaluator vs beam-search MPC. Does not use oracle information."""
from __future__ import annotations
import argparse,csv,random,time,statistics,json
from pathlib import Path
import numpy as np
from agent_env import create_env
from models.action_evaluator import ActionEvaluator
from models.mpc_controller import MPCController
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor


def run_episode(seed,steps,controller):
    env=create_env(render_mode='none');compute=0.;decisions=0;swaps=0;prev=None
    try:
        obs,info=env.reset(seed=seed)
        for step in range(steps):
            t=time.perf_counter();r=controller.evaluate(obs,current_action=prev)
            compute+=time.perf_counter()-t;decisions+=1
            a=r.best_action;swaps+=int(prev is not None and a!=prev)
            obs,_,term,trunc,info=env.step(np.asarray(a,dtype=np.float32));prev=a
            if term or trunc:break
        return dict(time_s=float(info['time_elapsed']),game_state=info.get('game_state','unknown'),
            decisions=decisions,compute_s=compute,switches=swaps)
    finally:env.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--segments',type=int,nargs='+',default=[2,3,3])
    p.add_argument('--beam-width',type=int,default=3)
    p.add_argument('--max-nodes',type=int,default=1000)
    p.add_argument('--out',default='logs/mpc_benchmark')
    args=p.parse_args()
    if args.episodes<1 or args.steps<1 or args.beam_width<1:p.error('invalid positive argument')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    sim=ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv('map_data/object_tiles.csv'))
    baseline=ActionEvaluator(sim,horizon_steps=sum(args.segments),robust_mode='none')
    mpc=MPCController(sim,segment_steps=args.segments,beam_width=args.beam_width,max_nodes=args.max_nodes)
    rng=random.Random(args.seed);rows=[]
    for ep in range(args.episodes):
        seed=rng.randrange(2**32);rng.randrange(2**32)
        b=run_episode(seed,args.steps,baseline);m=run_episode(seed,args.steps,mpc)
        rows.append(dict(episode=ep,env_seed=seed,
            **{'baseline_'+k:v for k,v in b.items()},
            **{'mpc_'+k:v for k,v in m.items()},delta_time_s=m['time_s']-b['time_s']))
        print('PROGRESS',ep+1,'/',args.episodes,'baseline',round(statistics.mean(r['baseline_time_s'] for r in rows),3),
              'mpc',round(statistics.mean(r['mpc_time_s'] for r in rows),3),flush=True) if (ep+1)%5==0 else None
    with (out/'episode_pairs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary=dict(episodes=args.episodes,configuration=vars(args),
        baseline_mean_time_s=statistics.mean(r['baseline_time_s'] for r in rows),
        mpc_mean_time_s=statistics.mean(r['mpc_time_s'] for r in rows),
        mpc_wins=sum(r['delta_time_s']>0 for r in rows),ties=sum(r['delta_time_s']==0 for r in rows),
        baseline_mean_compute_ms_per_decision=1000*sum(r['baseline_compute_s'] for r in rows)/sum(r['baseline_decisions'] for r in rows),
        mpc_mean_compute_ms_per_decision=1000*sum(r['mpc_compute_s'] for r in rows)/sum(r['mpc_decisions'] for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL SUMMARY',json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
