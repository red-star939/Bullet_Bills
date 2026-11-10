"""Collect observation/action supervision from our verified ActionEvaluator v2.

Output .npz has state rows matching models.ppo_policy.StateEncoder and
8-action indices matching action_evaluator.DEFAULT_ACTIONS.
No engine-private information is used as policy input.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
from models.ppo_policy import ACTION_TABLE, StateEncoder, load_platforms, SCHEMA_VERSION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--steps',type=int,default=900)
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--out',default='data/ppo_teacher_v2.npz')
    p.add_argument('--map',default='map_data/object_tiles.csv')
    p.add_argument('--horizon-steps',type=int,default=8)
    p.add_argument('--extended-horizon-steps',type=int,default=20)
    p.add_argument('--audit-top-k',type=int,default=4)
    args=p.parse_args()
    from agent_env import create_env
    from models.mario_motion_predictor import MarioMotionPredictor
    from models.forward_simulator import ForwardSimulator
    from models.action_evaluator_v2 import ActionEvaluatorV2
    enc=StateEncoder(load_platforms(args.map))
    simulator=ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv(args.map))
    teacher=ActionEvaluatorV2(simulator,horizon_steps=args.horizon_steps,
                              extended_horizon_steps=args.extended_horizon_steps,
                              audit_top_k=args.audit_top_k,robust_mode='none')
    states=[];labels=[];seeds=[];episode_times=[]
    rng=random.Random(args.seed)
    for epi in range(args.episodes):
        seed=rng.randrange(2**32)
        env=create_env(render_mode='none')
        prev_action=None
        try:
            obs,info=env.reset(seed=seed)
            for step in range(args.steps):
                state=enc.encode(obs,float(info.get('time_elapsed',0.)))
                eval_result=teacher.evaluate(obs,current_action=prev_action)
                action=np.asarray(eval_result.best_action,np.float32)
                matches=np.where(np.all(ACTION_TABLE==action,axis=1))[0]
                if len(matches)!=1:
                    raise RuntimeError(f'Teacher chose an action outside discrete action table: {action}')
                states.append(state);labels.append(int(matches[0]));seeds.append(seed)
                prev_action=tuple(float(x) for x in action)
                obs,_,term,trunc,info=env.step(action)
                if term or trunc:break
            episode_times.append(float(info['time_elapsed']))
        finally:env.close()
        if (epi+1)%10==0:print('TEACHER',epi+1,'/',args.episodes,'SAMPLES',len(labels),flush=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,states=np.asarray(states,np.float32),actions=np.asarray(labels,np.int64),
                        env_seeds=np.asarray(seeds,np.uint64),encoder_version=np.asarray(SCHEMA_VERSION),
                        action_table=ACTION_TABLE)
    metadata={'episodes':args.episodes,'samples':len(labels), 'seed':args.seed,
              'mean_teacher_capped_survival_s':float(np.mean(episode_times)),
              'schema':SCHEMA_VERSION, 'actions':len(ACTION_TABLE), 'output':str(out)}
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('TEACHER DATASET',metadata)

if __name__=='__main__':main()
