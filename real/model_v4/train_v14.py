"""Standalone Model V14 launcher. Run from mario_pj project root."""
import argparse
from pathlib import Path
import torch
from agent_ppo_v14_model_v4 import train, OBS_DIM, ActorCritic

HERE = Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--init',type=Path,default=HERE/'model_v14_initial.pt')
    p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--eval-interval',type=int,default=16384)
    p.add_argument('--eval-episodes',type=int,default=10)
    p.add_argument('--imit0',type=float,default=1.0)
    p.add_argument('--imit1',type=float,default=0.05)
    p.add_argument('--tag',default='')
    a=p.parse_args()
    if not a.init.is_file(): raise FileNotFoundError(a.init)
    if a.steps<1 or a.envs<1 or a.eval_interval<1 or a.eval_episodes<1:raise ValueError('Steps/envs/eval must be positive')
    ck=torch.load(a.init,map_location='cpu',weights_only=True)
    if ck.get('obs_dim')!=OBS_DIM or 'pair_head.weight' not in ck['model']:
        raise ValueError('V14 178D/15-action checkpoint required; PPO v10 and old V4 not compatible')
    import secrets
    seed=a.seed if a.seed is not None else secrets.randbelow(2**31-1)
    torch.set_num_threads(1)
    print(f'V14_NEW_MODEL seed={seed} init={a.init} envs={a.envs} requested_steps={a.steps} actual_steps_rounded_to_4096',flush=True)
    train(total_steps=a.steps,seed=seed,tag=a.tag,n_envs=a.envs,init=str(a.init),lr=a.lr,
          imit0=a.imit0,imit1=a.imit1,eval_interval=a.eval_interval,
          eval_episodes=a.eval_episodes,log_every=2,output_dir=str(HERE),checkpoint_dir=str(HERE/'checkpoints'))

if __name__=='__main__':main()
