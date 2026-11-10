"""Offline integration: fake env to test actual rollout -> PPO update -> .pt, NOT game performance."""
from __future__ import annotations
import argparse
import sys
import tempfile
import types
from pathlib import Path
import numpy as np
from train_ppo import train
from models.ppo_policy import load_policy

class FakeEnv:
    def __init__(self):self.n=0
    def reset(self,seed=None):
        self.n=0
        return self._obs(),{'time_elapsed':0.,'game_state':'playing'}
    def _obs(self):
        return {'mario':np.array([600,300,648,348,0,0,0,0],np.float32),
                'bullet_bills':np.zeros((0,5),np.float32)}
    def step(self,action):
        assert action.shape==(2,) and action.dtype==np.float32
        self.n+=1
        done=self.n==5
        return self._obs(),0.,done,False,{'time_elapsed':self.n/15.,'game_state':'game_over' if done else 'playing'}
    def close(self):pass


def test():
    sys.modules['agent_env']=types.SimpleNamespace(create_env=lambda render_mode='none':FakeEnv())
    with tempfile.TemporaryDirectory() as d:
        args=argparse.Namespace(seed=123,device='cpu',map=str(Path(__file__).parent/'map_data/object_tiles.csv'),
           out=d,total_steps=32,rollout_steps=16,max_episode_steps=100,
           lr=3e-4,epochs=2,minibatch=8,clip=.2,vf_coef=.5,ent_coef=.01,
           max_grad_norm=.5,target_kl=.03,gamma=.995,gae_lambda=.95,
           alive_reward=.01,death_penalty=-1.,timeout_bonus=0.,success_bonus=1.,
           log_interval=16,save_interval=16,resume=None)
        train(args)
        net,ck=load_policy(Path(d)/'mario_ppo_latest.pt')
        assert ck['trained'] and ck['total_env_steps']==32
        assert 'optimizer_state_dict' in ck
        # A checkpoint should also work when resumed for another rollout.
        args.resume=str(Path(d)/'mario_ppo_latest.pt')
        args.total_steps=48
        train(args)
        _,ck=load_policy(Path(d)/'mario_ppo_latest.pt')
        assert ck['total_env_steps']==48
    print('PPO fake-environment rollout / train / resume smoke passed!')

if __name__=='__main__':test()
