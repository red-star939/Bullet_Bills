"""Test spawn-based PPO vector environment and GAE with synthetic game. NOT game performance."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
from models.ppo_policy import load_policy
from train_ppo_parallel import evaluate_workers_gae, train


class FakeEnv:
    def __init__(self):
        self.count = 0
    def reset(self, seed=None):
        self.count = 0
        return self._obs(), {'time_elapsed': 0., 'game_state': 'playing'}
    def _obs(self):
        return {'mario': np.asarray([600., 300., 648., 348., 0., 0., 0., 0.], np.float32),
                'bullet_bills': np.zeros((0,5), np.float32)}
    def step(self, action):
        assert action.dtype == np.float32 and action.shape == (2,)
        self.count += 1
        done = self.count == 5
        return self._obs(), 0., done, False, {
            'time_elapsed': self.count / 15., 'game_state': 'game_over' if done else 'playing'}
    def close(self):
        pass


def make_fake_env(render_mode='none'):
    assert render_mode == 'none'
    return FakeEnv()


def test():
    # GAE must not propagate into the next episode or a different env.
    data = [[dict(state=np.zeros(79,np.float32),action=0,logp=-2.,reward=1.,value=0.,
                  next_value=0., terminated=False, boundary=False),
             dict(state=np.zeros(79,np.float32),action=0,logp=-2.,reward=1.,value=0.,
                  next_value=0., terminated=True, boundary=True)],
            [dict(state=np.zeros(79,np.float32),action=0,logp=-2.,reward=3.,value=0.,
                  next_value=0., terminated=False, boundary=True)]]
    batch = evaluate_workers_gae(data, 1., 1.)
    assert np.allclose(batch['advantages'], [2., 1., 3.])
    with tempfile.TemporaryDirectory() as d:
        a = argparse.Namespace(num_envs=4, torch_threads=1,
            env_factory='test_ppo_parallel_smoke:make_fake_env',
            total_steps=68, rollout_steps=24, max_episode_steps=20,
            lr=3e-4, epochs=2, minibatch=8, clip=.2, vf_coef=.5, ent_coef=.01,
            max_grad_norm=.5, target_kl=.03, gamma=.995, gae_lambda=.95,
            alive_reward=.01, death_penalty=-1., timeout_bonus=0., success_bonus=1.,
            seed=123, device='cpu', map=str(Path(__file__).parent / 'map_data/object_tiles.csv'),
            out=d, resume=None, log_interval=24, save_interval=24)
        train(a)
        _, ck = load_policy(Path(d)/'mario_ppo_latest.pt')
        assert ck['trained'] and ck['total_env_steps']==68
        assert ck['extra']['args']['num_envs']==4
        a.resume=str(Path(d)/'mario_ppo_latest.pt')
        a.total_steps=80
        train(a)
        _, ck=load_policy(Path(d)/'mario_ppo_latest.pt')
        assert ck['total_env_steps']==80
    print('Spawn parallel PPO smoke, GAE boundaries, checkpoint/resume passed!')


if __name__=='__main__':
    test()
