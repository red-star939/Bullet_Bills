"""Headless viewer smoke: fake game env, real PPO policy checkpoint."""
from __future__ import annotations
import argparse
import sys
import types
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

class FakeEnv:
    def __init__(self):
        self.n = 0
        self.closed = False
    def reset(self, seed=None):
        self.n = 0
        return self.obs(), {'time_elapsed': 0.0}
    def obs(self):
        return {'mario': np.array([200, 200, 248, 248, 0, 0, 0, 0], np.float32),
                'bullet_bills': np.zeros((0,5), np.float32)}
    def step(self, action):
        assert np.asarray(action).shape == (2,)
        self.n += 1
        return self.obs(), 0., self.n == 3, False, {'time_elapsed':self.n*4/60, 'game_state':'game_over' if self.n == 3 else 'playing'}
    def render(self):
        return None
    def close(self):
        self.closed = True

mod = types.ModuleType('agent_env')
mod.create_env = lambda render_mode='human': FakeEnv()
sys.modules['agent_env'] = mod

from watch_ppo import run
args = argparse.Namespace(model=str(BASE/'models/mario_ppo_init.pt'), render='human',
                          map=str(BASE/'map_data/object_tiles.csv'), episodes=2,
                          steps=5, seed=12, fps=100000, stochastic=False,
                          allow_untrained=True, show_actions=False, print_every=15)
run(args)
print('watch_ppo fake-env smoke passed!')
