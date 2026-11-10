"""Mocked regression of original Pygame human renderer integration."""
import argparse
import sys
import types
from pathlib import Path
import numpy as np

class FakeEnv:
    def __init__(self):
        self.actions = []
        self.closed = False
    def reset(self, seed=None):
        self.actions.clear()
        return self.obs(), {'time_elapsed': 0.0}
    def obs(self):
        return {'mario': np.array([200, 200, 248, 248, 0, 0, 0, 0], dtype=np.float32),
                'bullet_bills': np.zeros((0, 5), dtype=np.float32)}
    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        n = len(self.actions)
        return self.obs(), 0., n == 3, False, {'time_elapsed': n / 15, 'game_state': 'game_over' if n == 3 else 'playing'}
    def close(self): self.closed = True

created = []
def fake_make(**kwargs):
    assert kwargs == {'id':'kymnasium/BulletBill-Continuous-Hard-Stage-2', 'render_mode':'human', 'bgm':False}
    e=FakeEnv();created.append(e);return e

gym=types.ModuleType('gymnasium');gym.make=fake_make
sys.modules['gymnasium']=gym
sys.modules['kymnasium']=types.ModuleType('kymnasium')
pygame=types.ModuleType('pygame')
pygame.QUIT=256; pygame.KEYDOWN=768; pygame.K_ESCAPE=27
pygame.event=types.SimpleNamespace(get=lambda: [])
sys.modules['pygame']=pygame

import watch_ppo
args=argparse.Namespace(model=str(Path(__file__).parent/'models/mario_ppo_init.pt'), render='human',
                        map=str(Path(__file__).parent/'map_data/object_tiles.csv'), episodes=2, steps=4,
                        seed=1, fps=1e6, stochastic=False, allow_untrained=True, show_actions=False, print_every=15)
watch_ppo.run(args)
assert len(created)==1 and created[0].closed
assert len(created[0].actions)==3
for action in created[0].actions:
    assert action.shape==(2,) and action.dtype==np.float32
print('Original human renderer checkpoint playback mock passed!')
