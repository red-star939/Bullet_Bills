import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_v4_net import ModelV4, env_action, obs_vector
from train_initial_50k import run

class DummyEnv:
    def __init__(self):
        self.t = 0
    def reset(self, seed=None):
        self.t = 0
        return {'mario': np.ones(8, dtype=np.float32), 'bullet_bills': np.zeros((3, 4))}, {'time_elapsed': 0.}
    def step(self, action):
        self.t += 1
        assert len(action) == 2
        return {'mario': np.ones(8, dtype=np.float32) * self.t, 'bullet_bills': np.zeros((3,4))}, 0., self.t == 7, False, {'game_state': 'game_over' if self.t == 7 else 'playing', 'time_elapsed': self.t/15}
    def close(self):
        pass

class InitialTests(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(obs_vector({'mario': [0, 1]}).shape, (256,))
        self.assertEqual(env_action(2, 4).tolist(), [1., 1.])
        m=ModelV4()
        j, a, lp, v = m.act(torch.zeros((3,256)))
        self.assertEqual(tuple(lp.shape), (3,))

    def test_training_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            base=dict(total_steps=64,envs=2,batch_steps=8,seed=42,lr=3e-4,device='cpu',eval_episodes=0,checkpoint_every=16,resume=None,
                      out=d/'model_v4_initial_50k.pt',checkpoint_dir=d/'checkpoints')
            run(SimpleNamespace(**base), env_factory=DummyEnv)
            self.assertTrue(base['out'].exists())
            self.assertTrue((d/'checkpoints/model_v4_resume.pt').exists())
            ck=torch.load(base['out'], weights_only=False)
            self.assertTrue(ck['trained'])
            self.assertEqual(ck['steps'],64)
            base.update(total_steps=96,resume=d/'checkpoints/model_v4_resume.pt')
            run(SimpleNamespace(**base), env_factory=DummyEnv)
            self.assertEqual(torch.load(base['out'],weights_only=False)['steps'],96)

if __name__ == '__main__': unittest.main()
