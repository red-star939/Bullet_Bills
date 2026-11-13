"""Offline training smoke tests. Uses stubs only when gymnasium/kymnasium absent."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))
if importlib.util.find_spec('gymnasium') is None:
    sys.modules['gymnasium'] = types.ModuleType('gymnasium')
if importlib.util.find_spec('kymnasium') is None:
    mod = types.ModuleType('kymnasium')
    mod.Agent = type('Agent', (), {})
    sys.modules['kymnasium'] = mod

from agent_ppo_v10 import ActorCritic, OBS_DIM
from train_models2_v41_ppo import (auxiliary_bc, load_checkpoint, save_pt, seed_sets)


class TrainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(234)
        torch.set_num_threads(1)

    def test_checkpoint_roundtrip_and_real_gradients(self):
        with tempfile.TemporaryDirectory() as d:
            initial = Path(d) / 'original.pt'
            updated = Path(d) / 'new.pt'
            net = ActorCritic(obs_dim=OBS_DIM, hidden=32)
            save_pt(initial, net, 32, {'steps': 0})
            loaded, ck = load_checkpoint(initial)
            obs = torch.randn(64, OBS_DIM)
            with torch.no_grad():
                j, m, lp = loaded.act(obs)
                baseline = {k:v.detach().clone() for k,v in loaded.state_dict().items()}
            opt = torch.optim.Adam(loaded.parameters(), lr=1e-3)
            logp, ej, em, v = loaded.evaluate(obs, j, m)
            loss = -(logp - torch.from_numpy(np.zeros(len(obs),dtype=np.float32))).mean() + v.square().mean()
            opt.zero_grad();loss.backward();opt.step()
            save_pt(updated, loaded, 32, {'steps': 64})
            restored, _ = load_checkpoint(updated)
            self.assertEqual(restored.actor_body[0].weight.shape[1], OBS_DIM)
            self.assertTrue(any(not torch.equal(v, baseline[k]) for k,v in restored.state_dict().items()))
            self.assertTrue(initial.exists())

    def test_teacher_auxiliary_loss_has_gradients(self):
        m = ActorCritic(obs_dim=OBS_DIM, hidden=32)
        data = (torch.randn(20, OBS_DIM), torch.randint(0, 5, (20,)),
                torch.rand(20)*2-1, torch.randint(0, 2, (20,)).float())
        result = auxiliary_bc(m, data, np.random.default_rng(3), 8, 2.0)
        self.assertTrue(torch.isfinite(result))
        result.backward()
        self.assertIsNotNone(m.jump_head.weight.grad)

    def test_random_disjoint_evaluation_seed_sets(self):
        a, b = seed_sets(8172, 7, 9)
        self.assertEqual(len(set(a + b)), 16)
        self.assertEqual((a,b), seed_sets(8172,7,9))


if __name__ == '__main__': unittest.main()

class IntegrationTest(unittest.TestCase):
    def test_full_ppo_loop_fake_env(self):
        from unittest.mock import patch
        from train_models2_v41_ppo import parse_args, train
        class FakeFE:
            def reset(self): pass
            def __call__(self, obs, info):
                x=np.zeros(OBS_DIM,dtype=np.float32)
                x[0]=obs['mario'][0]/1000
                x[1]=info['time_elapsed']/20
                return x,0.1,0.2
        class FakeEnv:
            def reset(self, seed=None):
                self.step_num=0
                self.sd=seed
                return {'mario':np.array([500,672,548,720,0,0,0,0],dtype=np.float32)}, {'time_elapsed':0.}
            def step(self, action):
                self.step_num+=1
                m=np.array([500+np.clip(action[0],-1,1)*5,672,548,720,0,0,0,0],dtype=np.float32)
                return {'mario':m},0., self.step_num>=3,False, {'time_elapsed':self.step_num/15.,'game_state':'game_over' if self.step_num>=3 else 'playing'}
            def close(self): pass
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            original=root/'init.pt';dest=root/'best.pt'
            save_pt(original, ActorCritic(obs_dim=OBS_DIM,hidden=32),32,{})
            argv=['--init',str(original),'--out',str(dest),'--log-dir',str(root/'logs'),
                  '--total-steps','16','--envs','2','--batch-steps','4','--eval-episodes','2',
                  '--final-episodes','2','--eval-max-steps','3','--eval-interval','4',
                  '--epochs','1','--minibatches','2','--log-every','1']
            args=parse_args(argv)
            with patch('train_models2_v41_ppo.v10.FeatureExtractor',FakeFE),patch('train_models2_v41_ppo.gym.make',return_value=FakeEnv(),create=True):
                result=train(args)
            self.assertTrue(dest.is_file())
            self.assertTrue((root/'logs'/'final_holdout.csv').is_file())
            self.assertGreaterEqual(result['completed_steps'],16)
            self.assertTrue(set(result['eval_seeds']).isdisjoint(set(result['final_holdout_seeds'])))
