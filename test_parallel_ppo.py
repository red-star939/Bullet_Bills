"""Spawn smoke tests via --test-env, no game installation necessary."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
sys.path.insert(0, str(SOURCE))
if importlib.util.find_spec('gymnasium') is None:
    sys.modules['gymnasium'] = types.ModuleType('gymnasium')
if importlib.util.find_spec('kymnasium') is None:
    stub = types.ModuleType('kymnasium')
    stub.Agent = type('Agent', (), {})
    sys.modules['kymnasium'] = stub

import agent_ppo_v10 as v10
from train_models2_v41_ppo import save_pt


class ParallelTrainingTests(unittest.TestCase):
    def test_spawn_workers_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init = root / 'original.pt'
            best = root / 'best.pt'
            save_pt(init, v10.ActorCritic(obs_dim=v10.OBS_DIM, hidden=32), 32, {})
            base = [sys.executable, str(HERE / 'train_models2_v41_ppo_parallel.py'),
                    '--test-env', '--init', str(init), '--out', str(best),
                    '--log-dir', str(root / 'logs'), '--envs', '3',
                    '--batch-steps', '4', '--epochs', '1', '--minibatches', '2',
                    '--eval-episodes', '2', '--final-episodes', '2',
                    '--eval-interval', '12', '--log-every', '1', '--eval-max-steps', '3']
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(SOURCE))
            result = subprocess.run(base + ['--total-steps', '24'], capture_output=True,
                                    text=True, timeout=100, cwd=SOURCE, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(best.is_file())
            full = root / 'best_resume.pt'
            self.assertTrue(full.is_file())
            ck1 = torch.load(full, map_location='cpu', weights_only=False)
            self.assertIn('optimizer', ck1)
            self.assertEqual(ck1['completed'], 24)
            resumed = subprocess.run(base + ['--total-steps', '48', '--resume', str(full)],
                                     capture_output=True, text=True, timeout=100, cwd=SOURCE, env=env)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            ck2 = torch.load(full, map_location='cpu', weights_only=False)
            self.assertEqual(ck2['completed'], 48)
            self.assertTrue(set(ck2['eval_seeds']).isdisjoint(set(ck2['used_train'])))
            self.assertTrue(set(ck2['final_seeds']).isdisjoint(set(ck2['used_train'])))
            self.assertEqual(ck1['eval_seeds'], ck2['eval_seeds'])
            self.assertEqual(ck1['final_seeds'], ck2['final_seeds'])
            changed = any(not torch.equal(ck1['model'][k], ck2['model'][k]) for k in ck1['model'])
            self.assertTrue(changed)
            self.assertTrue((root / 'logs' / 'final_holdout.csv').exists())


if __name__ == '__main__':
    unittest.main()
