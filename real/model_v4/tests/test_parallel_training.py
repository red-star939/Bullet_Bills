import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train_parallel_1000ep as tr
from model_v4_net import ModelV4, make_payload


class TinyEnvironment:
    def __init__(self): self.time=0; self.seed=0
    def reset(self, seed):
        self.seed=seed; self.time=0
        return np.array([float(seed % 100)/100, 1.0]), {}
    def step(self, action):
        self.time+=1
        return np.array([float(self.time), float(action[0])]), 0., self.time>=4, False, {'time_elapsed':self.time/50., 'game_state':'game_over' if self.time>=4 else 'playing'}
    def close(self): pass


def fake_worker(i, tasks, results, max_steps):
    torch.set_num_threads(1)
    env=TinyEnvironment()
    model=ModelV4().eval()
    while True:
        task=tasks.get()
        if task is None: break
        idx, seed, state=task
        model.load_state_dict(state)
        torch.manual_seed(int(seed))
        roll=tr.run_episode(env,model,seed,max_steps)
        results.put((idx,seed,roll,None))


class Tests(unittest.TestCase):
    def test_gae_terminal_and_truncated(self):
        x={'v':np.zeros(2,np.float32),'rw':np.ones(2,np.float32),'terminal':True,'bootstrap':100}
        a,returns=tr.calculate_advantages(x, 1,1)
        self.assertEqual(returns.tolist(),[2,1])
        x['terminal']=False
        a,returns=tr.calculate_advantages(x,1,1)
        self.assertEqual(returns.tolist(),[102,101])

    def test_train_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            src=root/'untrained.pt'
            m=ModelV4()
            torch.save(make_payload(m,steps=0,trained=False,seed=4),src)
            def args(episodes,resume=None):
                return tr.parse_args(['--init',str(src),'--episodes',str(episodes),'--envs','3',
                                     '--max-steps','4','--device','cpu','--seed','1000',
                                     '--ppo-epochs','1','--minibatch','12','--max-train-transitions','48',
                                     '--checkpoint-every','3','--checkpoint-dir',str(root/'checkpoints'),
                                     '--out',str(root/'v4.pt')]+(['--resume',str(resume)] if resume else []))
            first=tr.main(args(6),worker_target=fake_worker)
            self.assertEqual(first['completed_episodes'],6)
            ck=root/'checkpoints/model_v4_1000ep_resume.pt'
            self.assertTrue(ck.exists())
            second=tr.main(args(9,ck),worker_target=fake_worker)
            self.assertEqual(second['completed_episodes'],9)
            self.assertEqual(second['environment_steps'],36)
            final=tr.read_checkpoint(root/'v4.pt')
            self.assertEqual(final['architecture'],'independent_model_v4')
            self.assertEqual(final['completed_episodes'],9)
            seeds=json.loads((root/'checkpoints/seed_plan.json').read_text())
            self.assertEqual(len(set(seeds['train_seeds']+seeds['eval_seeds'])),39)

if __name__=='__main__': unittest.main()
