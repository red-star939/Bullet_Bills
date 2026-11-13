"""Synthetic smoke test for the distillation script (no game runtime needed)."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn

class MiniActorCritic(nn.Module):
    def __init__(self,obs_dim=148,hidden=16):
        super().__init__()
        self.actor_body=nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh())
        self.jump_head=nn.Linear(hidden,5)
        self.move_head=nn.Linear(hidden,1)
        self.move_log_std=nn.Parameter(torch.tensor([-0.5]))
        self.critic=nn.Linear(obs_dim,1)

class TestDistillation(unittest.TestCase):
    def test_pt_weights_change_and_load(self):
        mock=types.ModuleType('agent_ppo_v10')
        mock.ActorCritic=MiniActorCritic
        mock.JUMP_LEVELS=np.array([0,.5,.6,.75,1],dtype=np.float32)
        old=sys.modules.get('agent_ppo_v10')
        sys.modules['agent_ppo_v10']=mock
        try:
            import importlib.util
            path=Path(__file__).with_name('train_models2_v41_distill.py')
            spec=importlib.util.spec_from_file_location('test_distill_implementation',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp)
                model=MiniActorCritic()
                src=d/'base.pt';out=d/'trained.pt';data=d/'sample.npz'
                torch.save({'model':model.state_dict(),'hidden':16,'obs_dim':148},src)
                rng=np.random.default_rng(99)
                states=rng.normal(size=(70,148)).astype(np.float32)
                actions=np.zeros((70,2),dtype=np.float32)
                actions[:,0]=.5
                actions[:,1]=.75
                episodes=np.repeat(np.arange(7),10)
                overrides=np.zeros(70,dtype=np.uint8);overrides[::7]=1
                np.savez_compressed(data,states=states,actions=actions,episodes=episodes,override=overrides)
                module.distill(str(src),[str(data)],str(out),epochs=2,batch=16)
                target=torch.load(out,map_location='cpu',weights_only=True)
                self.assertEqual(target['obs_dim'],148)
                self.assertTrue(any(not torch.equal(model.state_dict()[k],v) for k,v in target['model'].items()))
                target_model=MiniActorCritic();target_model.load_state_dict(target['model'])
        finally:
            if old is None:sys.modules.pop('agent_ppo_v10',None)
            else:sys.modules['agent_ppo_v10']=old

if __name__=='__main__':unittest.main()
