import importlib.util
import queue
import sys
import threading
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).parent/'real'/'model_v4'
sys.path.insert(0,str(ROOT))
import train_parallel_500k_diagnostic as tr
from model_v4_net import ModelV4

class FakeEnv:
    def __init__(self): self.n=0; self.x=648.; self.time=0.
    def reset(self,seed=None): self.n=0; self.x=648.;self.time=0.;return self.obs(),{}
    def obs(self):return {'mario':np.array([self.x,12.,self.x+20.,32.],dtype=np.float32),'time':np.array([self.time],dtype=np.float32)}
    def step(self,action):
        self.n+=1; self.time+=1/15; self.x+=float(action[0])*5
        done=self.n>=3
        return self.obs(),0.,done,False,{'time_elapsed':self.time,'game_state':'game_over' if done else 'playing'}
    def close(self):pass

def test():
    tr.make_env=lambda: FakeEnv()
    req=queue.Queue();res=queue.Queue()
    m=ModelV4()
    req.put((0,m.state_dict(),8));req.put(None)
    tr.worker(0,req,res,1234,8,99)
    i,gen,roll,error=res.get_nowait()
    assert error is None, error
    assert roll['s'].shape==(8,256)
    assert np.isfinite(roll['mu']).all()
    assert np.isfinite(roll['dx']).all()
    assert int(roll['death_flag'].sum())>=1
    args=tr.parse_args(['--minibatch','4','--ppo-epochs','2'])
    opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    stats=tr.update(m,opt,[roll],args,torch.device('cpu'))
    for key in ('approx_kl','actor_loss','value_loss','grad_norm','mu_abs','actual_dx_abs','deaths'):
        assert key in stats,key
    print('PASS',stats)
if __name__=='__main__':test()
