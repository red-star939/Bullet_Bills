import sys,unittest,tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_v41_net import OBS_DIM, ARCH, ModelV41, obs_vector, env_action, make_payload
from eval_v41 import load_model,run,parse_args
from train_v41 import gae,update,parse_args as train_parse

class FakeEnv:
    def reset(self,seed=None):
        self.t=0;self.x=650.0
        return self.obs(),{}
    def obs(self):
        return {'mario':np.array([self.x-24,624,self.x+24,672,0,0,0,0],np.float32),
                'bullet_bills':np.array([[850,625,898,673,-180],[0,0,0,0,0]],np.float32),
                'stars':np.zeros((2,4),np.float32)}
    def step(self,a):
        self.x+=float(a[0])*5;self.t+=1
        return self.obs(),0.0,self.t>=6,False,{'time_elapsed':self.t/50,'game_state':'game_over'}
    def close(self):pass

class Tests(unittest.TestCase):
    def test_features(self):
        env=FakeEnv(); env.reset(); ob=env.obs(); vec=obs_vector(ob)
        self.assertEqual(vec.shape,(64,));self.assertTrue(np.isfinite(vec).all());self.assertEqual(vec[12],1)
        ob2={**ob,'bullet_bills':np.zeros((2,5),np.float32)}
        self.assertEqual(obs_vector(ob2)[12],0.)
    def test_action_mapping(self):
        self.assertEqual([float(env_action(i,0)[0]) for i in range(3)],[-1.,0.,1.])
    def test_policy(self):
        m=ModelV41();x=torch.zeros(16,OBS_DIM)
        j,move,lp,v=m.act(x)
        self.assertTrue(torch.all((move>=0)&(move<3)).item())
        self.assertTrue(torch.all((j>=0)&(j<5)).item())
        l,e,v2=m.evaluate_actions(x,j,move)
        self.assertTrue(torch.allclose(lp,l))
        self.assertEqual(v2.shape,(16,))
    def test_evaluation(self):
        m=ModelV41()
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'v41.pt';torch.save(make_payload(m,steps=0,trained=False,seed=21),f)
            a=parse_args(['--model',str(f),'--mode','none','--episodes','2','--max-steps','10','--out',td])
            out=run(a,env_factory=lambda mode:FakeEnv())
            self.assertEqual(out['completed_episodes'],2)
            self.assertEqual(out['model_steps'],0)
    def test_ppo_update(self):
        model=ModelV41();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        x=np.zeros((8,OBS_DIM),np.float32)
        with torch.no_grad():
            j,m,lp,v=model.act(torch.tensor(x))
        r={'s':x,'j':j.numpy(),'m':m.numpy(),'lp':lp.numpy(),'v':v.numpy(),
           'rw':np.ones(8,np.float32),'done':np.zeros(8,np.int64),
           'timeout':np.zeros(8,np.int64),'timeout_value':np.zeros(8,np.float32),'bootstrap':0.0}
        args=train_parse(['--total-steps','8','--envs','1','--batch-steps','8'])
        before={k:v.clone() for k,v in model.state_dict().items()}
        stats=update(model,optimizer,[r],args,torch.device('cpu'))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items()))
        self.assertIn('move_neutral_pct',stats)

if __name__=='__main__':unittest.main()
