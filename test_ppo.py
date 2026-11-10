"""PPO offline smoke tests; no game dependencies required."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import torch
from models.ppo_policy import (ACTION_TABLE, OBS_DIM, ActorCritic, StateEncoder,
                               checkpoint_payload, load_platforms, load_policy)
from train_ppo import compute_gae, ppo_update


def test():
    torch.manual_seed(123)
    assert len(ACTION_TABLE)==8 and ACTION_TABLE.shape==(8,2)
    enc=StateEncoder(load_platforms(Path(__file__).parent/'map_data/object_tiles.csv'))
    state=enc.encode({'mario':np.array([600,350,648,398,0,0,0,0],np.float32),
                      'bullet_bills':np.array([[1000,360,1048,408,-240]],np.float32)},1.)
    assert state.shape==(OBS_DIM,) and np.isfinite(state).all()
    model=ActorCritic()
    st=torch.as_tensor(np.tile(state,(64,1)))
    with torch.no_grad():act,logp,val=model.act(st)
    adv,ret=compute_gae([1.,1.,1.],[0.,0.,0.],[0.,0.,0.],
                        [False,True,False],[False,True,True],gamma=.9,lam=1.)
    assert np.allclose(adv,[1.9,1.,1.])
    # True truncation bootstraps critic but never crosses episode boundary.
    adv2,_=compute_gae([1.,50.],[0.,0.],[3.,0.],
                       [False,False],[True,True],gamma=.9,lam=1.)
    assert np.allclose(adv2,[3.7,50.])
    with torch.no_grad():initial=model(st)[0].log_prob(act)
    stats=ppo_update(model,torch.optim.Adam(model.parameters(),lr=1e-3),
           {'states':st.numpy(),'actions':act.numpy(),'logps':initial.numpy(),
            'advantages':np.linspace(-1,1,64,dtype=np.float32),'returns':np.ones(64,np.float32)},
           epochs=2,minibatch=32)
    assert all(np.isfinite(v) for v in stats.values())
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'p.pt'
        torch.save(checkpoint_payload(model,trained=False,total_steps=0),path)
        loaded,ck=load_policy(path)
        assert not ck['trained'] and ck['total_env_steps']==0
        with torch.no_grad():
            assert torch.allclose(loaded(st)[1],model(st)[1])
    print('PPO policy / encoder / GAE / update / checkpoint regression passed!')

if __name__=='__main__':test()
