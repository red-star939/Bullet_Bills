"""Distill observed v4.1 teacher actions into PPO v10 actor checkpoint.

Requires the user's original v10 .pt and collected teacher transitions. This
updates neural weights (unlike a metadata-only .pt wrapper). It does not train
PPO using rewards; it is supervised policy distillation / behavior cloning.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from agent_ppo_v10 import ActorCritic, JUMP_LEVELS


def load_data(paths):
    all_s, all_a, all_w, all_ep = [], [], [], []
    offset = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as ds:
            s = ds['states'].astype(np.float32)
            a = ds['actions'].astype(np.float32)
            ov = ds['override'].astype(np.float32)
            ep = ds['episodes'].astype(np.int64)
            if s.ndim != 2 or s.shape[1] != 148 or a.shape != (len(s), 2) or len(ov) != len(s):
                raise ValueError(f'Invalid v10 teacher dataset: {path}')
            if not len(s) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(a)):
                raise ValueError(f'Empty or non-finite teacher dataset: {path}')
            all_s.append(s); all_a.append(a); all_w.append(ov)
            all_ep.append(ep + offset)
            offset += int(ep.max()) + 1
    return tuple(np.concatenate(parts) for parts in (all_s, all_a, all_w, all_ep))


def distill(src, files, out, *, epochs=12, lr=2e-5, batch=512, override_weight=8.,
            preserve_kl=0.25, seed=314159, val_fraction=.2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    source, target = Path(src).resolve(), Path(out).resolve()
    if source == target or target.exists():
        raise FileExistsError('Output must be a NEW filepath; input and existing outputs are protected')
    ck = torch.load(source, map_location='cpu', weights_only=True)
    if not all(k in ck for k in ('model', 'hidden', 'obs_dim')) or ck['obs_dim'] != 148:
        raise ValueError('Expected PPO v10 checkpoint with 148 features')
    teacher = ActorCritic(obs_dim=ck['obs_dim'], hidden=ck['hidden'])
    teacher.load_state_dict(ck['model']); teacher.eval()
    student = copy.deepcopy(teacher)
    states, actions, overrides, episodes = load_data(files)
    uniq = np.unique(episodes)
    if len(uniq) < 2:
        raise ValueError('Collect at least 2 separate episodes for train/validation split')
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    nv = max(1, min(len(uniq)-1, math.ceil(len(uniq) * val_fraction)))
    val_sel = np.isin(episodes, uniq[:nv])
    tr_idx = np.flatnonzero(~val_sel); va_idx = np.flatnonzero(val_sel)
    x = torch.from_numpy(states)
    y_move = torch.from_numpy(np.clip(actions[:,0], -1, 1))
    # Environment sends categorical jump level, map to closest valid action index.
    levels = np.asarray(JUMP_LEVELS)
    jump_idx = np.abs(actions[:,1,None]-levels[None,:]).argmin(1)
    if np.max(np.abs(actions[:,1]-levels[jump_idx])) > .08:
        raise ValueError('Invalid jump values in teacher actions')
    y_jump = torch.from_numpy(jump_idx.astype(np.int64))
    w = torch.from_numpy(1. + (max(1.,override_weight)-1.) * overrides)
    optimizer = torch.optim.Adam([*student.actor_body.parameters(), *student.jump_head.parameters(),
                                  *student.move_head.parameters()], lr=lr)
    # Critic and log std stay unchanged: behavior cloning has no return targets.
    record=[]; best=math.inf; best_actor=None
    def loss_at(ids, grad):
        losses=[]
        iterable=[ids[i:i+batch] for i in range(0,len(ids),batch)]
        for ix in iterable:
            obs=x[ix]; z=student.actor_body(obs)
            logits=student.jump_head(z)
            m=student.move_head(z).flatten()
            with torch.no_grad():
                zh=teacher.actor_body(obs)
                old_logits=teacher.jump_head(zh)
                old_m=teacher.move_head(zh).flatten()
                old_probs=F.softmax(old_logits,dim=-1)
            ce=F.cross_entropy(logits,y_jump[ix],reduction='none')
            mse=(m-y_move[ix]).square()
            imitate=((ce+mse)*w[ix]).sum()/w[ix].sum()
            kl=(old_probs*(F.log_softmax(old_logits,dim=-1)-F.log_softmax(logits,dim=-1))).sum(-1).mean()
            penalty=kl+(m-old_m).square().mean()
            loss=imitate+preserve_kl*penalty
            if grad:
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.0);optimizer.step()
            losses.append((len(ix),float(loss.detach()),float(imitate.detach())))
        return [sum(n*a for n,a,*_ in losses)/len(ids),sum(n*b for n,_,b in losses)/len(ids)]
    for epoch in range(1,epochs+1):
        student.train()
        train_ids=rng.permutation(tr_idx)
        train_loss, train_bc=loss_at(train_ids,True)
        student.eval()
        with torch.no_grad():
            val_loss,val_bc=loss_at(va_idx,False)
        record.append(dict(epoch=epoch,train_loss=train_loss,val_loss=val_loss,train_bc=train_bc,val_bc=val_bc))
        print(f'EPOCH {epoch:02d} train={train_loss:.5f} val={val_loss:.5f} val_bc={val_bc:.5f}',flush=True)
        if val_loss < best:
            best=val_loss
            best_actor={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    if best_actor is None:
        raise RuntimeError('Training did not produce a checkpoint')
    target.parent.mkdir(parents=True,exist_ok=True)
    saved={'model':best_actor,'obs_dim':ck['obs_dim'],'hidden':ck['hidden'],
           'distillation':{'format':'models2_v41_distilled_v1','source':str(src),
             'teacher':'v10+spawn_shield+v4.1 explore','training':'behavior_cloning',
             'total_transitions':int(len(states)),'override_transitions':int(overrides.sum()),
             'train_episodes':int(len(uniq)-nv),'val_episodes':int(nv),
             'epochs':epochs,'lr':lr,'override_weight':override_weight,
             'preserve_kl':preserve_kl,'best_val_loss':float(best),
             'note':'No environment reward optimization. Requires independent gameplay evaluation.'}}
    torch.save(saved,target)
    loaded=torch.load(target,map_location='cpu',weights_only=True)
    assert loaded['model'].keys()==ck['model'].keys()
    changed=sum(not torch.equal(loaded['model'][k],ck['model'][k]) for k in ck['model'])
    if changed==0:
        raise RuntimeError('No neural weights changed: refusing to label output distilled')
    report=target.with_suffix('.training.json')
    report.write_text(json.dumps({'source':src,'output':str(target),'weights_changed':changed,
       'train_samples':int(len(tr_idx)),'val_samples':int(len(va_idx)),
       'override_samples':int(overrides.sum()),'history':record},indent=2),encoding='utf-8')
    print(f'NEW_PT {target} changed_tensors={changed} samples={len(states)} overrides={int(overrides.sum())}')
    print(f'TRAINING_REPORT {report}')
    if overrides.sum()==0:
        print('WARNING v4.1 never overrode actions; no v4.1-specific improvement signal was observed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='models/bullet_bill_ppo_v10_best.pt')
    p.add_argument('--data',nargs='+',required=True)
    p.add_argument('--out',default='models2/bullet_bill_ppo_v10_v41_distilled.pt')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--override-weight',type=float,default=8.)
    p.add_argument('--preserve-kl',type=float,default=.25)
    p.add_argument('--seed',type=int,default=314159)
    a=p.parse_args()
    if a.epochs<1 or a.batch<1 or a.lr<=0 or a.override_weight<1 or a.preserve_kl<0:p.error('Invalid training parameters')
    distill(a.model,a.data,a.out,epochs=a.epochs,lr=a.lr,batch=a.batch,
           override_weight=a.override_weight,preserve_kl=a.preserve_kl,seed=a.seed)
if __name__=='__main__':main()
