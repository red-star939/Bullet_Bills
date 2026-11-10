"""Optional warm-start of *our PPO actor* using ActionEvaluator v2 demonstrations.

This is supervised behavior cloning, NOT PPO reinforcement learning. The
checkpoint is explicitly labeled training_method=teacher_bc; run train_ppo.py
--resume on it for PPO fine-tuning with real game experience.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from models.ppo_policy import ACTION_TABLE,OBS_DIM,SCHEMA_VERSION,ActorCritic,checkpoint_payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--out',default='models/mario_ppo_bc_init.pt')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=20261110)
    args=p.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    with np.load(args.dataset,allow_pickle=False) as data:
        states=data['states'];actions=data['actions'];table=data['action_table'];schema=data['encoder_version'].item()
    if states.ndim!=2 or states.shape[1]!=OBS_DIM or len(states)==0:
        raise ValueError('Invalid state dimensions or empty teacher dataset')
    if schema!=SCHEMA_VERSION or not np.array_equal(table,ACTION_TABLE):
        raise ValueError('Teacher dataset schema mismatch')
    if np.any(actions<0) or np.any(actions>=len(ACTION_TABLE)):
        raise ValueError('Teacher action IDs outside action table')
    s=torch.as_tensor(states,dtype=torch.float32); a=torch.as_tensor(actions,dtype=torch.long)
    model=ActorCritic();optimizer=torch.optim.Adam(model.actor_body.parameters(),lr=args.lr)
    optimizer.add_param_group({'params':model.actor_head.parameters()})
    # Separate validation episodes by seed, not random frames from the same episode.
    with np.load(args.dataset,allow_pickle=False) as data:
        seeds=data['env_seeds']
    unique=np.unique(seeds);split=max(1,int(len(unique)*.9))
    rng=np.random.default_rng(args.seed);rng.shuffle(unique)
    if len(unique)>1:
        train_mask=np.isin(seeds,unique[:split]);val_mask=~train_mask
    else:
        train_mask=np.ones(len(states),bool);val_mask=~train_mask
    train_ix=torch.as_tensor(np.flatnonzero(train_mask))
    val_ix=torch.as_tensor(np.flatnonzero(val_mask))
    for epoch in range(args.epochs):
        model.train()
        ix=train_ix[torch.randperm(len(train_ix))]
        for batch in ix.split(args.batch_size):
            logits=model.actor_head(model.actor_body(s[batch]))
            loss=F.cross_entropy(logits,a[batch]);optimizer.zero_grad(set_to_none=True)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        model.eval()
        with torch.no_grad():
            accuracy=float((model.actor_head(model.actor_body(s[val_ix])).argmax(-1)==a[val_ix]).float().mean().item()) if len(val_ix) else None
        print('BC EPOCH',epoch+1,'VALIDATION_ACCURACY',accuracy,flush=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    ck=checkpoint_payload(model,trained=True,total_steps=0,
                          extra={'training_method':'teacher_behavior_cloning_not_ppo',
                                 'dataset':args.dataset,'epochs':args.epochs,
                                 'validation_action_accuracy':accuracy})
    torch.save(ck,out)
    out.with_suffix('.json').write_text(json.dumps({'dataset':args.dataset,
         'training_method':'behavior_cloning','validation_accuracy':accuracy},indent=2),encoding='utf-8')
    print('TEACHER_PRETRAINED',out)

if __name__=='__main__':main()
