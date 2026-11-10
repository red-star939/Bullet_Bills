"""Train our discrete-action PPO on the real BulletBill Continuous Stage 2 environment.

Requires agent_env.py + kymnasium in the Windows project. Use `--resume` to
resume a *training checkpoint* with optimizer and RNG saved; a raw initial
checkpoint can also be loaded but begins with a fresh optimizer.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.ppo_policy import (ACTION_TABLE, OBS_DIM, ActorCritic, StateEncoder,
                                checkpoint_payload, load_platforms, load_policy)


def compute_gae(rewards, values, next_values, terminated, boundary, *, gamma=.995, lam=.95):
    """Correct bootstrapping on time limit; no GAE leakage across episode ends."""
    rewards = np.asarray(rewards, np.float32)
    values = np.asarray(values, np.float32)
    next_values = np.asarray(next_values, np.float32)
    term = np.asarray(terminated, np.bool_)
    bound = np.asarray(boundary, np.bool_)
    adv = np.empty_like(rewards)
    last = 0.
    for t in range(len(rewards)-1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * (not term[t]) - values[t]
        last = delta + gamma * lam * (not bound[t]) * last
        adv[t] = last
    return adv, adv + values


def ppo_update(net, optimizer, batch, *, epochs=4, minibatch=256, clip=.2,
               vf_coef=.5, ent_coef=.01, max_grad_norm=.5, target_kl=.03):
    device = next(net.parameters()).device
    states = torch.as_tensor(np.asarray(batch['states']), device=device)
    actions = torch.as_tensor(batch['actions'], dtype=torch.long, device=device)
    old_logprob = torch.as_tensor(batch['logps'], dtype=torch.float32, device=device)
    adv = torch.as_tensor(batch['advantages'], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch['returns'], dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    stats = []
    for epoch in range(epochs):
        permutation = torch.randperm(len(states), device=device)
        for ix in permutation.split(minibatch):
            dist, val = net(states[ix])
            logp = dist.log_prob(actions[ix])
            ratio = torch.exp(logp - old_logprob[ix])
            clipped = torch.clamp(ratio, 1.-clip, 1.+clip)
            pg = -torch.minimum(ratio * adv[ix], clipped * adv[ix]).mean()
            vl = 0.5 * (val - returns[ix]).square().mean()
            entropy = dist.entropy().mean()
            loss = pg + vf_coef * vl - ent_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kl = ((ratio - 1.) - (logp-old_logprob[ix])).mean().item()
                stats.append((pg.item(), vl.item(), entropy.item(), kl))
        if stats and np.mean([s[3] for s in stats[-max(1, (len(states)+minibatch-1)//minibatch):]]) > target_kl:
            break
    return {'policy_loss':float(np.mean([s[0] for s in stats])),
            'value_loss':float(np.mean([s[1] for s in stats])),
            'entropy':float(np.mean([s[2] for s in stats])),
            'approx_kl':float(np.mean([s[3] for s in stats]))}


def train(args):
    from agent_env import create_env  # game dependency only when training
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    encoder = StateEncoder(load_platforms(args.map))
    net = ActorCritic().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    global_steps = 0
    rng = np.random.default_rng(args.seed)
    if args.resume:
        net, ck = load_policy(args.resume, args.device)
        net.train()
        global_steps = int(ck.get('total_env_steps', 0))
        if 'optimizer_state_dict' in ck:
            opt.load_state_dict(ck['optimizer_state_dict'])
        if ck.get('extra', {}).get('rng_numpy') is not None:
            rng.bit_generator.state = ck['extra']['rng_numpy']
        if ck.get('extra', {}).get('rng_torch') is not None:
            torch.set_rng_state(ck['extra']['rng_torch'])
        if ck.get('extra', {}).get('rng_python') is not None:
            random.setstate(ck['extra']['rng_python'])
    # Always create env after checkpoint inspection. New episode seeds are
    # generated from the reproducible RNG, never from the evaluation split.
    env = create_env(render_mode='none')
    episode_count = 0; returns_history = []; survival_history = []
    obs = None; info = None; ep_len = 0; ep_return = 0.
    try:
        while global_steps < args.total_steps:
            states=[]; actions=[]; logps=[]; values=[]; next_values=[]
            rewards=[]; terminated=[]; boundaries=[]
            while len(states) < args.rollout_steps and global_steps < args.total_steps:
                if obs is None:
                    obs, info = env.reset(seed=int(rng.integers(0, 2**32, dtype=np.uint64)))
                    ep_len=0; ep_return=0.
                state = encoder.encode(obs, float(info.get('time_elapsed',0.)))
                with torch.no_grad():
                    a, lp, val = net.act(torch.as_tensor(state, device=device).unsqueeze(0))
                idx = int(a.item()); action = ACTION_TABLE[idx].copy()
                next_obs, env_reward, term, trunc, next_info = env.step(action)
                # Environment reward is documented as always zero; use explicit
                # survival-shaping reward, and do not infer failure from time cap.
                game_state = str(next_info.get('game_state', 'unknown'))
                is_death = bool(term and game_state == 'game_over')
                is_success = bool(term and game_state in ('game_clear', 'clear', 'win'))
                r = args.alive_reward + (args.death_penalty if is_death else 0.)
                if is_success:
                    r += args.success_bonus
                if trunc and not term:
                    r += args.timeout_bonus
                new_state = encoder.encode(next_obs, float(next_info.get('time_elapsed',0.)))
                with torch.no_grad():
                    next_val = net(torch.as_tensor(new_state, device=device).unsqueeze(0))[1].item()
                ep_len+=1; ep_return+=r; global_steps+=1
                cutoff = ep_len >= args.max_episode_steps
                boundary = bool(term or trunc or cutoff)
                states.append(state);actions.append(idx);logps.append(float(lp.item()))
                values.append(float(val.item()));next_values.append(float(next_val))
                rewards.append(r);terminated.append(bool(term));boundaries.append(boundary)
                if boundary:
                    episode_count+=1
                    returns_history.append(ep_return)
                    survival_history.append(float(next_info.get('time_elapsed',0.)))
                    obs=None
                else:
                    obs,info=next_obs,next_info
            adv,ret = compute_gae(rewards,values,next_values,terminated,boundaries,
                                  gamma=args.gamma,lam=args.gae_lambda)
            batch={'states':states,'actions':actions,'logps':logps,
                   'advantages':adv,'returns':ret}
            stats=ppo_update(net,opt,batch,epochs=args.epochs,minibatch=args.minibatch,
                             clip=args.clip,vf_coef=args.vf_coef,ent_coef=args.ent_coef,
                             max_grad_norm=args.max_grad_norm,target_kl=args.target_kl)
            if global_steps % args.log_interval < len(states) or global_steps >= args.total_steps:
                print('STEPS',global_steps,'EPISODES',episode_count,
                      'LAST20_SURVIVAL',round(float(np.mean(survival_history[-20:])),3) if survival_history else 'NA',
                      'KL',round(stats['approx_kl'],5),'ENTROPY',round(stats['entropy'],3),flush=True)
            if global_steps % args.save_interval < len(states) or global_steps >= args.total_steps:
                extra={'args':vars(args), 'rng_numpy':rng.bit_generator.state,
                       'rng_torch':torch.get_rng_state(), 'rng_python':random.getstate(),
                       'episodes_seen':episode_count,
                       'recent_survival_s':survival_history[-100:]}
                ck=checkpoint_payload(net,trained=True,total_steps=global_steps,optimizer=opt,extra=extra)
                checkpoint = out / f'mario_ppo_step_{global_steps:09d}.pt'
                torch.save(ck,checkpoint)
                torch.save(ck,out/'mario_ppo_latest.pt')
                (out/'latest_metrics.json').write_text(json.dumps({'global_steps':global_steps,
                    'episode_count':episode_count, 'recent_mean_survival_s':float(np.mean(survival_history[-100:])) if survival_history else None,
                    'stats':stats},indent=2),encoding='utf-8')
                print('CHECKPOINT',checkpoint,flush=True)
    finally:
        env.close()


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--total-steps',type=int,default=100_000)
    p.add_argument('--rollout-steps',type=int,default=2048)
    p.add_argument('--max-episode-steps',type=int,default=900)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--minibatch',type=int,default=256)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gamma',type=float,default=.995)
    p.add_argument('--gae-lambda',type=float,default=.95)
    p.add_argument('--clip',type=float,default=.2)
    p.add_argument('--vf-coef',type=float,default=.5)
    p.add_argument('--ent-coef',type=float,default=.01)
    p.add_argument('--max-grad-norm',type=float,default=.5)
    p.add_argument('--target-kl',type=float,default=.03)
    p.add_argument('--alive-reward',type=float,default=.01)
    p.add_argument('--death-penalty',type=float,default=-1.)
    p.add_argument('--timeout-bonus',type=float,default=0.)
    p.add_argument('--success-bonus',type=float,default=1.)
    p.add_argument('--seed',type=int,default=20261110)
    p.add_argument('--map',default='map_data/object_tiles.csv')
    p.add_argument('--out',default='models/ppo_runs')
    p.add_argument('--resume',default=None)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--log-interval',type=int,default=10000)
    p.add_argument('--save-interval',type=int,default=25000)
    args=p.parse_args()
    if any(getattr(args,k)<1 for k in ['total_steps','rollout_steps','max_episode_steps','epochs','minibatch','save_interval','log_interval']):
        p.error('counts must be positive')
    return args

if __name__=='__main__':
    train(parse())
