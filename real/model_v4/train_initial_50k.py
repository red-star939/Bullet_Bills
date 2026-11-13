"""Train a new, independent V4 policy for 50,000 environment steps.

Final model: real/model_v4/model_v4_initial_50k.pt
Resume/periodic snapshots: real/model_v4/checkpoints/

Environment physics remain CPU. GPU accelerates network updates if requested.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from model_v4_net import ModelV4, OBS_DIM, HIDDEN, obs_vector, env_action, make_payload
from methods.ppo_core import gae, clipped_policy_loss

ROOT = Path(__file__).resolve().parent
ENV_ID = 'kymnasium/BulletBill-Continuous-Hard-Stage-2'


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--total-steps', type=int, default=50000)
    p.add_argument('--envs', type=int, default=4)
    p.add_argument('--batch-steps', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261112)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    p.add_argument('--eval-episodes', type=int, default=5)
    p.add_argument('--checkpoint-every', type=int, default=10000)
    p.add_argument('--resume', type=Path)
    p.add_argument('--out', type=Path, default=ROOT / 'model_v4_initial_50k.pt')
    p.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints')
    return p.parse_args(argv)


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_env():
    try:
        import gymnasium as gym
        import kymnasium  # noqa: F401 environment registration
    except ImportError as exc:
        raise RuntimeError('Install gymnasium and kymnasium in your Windows project environment') from exc
    return gym.make(id=ENV_ID, render_mode='none', bgm=False)


def evaluate(model, seeds, device):
    env = make_env()
    times = []
    try:
        for seed in seeds:
            obs, info = env.reset(seed=int(seed))
            for _ in range(2701):
                x = torch.as_tensor(obs_vector(obs), device=device).unsqueeze(0)
                with torch.no_grad():
                    j, m, _, _ = model.act(x, deterministic=True)
                obs, _, term, trunc, info = env.step(env_action(m.item(), j.item()))
                if term or trunc:
                    break
            times.append(float(info.get('time_elapsed', 0.)))
    finally:
        env.close()
    return {'mean_survival_s': float(np.mean(times)), 'times_s': times,
            'reached_100_s': int(sum(t >= 100 for t in times)),
            'reached_150_s': int(sum(t >= 150 for t in times))}


def run(args, env_factory=make_env):
    if args.total_steps <= 0 or args.envs <= 0 or args.batch_steps <= 0:
        raise ValueError('total-steps, envs and batch-steps must be positive')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.resolve().parent == args.checkpoint_dir.resolve():
        raise ValueError('Final .pt must be outside the checkpoints directory')
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model = ModelV4().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    steps = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ck.get('architecture') != 'independent_model_v4':
            raise ValueError('Expected independent Model V4 checkpoint')
        model.load_state_dict(ck['model'])
        optim.load_state_dict(ck['optimizer'])
        for state in optim.state.values():
            for key, val in state.items():
                if torch.is_tensor(val):
                    state[key] = val.to(device)
        steps = int(ck['steps'])
        if 'torch_rng' in ck:
            torch.set_rng_state(ck['torch_rng'])
        if 'numpy_rng' in ck:
            np.random.set_state(ck['numpy_rng'])
        if 'python_rng' in ck:
            random.setstate(ck['python_rng'])
        print(f'RESUMED steps={steps} (environment episodes restart)')
    envs = [env_factory() for _ in range(args.envs)]
    episodes = np.zeros(args.envs, dtype=np.int64)
    eval_seeds = [args.seed + 1_000_000 + i for i in range(args.eval_episodes)]
    def reset_env(i):
        ep_seed = args.seed + i * 10_000_000 + int(episodes[i])
        episodes[i] += 1
        o, info = envs[i].reset(seed=ep_seed)
        return obs_vector(o)
    obs = np.stack([reset_env(i) for i in range(args.envs)])
    t0 = time.perf_counter()
    next_ckpt = (steps // args.checkpoint_every + 1) * args.checkpoint_every
    try:
        while steps < args.total_steps:
            n = min(args.batch_steps, max(1, (args.total_steps - steps) // args.envs))
            S, J, M, L, V, R, D = [], [], [], [], [], [], []
            for _ in range(n):
                x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    jumps, moves, logp, values = model.act(x)
                j = jumps.cpu().numpy()
                m = moves.cpu().numpy()
                l = logp.cpu().numpy()
                v = values.cpu().numpy()
                new_obs = np.zeros_like(obs)
                rewards = np.zeros(args.envs, np.float32)
                dones = np.zeros(args.envs, np.float32)
                for i, env in enumerate(envs):
                    o, env_reward, term, trunc, info = env.step(env_action(m[i], j[i]))
                    dead = bool(term and info.get('game_state') == 'game_over')
                    # Survival-centric baseline, uses environment reward only if finite.
                    rewards[i] = 0.1 - 10.0 * float(dead) + (2.0 if term and not dead else 0.0)
                    dones[i] = float(term or trunc)
                    new_obs[i] = reset_env(i) if dones[i] else obs_vector(o)
                S.append(obs.copy()); J.append(j); M.append(m); L.append(l); V.append(v)
                R.append(rewards); D.append(dones)
                obs = new_obs
            steps += n * args.envs
            with torch.no_grad():
                last_v = model.act(torch.as_tensor(obs, device=device))[3]
            r = torch.as_tensor(np.array(R), device=device)
            val = torch.as_tensor(np.array(V), device=device)
            done = torch.as_tensor(np.array(D), device=device)
            next_val = torch.cat([val[1:], last_v[None]], dim=0)
            advantages, returns = gae(r, val, next_val, done)
            batch_obs = torch.as_tensor(np.array(S).reshape(-1, OBS_DIM), device=device)
            bj = torch.as_tensor(np.array(J).reshape(-1), device=device, dtype=torch.long)
            bm = torch.as_tensor(np.array(M).reshape(-1), device=device)
            blogp = torch.as_tensor(np.array(L).reshape(-1), device=device)
            badv = advantages.reshape(-1)
            bret = returns.reshape(-1)
            badv = (badv - badv.mean()) / (badv.std(unbiased=False) + 1e-8)
            inds = torch.randperm(len(badv), device=device)
            total_loss = 0.
            num = 0
            for _ in range(4):
                inds = inds[torch.randperm(len(inds), device=device)]
                for idx in inds.split(min(256, len(inds))):
                    newlogp, entropy, pred_value = model.evaluate_actions(batch_obs[idx], bj[idx], bm[idx])
                    pg = clipped_policy_loss(newlogp, blogp[idx], badv[idx])
                    loss = pg + 0.5 * (pred_value - bret[idx]).square().mean() - 0.01 * entropy.mean()
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optim.step()
                    total_loss += float(loss.detach())
                    num += 1
            print(f'UPDATE steps={steps}/{args.total_steps} loss={total_loss/max(num,1):.4f} elapsed={time.perf_counter()-t0:.1f}s', flush=True)
            if steps >= next_ckpt:
                snapshot = make_payload(model, steps=steps, trained=True, seed=args.seed,
                                        optimizer=optim.state_dict(), torch_rng=torch.get_rng_state(),
                                        numpy_rng=np.random.get_state(), python_rng=random.getstate())
                atomic_save(snapshot, args.checkpoint_dir / 'model_v4_resume.pt')
                next_ckpt = (steps // args.checkpoint_every + 1) * args.checkpoint_every
        final = make_payload(model, steps=steps, trained=True, seed=args.seed)
        atomic_save(final, args.out)
        print(f'FINAL_PT {args.out} steps={steps}', flush=True)
        if args.eval_episodes > 0 and env_factory is make_env:
            stats = evaluate(model, eval_seeds, device)
            (args.checkpoint_dir / 'initial_50k_eval.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
            print('EVAL', json.dumps(stats, ensure_ascii=False), flush=True)
        return model
    finally:
        for env in envs:
            env.close()


if __name__ == '__main__':
    run(parse_args())
