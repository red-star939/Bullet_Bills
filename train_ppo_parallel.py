"""Windows-spawn-safe multi-process rollout collector for BulletBill PPO v1.

One learner, N CPU environment workers. Run from the project root:
  .\\.venv\\Scripts\\python.exe train_ppo_parallel.py --num-envs 16 --total-steps 1000000

Environment instances must be created *inside* worker processes. All PPO
inference, policy updates and checkpointing occur in the parent process.
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import random
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import torch

from models.ppo_policy import ACTION_TABLE, ActorCritic, StateEncoder, checkpoint_payload, load_platforms, load_policy
from train_ppo import compute_gae, ppo_update


def _worker(connection, env_factory: str):
    """Top-level function for Windows 'spawn'. No torch inference in workers."""
    env = None
    try:
        module_name, function_name = env_factory.split(':', 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        env = factory(render_mode='none')
        while True:
            command, data = connection.recv()
            if command == 'reset':
                obs, info = env.reset(seed=int(data))
                connection.send(('ok', (obs, info)))
            elif command == 'step':
                action = ACTION_TABLE[int(data)].copy()
                result = env.step(action)
                connection.send(('ok', result))
            elif command == 'close':
                connection.send(('ok', None))
                break
            else:
                raise ValueError(f'Unknown worker command: {command}')
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class EnvPool:
    def __init__(self, n: int, factory: str):
        ctx = mp.get_context('spawn')
        self.conns = []
        self.processes = []
        try:
            for _ in range(n):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, factory), daemon=True)
                p.start()
                child.close()
                self.conns.append(parent)
                self.processes.append(p)
        except BaseException:
            self.close()
            raise

    def execute(self, indices, commands):
        for i, command in zip(indices, commands):
            try:
                self.conns[i].send(command)
            except (OSError, EOFError, BrokenPipeError) as exc:
                raise RuntimeError(f'Worker {i} failed on send: exitcode={self.processes[i].exitcode}') from exc
        output = []
        for i in indices:
            try:
                if not self.conns[i].poll(120):
                    raise RuntimeError(f'Worker {i} timeout (exitcode={self.processes[i].exitcode})')
                status, payload = self.conns[i].recv()
            except (OSError, EOFError, BrokenPipeError) as exc:
                raise RuntimeError(f'Worker {i} failed on receive: exitcode={self.processes[i].exitcode}') from exc
            if status != 'ok':
                raise RuntimeError(f'Worker {i} failed:\n{payload}')
            output.append(payload)
        return output

    def close(self):
        for conn in self.conns:
            try:
                conn.send(('close', None))
            except (OSError, EOFError):
                pass
        for conn in self.conns:
            try:
                conn.close()
            except OSError:
                pass
        for p in self.processes:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)


def evaluate_workers_gae(per_worker, gamma, lam):
    """Never propagate GAE between different envs or across episode boundaries."""
    flat = {k: [] for k in ('states', 'actions', 'logps', 'advantages', 'returns')}
    for worker_rollout in per_worker:
        if not worker_rollout:
            continue
        adv, ret = compute_gae(
            [x['reward'] for x in worker_rollout],
            [x['value'] for x in worker_rollout],
            [x['next_value'] for x in worker_rollout],
            [x['terminated'] for x in worker_rollout],
            [x['boundary'] for x in worker_rollout],
            gamma=gamma, lam=lam,
        )
        for j, transition in enumerate(worker_rollout):
            flat['states'].append(transition['state'])
            flat['actions'].append(transition['action'])
            flat['logps'].append(transition['logp'])
            flat['advantages'].append(adv[j])
            flat['returns'].append(ret[j])
    return flat


def train(args):
    if args.num_envs < 1:
        raise ValueError('--num-envs must be positive')
    # A single learner should not spawn 16 intra-op Torch threads as well.
    torch.set_num_threads(max(1, args.torch_threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    encoder = StateEncoder(load_platforms(args.map))
    device = torch.device(args.device)
    net = ActorCritic().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    global_steps = 0
    episode_count = 0
    recent_survival = deque(maxlen=100)
    recent_returns = deque(maxlen=100)
    if args.resume:
        net, ck = load_policy(args.resume, args.device)
        net.train()
        global_steps = int(ck.get('total_env_steps', 0))
        if 'optimizer_state_dict' in ck:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
        extra = ck.get('extra') or {}
        if extra.get('rng_numpy') is not None:
            rng.bit_generator.state = extra['rng_numpy']
        if extra.get('rng_torch') is not None:
            torch.set_rng_state(extra['rng_torch'])
        if extra.get('rng_python') is not None:
            random.setstate(extra['rng_python'])
        episode_count = int(extra.get('episodes_seen', 0))
        recent_survival.extend(extra.get('recent_survival_s', []))
    net.train()
    start = time.perf_counter()
    obs = [None] * args.num_envs
    info = [None] * args.num_envs
    ep_len = np.zeros(args.num_envs, dtype=np.int32)
    ep_return = np.zeros(args.num_envs, dtype=np.float64)
    next_log = ((global_steps // args.log_interval) + 1) * args.log_interval
    next_save = ((global_steps // args.save_interval) + 1) * args.save_interval
    with_pool = EnvPool(args.num_envs, args.env_factory)
    try:
        ids = list(range(args.num_envs))
        results = with_pool.execute(ids, [('reset', int(rng.integers(0, 2**32, dtype=np.uint64))) for _ in ids])
        for i, (o, inf) in enumerate(results):
            obs[i], info[i] = o, inf
        while global_steps < args.total_steps:
            per_worker = [[] for _ in ids]
            rollout_steps = 0
            # Count rollout budget in environment transitions, not vector ticks.
            while rollout_steps < args.rollout_steps and global_steps < args.total_steps:
                count = min(args.num_envs, args.rollout_steps - rollout_steps,
                            args.total_steps - global_steps)
                active = ids[:count]
                states = np.stack([encoder.encode(obs[i], float(info[i].get('time_elapsed', 0.)))
                                   for i in active], axis=0)
                with torch.no_grad():
                    acts, lps, vals = net.act(torch.as_tensor(states, device=device))
                action_ids = acts.cpu().numpy().astype(int)
                logps = lps.cpu().numpy()
                values = vals.cpu().numpy()
                transitions = with_pool.execute(active, [('step', int(a)) for a in action_ids])
                next_states = np.stack([
                    encoder.encode(nobs, float(ninfo.get('time_elapsed', 0.)))
                    for nobs, _, _, _, ninfo in transitions
                ])
                with torch.no_grad():
                    next_values = net(torch.as_tensor(next_states, device=device))[1].cpu().numpy()
                resets = []
                for j, i in enumerate(active):
                    nobs, _, term, trunc, ninfo = transitions[j]
                    term, trunc = bool(term), bool(trunc)
                    game_state = str(ninfo.get('game_state', 'unknown'))
                    death = term and game_state == 'game_over'
                    success = term and game_state in ('game_clear', 'clear', 'win')
                    reward = args.alive_reward + (args.death_penalty if death else 0.)
                    if success:
                        reward += args.success_bonus
                    if trunc and not term:
                        reward += args.timeout_bonus
                    ep_len[i] += 1
                    ep_return[i] += reward
                    cutoff = ep_len[i] >= args.max_episode_steps
                    boundary = bool(term or trunc or cutoff)
                    per_worker[i].append(dict(state=states[j], action=int(action_ids[j]),
                                              logp=float(logps[j]), value=float(values[j]),
                                              next_value=float(next_values[j]), reward=reward,
                                              terminated=term, boundary=boundary))
                    if boundary:
                        episode_count += 1
                        recent_survival.append(float(ninfo.get('time_elapsed', 0.)))
                        recent_returns.append(float(ep_return[i]))
                        resets.append(i)
                        ep_len[i] = 0
                        ep_return[i] = 0.
                    else:
                        obs[i], info[i] = nobs, ninfo
                if resets:
                    results = with_pool.execute(resets, [
                        ('reset', int(rng.integers(0, 2**32, dtype=np.uint64))) for _ in resets
                    ])
                    for i, (o, inf) in zip(resets, results):
                        obs[i], info[i] = o, inf
                global_steps += count
                rollout_steps += count
            batch = evaluate_workers_gae(per_worker, args.gamma, args.gae_lambda)
            stats = ppo_update(net, optimizer, batch, epochs=args.epochs,
                               minibatch=args.minibatch, clip=args.clip,
                               vf_coef=args.vf_coef, ent_coef=args.ent_coef,
                               max_grad_norm=args.max_grad_norm, target_kl=args.target_kl)
            if global_steps >= next_log or global_steps >= args.total_steps:
                elapsed = time.perf_counter() - start
                print('STEPS', global_steps, 'EPISODES', episode_count,
                      'LAST100_SURVIVAL', round(float(np.mean(recent_survival)), 3) if recent_survival else 'NA',
                      'ENV_STEPS_PER_S', round((global_steps - (int(ck.get('total_env_steps',0)) if args.resume else 0)) / max(elapsed,1e-6), 1),
                      'KL', round(stats['approx_kl'], 5), 'ENTROPY', round(stats['entropy'], 3), flush=True)
                while next_log <= global_steps:
                    next_log += args.log_interval
            if global_steps >= next_save or global_steps >= args.total_steps:
                extra = {'args': vars(args), 'rng_numpy': rng.bit_generator.state,
                         'rng_torch': torch.get_rng_state(), 'rng_python': random.getstate(),
                         'episodes_seen': episode_count, 'recent_survival_s': list(recent_survival)}
                payload = checkpoint_payload(net, trained=True, total_steps=global_steps,
                                             optimizer=optimizer, extra=extra)
                destination = out / f'mario_ppo_step_{global_steps:09d}.pt'
                torch.save(payload, destination)
                torch.save(payload, out / 'mario_ppo_latest.pt')
                (out / 'latest_metrics.json').write_text(json.dumps({
                    'global_steps': global_steps, 'episode_count': episode_count,
                    'recent_mean_survival_s': float(np.mean(recent_survival)) if recent_survival else None,
                    'recent_mean_reward': float(np.mean(recent_returns)) if recent_returns else None,
                    'env_steps_per_second': (global_steps - (int(ck.get('total_env_steps',0)) if args.resume else 0)) /
                                            max(time.perf_counter() - start, 1e-6),
                    'stats': stats, 'num_envs': args.num_envs,
                }, indent=2), encoding='utf-8')
                print('CHECKPOINT', destination, flush=True)
                while next_save <= global_steps:
                    next_save += args.save_interval
    finally:
        with_pool.close()


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--torch-threads', type=int, default=1,
                   help='PyTorch threads in learner. Keep 1-2 for 16 env workers.')
    p.add_argument('--env-factory', default='agent_env:create_env',
                   help='Importable factory module:function, called in each worker.')
    p.add_argument('--total-steps', type=int, default=1_000_000)
    p.add_argument('--rollout-steps', type=int, default=4096,
                   help='Total transitions across all workers per PPO update.')
    p.add_argument('--max-episode-steps', type=int, default=900)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatch', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=.995)
    p.add_argument('--gae-lambda', type=float, default=.95)
    p.add_argument('--clip', type=float, default=.2)
    p.add_argument('--vf-coef', type=float, default=.5)
    p.add_argument('--ent-coef', type=float, default=.01)
    p.add_argument('--max-grad-norm', type=float, default=.5)
    p.add_argument('--target-kl', type=float, default=.03)
    p.add_argument('--alive-reward', type=float, default=.01)
    p.add_argument('--death-penalty', type=float, default=-1.)
    p.add_argument('--timeout-bonus', type=float, default=0.)
    p.add_argument('--success-bonus', type=float, default=1.)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--map', default='map_data/object_tiles.csv')
    p.add_argument('--out', default='models/ppo_runs_parallel')
    p.add_argument('--resume', default=None)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--log-interval', type=int, default=10000)
    p.add_argument('--save-interval', type=int, default=25000)
    a = p.parse_args()
    if any(getattr(a, k) < 1 for k in ('num_envs', 'total_steps', 'rollout_steps',
            'max_episode_steps', 'epochs', 'minibatch', 'log_interval', 'save_interval')):
        p.error('count parameters must be positive')
    return a


if __name__ == '__main__':
    mp.freeze_support()
    train(parse())
