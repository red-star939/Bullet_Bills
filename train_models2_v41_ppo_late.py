"""Late-game 120-180s PPO curriculum using policy-driven real-state warmup.

Workers own gym environments and expensive 148D FeatureExtractor; learner owns PPO,
seed selection, optimizer, evaluation. Uses on-policy transitions only.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import secrets
import time
import traceback
import sys
import types
import importlib.util
from collections import deque
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import torch
# Test-only dependency stubs for the isolated multiprocessing smoke test.
if '--test-env' in sys.argv:
    if importlib.util.find_spec('gymnasium') is None:
        sys.modules['gymnasium'] = types.ModuleType('gymnasium')
    if importlib.util.find_spec('kymnasium') is None:
        stub = types.ModuleType('kymnasium')
        stub.Agent = type('Agent', (), {})
        sys.modules['kymnasium'] = stub
from train_models2_v41_ppo import (
    load_checkpoint, save_pt, seed_sets, teacher_data, auxiliary_bc,
    predict_action, evaluate_model, summarize, write_rows,
)
import agent_ppo_v10 as v10


class _FakeEnv:
    """Only used by the multiprocessing integration test; never for real training."""
    def reset(self, seed=None):
        self.n = 0
        return {'mario': np.array([504, 672, 552, 720, 0, 0, 0, 0], np.float32)}, {'time_elapsed': 0.0}

    def step(self, action):
        self.n += 1
        o = {'mario': np.array([504 + self.n, 672, 552 + self.n, 720, 0, 0, 0, 0], np.float32)}
        term = self.n >= 3
        return o, 0., term, False, {'time_elapsed': self.n / 15., 'game_state': 'game_over' if term else 'playing'}

    def close(self):
        pass


class _FakeFE:
    def reset(self):
        pass

    def __call__(self, obs, info):
        x = np.zeros(v10.OBS_DIM, np.float32)
        x[0] = obs['mario'][0] / 1000
        x[1] = info['time_elapsed'] / 20
        return x, 0.1, 0.2


def _worker(conn, worker_idx, max_episode_steps, test_env=False):
    """No policy model is loaded in children. Avoids 16x PyTorch duplication."""
    env = None
    try:
        if test_env:
            env, fe = _FakeEnv(), _FakeFE()
        else:
            import gymnasium as gym
            env = gym.make(id=v10.ENV_ID, render_mode='none', bgm=False)
            fe = v10.FeatureExtractor()
        obs, info, phi, steps, active_seed = None, None, 0., 0, None
        conn.send(('ready', worker_idx))
        while True:
            cmd, data = conn.recv()
            if cmd == 'close':
                break
            if cmd == 'reset':
                active_seed = int(data)
                obs, info = env.reset(seed=active_seed)
                fe.reset()
                s, edge, bullet = fe(obs, info)
                phi = v10.potential(edge, bullet)
                steps = 0
                conn.send(('reset', (np.asarray(s, np.float32), float(phi), float(info.get('time_elapsed', 0.)))))
                continue
            if cmd != 'step':
                raise ValueError(f'Unexpected worker command {cmd!r}')
            if obs is None:
                raise RuntimeError('Must reset worker before stepping')
            jump, move = data
            action = v10.to_env_action(float(move), int(jump))
            on_ground = float(obs['mario'][7]) < 0.5
            prev_inv = float(obs['mario'][6]) > 0.5
            new_obs, _, term, trunc, new_info = env.step(action)
            steps += 1
            s2, edge2, bullet2 = fe(new_obs, new_info)
            done = bool(term or trunc or steps >= max_episode_steps)
            dead = bool(term and new_info.get('game_state') == 'game_over')
            phi2 = 0. if done else v10.potential(edge2, bullet2)
            reward = v10.SURVIVE + v10.SHAPE * (v10.GAMMA * phi2 - phi)
            if on_ground and action[1] > 0:
                reward += v10.JUMP_COST
            if dead:
                reward += v10.DEATH
            if float(new_obs['mario'][6]) > 0.5 and not prev_inv:
                when = float(new_info.get('time_elapsed', 0.))
                reward -= v10.STAR_EARLY_PENALTY * max(0., 1. - when / v10.STAR_FREE_TIME)
            event = None
            if done:
                event = {'survival_s': float(new_info.get('time_elapsed', steps * v10.DT)),
                         'death': dead, 'seed': active_seed, 'episode_steps': steps}
            obs, info, phi = new_obs, new_info, phi2
            conn.send(('step', (np.asarray(s2, np.float32), float(reward), float(done), event, float(new_info.get('time_elapsed', steps * v10.DT)))))
    except BaseException:
        try:
            conn.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        conn.close()


class ParallelEnv:
    def __init__(self, n, max_episode_steps, test_env=False, start_method='spawn'):
        ctx = mp.get_context(start_method)
        self.pipes = []
        self.processes = []
        try:
            for i in range(n):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, i, max_episode_steps, test_env), daemon=True)
                p.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(p)
            for i in range(n):
                tag, data = self._receive(i)
                if tag != 'ready':
                    raise RuntimeError(f'Worker {i} failed startup: {tag} {data}')
        except BaseException:
            self.close()
            raise

    def _receive(self, i):
        try:
            tag, data = self.pipes[i].recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise RuntimeError(f'Worker {i} disconnected (exit={self.processes[i].exitcode})') from exc
        if tag == 'error':
            raise RuntimeError(f'Worker {i} failed:\n{data}')
        return tag, data

    def reset(self, indices, seeds):
        for i, seed in zip(indices, seeds):
            self.pipes[i].send(('reset', int(seed)))
        result = []
        for i in indices:
            tag, data = self._receive(i)
            if tag != 'reset':
                raise RuntimeError(f'Unexpected {tag} from worker {i} on reset')
            result.append(data)
        return result

    def step_selected(self, indices, jumps, moves):
        for i, j, m in zip(indices, jumps, moves):
            self.pipes[i].send(('step', (int(j), float(m))))
        out = []
        for i in indices:
            tag, data = self._receive(i)
            if tag != 'step':
                raise RuntimeError(f'Unexpected {tag} from worker {i} on step')
            out.append(data)
        return out

    def step(self, jumps, moves):
        return self.step_selected(list(range(len(self.pipes))), jumps, moves)

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except (OSError, BrokenPipeError, EOFError):
                pass
        for p in self.processes:
            p.join(timeout=2.)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.)
        for pipe in self.pipes:
            pipe.close()


def _atomic_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def _resume_path(out):
    return out.with_name(out.stem + '_resume.pt')


def _last_path(out):
    return out.with_name(out.stem + '_last.pt')


def _full_save(path, *, model, optimizer, baseline, rng, completed, update, best, last_eval,
               master_seed, eval_seeds, final_seeds, used_train, records, episode_records, episode_times,
               args, init):
    """Optimizer + PRNG state resume, at rollout boundary. Environment states reset on resume."""
    _atomic_torch(path, {
        'version': 1,
        'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict(),
        'baseline': {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()},
        'rng_state': rng.bit_generator.state,
        'torch_rng_state': torch.get_rng_state(),
        'completed': completed, 'update': update, 'best': best, 'last_eval': last_eval,
        'master_seed': master_seed, 'eval_seeds': eval_seeds, 'final_seeds': final_seeds,
        'used_train': sorted(used_train), 'records': records,
        'episode_records': episode_records, 'episode_times': list(episode_times),
        'init': str(init), 'out': str(args.out),
        'hidden': model.jump_head.in_features, 'envs': args.envs,
        'batch_steps': args.batch_steps, 'max_episode_steps': args.max_episode_steps,
        'teacher_data': str(args.teacher_data) if args.teacher_data else None,
        'total_steps_at_save': args.total_steps,
        'curriculum_min_s': args.curriculum_min_s,
        'curriculum_max_s': args.curriculum_max_s,
    })


def train(args):
    args.log_dir.mkdir(parents=True, exist_ok=True)  # before writing seed files
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.init.resolve() == args.out.resolve():
        raise ValueError('Output must not overwrite the init checkpoint')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    test_env = getattr(args, 'test_env', False)

    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ck.get('version') != 1:
            raise ValueError('Unsupported resume checkpoint')
        for name in ('envs', 'batch_steps', 'max_episode_steps'):
            if ck[name] != getattr(args, name):
                raise ValueError(f'Resume {name} mismatch: {ck[name]} != {getattr(args,name)}')
        if Path(ck['out']).resolve() != args.out.resolve():
            raise ValueError('Resume output checkpoint path must be unchanged')
        if not args.out.is_file():
            raise FileNotFoundError('Best .pt must exist next to full resume checkpoint')
        for name in ('curriculum_min_s', 'curriculum_max_s'):
            if float(ck.get(name, -1.)) != float(getattr(args, name)):
                raise ValueError(f'Resume {name} mismatch: {ck.get(name)} != {getattr(args,name)}')
        if ck['teacher_data'] != (str(args.teacher_data) if args.teacher_data else None):
            raise ValueError('Resume teacher data path mismatch')
        completed = int(ck['completed'])
        if args.total_steps <= completed:
            raise ValueError(f'--total-steps must exceed already-completed {completed}; total is cumulative')
        master_seed = int(ck['master_seed'])
        rng = np.random.default_rng()
        rng.bit_generator.state = ck['rng_state']
        hidden = int(ck['hidden'])
        model = v10.ActorCritic(obs_dim=v10.OBS_DIM, hidden=hidden)
        model.load_state_dict(ck['model'])
        baseline = v10.ActorCritic(obs_dim=v10.OBS_DIM, hidden=hidden)
        baseline.load_state_dict(ck['baseline'])
        best = float(ck['best'])
        last_eval_at = int(ck['last_eval'])
        update_start = int(ck['update'])
        eval_seeds = ck['eval_seeds']
        final_seeds = ck['final_seeds']
        used_train = set(ck['used_train'])
        records = ck['records']
        episode_records = ck['episode_records']
        episode_times = deque(ck['episode_times'], maxlen=100)
        print(f'RESUME completed={completed} target_total={args.total_steps} from={args.resume}', flush=True)
    else:
        master_seed = args.seed if args.seed is not None else secrets.randbelow(2_147_483_647)
        np.random.seed(master_seed)
        torch.manual_seed(master_seed)
        rng = np.random.default_rng(master_seed)
        eval_seeds, final_seeds = seed_sets(master_seed, args.eval_episodes, args.final_episodes)
        model, original = load_checkpoint(args.init)
        hidden = int(original['hidden'])
        baseline = v10.ActorCritic(obs_dim=v10.OBS_DIM, hidden=hidden)
        baseline.load_state_dict(original['model'])
        completed, update_start, last_eval_at = 0, 0, 0
        used_train, records, episode_records = set(), [], []
        episode_times = deque(maxlen=100)
        best = float('-inf')
    baseline.eval()
    for param in baseline.parameters():
        param.requires_grad_(False)
    # Teacher subsampling must not alter restored training RNG sequence.
    teacher_rng = np.random.default_rng(master_seed ^ 0x5F3759DF)
    teacher = teacher_data(args.teacher_data, v10.OBS_DIM, args.max_teacher_rows, teacher_rng)
    if args.resume:
        torch.set_rng_state(ck['torch_rng_state'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    if args.resume:
        optimizer.load_state_dict(ck['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.lr
    forbidden = set(eval_seeds + final_seeds)
    def next_seed():
        for _ in range(10000):
            s = int(rng.integers(0, 2_147_483_647))
            if s not in forbidden and s not in used_train:
                used_train.add(s)
                return s
        raise RuntimeError('Failed to allocate unused random environment seed')

    _atomic_json(args.log_dir / 'eval_seeds.json', eval_seeds)
    _atomic_json(args.log_dir / 'final_holdout_seeds.json', final_seeds)
    _atomic_json(args.log_dir / 'seeds.json', {'master_seed': master_seed,
                'eval_seeds': eval_seeds, 'final_holdout_seeds': final_seeds})
    if not args.resume:
        # First baseline evaluation before workers start. Test mode patches evaluation.
        initial_rows = _evaluate(model, eval_seeds, args.eval_max_steps, test_env)
        best = summarize(initial_rows)['mean_s']
        write_rows(args.log_dir / 'eval_initial.csv', initial_rows)
        save_pt(args.out, model, hidden, {'steps': 0, 'master_seed': master_seed,
                         'eval_mean_s': best, 'training': 'parallel_ppo_v10_reward'})
        print(f'BASELINE mean={best:.3f}s', flush=True)

    batch = args.envs * args.batch_steps
    updates_est = math.ceil((args.total_steps - completed) / batch)
    print(f'START env_workers={args.envs} method=spawn batch={batch} '
          f'target_total={args.total_steps} updates_remaining≈{updates_est} '
          f'teacher_rows={0 if teacher is None else len(teacher[0])}', flush=True)
    workers = ParallelEnv(args.envs, args.max_episode_steps, test_env=test_env)
    try:
        initial_seeds = [next_seed() for _ in range(args.envs)]
        reset_values = workers.reset(list(range(args.envs)), initial_seeds)
        states = np.stack([x[0] for x in reset_values]).astype(np.float32)
        warmup_stats = {'attempts': 0, 'failures': 0, 'completed': 0, 'env_steps': 0,
                        'reached_target_times': []}

        def prepare_late(indices):
            """Roll *real* full episodes to random late timestamps. No warmup samples enter PPO.

            Uses current policy with deterministic actions; each failed pre-120s run is
            reset with a fresh seed. Limits unsuccessful attempts per target/worker.
            """
            if not indices:
                return
            targets = {i: float(rng.uniform(args.curriculum_min_s,
                                           args.curriculum_max_s)) for i in indices}
            counts = {i: 0 for i in indices}
            alive = set(indices)
            times = {i: 0. for i in indices}
            while alive:
                active = sorted(alive)
                model.eval()
                js, ms, _, _ = predict_action(model, states[active], deterministic=True)
                results = workers.step_selected(active, js, ms)
                warmup_stats['env_steps'] += len(active)
                reset_ids = []
                for i, (state, reward, done, event, now_s) in zip(active, results):
                    states[i] = state
                    times[i] = now_s
                    if done:
                        counts[i] += 1
                        warmup_stats['attempts'] += 1
                        warmup_stats['failures'] += 1
                        if counts[i] >= args.warmup_max_attempts:
                            raise RuntimeError(
                                f'Worker {i}: failed to survive to target {targets[i]:.1f}s '
                                f'in {counts[i]} policy warmup episodes. This model cannot '
                                'reliably generate late-game states. Consider earlier targets '
                                '(e.g. 80-110s) or environment snapshot/reset support. '
                                'No synthetic time teleport was applied.')
                        reset_ids.append(i)
                    elif now_s >= targets[i]:
                        alive.remove(i)
                        warmup_stats['completed'] += 1
                        warmup_stats['reached_target_times'].append(now_s)
                if reset_ids:
                    new_seeds = [next_seed() for _ in reset_ids]
                    vals = workers.reset(reset_ids, new_seeds)
                    for i, val in zip(reset_ids, vals):
                        states[i] = val[0]
                        times[i] = val[2]
                if warmup_stats['env_steps'] and warmup_stats['env_steps'] % 250000 < len(active):
                    print(f'WARMUP steps_excluded={warmup_stats["env_steps"]} '
                          f'ready={len(indices)-len(alive)}/{len(indices)} '
                          f'failed_episodes={warmup_stats["failures"]}', flush=True)

        print(f'CURRICULUM real state warmup: [{args.curriculum_min_s}, '
              f'{args.curriculum_max_s}]s; no PPO before target. '
              f'Max attempts per worker={args.warmup_max_attempts}', flush=True)
        prepare_late(list(range(args.envs)))
        last_log_at = time.monotonic()
        last_checkpoint_at = completed
        update = update_start
        while completed < args.total_steps:
            update += 1
            model.train()
            T = min(args.batch_steps, math.ceil((args.total_steps - completed) / args.envs))
            S = np.zeros((T, args.envs, v10.OBS_DIM), np.float32)
            J = np.zeros((T, args.envs), np.int64)
            M = np.zeros((T, args.envs), np.float32)
            LP = np.zeros((T, args.envs), np.float32)
            V = np.zeros((T, args.envs), np.float32)
            R = np.zeros((T, args.envs), np.float32)
            D = np.zeros((T, args.envs), np.float32)
            for t in range(T):
                jumps, moves, lps, values = predict_action(model, states)
                S[t], J[t], M[t], LP[t], V[t] = states, jumps, moves, lps, values
                results = workers.step(jumps, moves)
                finished = []
                for i, (new_state, rew, done, event, elapsed_s) in enumerate(results):
                    states[i] = new_state
                    R[t, i] = rew
                    D[t, i] = done
                    if event is not None:
                        episode_times.append(event['survival_s'])
                        event['update'] = update
                        episode_records.append(event)
                        finished.append(i)
                if finished:
                    seeds = [next_seed() for _ in finished]
                    for i, (new_state, phi, now_s) in zip(finished, workers.reset(finished, seeds)):
                        states[i] = new_state
                    prepare_late(finished)
            completed += T * args.envs
            with torch.no_grad():
                last_value = model.value(torch.from_numpy(states)).numpy()
            adv = v10.compute_gae(R, V, D, last_value, v10.GAMMA, args.gae_lambda)
            bs = torch.from_numpy(S.reshape(-1, v10.OBS_DIM))
            bj = torch.from_numpy(J.reshape(-1))
            bm = torch.from_numpy(M.reshape(-1))
            blp = torch.from_numpy(LP.reshape(-1))
            ba = torch.from_numpy(adv.reshape(-1))
            br = torch.from_numpy((adv + V).reshape(-1))
            n = len(bs)
            mb = max(1, math.ceil(n / args.minibatches))
            losses, clips = [], []
            fraction = min(1., completed / args.total_steps)
            bc_coeff = (args.bc_coef * max(0., 1. - fraction / max(args.bc_decay, 1e-6))) if teacher else 0.
            for _ in range(args.epochs):
                indices = rng.permutation(n)
                for k in range(0, n, mb):
                    ids = torch.from_numpy(indices[k:k + mb])
                    new_lp, ent_j, ent_m, new_v = model.evaluate(bs[ids], bj[ids], bm[ids])
                    ratio = torch.exp(new_lp - blp[ids])
                    normalized = ba[ids]
                    normalized = (normalized - normalized.mean()) / (normalized.std(unbiased=False) + 1e-8)
                    policy_loss = torch.maximum(-normalized * ratio,
                        -normalized * ratio.clamp(1. - args.clip, 1. + args.clip)).mean()
                    value_loss = 0.5 * (new_v - br[ids]).square().mean()
                    loss = policy_loss + args.vf_coef * value_loss - args.ent_jump * ent_j.mean() - args.ent_move * ent_m.mean()
                    if bc_coeff > 0:
                        loss = loss + bc_coeff * auxiliary_bc(model, teacher, rng, args.bc_batch,
                            args.override_weight, baseline, args.preserve_kl)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    clips.append(float(((ratio.detach() - 1).abs() > args.clip).float().mean()))
            mean_recent = float(np.mean(episode_times)) if episode_times else None
            rec = {'update': update, 'steps': completed, 'recent_mean_s': mean_recent,
                   'loss': float(np.mean(losses)), 'clip_fraction': float(np.mean(clips)),
                   'bc_coef': bc_coeff, 'elapsed_since_log_s': round(time.monotonic() - last_log_at, 2)}
            records.append(rec)
            if update % args.log_every == 0 or update == update_start + 1:
                print(f'UPDATE {update} steps={completed}/{args.total_steps} recent_mean={mean_recent} '
                      f'loss={rec["loss"]:.4f} clip={rec["clip_fraction"]:.3f} '
                      f'wall={rec["elapsed_since_log_s"]:.1f}s', flush=True)
                last_log_at = time.monotonic()
            do_eval = completed - last_eval_at >= args.eval_interval or completed >= args.total_steps
            if do_eval:
                last_eval_at = completed
                rows = _evaluate(model, eval_seeds, args.eval_max_steps, test_env)
                score = summarize(rows)
                write_rows(args.log_dir / f'eval_step_{completed}.csv', rows)
                print(f'EVAL steps={completed} mean={score["mean_s"]:.3f}s '
                      f'reached150={score["reached_150"]}/{len(rows)} best={best:.3f}s', flush=True)
                if score['mean_s'] > best:
                    best = score['mean_s']
                    save_pt(args.out, model, hidden, {'steps': completed, 'master_seed': master_seed,
                        'eval_mean_s': best, 'training': 'parallel_ppo_v10_reward'})
                    print(f'BEST_PT {args.out} steps={completed} mean={best:.3f}s', flush=True)
                save_pt(_last_path(args.out), model, hidden, {'steps': completed, 'master_seed': master_seed,
                    'eval_mean_s': score['mean_s'], 'training': 'parallel_ppo_v10_reward'})
            do_checkpoint = (completed - last_checkpoint_at >= args.checkpoint_interval or do_eval)
            if do_checkpoint:
                _full_save(_resume_path(args.out), model=model, optimizer=optimizer, baseline=baseline,
                   rng=rng, completed=completed, update=update, best=best, last_eval=last_eval_at,
                   master_seed=master_seed, eval_seeds=eval_seeds, final_seeds=final_seeds,
                   used_train=used_train, records=records, episode_records=episode_records,
                   episode_times=episode_times, args=args, init=args.init)
                _atomic_json(args.log_dir / 'history.json', records)
                _atomic_json(args.log_dir / 'train_episodes.json', episode_records)
                last_checkpoint_at = completed
        best_model, _ = load_checkpoint(args.out)
        final_rows = _evaluate(best_model, final_seeds, args.eval_max_steps, test_env)
        write_rows(args.log_dir / 'final_holdout.csv', final_rows)
        summary = {'output_pt': str(args.out), 'last_pt': str(_last_path(args.out)),
                   'resume_pt': str(_resume_path(args.out)), 'completed_steps': completed,
                   'best_eval_mean_s': best, 'final_holdout': summarize(final_rows),
                   'worker_count': args.envs, 'master_seed': master_seed,
                   'train_seed_count': len(used_train), 'notes': 'Resume preserves optimizer and PRNG; episode states reset'}
        summary['late_curriculum'] = {'min_s': args.curriculum_min_s,
             'max_s': args.curriculum_max_s, 'warmup': warmup_stats}
        _atomic_json(args.log_dir / 'summary.json', summary)
        _atomic_json(args.log_dir / 'train_seed_history.json', sorted(used_train))
        print('FINAL', json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    finally:
        workers.close()


def _evaluate(model, seeds, max_steps, test_env):
    if not test_env:
        return evaluate_model(model, seeds, max_steps)
    return [{'seed': int(s), 'survival_s': 0.2, 'steps': 3, 'status': 'terminated'} for s in seeds]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', type=Path, default=Path('models2/bullet_bill_ppo_v10_v41_distilled.pt'))
    p.add_argument('--out', type=Path, default=Path('models2/bullet_bill_ppo_v10_v41_parallel_best.pt'))
    p.add_argument('--teacher-data', type=Path)
    p.add_argument('--resume', type=Path, default=None,
                   help='Full training checkpoint; --total-steps is the CUMULATIVE target')
    p.add_argument('--log-dir', type=Path, default=Path('logs/models2_v41_ppo_parallel'))
    p.add_argument('--total-steps', type=int, default=5_000_000)
    p.add_argument('--envs', type=int, default=12)
    p.add_argument('--batch-steps', type=int, default=256)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatches', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--clip', type=float, default=0.15)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--ent-jump', type=float, default=0.005)
    p.add_argument('--ent-move', type=float, default=0.0005)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--bc-coef', type=float, default=0.02)
    p.add_argument('--bc-decay', type=float, default=0.5)
    p.add_argument('--bc-batch', type=int, default=256)
    p.add_argument('--override-weight', type=float, default=2.)
    p.add_argument('--preserve-kl', type=float, default=0.1)
    p.add_argument('--max-teacher-rows', type=int, default=100_000)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--final-episodes', type=int, default=50)
    p.add_argument('--eval-max-steps', type=int, default=9000)
    p.add_argument('--max-episode-steps', type=int, default=9000)
    p.add_argument('--eval-interval', type=int, default=50_000)
    p.add_argument('--checkpoint-interval', type=int, default=10_000)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--curriculum-min-s', type=float, default=120.)
    p.add_argument('--curriculum-max-s', type=float, default=165.,
                   help='Upper START time, must be <180s terminal. Training extends toward 180s.')
    p.add_argument('--warmup-max-attempts', type=int, default=25,
                   help='Fail fast if no valid 120+s start can be reached')
    p.add_argument('--test-env', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if min(args.total_steps, args.envs, args.batch_steps, args.eval_interval, args.checkpoint_interval,
           args.eval_episodes, args.final_episodes, args.threads, args.epochs,
           args.minibatches, args.max_episode_steps, args.eval_max_steps) < 1:
        p.error('Steps, episode counts, workers and threads must be positive')
    if (args.curriculum_min_s < 0 or args.curriculum_max_s >= 180 or
            args.curriculum_min_s > args.curriculum_max_s or args.warmup_max_attempts < 1):
        p.error('Curriculum requires 0 <= min <= max < 180 and positive max attempts')
    if args.max_episode_steps * v10.DT <= args.curriculum_max_s:
        p.error('max-episode-steps is too small to reach curriculum maximum start time')
    if args.lr <= 0 or args.bc_coef < 0 or args.preserve_kl < 0 or args.override_weight < 1:
        p.error('Invalid learning hyperparameter')
    return args


if __name__ == '__main__':
    mp.freeze_support()
    train(parse_args())
