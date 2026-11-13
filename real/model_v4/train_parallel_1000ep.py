"""Independent Model V4: asynchronous per-episode PPO data collection with 16 CPU workers.

Episode count is the training budget (not environment steps). All 16 workers use an
identical frozen policy snapshot per wave. PPO updates only after that wave finishes.
Original ModelV4 checkpoints and standalone evaluator remain compatible.
"""
from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import random
import secrets
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from model_v4_net import ModelV4, OBS_DIM, HIDDEN, obs_vector, env_action, make_payload

ROOT = Path(__file__).resolve().parent
ENV_ID = 'kymnasium/BulletBill-Continuous-Hard-Stage-2'


def environment():
    import gymnasium as gym
    import kymnasium  # noqa: F401
    return gym.make(id=ENV_ID, render_mode='none', bgm=False)


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_episode(env, model, seed, max_steps):
    """Run single episode, preserve exact stochastic actions and log-probs for PPO."""
    observation, info = env.reset(seed=int(seed))
    s, j, m, lp, v, rw, end = [], [], [], [], [], [], []
    ended, truncated = False, False
    with torch.inference_mode():
        for step in range(max_steps):
            feature = obs_vector(observation)
            x = torch.from_numpy(feature).unsqueeze(0)
            jump, move, logprob, value = model.act(x)
            ji, mi = int(jump.item()), float(move.item())
            observation, _, term, trunc, info = env.step(env_action(mi, ji))
            dead = bool(term and info.get('game_state') == 'game_over')
            s.append(feature); j.append(ji); m.append(mi)
            lp.append(float(logprob.item())); v.append(float(value.item()))
            rw.append(0.1 - 10.0 * float(dead) + (2.0 if term and not dead else 0.0))
            end.append(bool(term or trunc)); ended, truncated = bool(term), bool(trunc)
            if term or trunc:
                break
        bootstrap = 0.0
        # If max_steps or a Gym truncation was hit, the state is nonterminal.
        if not ended:
            feature = obs_vector(observation)
            bootstrap = float(model.act(torch.from_numpy(feature).unsqueeze(0), deterministic=True)[3].item())
    return dict(s=np.asarray(s, np.float32), j=np.asarray(j, np.int64),
                m=np.asarray(m, np.float32), lp=np.asarray(lp, np.float32),
                v=np.asarray(v, np.float32), rw=np.asarray(rw, np.float32),
                terminal=bool(ended), bootstrap=float(bootstrap),
                seconds=float(info.get('time_elapsed', 0)),
                reached150=bool(float(info.get('time_elapsed', 0)) >= 150),
                step_count=len(s), truncated=bool(truncated))


def worker(worker_id, task_queue, result_queue, max_steps):
    torch.set_num_threads(1)
    try:
        env = environment()
        model = ModelV4().cpu().eval()
        try:
            while True:
                task = task_queue.get()
                if task is None:
                    break
                index, seed, state = task
                try:
                    model.load_state_dict(state, strict=True)
                    # Seed RNG separately for action sampling. Resets affect environment RNG.
                    torch.manual_seed(int(seed))
                    rollout = run_episode(env, model, seed, max_steps)
                    result_queue.put((index, seed, rollout, None))
                except Exception:
                    result_queue.put((index, seed, None, traceback.format_exc()))
        finally:
            env.close()
    except Exception:
        result_queue.put((-1, -1, None, f'Worker {worker_id} initialization failure:\n{traceback.format_exc()}'))


def calculate_advantages(roll, gamma, lam):
    """Episode-boundary aware GAE, with truncation bootstrapping."""
    values = roll['v']; rewards = roll['rw']; n = len(rewards)
    adv = np.zeros(n, np.float32)
    carry = 0.0
    for t in range(n - 1, -1, -1):
        next_value = (roll['bootstrap'] if t == n-1 else float(values[t+1]))
        if t == n-1 and roll['terminal']:
            next_value = 0.0
        delta = float(rewards[t]) + gamma * next_value - float(values[t])
        carry = delta + gamma * lam * carry
        adv[t] = carry
    return adv, adv + values


def train_wave(model, optimizer, trajectories, device, args, rng):
    parts = []
    for tr in trajectories:
        adv, ret = calculate_advantages(tr, args.gamma, args.gae_lambda)
        parts.append((tr['s'], tr['j'], tr['m'], tr['lp'], adv, ret))
    # Flatten across episodes, but retain boundaries in GAE above.
    data = [np.concatenate([p[k] for p in parts]) for k in range(6)]
    total = len(data[0])
    if total > args.max_train_transitions:
        idx = rng.choice(total, size=args.max_train_transitions, replace=False)
        data = [a[idx] for a in data]
    s, j, m, old_lp, advantage, returns = [torch.as_tensor(a, device=device) for a in data]
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    model.train()
    losses, cliprates = [], []
    for _ in range(args.ppo_epochs):
        indices = torch.randperm(len(s), device=device)
        for ix in indices.split(args.minibatch):
            new_lp, entropy, value = model.evaluate_actions(s[ix], j[ix], m[ix])
            ratio = torch.exp(new_lp - old_lp[ix])
            pg = -torch.minimum(ratio * advantage[ix],
                                ratio.clamp(1-args.clip, 1+args.clip) * advantage[ix]).mean()
            loss = pg + args.value_coef * (value-returns[ix]).square().mean() - args.entropy_coef*entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            cliprates.append(float(((ratio-1).abs()>args.clip).float().mean().detach().cpu()))
    model.eval()
    return dict(loss=float(np.mean(losses)), clip=float(np.mean(cliprates)),
                used_transitions=len(s), collected_transitions=total)


def read_checkpoint(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('architecture') != 'independent_model_v4':
        raise ValueError('Only independent Model V4 checkpoints are allowed; PPO v10 is excluded')
    return ck


def main(args, worker_target=worker, env_factory=None):
    if args.episodes <= 0 or args.envs <= 0 or args.max_steps <= 0 or args.checkpoint_every <= 0:
        raise ValueError('episode, env, max_steps, checkpoint_every must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    torch.set_num_threads(1)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.parent.resolve() == args.checkpoint_dir.resolve():
        raise ValueError('Final model should be outside checkpoints/')
    if not args.resume and not args.init.is_file():
        raise FileNotFoundError(f'Initial checkpoint missing: {args.init}. Use --init with an existing Model V4 .pt')
    ck = read_checkpoint(args.resume if args.resume else args.init)
    model = ModelV4(int(ck.get('obs_dim', OBS_DIM)), int(ck.get('hidden', HIDDEN))).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    completed, total_steps = 0, 0
    master_seed = int(args.seed if args.seed is not None else secrets.randbelow(2**31-10**6)+1)
    np_rng = np.random.default_rng(master_seed)
    if args.resume:
        if 'optimizer' not in ck or 'episode_seeds' not in ck:
            raise ValueError('This is not a full 1000-episode resume checkpoint')
        if ck.get('trainer') != 'model_v4_parallel_episodes_v1':
            raise ValueError('Unsupported resume trainer')
        completed = int(ck['completed_episodes'])
        total_steps = int(ck['environment_steps'])
        master_seed = int(ck['master_seed'])
        episode_seeds = list(map(int, ck['episode_seeds']))
        eval_seeds = list(map(int, ck['eval_seeds']))
        if args.episodes > len(episode_seeds):
            # Preserve the original training/evaluation seeds when extending a run.
            used = set(episode_seeds) | set(eval_seeds)
            while len(episode_seeds) < args.episodes:
                candidate = int(np_rng.integers(1, 2_147_483_000))
                if candidate not in used:
                    used.add(candidate)
                    episode_seeds.append(candidate)
        if args.episodes < completed:
            raise ValueError('Target episode count is below checkpoint progress')
        optimizer.load_state_dict(ck['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        if 'torch_rng' in ck:
            torch.set_rng_state(ck['torch_rng'])
        if 'numpy_rng' in ck:
            np_rng.bit_generator.state = ck['numpy_rng']
        print(f'RESUME episodes={completed}/{args.episodes} env_steps={total_steps}', flush=True)
    else:
        # One run-wide collision-free list (train and holdout disjoint).
        seed_pool = np_rng.choice(2_147_482_999, size=args.episodes+args.eval_episodes, replace=False) + 1
        episode_seeds = seed_pool[:args.episodes].tolist()
        eval_seeds = seed_pool[args.episodes:].tolist()
        torch.manual_seed(master_seed)
    (args.checkpoint_dir / 'seed_plan.json').write_text(json.dumps({
        'master_seed': master_seed, 'train_seeds': episode_seeds, 'eval_seeds': eval_seeds}, indent=2), encoding='utf-8')
    def save_resume():
        payload = make_payload(model, steps=total_steps, trained=True, seed=master_seed,
                               optimizer=optimizer.state_dict(), trainer='model_v4_parallel_episodes_v1',
                               completed_episodes=completed, environment_steps=total_steps,
                               episode_seeds=episode_seeds, eval_seeds=eval_seeds,
                               master_seed=master_seed, torch_rng=torch.get_rng_state(),
                               numpy_rng=np_rng.bit_generator.state)
        atomic_save(payload, args.checkpoint_dir / 'model_v4_1000ep_resume.pt')
    print(f'START episodes={args.episodes} completed={completed} workers={args.envs} '
          f'device={device} master_seed={master_seed} max_steps={args.max_steps}', flush=True)
    ctx = mp.get_context('spawn')
    task_queues = [ctx.Queue(maxsize=1) for _ in range(args.envs)]
    result_queue = ctx.Queue(maxsize=args.envs * 2)
    processes = [ctx.Process(target=worker_target, args=(i, task_queues[i], result_queue, args.max_steps), daemon=True)
                 for i in range(args.envs)]
    for process in processes:
        process.start()
    t0 = time.perf_counter()
    try:
        while completed < args.episodes:
            count = min(args.envs, args.episodes-completed)
            # All workers in a wave receive one consistent, frozen policy version.
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            for i in range(count):
                task_queues[i].put((completed+i, episode_seeds[completed+i], state))
            trajectories = {}
            while len(trajectories) < count:
                try:
                    index, seed, roll, err = result_queue.get(timeout=5)
                except queue.Empty:
                    if any(not processes[i].is_alive() for i in range(count)):
                        raise RuntimeError('Worker unexpectedly exited; check child logs and environment install')
                    continue
                if err:
                    raise RuntimeError(f'Worker failed episode={index} seed={seed}:\n{err}')
                trajectories[index] = roll
            ordered = [trajectories[completed+i] for i in range(count)]
            stats = train_wave(model, optimizer, ordered, device, args, np_rng)
            total_steps += sum(x['step_count'] for x in ordered)
            completed += count
            record = {'completed_episodes': completed, 'environment_steps': total_steps,
                      'mean_survival_s': float(np.mean([x['seconds'] for x in ordered])),
                      'reached150': sum(x['reached150'] for x in ordered),
                      'elapsed_s': round(time.perf_counter()-t0,2), **stats}
            print('WAVE ' + json.dumps(record, ensure_ascii=False), flush=True)
            with (args.checkpoint_dir / 'training_waves.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
            episode_csv = args.checkpoint_dir / 'training_episodes.csv'
            create_header = not episode_csv.exists() or episode_csv.stat().st_size == 0
            with episode_csv.open('a', encoding='utf-8-sig' if create_header else 'utf-8', newline='') as f:
                writer = csv.writer(f)
                if create_header:
                    writer.writerow(['episode', 'seed', 'survival_s', 'environment_steps', 'reached150', 'time_limit'])
                for i, trajectory in enumerate(ordered):
                    writer.writerow([completed-count+i+1, episode_seeds[completed-count+i],
                                     trajectory['seconds'], trajectory['step_count'],
                                     int(trajectory['reached150']), int(not trajectory['terminal'])])
            if completed % args.checkpoint_every < count or completed == args.episodes:
                save_resume()
                print(f'CHECKPOINT episodes={completed} path={args.checkpoint_dir / "model_v4_1000ep_resume.pt"}', flush=True)
        atomic_save(make_payload(model, steps=total_steps, trained=True, seed=master_seed,
                                 completed_episodes=completed, trainer='model_v4_parallel_episodes_v1'), args.out)
        print(f'FINAL_PT {args.out} completed_episodes={completed} environment_steps={total_steps}', flush=True)
        return {'completed_episodes': completed, 'environment_steps': total_steps, 'model': str(args.out)}
    finally:
        for q in task_queues:
            try: q.put_nowait(None)
            except queue.Full: pass
        for process in processes:
            process.join(timeout=2)
            if process.is_alive(): process.terminate(); process.join(timeout=2)
        for q in task_queues: q.close()
        result_queue.close()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Independent Model V4; 16-worker PPO training, 1000 episodes')
    p.add_argument('--init', type=Path, default=ROOT / 'model_v4_initial_50k.pt')
    p.add_argument('--resume', type=Path)
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--envs', type=int, default=16)
    p.add_argument('--max-steps', type=int, default=9000)
    p.add_argument('--seed', type=int, default=None, help='Omit for cryptographically randomized run seed')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', choices=('cpu','cuda','auto'), default='cpu')
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--ppo-epochs', type=int, default=2)
    p.add_argument('--minibatch', type=int, default=2048)
    p.add_argument('--max-train-transitions', type=int, default=32768)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--entropy-coef', type=float, default=0.01)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--checkpoint-every', type=int, default=64)
    p.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints')
    p.add_argument('--eval-episodes', type=int, default=30, help='Reserved disjoint evaluation seeds; use separate simulator after training')
    p.add_argument('--out', type=Path, default=ROOT / 'model_v4_1000ep.pt')
    return p.parse_args(argv)

if __name__ == '__main__':
    mp.freeze_support()
    main(parse_args())
