"""Parallel CPU environment/Numba teacher, optional CUDA PPO learner for Model V14.

Run from mario_pj project root. Standalone V15 continuation trainer.
V15 178D feature extractor, 15-action PPO/DAgger with late-start curriculum.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import secrets
import time
import traceback
from pathlib import Path

# Set before importing numpy/torch in worker processes.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from torch.distributions import Categorical
from agent_ppo_v15_model_v4 import (
    ActorCritic, ENV_ID, FeatureExtractor, OBS_DIM, N_ACT, PAIR_OFF,
    pair_to_env, potential, SURVIVE, SHAPE, GAMMA, JUMP_COST, DEATH,
    STAR_EARLY_PENALTY, STAR_FREE_TIME, HAS_NUMBA, TRAIN_DURATION, late_start, find_core,
)

HERE = Path(__file__).resolve().parent


def save_atomic(obj, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, target)


def cpu_state(model):
    return {k: v.detach().to('cpu').clone() for k, v in model.state_dict().items()}


def worker(wid, input_queue, output_queue, seed, max_episode_steps, p_late, t_lo, t_hi):
    torch.set_num_threads(1)
    env = None
    try:
        import gymnasium as gym
        import kymnasium  # registers env
        env = gym.make(id=ENV_ID, render_mode='none', bgm=False)
        find_core(env).game_duration = TRAIN_DURATION
        model = ActorCritic().cpu().eval()
        fe = FeatureExtractor()
        rng = np.random.default_rng(seed + 1000003 * wid)
        obs = info = state = None
        phi = 0.0
        epsteps = 0
        start_t = 0.0
        while True:
            msg = input_queue.get()
            if msg is None:
                return
            wave, weights, count = msg
            model.load_state_dict(weights, strict=True)
            # Across a wave boundary keep the current env state and feature extractor.
            rows = {k: [] for k in ('s', 'a', 'logp', 'v', 'r', 'done')}
            finished = []
            finished_normal = []
            finished_late = []
            sat_time = 0.0
            sat_deaths = 0
            starts_late = 0
            acts = np.zeros(N_ACT, np.int64)
            reward_sum = 0.0
            with torch.inference_mode():
                for _ in range(count):
                    if obs is None:
                        sd = int(rng.integers(1, 2147483000))
                        obs, info = env.reset(seed=sd)
                        start_t = 0.0
                        if rng.random() < p_late:
                            obs, info, start_t = late_start(env, rng, t_lo, t_hi)
                            starts_late += 1
                        fe.reset()
                        state, ed, bd = fe(obs, info)
                        phi = potential(ed, bd)
                        epsteps = 0
                    x = torch.from_numpy(state).unsqueeze(0)
                    action, _, logp = model.act(x)
                    value = model.value(x)
                    k = int(action.item())
                    old_obs = obs
                    env_a = pair_to_env(k)
                    on_ground = old_obs['mario'][7] < 0.5
                    obs2, _, terminated, truncated, info2 = env.step(env_a)
                    state2, ed2, bd2 = fe(obs2, info2)
                    epsteps += 1
                    cutoff = epsteps >= max_episode_steps
                    done = bool(terminated or truncated or cutoff)
                    dead = bool(terminated and info2.get('game_state') == 'game_over')
                    phi2 = 0.0 if done else potential(ed2, bd2)
                    reward = SURVIVE + SHAPE * (GAMMA * phi2 - phi)
                    if on_ground and env_a[1] > 0:
                        reward += JUMP_COST
                    if dead:
                        reward += DEATH
                    if obs2['mario'][6] > 0.5 and old_obs['mario'][6] < 0.5:
                        reward -= STAR_EARLY_PENALTY * max(0.0, 1.0 - float(info2['time_elapsed']) / STAR_FREE_TIME)
                    rows['s'].append(state)
                    rows['a'].append(k)
                    rows['logp'].append(float(logp.item()))
                    rows['v'].append(float(value.item()))
                    rows['r'].append(float(reward))
                    rows['done'].append(done)
                    acts[k] += 1
                    reward_sum += reward
                    if done:
                        end_t = float(info2.get('time_elapsed', 0.0))
                        finished.append(end_t - start_t)
                        if start_t > 0.0:
                            finished_late.append(end_t - start_t)
                        else:
                            finished_normal.append(end_t)
                        sat_from = max(80.0, start_t + 5.0)
                        sat_time += max(0.0, end_t - sat_from)
                        sat_deaths += int(dead and end_t >= sat_from)
                        obs = info = state = None
                        phi = 0.0
                    else:
                        obs, info, state, phi = obs2, info2, state2, phi2
                if state is None:
                    last_value = 0.0
                else:
                    last_value = float(model.value(torch.from_numpy(state).unsqueeze(0)).item())
            result = {
                's': np.stack(rows['s']).astype(np.float32),
                'a': np.asarray(rows['a'], dtype=np.int64),
                'logp': np.asarray(rows['logp'], dtype=np.float32),
                'v': np.asarray(rows['v'], dtype=np.float32),
                'r': np.asarray(rows['r'], dtype=np.float32),
                'done': np.asarray(rows['done'], dtype=np.float32),
                'last_value': last_value,
                'finished': finished,
                'finished_normal': finished_normal,
                'finished_late': finished_late,
                'sat_time': sat_time, 'sat_deaths': sat_deaths, 'starts_late': starts_late,
                'acts': acts,
                'reward_sum': reward_sum,
            }
            output_queue.put((wid, wave, result, None))
    except BaseException:
        output_queue.put((wid, -1, None, traceback.format_exc()))
    finally:
        if env is not None:
            env.close()


def gae_single(roll, gamma, lam):
    r, v, d = roll['r'], roll['v'], roll['done']
    n = len(r)
    advantages = np.empty(n, dtype=np.float32)
    carry = 0.0
    for i in range(n - 1, -1, -1):
        next_value = roll['last_value'] if i == n - 1 else float(v[i + 1])
        mask = 1.0 - float(d[i])
        delta = float(r[i]) + gamma * next_value * mask - float(v[i])
        carry = delta + gamma * lam * mask * carry
        advantages[i] = carry
    return advantages, advantages + v


def ppo_step(model, optim, rolls, args, device, imit_coef, entropy_coef):
    data = {k: [] for k in ('s', 'a', 'logp', 'adv', 'ret')}
    for roll in rolls:
        adv, ret = gae_single(roll, GAMMA, args.gae_lambda)
        for key, val in [('s', roll['s']), ('a', roll['a']), ('logp', roll['logp']), ('adv', adv), ('ret', ret)]:
            data[key].append(val)
    data = {k: torch.from_numpy(np.concatenate(v)).to(device) for k, v in data.items()}
    adv = data['adv']
    data['adv'] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    total = len(adv)
    model.train()
    results = []
    for _ in range(args.epochs):
        perm = torch.randperm(total, device=device)
        for ix in perm.split(args.minibatch):
            ob = data['s'][ix]
            logits = model.logits(ob)
            dist = Categorical(logits=logits)
            lp = dist.log_prob(data['a'][ix])
            ent = dist.entropy().mean()
            value = model.value(ob)
            ratio = torch.exp(lp - data['logp'][ix])
            a = data['adv'][ix]
            loss_actor = torch.max(-a * ratio, -a * ratio.clamp(1. - args.clip, 1. + args.clip)).mean()
            loss_critic = 0.5 * (value - data['ret'][ix]).square().mean()
            scores = ob[:, PAIR_OFF:PAIR_OFF + N_ACT]
            best_mask = scores >= (scores.amax(dim=1, keepdim=True) - 1e-6)
            imit_loss = (torch.logsumexp(logits, dim=1) -
                         torch.logsumexp(logits.masked_fill(~best_mask, -1e9), dim=1)).mean()
            loss = loss_actor + args.vf_coef * loss_critic - entropy_coef * ent + imit_coef * imit_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            with torch.no_grad():
                kl = ((ratio - 1) - (lp - data['logp'][ix])).mean()
                clipfrac = ((ratio - 1).abs() > args.clip).float().mean()
                crisis = (scores < scores.amax(dim=1, keepdim=True) - 0.1).any(dim=1)
                chosen_best = best_mask.gather(1, logits.argmax(dim=1, keepdim=True)).squeeze(1)
                correct = (chosen_best & crisis).float().sum() / crisis.float().sum().clamp_min(1)
                results.append([float(loss_actor), float(loss_critic), float(imit_loss), float(ent),
                                float(kl), float(clipfrac), float(grad), float(crisis.float().mean()), float(correct)])
    model.eval()
    return dict(zip(('actor_loss', 'value_loss', 'imit_loss', 'entropy', 'approx_kl',
                     'clipfrac', 'grad_norm', 'crisis_pct', 'crisis_correct_pct'),
                    [round(v, 6) if i < 7 else round(v * 100, 2)
                     for i, v in enumerate(np.mean(results, axis=0))]))


def evaluate(model, episodes, seed):
    import gymnasium as gym
    import kymnasium
    from agent_ppo_v15_model_v4 import BulletBillAgent
    agent = BulletBillAgent(model)
    agent.deterministic = True
    env = gym.make(id=ENV_ID, render_mode='none', bgm=False)
    times = []
    try:
        for i in range(episodes):
            obs, info = env.reset(seed=seed + i)
            agent.fe.reset()
            while True:
                obs, _, term, trunc, info = env.step(agent.act(obs, info))
                if term or trunc:
                    break
            times.append(float(info['time_elapsed']))
    finally:
        env.close()
    return {'eval_mean_s': round(float(np.mean(times)), 3), 'eval_median_s': round(float(np.median(times)), 3),
            'eval_max_s': round(float(np.max(times)), 3), 'eval_ge_120': int(sum(t >= 120 for t in times)),
            'eval_ge_150': int(sum(t >= 150 for t in times))}


def main(argv=None):
    p = argparse.ArgumentParser(description='V15 continuation: 16 CPU workers + CUDA PPO, late-start curriculum')
    p.add_argument('--init', type=Path, default=Path('v/model_v14_10m.pt'))
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--steps', type=int, default=8003584, help='ADDITIONAL transitions (not absolute); batch-multiple recommended')
    p.add_argument('--envs', type=int, default=16)
    p.add_argument('--batch-steps', type=int, default=256)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatch', type=int, default=1024)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--ent0', type=float, default=0.01)
    p.add_argument('--ent1', type=float, default=0.003)
    p.add_argument('--p-late', type=float, default=0.30)
    p.add_argument('--t-lo', type=float, default=60.0)
    p.add_argument('--t-hi', type=float, default=120.0)
    p.add_argument('--imit-steps', type=int, default=5000000, help='additional steps over which imitation decays')
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--imit0', type=float, default=0.05, help='continuation default avoids resetting teacher weight to 1.0')
    p.add_argument('--imit1', type=float, default=0.05)
    p.add_argument('--eval-every', type=int, default=0, help='0=disable periodic evaluation; evaluate after training separately')
    p.add_argument('--eval-episodes', type=int, default=30)
    p.add_argument('--eval-seed', type=int, default=20000)
    p.add_argument('--checkpoint-every', type=int, default=16384)
    p.add_argument('--max-episode-steps', type=int, default=9000)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--out', type=Path, default=Path('v/model_v15_from_v14_8m.pt'))
    p.add_argument('--best-out', type=Path, default=Path('v/model_v15_from_v14_best.pt'))
    p.add_argument('--checkpoint-dir', type=Path, default=Path('v/checkpoints'))
    a = p.parse_args(argv)
    if a.steps <= 0 or a.envs <= 0 or a.batch_steps <= 0 or a.epochs <= 0 or a.minibatch <= 0:
        p.error('steps, envs, batch-steps, epochs, minibatch must be positive')
    if not 0 < a.gae_lambda <= 1 or a.eval_every < 0 or a.checkpoint_every < 1:
        p.error('invalid gae-lambda/eval-every/checkpoint-every')
    if not 0 <= a.p_late <= 1 or not 0 < a.t_lo < a.t_hi < TRAIN_DURATION:
        p.error('invalid late-start parameters')
    if a.imit_steps <= 0 or min(a.ent0, a.ent1, a.imit0, a.imit1) < 0:
        p.error('invalid schedule parameters')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('requested cuda but torch.cuda.is_available() is False')
    if a.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(a.device)
    if not HAS_NUMBA:
        print('WARNING numba is unavailable. V15 lookahead simulation can be extremely slow.', flush=True)
    torch.set_num_threads(1)
    initial = a.resume if a.resume else a.init
    if not initial.exists():
        p.error(f'checkpoint not found: {initial}')
    ck = torch.load(initial, map_location='cpu', weights_only=False)
    if ck.get('obs_dim') != OBS_DIM or ck.get('hidden', 256) != 256 or 'pair_head.weight' not in ck['model']:
        p.error('requires V14/V15 178D/15-action checkpoint')
    model = ActorCritic().to(device)
    model.load_state_dict(ck['model'], strict=True)
    optim = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    total_done = 0
    best_mean = -float('inf')
    if a.resume:
        # Restores optimizer and schedule progress; --steps is the new additional amount.
        if ck.get('trainer') != 'v15_fast_parallel_v1':
            p.error('resume requires v15_fast_parallel_v1 optimizer checkpoint')
        optim.load_state_dict(ck['optimizer'])
        for state in optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        total_done = int(ck['total_steps'])
        schedule_step = int(ck.get('schedule_step', 0))
        planned_steps = int(ck.get('planned_steps', a.steps))
        best_mean = float(ck.get('best_mean', -float('inf')))
    if not a.resume:
        schedule_step = 0
        planned_steps = a.steps
    seed = int(a.seed if a.seed is not None else secrets.randbelow(2**31))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ctx = mp.get_context('spawn')
    inputs = [ctx.Queue(maxsize=2) for _ in range(a.envs)]
    outputs = ctx.Queue(maxsize=max(2 * a.envs, 4))
    procs = [ctx.Process(target=worker, args=(i, inputs[i], outputs, seed, a.max_episode_steps, a.p_late, a.t_lo, a.t_hi), daemon=True)
             for i in range(a.envs)]
    for proc in procs:
        proc.start()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.best_out.parent.mkdir(parents=True, exist_ok=True)
    a.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = a.checkpoint_dir / 'model_v15_fast_resume.pt'
    resolved = [p.resolve() for p in [a.out, a.best_out, checkpoint_path]]
    if len(set(resolved)) < 3:
        raise ValueError('out, best-out, checkpoint file paths must differ')
    batch = a.envs * a.batch_steps
    num_waves = (a.steps + batch - 1) // batch
    start_time = time.perf_counter()
    print('START ' + json.dumps({'init': str(initial), 'device': str(device), 'workers': a.envs,
                                'numba': HAS_NUMBA, 'batch': batch, 'additional_steps': a.steps,
                                'starting_steps': total_done, 'p_late': a.p_late, 'imit0': a.imit0, 'imit1': a.imit1, 'ent0': a.ent0, 'ent1': a.ent1}, ensure_ascii=False), flush=True)
    new_done = 0
    wave = 0
    last_ckpt = 0
    last_eval = 0
    try:
        while new_done < a.steps:
            remain = a.steps - new_done
            targets = [max(0, min(a.batch_steps, remain - i * a.batch_steps)) for i in range(a.envs)]
            active = [i for i, count in enumerate(targets) if count]
            weights = cpu_state(model)
            for i in active:
                inputs[i].put((wave, weights, targets[i]))
            responses = {}
            while len(responses) < len(active):
                try:
                    i, got_wave, result, error = outputs.get(timeout=10)
                except queue.Empty:
                    if any(not procs[i].is_alive() for i in active):
                        raise RuntimeError('Worker exited without sending its result')
                    continue
                if error:
                    raise RuntimeError(f'Worker {i}:\n{error}')
                if got_wave != wave or i not in active:
                    raise RuntimeError(f'Unexpected result worker={i}, wave={got_wave}, expected={wave}')
                responses[i] = result
            rolls = [responses[i] for i in active]
            # Schedules use additional transitions, preserved across --resume of this run.
            fraction = min(1.0, schedule_step / max(planned_steps, 1))
            imit_fraction = min(1.0, schedule_step / a.imit_steps)
            imit_coef = a.imit0 + (a.imit1 - a.imit0) * imit_fraction
            ent_coef = a.ent0 + (a.ent1 - a.ent0) * fraction
            metrics = ppo_step(model, optim, rolls, a, device, imit_coef, ent_coef)
            n = sum(len(r['r']) for r in rolls)
            new_done += n
            total_done += n
            schedule_step += n
            wave += 1
            endings = [t for r in rolls for t in r['finished']]
            normal = [t for r in rolls for t in r['finished_normal']]
            late = [t for r in rolls for t in r['finished_late']]
            sat_t = sum(r['sat_time'] for r in rolls)
            sat_d = sum(r['sat_deaths'] for r in rolls)
            action_counts = np.sum([r['acts'] for r in rolls], axis=0)
            row = {'wave': wave, 'steps': total_done, 'new_steps': new_done,
                   'target_new_steps': a.steps, 'episodes_ended': len(endings),
                   'mean_ended_s': round(float(np.mean(endings)), 3) if endings else None,
                   'normal_ended':len(normal), 'normal_mean_s':round(float(np.mean(normal)), 3) if normal else None,
                   'late_ended':len(late), 'late_mean_from_start_s':round(float(np.mean(late)), 3) if late else None,
                   'late_starts':sum(r['starts_late'] for r in rolls),
                   'sat_hazard_per_s':round(sat_d/sat_t, 6) if sat_t else None,
                   'moves_left_pct': round(float(action_counts[0::3].sum() / n * 100), 2),
                   'moves_neutral_pct': round(float(action_counts[1::3].sum() / n * 100), 2),
                   'moves_right_pct': round(float(action_counts[2::3].sum() / n * 100), 2),
                   'imit_coef': round(float(imit_coef), 4), 'ent_coef': round(float(ent_coef), 5), 'wall_s': round(time.perf_counter() - start_time, 1),
                   'steps_per_s': round(new_done / max(0.001, time.perf_counter() - start_time), 1), **metrics}
            print('UPDATE ' + json.dumps(row, ensure_ascii=False), flush=True)
            if new_done - last_ckpt >= a.checkpoint_every or new_done == a.steps:
                payload = {'model': cpu_state(model), 'obs_dim': OBS_DIM, 'hidden': 256,
                           'trainer': 'v15_fast_parallel_v1', 'optimizer': optim.state_dict(), 'schedule_step': schedule_step, 'planned_steps': planned_steps,
                           'total_steps': total_done, 'best_mean': best_mean, 'seed': seed, 'config': vars(a)}
                save_atomic(payload, checkpoint_path)
                last_ckpt = new_done
                print('CHECKPOINT ' + str(checkpoint_path), flush=True)
            if a.eval_every and (new_done - last_eval >= a.eval_every or new_done == a.steps):
                # Save a CPU copy to avoid altering device of training model.
                eval_model = ActorCritic()
                eval_model.load_state_dict(cpu_state(model))
                eval_model.eval()
                result = evaluate(eval_model, a.eval_episodes, a.eval_seed)
                print('EVAL ' + json.dumps(result, ensure_ascii=False), flush=True)
                snapshot = a.checkpoint_dir / f'model_v15_step{total_done}.pt'
                save_atomic({'model': cpu_state(model), 'obs_dim': OBS_DIM, 'hidden': 256, 'total_steps':total_done}, snapshot)
                print('SNAPSHOT ' + str(snapshot), flush=True)
                last_eval = new_done
                if result['eval_mean_s'] > best_mean:
                    best_mean = result['eval_mean_s']
                    save_atomic({'model': cpu_state(model), 'obs_dim': OBS_DIM, 'hidden': 256,
                                 'total_steps': total_done, 'trained': True, 'trainer': 'v15_fast_parallel_v1'}, a.best_out)
                    print('BEST ' + str(a.best_out), flush=True)
        save_atomic({'model': cpu_state(model), 'obs_dim': OBS_DIM, 'hidden': 256,
                     'total_steps': total_done, 'trained': True, 'trainer': 'v15_fast_parallel_v1'}, a.out)
        print('FINAL ' + json.dumps({'model': str(a.out), 'steps': total_done, 'wall_s': round(time.perf_counter() - start_time, 2)}, ensure_ascii=False), flush=True)
    finally:
        for q in inputs:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for proc in procs:
            proc.join(timeout=1)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
        for q in inputs:
            q.close()
        outputs.close()


if __name__ == '__main__':
    mp.freeze_support()
    main()
