"""Continue PPO-v10 from a pretrained .pt with real environment rewards.

Optional v4.1 distilled dataset contributes a decaying auxiliary BC loss; it is
NOT used as fake on-policy experience. Inference of saved .pt needs no overlay.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import secrets
import time
from collections import deque
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import agent_ppo_v10 as v10


def load_checkpoint(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if not all(k in ck for k in ('model', 'obs_dim', 'hidden')) or int(ck['obs_dim']) != v10.OBS_DIM:
        raise ValueError(f'Expected {v10.OBS_DIM}-D PPO-v10 checkpoint: {path}')
    model = v10.ActorCritic(obs_dim=ck['obs_dim'], hidden=ck['hidden'])
    model.load_state_dict(ck['model'])
    return model, ck


def save_pt(path, model, hidden, metadata):
    """Save in BulletBillAgent.load()-compatible format. Atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    data = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'obs_dim': v10.OBS_DIM, 'hidden': hidden, 'finetune': metadata}
    torch.save(data, tmp)
    os.replace(tmp, path)


def seed_sets(seed, n_eval, n_final):
    """Disjoint evaluation and final holdout seeds, different from episodic training stream."""
    g = np.random.default_rng(seed)
    chosen = g.choice(2_147_483_647, size=n_eval + n_final, replace=False).astype(int).tolist()
    return chosen[:n_eval], chosen[n_eval:]


def teacher_data(path, obs_dim, max_rows, rng):
    if not path:
        return None
    with np.load(path, allow_pickle=False) as ds:
        s = np.asarray(ds['states'], dtype=np.float32)
        a = np.asarray(ds['actions'], dtype=np.float32)
        over = np.asarray(ds['override'], dtype=np.float32)
    if s.ndim != 2 or s.shape[1] != obs_dim or a.shape != (len(s), 2) or over.shape != (len(s),):
        raise ValueError('Invalid v4.1 dataset arrays; expected states[N,148], actions[N,2], override[N]')
    if not len(s) or not np.isfinite(s).all() or not np.isfinite(a).all():
        raise ValueError('Teacher dataset must have finite values and nonzero length')
    if len(s) > max_rows:
        ix = rng.choice(len(s), size=max_rows, replace=False)
        s, a, over = s[ix], a[ix], over[ix]
    levels = np.asarray(v10.JUMP_LEVELS, dtype=np.float32)
    j = np.abs(a[:, 1, None] - levels[None, :]).argmin(axis=1)
    if (np.abs(a[:, 1] - levels[j]) > 0.08).any():
        raise ValueError('Teacher dataset has invalid jump levels')
    return (torch.from_numpy(s), torch.from_numpy(j.astype(np.int64)),
            torch.from_numpy(np.clip(a[:, 0], -1., 1.)), torch.from_numpy(np.clip(over, 0, 1)))


def auxiliary_bc(model, teacher, rng, batch, override_weight, reference=None, preserve=0.0):
    if teacher is None:
        return torch.zeros((), dtype=torch.float32)
    s, j, m, ov = teacher
    ids = torch.from_numpy(rng.integers(0, len(s), size=min(batch, len(s))))
    x, yj, ym, yo = s[ids], j[ids], m[ids], ov[ids]
    dj, dm = model.dists(x)
    # Match actions using the policy's un-clipped Gaussian likelihood convention.
    loss = -(dj.log_prob(yj) + dm.log_prob(ym))
    w = 1 + (max(1., override_weight) - 1) * yo
    result = (w * loss).sum() / w.sum()
    if reference is not None and preserve > 0:
        with torch.no_grad():
            old_j, old_m = reference.dists(x)
        kl_jump = torch.distributions.kl.kl_divergence(old_j, dj).mean()
        kl_move = torch.distributions.kl.kl_divergence(old_m, dm).mean()
        result = result + preserve * (kl_jump + kl_move)
    return result


def predict_action(model, s, deterministic=False):
    with torch.no_grad():
        t = torch.from_numpy(s)
        j, m, logp = model.act(t, deterministic=deterministic)
        value = model.value(t)
    return j.numpy(), m.numpy(), logp.numpy(), value.numpy()


def evaluate_model(model, seeds, max_steps):
    """Fresh extractor for each episode; no v4.1 runtime controller."""
    env = gym.make(id=v10.ENV_ID, render_mode='none', bgm=False)
    out = []
    try:
        model.eval()
        for sd in seeds:
            obs, info = env.reset(seed=int(sd))
            fe = v10.FeatureExtractor()
            fe.reset()
            status = 'step_limit'
            for t in range(max_steps):
                s, _, _ = fe(obs, info)
                j, m, _, _ = predict_action(model, np.asarray(s, dtype=np.float32)[None, :], deterministic=True)
                act = v10.to_env_action(float(m[0]), int(j[0]))
                obs, _, term, trunc, info = env.step(act)
                if term or trunc:
                    status = 'terminated' if term else 'truncated'
                    break
            out.append({'seed': int(sd), 'survival_s': float(info.get('time_elapsed', (t + 1) * v10.DT)),
                        'steps': t + 1, 'status': status})
    finally:
        env.close()
    return out


def summarize(rows):
    v = np.asarray([r['survival_s'] for r in rows], dtype=float)
    return {'mean_s': float(v.mean()), 'median_s': float(np.median(v)), 'min_s': float(v.min()),
            'max_s': float(v.max()), 'reached_120': int((v >= 120).sum()),
            'reached_150': int((v >= 150).sum()), 'episodes': len(v)}


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['seed', 'survival_s', 'steps', 'status'])
        wr.writeheader(); wr.writerows(rows)


def train(args):
    if args.init.resolve() == args.out.resolve():
        raise ValueError('Output file must be different from input checkpoint')
    if args.batch_steps < 1 or args.envs < 1 or args.total_steps < 1:
        raise ValueError('steps and env count must be positive')
    torch.set_num_threads(args.threads)
    master_seed = args.seed if args.seed is not None else secrets.randbelow(2_147_483_647)
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    rng = np.random.default_rng(master_seed)
    eval_seeds, final_seeds = seed_sets(master_seed, args.eval_episodes, args.final_episodes)
    (args.log_dir / 'eval_seeds.json').write_text(json.dumps(eval_seeds, indent=2), encoding='utf-8')
    (args.log_dir / 'final_holdout_seeds.json').write_text(json.dumps(final_seeds, indent=2), encoding='utf-8')
    # Use a different stream for training resets. Block the reserved evaluation seeds.
    forbidden = set(eval_seeds + final_seeds)
    used_train = set()
    def next_seed():
        while True:
            s = int(rng.integers(0, 2_147_483_647))
            if s not in forbidden and s not in used_train:
                used_train.add(s)
                return s

    model, ck = load_checkpoint(args.init)
    hidden = int(ck['hidden'])
    baseline = v10.ActorCritic(obs_dim=v10.OBS_DIM, hidden=hidden)
    baseline.load_state_dict(ck['model'])
    baseline.eval()
    for p in baseline.parameters(): p.requires_grad_(False)
    teacher = teacher_data(args.teacher_data, v10.OBS_DIM, args.max_teacher_rows, rng)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / 'seeds.json').write_text(json.dumps({
        'master_seed': master_seed, 'eval_seeds': eval_seeds, 'final_holdout_seeds': final_seeds,
        'notes': 'Final holdout seeds must not be used for tuning or checkpoint selection'}, indent=2), encoding='utf-8')
    batch = args.envs * args.batch_steps
    updates = math.ceil(args.total_steps / batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    envs = [gym.make(id=v10.ENV_ID, render_mode='none', bgm=False) for _ in range(args.envs)]
    fes = [v10.FeatureExtractor() for _ in range(args.envs)]
    obs_list = []
    states = np.zeros((args.envs, v10.OBS_DIM), dtype=np.float32)
    phis = np.zeros(args.envs, dtype=np.float32)
    episode_steps = np.zeros(args.envs, np.int64)
    active_seeds = np.zeros(args.envs, np.int64)
    episode_times = deque(maxlen=100)
    episode_records = []
    try:
        for i, env in enumerate(envs):
            sd = next_seed()
            active_seeds[i] = sd
            obs, inf = env.reset(seed=sd)
            fes[i].reset()
            s, edge, bul = fes[i](obs, inf)
            states[i], phis[i] = s, v10.potential(edge, bul)
            obs_list.append(obs)
        print(f'START init={args.init} target={args.out} steps={args.total_steps} batch={batch} '
              f'updates={updates} envs={args.envs} random_master_seed={master_seed} '
              f'teacher_samples={0 if teacher is None else len(teacher[0])}', flush=True)
        base_rows = evaluate_model(model, eval_seeds, args.eval_max_steps)
        best = summarize(base_rows)['mean_s']
        write_rows(args.log_dir / 'eval_initial.csv', base_rows)
        print(f'BASELINE mean={best:.3f}s', flush=True)
        meta = {'init': str(args.init), 'master_seed': master_seed, 'steps': 0,
                'eval_mean_s': best, 'training': 'on_policy_ppo_v10_reward', 'auxiliary_bc': bool(teacher)}
        save_pt(args.out, model, hidden, meta)
        completed = 0
        last_eval_at = 0
        records = []
        for update in range(1, updates + 1):
            model.train()
            T = min(args.batch_steps, max(1, math.ceil((args.total_steps - completed) / args.envs)))
            S = np.zeros((T, args.envs, v10.OBS_DIM), np.float32)
            J = np.zeros((T, args.envs), np.int64)
            M = np.zeros((T, args.envs), np.float32)
            LP = np.zeros((T, args.envs), np.float32)
            V = np.zeros((T, args.envs), np.float32)
            R = np.zeros((T, args.envs), np.float32)
            D = np.zeros((T, args.envs), np.float32)
            for t in range(T):
                j, m, lp, value = predict_action(model, states)
                S[t], J[t], M[t], LP[t], V[t] = states, j, m, lp, value
                for i, env in enumerate(envs):
                    action = v10.to_env_action(float(m[i]), int(j[i]))
                    on_ground = float(obs_list[i]['mario'][7]) < 0.5
                    prev_inv = float(obs_list[i]['mario'][6]) > 0.5
                    obs, _, term, trunc, info = env.step(action)
                    episode_steps[i] += 1
                    s2, edge2, bul2 = fes[i](obs, info)
                    done = bool(term or trunc or episode_steps[i] >= args.max_episode_steps)
                    dead = bool(term and info.get('game_state') == 'game_over')
                    phi2 = 0. if done else v10.potential(edge2, bul2)
                    reward = v10.SURVIVE + v10.SHAPE * (v10.GAMMA * phi2 - phis[i])
                    if on_ground and action[1] > 0: reward += v10.JUMP_COST
                    if dead: reward += v10.DEATH
                    if float(obs['mario'][6]) > 0.5 and not prev_inv:
                        when = float(info.get('time_elapsed', 0.))
                        reward -= v10.STAR_EARLY_PENALTY * max(0., 1. - when / v10.STAR_FREE_TIME)
                    R[t, i], D[t, i] = reward, float(done)
                    if done:
                        elapsed = float(info.get('time_elapsed', episode_steps[i] * v10.DT))
                        episode_times.append(elapsed)
                        episode_records.append({'update': update, 'survival_s': elapsed,
                                                'death': dead, 'seed': int(active_seeds[i])})
                        sd = next_seed()
                        active_seeds[i] = sd
                        obs, info = env.reset(seed=sd)
                        fes[i].reset()
                        s2, edge2, bul2 = fes[i](obs, info)
                        phi2 = v10.potential(edge2, bul2)
                        episode_steps[i] = 0
                    obs_list[i], states[i], phis[i] = obs, s2, phi2
            completed += T * args.envs
            with torch.no_grad(): last_value = model.value(torch.from_numpy(states)).numpy()
            adv = v10.compute_gae(R, V, D, last_value, v10.GAMMA, args.gae_lambda)
            ret = adv + V
            bs = torch.from_numpy(S.reshape(-1, v10.OBS_DIM))
            bj = torch.from_numpy(J.reshape(-1))
            bm = torch.from_numpy(M.reshape(-1))
            blp = torch.from_numpy(LP.reshape(-1))
            ba = torch.from_numpy(adv.reshape(-1))
            br = torch.from_numpy(ret.reshape(-1))
            n = len(bs)
            mb = max(1, math.ceil(n / args.minibatches))
            entropy_stat = []
            clip_stat = []
            grad_loss = []
            fraction = min(1., completed / args.total_steps)
            bc_coeff = args.bc_coef * max(0., 1. - fraction / max(args.bc_decay, 1e-6)) if teacher else 0.
            for ep in range(args.epochs):
                ix = rng.permutation(n)
                for k in range(0, n, mb):
                    ids = torch.from_numpy(ix[k:k + mb])
                    logp, ent_j, ent_m, new_value = model.evaluate(bs[ids], bj[ids], bm[ids])
                    ratio = torch.exp(logp - blp[ids])
                    norm_adv = ba[ids]
                    norm_adv = (norm_adv - norm_adv.mean()) / (norm_adv.std(unbiased=False) + 1e-8)
                    loss_pi = torch.maximum(-norm_adv * ratio,
                                            -norm_adv * ratio.clamp(1. - args.clip, 1. + args.clip)).mean()
                    value_loss = 0.5 * (new_value - br[ids]).square().mean()
                    loss = (loss_pi + args.vf_coef * value_loss
                            - args.ent_jump * ent_j.mean() - args.ent_move * ent_m.mean())
                    if bc_coeff > 0:
                        loss = loss + bc_coeff * auxiliary_bc(model, teacher, rng, args.bc_batch,
                                                               args.override_weight, baseline, args.preserve_kl)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    grad_loss.append(float(loss.detach()))
                    entropy_stat.append(float(ent_j.detach().mean()))
                    clip_stat.append(float(((ratio.detach() - 1.).abs() > args.clip).float().mean()))
            mean_recent = float(np.mean(episode_times)) if episode_times else None
            record = {'update': update, 'steps': completed, 'recent_mean_s': mean_recent,
                      'training_loss': float(np.mean(grad_loss)), 'jump_entropy': float(np.mean(entropy_stat)),
                      'clip_fraction': float(np.mean(clip_stat)), 'bc_coef': bc_coeff}
            records.append(record)
            if update % args.log_every == 0 or update == 1:
                print(f'UPDATE {update}/{updates} steps={completed} recent_mean={mean_recent} '
                      f'loss={record["training_loss"]:.4f} clip={record["clip_fraction"]:.3f} '
                      f'bc_coef={bc_coeff:.4f}', flush=True)
            if completed - last_eval_at >= args.eval_interval or update == updates:
                last_eval_at = completed
                rows = evaluate_model(model, eval_seeds, args.eval_max_steps)
                score = summarize(rows)
                print(f'EVAL steps={completed} mean={score["mean_s"]:.3f}s '
                      f'min={score["min_s"]:.3f}s reached150={score["reached_150"]}/{args.eval_episodes} '
                      f'best={best:.3f}s', flush=True)
                write_rows(args.log_dir / f'eval_step_{completed}.csv', rows)
                if score['mean_s'] > best:
                    best = score['mean_s']
                    save_pt(args.out, model, hidden, {'init': str(args.init), 'master_seed': master_seed,
                        'steps': completed, 'eval_mean_s': best, 'training': 'on_policy_ppo_v10_reward',
                        'auxiliary_bc': bool(teacher)})
                    print(f'BEST_PT {args.out} mean={best:.3f}s steps={completed}', flush=True)
                save_pt(args.out.with_name(args.out.stem + '_last.pt'), model, hidden,
                        {'init': str(args.init), 'master_seed': master_seed,
                         'steps': completed, 'eval_mean_s': score['mean_s'], 'training': 'on_policy_ppo_v10_reward'})
                (args.log_dir / 'history.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
                (args.log_dir / 'train_episodes.json').write_text(json.dumps(episode_records, indent=2), encoding='utf-8')
                # Evaluate final holdout ONCE after checkpoint selection, not each update.
        model_best, _ = load_checkpoint(args.out)
        final_rows = evaluate_model(model_best, final_seeds, args.eval_max_steps)
        write_rows(args.log_dir / 'final_holdout.csv', final_rows)
        result = {'output_pt': str(args.out), 'last_pt': str(args.out.with_name(args.out.stem + '_last.pt')),
                  'master_seed': master_seed, 'completed_steps': completed, 'best_validation_mean_s': best,
                  'final_holdout': summarize(final_rows), 'teacher_data': str(args.teacher_data) if teacher else None,
                  'train_seed_count': len(used_train), 'eval_seeds': eval_seeds,
                  'final_holdout_seeds': final_seeds}
        (args.log_dir / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        (args.log_dir / 'train_seed_history.json').write_text(json.dumps(sorted(used_train)), encoding='utf-8')
        print('FINAL', json.dumps(result, ensure_ascii=False), flush=True)
        return result
    finally:
        for env in envs: env.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', type=Path, default=Path('models2/bullet_bill_ppo_v10_v41_distilled.pt'))
    p.add_argument('--out', type=Path, default=Path('models2/bullet_bill_ppo_v10_v41_finetuned_best.pt'))
    p.add_argument('--teacher-data', type=Path, help='Optional v4.1 teacher.npz auxiliary BC during PPO')
    p.add_argument('--log-dir', type=Path, default=Path('logs/models2_v41_ppo'))
    p.add_argument('--total-steps', type=int, default=100_000)
    p.add_argument('--envs', type=int, default=4)
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
    p.add_argument('--eval-episodes', type=int, default=5)
    p.add_argument('--final-episodes', type=int, default=30)
    p.add_argument('--eval-max-steps', type=int, default=9000)
    p.add_argument('--max-episode-steps', type=int, default=9000)
    p.add_argument('--eval-interval', type=int, default=25_000)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=None, help='Omit to generate a random master seed')
    args = p.parse_args(argv)
    if args.lr <= 0 or args.eval_episodes < 1 or args.final_episodes < 1 or args.epochs < 1 or args.minibatches < 1 or args.threads < 1:
        p.error('Invalid hyperparameters')
    if args.total_steps < 1 or args.envs < 1 or args.batch_steps < 1 or args.eval_interval < 1:
        p.error('Invalid training/evaluation sizes')
    if args.bc_coef < 0 or args.bc_decay < 0 or args.override_weight < 1 or args.preserve_kl < 0:
        p.error('Invalid auxiliary BC parameters')
    return args


if __name__ == '__main__':
    train(parse_args())
