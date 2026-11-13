"""Inspect policy mean versus sampled actions and actual horizontal game motion.

Independent Model V4 only. Uses model_v4_net.py and eval_model_v4.py.
Does not change weights or create training transitions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from model_v4_net import obs_vector, env_action
from eval_model_v4 import load_model, make_env, get_time

HERE = Path(__file__).resolve().parent

FIELDS = [
    'mode','episode','seed','step','time_s','time_source',
    'mario_x_before','mario_x_after','actual_dx','actual_vx_after',
    'mu','std','raw_move','sent_move','move_clipped','jump_index','jump_value',
    'jump_confidence','jump_entropy','feature_nonzero','feature_l2','reward','terminated','truncated',
]


def mario_state(obs):
    """For BulletBill Stage 2, obs['mario'] == [left,top,right,bottom,vx,vy,...]."""
    if not isinstance(obs, dict) or 'mario' not in obs:
        return float('nan'), float('nan')
    mario = np.asarray(obs['mario'], dtype=np.float64).ravel()
    if len(mario) < 5:
        return float('nan'), float('nan')
    x = (mario[0] + mario[2]) / 2 if len(mario) >= 3 else mario[0]
    return float(x), float(mario[4])


def stats(rows):
    def arr(key):
        return np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    if not rows:
        return {'steps': 0}
    mu, raw, sent, dx = (arr(k) for k in ('mu','raw_move','sent_move','actual_dx'))
    valid = np.isfinite(dx)
    jumps = [sum(int(r['jump_index']) == i for r in rows) for i in range(5)]
    return {
        'steps': len(rows), 'mu_mean': float(np.mean(mu)), 'mu_abs_mean': float(np.mean(np.abs(mu))),
        'mu_near_zero_pct': float(np.mean(np.abs(mu) < .05)*100),
        'std_mean': float(np.mean(arr('std'))),
        'raw_action_abs_mean': float(np.mean(np.abs(raw))),
        'sent_action_abs_mean': float(np.mean(np.abs(sent))),
        'sent_action_neutral_pct': float(np.mean(np.abs(sent) < .05)*100),
        'clipped_pct': float(np.mean(arr('move_clipped'))*100),
        'actual_dx_abs_mean': float(np.mean(np.abs(dx[valid]))) if valid.any() else None,
        'actual_dx_zero_pct': float(np.mean(np.abs(dx[valid]) < .01)*100) if valid.any() else None,
        'actual_dx_available_pct': float(valid.mean()*100),
        'jump_counts': jumps, 'jump_fraction': [x/len(rows) for x in jumps],
        'jump_entropy_mean': float(np.mean(arr('jump_entropy'))),
        'feature_nonzero_mean': float(np.mean(arr('feature_nonzero'))),
    }


def run(args, env_factory=make_env):
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError('episodes and max-steps must be positive')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu')
    model, ck = load_model(args.model, device)
    rng = random.Random(args.seed)
    seeds = rng.sample(range(1, 2**31 - 1), args.episodes)
    args.out.mkdir(parents=True, exist_ok=True)
    records, episodes = [], []
    for mode in ('deterministic', 'stochastic'):
        env = env_factory(args.render)
        try:
            for epi, seed in enumerate(seeds, 1):
                obs, info = env.reset(seed=seed)
                if mode == 'stochastic':
                    torch.manual_seed(seed)  # reproducible sampling; same environment seeds across modes
                rows = []
                stopped = False
                start = time.perf_counter()
                for step in range(args.max_steps):
                    feature = obs_vector(obs, size=model.obs_dim)
                    inp = torch.from_numpy(feature).to(device=device).unsqueeze(0)
                    with torch.inference_mode():
                        jump_dist, move_dist, _ = model.distributions(inp)
                        mu = float(move_dist.mean[0].item())
                        std = float(move_dist.stddev.flatten()[0].item())
                        probs = jump_dist.probs[0].detach().cpu().numpy()
                        if mode == 'deterministic':
                            ji = int(np.argmax(probs)); raw = mu
                        else:
                            ji = int(jump_dist.sample()[0].item())
                            raw = float(move_dist.sample()[0].item())
                    action = env_action(raw, ji)
                    before_x, _ = mario_state(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    after_x, vx_after = mario_state(next_obs)
                    sim_t, t_src = get_time(info, step+1, args.game_fps)
                    item = {
                        'mode':mode, 'episode':epi, 'seed':seed, 'step':step+1,
                        'time_s':sim_t, 'time_source':t_src,
                        'mario_x_before':before_x, 'mario_x_after':after_x,
                        'actual_dx':after_x-before_x, 'actual_vx_after':vx_after,
                        'mu':mu, 'std':std, 'raw_move':raw, 'sent_move':float(action[0]),
                        'move_clipped':int(abs(raw - float(action[0])) > 1e-5),
                        'jump_index':ji, 'jump_value':float(action[1]),
                        'jump_confidence':float(probs[ji]),
                        'jump_entropy':float(-np.sum(probs*np.log(np.maximum(probs, 1e-12)))),
                        'feature_nonzero':int(np.count_nonzero(feature)),
                        'feature_l2':float(np.linalg.norm(feature)),
                        'reward':float(reward), 'terminated':int(terminated), 'truncated':int(truncated),
                    }
                    rows.append(item)
                    obs = next_obs
                    if terminated or truncated:
                        break
                records.extend(rows)
                total_s, _ = get_time(info, len(rows), args.game_fps)
                ep_summary = {'mode':mode, 'episode':epi, 'seed':seed,
                              'survival_s':total_s, 'steps':len(rows),
                              'status':'terminated' if rows and rows[-1]['terminated'] else 'truncated' if rows and rows[-1]['truncated'] else 'step_limit',
                              'wall_s':round(time.perf_counter()-start,3), **stats(rows)}
                episodes.append(ep_summary)
                print(f'{mode.upper()} {epi}/{len(seeds)} seed={seed} survival={total_s:.3f}s mu_abs={ep_summary["mu_abs_mean"]:.4f} sent_abs={ep_summary["sent_action_abs_mean"]:.4f} dx_abs={ep_summary["actual_dx_abs_mean"]}', flush=True)
        finally:
            env.close()
    with (args.out/'actions.csv').open('w', encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS); writer.writeheader(); writer.writerows(records)
    with (args.out/'episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=list(episodes[0]); writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        for record in episodes:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in record.items()})
    summary={'model':str(args.model), 'model_steps':ck.get('steps'), 'seed_base':args.seed,
             'paired_seeds':seeds, 'episodes_per_mode':args.episodes,
             'modes':{m:{**stats([r for r in records if r['mode']==m]),
                         'mean_survival_s':float(np.mean([e['survival_s'] for e in episodes if e['mode']==m]))}
                      for m in ('deterministic','stochastic')},
             'notes':['This is diagnostics, not a training run.',
                      'Training move_abs is sampled action magnitude; deterministic eval uses policy mu.',
                      "actual_dx is derived from obs['mario'] horizontal box center. Check this observation schema if unavailable."]}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('SUMMARY',json.dumps(summary['modes'],ensure_ascii=False),flush=True)
    print('SAVED',args.out/'actions.csv',args.out/'episodes.csv',args.out/'summary.json',flush=True)
    return summary


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Independent Model V4 action diagnostic, deterministic vs stochastic')
    p.add_argument('--model',type=Path,default=HERE/'model_v4_500k.pt')
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--seed',type=int,default=731209)
    p.add_argument('--max-steps',type=int,default=9000)
    p.add_argument('--render',choices=('none','human'),default='none')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--game-fps',type=float,default=50.0)
    p.add_argument('--out',type=Path,default=HERE/'eval_results'/'action_diagnostic')
    return p.parse_args(argv)

if __name__=='__main__':
    run(parse_args())
