r"""Paired-seed real-environment smoke test of ActionEvaluator vs random control.

Run from project root:
 .\.venv\Scripts\python.exe test_action_evaluator_env.py --episodes 20 --steps 150 --horizon-steps 8 --robust-mode top_k

The agent is a rule-based *known-bullet* baseline, not learned RL. It does not
predict random spawns. Episode survival comparisons are descriptive; no claim
of statistical superiority is made from small samples.
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from pathlib import Path

import numpy as np
from agent_env import create_env
from models.action_evaluator import ActionEvaluator
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor


def episode(seed, mode, max_steps, evaluator, random_seed):
    env = create_env(render_mode='none')
    random_gen = random.Random(random_seed)
    elapsed_action_score_s = 0.0
    steps = 0
    boundary_choices = 0
    action_switches = 0
    prev_action = None
    last_info = None
    try:
        obs, info = env.reset(seed=seed)
        last_info = info
        for step in range(max_steps):
            if mode == 'evaluator':
                tic = time.perf_counter()
                evaluated = evaluator.evaluate(obs, current_action=prev_action)
                elapsed_action_score_s += time.perf_counter() - tic
                action = evaluated.best_action
                boundary_choices += int(evaluated.best.robust_status == 'boundary_sensitive')
            else:
                action = (random_gen.choice([-1., -.5, 0., .5, 1.]),
                          random_gen.choice([0., 1.]))
            action_switches += int(prev_action is not None and action != prev_action)
            prev_action = action
            obs, _, terminated, truncated, last_info = env.step(
                np.asarray(action, dtype=np.float32))
            steps = step + 1
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        'steps': steps,
        'time_s': round(float(last_info['time_elapsed']), 6),
        'game_state': last_info.get('game_state', 'unknown'),
        'policy_compute_s': round(elapsed_action_score_s, 6),
        'boundary_choices': boundary_choices,
        'action_switches': action_switches,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--horizon-steps', type=int, default=8)
    p.add_argument('--robust-mode', choices=['all','top_k','none'], default='top_k')
    p.add_argument('--robust-top-k', type=int, default=3)
    p.add_argument('--out', default='logs/action_evaluator_smoke')
    args = p.parse_args()
    if args.episodes < 1 or args.steps < 1:
        p.error('--episodes and --steps must be >=1')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    forward = ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv(
        Path('map_data/object_tiles.csv')))
    evaluator = ActionEvaluator(forward, horizon_steps=args.horizon_steps,
                                robust_mode=args.robust_mode, robust_top_k=args.robust_top_k)
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.episodes):
        seed = rng.randrange(2**32)
        action_seed = rng.randrange(2**32)
        # Paired environment seeds; actions differ by policy by design.
        baseline = episode(seed, 'random', args.steps, evaluator, action_seed)
        policy = episode(seed, 'evaluator', args.steps, evaluator, action_seed)
        rows.append({
            'episode': i, 'env_seed': seed, 'random_action_seed': action_seed,
            **{'random_' + k: v for k, v in baseline.items()},
            **{'evaluator_' + k: v for k, v in policy.items()},
            'delta_time_s': round(policy['time_s'] - baseline['time_s'], 6),
        })
        if (i + 1) % 5 == 0:
            print('PROGRESS', i+1, '/', args.episodes,
                  'avg_random_time_s',round(statistics.mean(x['random_time_s'] for x in rows), 3),
                  'avg_eval_time_s',round(statistics.mean(x['evaluator_time_s'] for x in rows), 3),
                  flush=True)
    with (output / 'episode_pairs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    avg_random = statistics.mean(x['random_time_s'] for x in rows)
    avg_eval = statistics.mean(x['evaluator_time_s'] for x in rows)
    avg_compute = statistics.mean(x['evaluator_policy_compute_s'] for x in rows)
    print('PAIRED SUMMARY episodes', args.episodes,
          'avg_random_survival_s', round(avg_random, 3),
          'avg_evaluator_survival_s', round(avg_eval, 3),
          'mean_delta_s', round(avg_eval-avg_random, 3),
          'mean_policy_compute_s_per_episode', round(avg_compute, 3))
    print('OUTPUT', output / 'episode_pairs.csv')


if __name__ == '__main__':
    main()
