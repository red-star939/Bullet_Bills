"""Paired-seed *real environment* evaluation of ActionEvaluator v1 vs v2.

Run from project root, e.g.:
 .\\.venv\\Scripts\\python.exe benchmark_action_evaluator_v2.py --episodes 20 --steps 900

Data is right-censored at max_steps; policy decisions never use engine internals.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
from agent_env import create_env
from models.action_evaluator import ActionEvaluator
from models.action_evaluator_v2 import ActionEvaluatorV2
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor


def run_episode(seed, steps, policy, *, is_v2=False):
    env = create_env(render_mode='none')
    compute_seconds = 0.0
    decisions = 0
    switches = 0
    overrides = 0
    override_reasons = {}
    prev = None
    final_info = {}
    try:
        obs, final_info = env.reset(seed=seed)
        for _ in range(steps):
            t = time.perf_counter()
            result = policy.evaluate(obs, current_action=prev)
            compute_seconds += time.perf_counter() - t
            action = result.best_action
            if is_v2 and result.changed_from_baseline:
                overrides += 1
                override_reasons[result.change_reason] = override_reasons.get(result.change_reason, 0) + 1
            switches += int(prev is not None and action != prev)
            decisions += 1
            prev = action
            obs, _, terminated, truncated, final_info = env.step(np.asarray(action, dtype=np.float32))
            if terminated or truncated:
                break
        return {
            'time_s': round(float(final_info.get('time_elapsed', 0)), 6),
            'game_state': final_info.get('game_state', 'unknown'),
            'decisions': decisions,
            'policy_compute_s': round(compute_seconds, 6),
            'policy_compute_ms_per_decision': round(1000 * compute_seconds / max(1, decisions), 4),
            'switches': switches,
            'v2_fall_overrides': overrides,
            'v2_override_reasons': json.dumps(override_reasons, ensure_ascii=False, sort_keys=True),
        }
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--steps', type=int, default=900)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--horizon-steps', type=int, default=8)
    p.add_argument('--extended-horizon-steps', type=int, default=20)
    p.add_argument('--audit-top-k', type=int, default=4)
    p.add_argument('--report-every', type=int, default=10)
    p.add_argument('--out', default='logs/action_evaluator_v2_benchmark')
    args = p.parse_args()
    if min(args.episodes,args.steps,args.horizon_steps,args.audit_top_k,args.report_every) <= 0:
        p.error('positive parameters required')
    if args.extended_horizon_steps < args.horizon_steps:
        p.error('extended horizon must be >= baseline horizon')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sim = ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv('map_data/object_tiles.csv'))
    v1 = ActionEvaluator(sim, horizon_steps=args.horizon_steps, robust_mode='none')
    v2 = ActionEvaluatorV2(sim, horizon_steps=args.horizon_steps,
                           extended_horizon_steps=args.extended_horizon_steps,
                           audit_top_k=args.audit_top_k, robust_mode='none')
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.episodes):
        seed = rng.randrange(2**32)
        rng.randrange(2**32)  # preserve earlier paired experiment seed sequence
        b = run_episode(seed, args.steps, v1)
        v = run_episode(seed, args.steps, v2, is_v2=True)
        rows.append({'episode':i, 'env_seed':seed,
                     **{'v1_'+k:value for k,value in b.items()},
                     **{'v2_'+k:value for k,value in v.items()},
                     'delta_time_s':round(v['time_s']-b['time_s'],6)})
        if (i+1) % args.report_every == 0:
            print('PROGRESS', i+1, '/', args.episodes,
                  'v1',round(statistics.mean(x['v1_time_s'] for x in rows),3),
                  'v2',round(statistics.mean(x['v2_time_s'] for x in rows),3),
                  flush=True)
    with (out/'episode_pairs.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    summary={
      'episodes':args.episodes, 'configuration':vars(args),
      'v1_mean_observed_time_s':statistics.mean(r['v1_time_s'] for r in rows),
      'v2_mean_observed_time_s':statistics.mean(r['v2_time_s'] for r in rows),
      'v2_wins':sum(r['delta_time_s']>0 for r in rows),
      'ties':sum(r['delta_time_s']==0 for r in rows),
      'v2_losses':sum(r['delta_time_s']<0 for r in rows),
      'v1_survived_to_step_limit':sum(r['v1_game_state']=='playing' for r in rows),
      'v2_survived_to_step_limit':sum(r['v2_game_state']=='playing' for r in rows),
      'v1_mean_compute_ms_per_decision':1000*sum(r['v1_policy_compute_s'] for r in rows)/sum(r['v1_decisions'] for r in rows),
      'v2_mean_compute_ms_per_decision':1000*sum(r['v2_policy_compute_s'] for r in rows)/sum(r['v2_decisions'] for r in rows),
      'v2_total_fall_overrides':sum(r['v2_v2_fall_overrides'] for r in rows),
      'note':'Time capped at step limit; longer nominal forecasts assume no unknown future spawns.'
    }
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL SUMMARY',json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
