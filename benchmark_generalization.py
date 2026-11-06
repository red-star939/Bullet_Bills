"""Leakage-free paired-seed generalization test: ActionEvaluator v1 vs v2.

The 'reference' split reproduces the old 100 environmental seeds (seed=20261106,
consuming the prior unused random seed). The 'holdout' split draws disjoint
seeds from an independent RNG. Never tune policy against holdout outcomes.

Usage (Windows PowerShell):
 .\\.venv\\Scripts\\python.exe benchmark_generalization.py --reference-episodes 100 --holdout-episodes 1000 --steps 900 --out logs/generalization_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable


FPS = 60
FRAME_SKIP = 4


def reference_seeds(n: int, seed: int = 20261106) -> list[int]:
    rng = random.Random(seed)
    result = []
    for _ in range(n):
        result.append(rng.randrange(2**32))
        rng.randrange(2**32)  # match existing paired benchmark generation
    return result


def holdout_seeds(n: int, *, seed: int = 20261107,
                  excluded: Iterable[int] = ()) -> list[int]:
    blocked = set(excluded)
    rng = random.Random(seed)
    result = []
    while len(result) < n:
        value = rng.randrange(2**32)
        if value not in blocked:
            result.append(value)
            blocked.add(value)
    return result


def bootstrap_paired_delta_ci(deltas: list[float], *, samples: int = 2000,
                              seed: int = 1234) -> tuple[float, float] | None:
    if not deltas:
        return None
    rng = random.Random(seed)
    n = len(deltas)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def summarize(rows: list[dict], *, steps: int, bootstraps: int = 2000) -> dict:
    if not rows:
        return {'episodes': 0}
    horizon = steps * FRAME_SKIP / FPS
    v1 = [r['v1_time_s'] for r in rows]
    v2 = [r['v2_time_s'] for r in rows]
    deltas = [b - a for a, b in zip(v1, v2)]
    def survivors(prefix: str, seconds: float) -> int | None:
        if seconds > horizon + 1e-8:
            return None
        # 'playing' at the limit is right-censored; reaching the threshold
        # counts as survival to the threshold, not proof of surviving after it.
        return sum(float(r[f'{prefix}_time_s']) >= seconds - 1e-5 for r in rows)
    def compute(prefix):
        ndec = sum(int(r[f'{prefix}_decisions']) for r in rows)
        return 1000 * sum(float(r[f'{prefix}_policy_compute_s']) for r in rows) / max(ndec, 1)
    return {
        'episodes': len(rows), 'max_observed_time_s': horizon,
        'v1_mean_capped_survival_s': statistics.mean(v1),
        'v2_mean_capped_survival_s': statistics.mean(v2),
        'mean_paired_delta_s': statistics.mean(deltas),
        'bootstrap_95pct_ci_paired_delta_s': bootstrap_paired_delta_ci(deltas, samples=bootstraps),
        'v2_wins': sum(x > 1e-5 for x in deltas),
        'ties': sum(abs(x) <= 1e-5 for x in deltas),
        'v2_losses': sum(x < -1e-5 for x in deltas),
        'v1_censored_at_limit': sum(r['v1_game_state'] == 'playing' for r in rows),
        'v2_censored_at_limit': sum(r['v2_game_state'] == 'playing' for r in rows),
        'survivors_at_s': {
            str(t): {'v1': survivors('v1', t), 'v2': survivors('v2', t)}
            for t in (10, 20, 40, 60) if t <= horizon + 1e-8},
        'v1_compute_ms_per_decision_weighted': compute('v1'),
        'v2_compute_ms_per_decision_weighted': compute('v2'),
        'v2_fall_overrides': sum(r['v2_v2_fall_overrides'] for r in rows),
        'warning': 'Capped survival time is not uncensored expected lifespan; '
                   'bootstrap intervals quantify sampled-seed variability only.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-episodes', type=int, default=100)
    parser.add_argument('--holdout-episodes', type=int, default=1000)
    parser.add_argument('--reference-seed', type=int, default=20261106)
    parser.add_argument('--holdout-seed', type=int, default=20261107)
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--horizon-steps', type=int, default=8)
    parser.add_argument('--extended-horizon-steps', type=int, default=20)
    parser.add_argument('--audit-top-k', type=int, default=4)
    parser.add_argument('--report-every', type=int, default=25)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--out', default='logs/generalization_v1')
    args = parser.parse_args()
    if args.reference_episodes < 0 or args.holdout_episodes < 0 or not (args.reference_episodes + args.holdout_episodes):
        parser.error('At least one split requires positive episodes')
    if min(args.steps, args.horizon_steps, args.audit_top_k, args.report_every, args.bootstrap_samples) <= 0:
        parser.error('Positive steps/horizons/top-k/report/bootstrap required')
    if args.extended_horizon_steps < args.horizon_steps:
        parser.error('extended horizon must be >= horizon')
    if args.reference_seed == args.holdout_seed:
        parser.error('reference and holdout RNG seeds must differ')

    # Live game dependency is needed only for an actual benchmark, not for
    # seed-generation and summary unit tests.
    from benchmark_action_evaluator_v2 import run_episode
    from models.action_evaluator import ActionEvaluator
    from models.action_evaluator_v2 import ActionEvaluatorV2
    from models.forward_simulator import ForwardSimulator
    from models.mario_motion_predictor import MarioMotionPredictor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = reference_seeds(args.reference_episodes, args.reference_seed)
    hold = holdout_seeds(args.holdout_episodes, seed=args.holdout_seed, excluded=ref)
    assert len(ref) == len(set(ref)) and len(hold) == len(set(hold)) and not (set(ref) & set(hold)), 'Seed collision'
    simulator = ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv('map_data/object_tiles.csv'))
    v1 = ActionEvaluator(simulator, horizon_steps=args.horizon_steps, robust_mode='none')
    v2 = ActionEvaluatorV2(simulator, horizon_steps=args.horizon_steps,
                           extended_horizon_steps=args.extended_horizon_steps,
                           audit_top_k=args.audit_top_k, robust_mode='none')
    rows = []
    total = len(ref) + len(hold)
    for split, seeds in (('reference', ref), ('holdout', hold)):
        for i, seed in enumerate(seeds):
            baseline = run_episode(seed, args.steps, v1)
            enhanced = run_episode(seed, args.steps, v2, is_v2=True)
            row = {'split': split, 'episode_in_split': i, 'env_seed': seed}
            row.update({'v1_' + k: value for k, value in baseline.items()})
            row.update({'v2_' + k: value for k, value in enhanced.items()})
            row['delta_time_s'] = round(enhanced['time_s'] - baseline['time_s'], 6)
            rows.append(row)
            if len(rows) % args.report_every == 0:
                print('PROGRESS', len(rows), '/', total, 'split', split,
                      'mean_delta_s', round(statistics.mean(r['delta_time_s'] for r in rows), 3), flush=True)
    if rows:
        with (out/'episode_pairs.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    group_ref = [r for r in rows if r['split'] == 'reference']
    group_hold = [r for r in rows if r['split'] == 'holdout']
    report = {
        'configuration': vars(args),
        'splits_disjoint': True,
        'reference': summarize(group_ref, steps=args.steps, bootstraps=args.bootstrap_samples),
        'holdout': summarize(group_hold, steps=args.steps, bootstraps=args.bootstrap_samples),
        'all': summarize(rows, steps=args.steps, bootstraps=args.bootstrap_samples),
        'seed_generation': 'reference: random.Random(reference_seed), draw env and discard next draw; '
                           'holdout: independent random.Random(holdout_seed), unique and disjoint',
        'method': 'Same environment seed for each policy; separate environment instances; '
                  'game RNG may diverge as trajectories diverge, so common seed is not identical future spawn sequence.',
    }
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FINAL', json.dumps({'reference': report['reference'], 'holdout': report['holdout']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
