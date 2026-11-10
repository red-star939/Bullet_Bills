"""Compare ActionEvaluator v2 and v3 in the REAL environment on matched seeds.

Records each episode immediately, supports --resume, separates reference and
holdout. A 1200-second cap is 18000 decisions at 60 fps / frame_skip 4.
If the environment itself truncates earlier, do NOT interpret it as a 20-minute
survival result. No claims of 600-second success without real observed results.
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

from models.action_evaluator_v2 import ActionEvaluatorV2
from models.action_evaluator_v3 import ActionEvaluatorV3
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor

FIELDS = ('split','episode','seed','v2_time_s','v3_time_s','delta_s',
          'v2_state','v3_state','v2_terminal','v3_terminal',
          'v2_decisions','v3_decisions','v2_ms_decision','v3_ms_decision',
          'v3_overrides','v3_recovery_probes')


def seeds(count, initial, excluded=()):
    rng = random.Random(initial)
    blocked = set(excluded)
    values = []
    while len(values) < count:
        s = rng.randrange(2**32)
        if s not in blocked:
            values.append(s); blocked.add(s)
    return values


def play(policy, seed, max_steps, is_v3=False):
    from agent_env import create_env
    env = create_env(render_mode='none')
    prev = None
    t_compute = 0.
    switches = 0
    probes = 0
    info = {}
    terminated = truncated = False
    decisions = 0
    try:
        obs, info = env.reset(seed=seed)
        for decisions in range(1, max_steps + 1):
            start = time.perf_counter()
            result = policy.evaluate(obs, current_action=prev)
            t_compute += time.perf_counter() - start
            action = result.best_action
            if is_v3:
                switches += int(result.changed_from_baseline)
                probes += result.extra_simulations
            prev = action
            obs, _, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
            if terminated or truncated:
                break
        return {
            'time_s': round(float(info.get('time_elapsed', decisions * 4 / 60)), 6),
            'state': str(info.get('game_state', 'unknown')),
            'terminal': ('terminated' if terminated else 'truncated' if truncated else 'step_cap'),
            'decisions': decisions,
            'ms_decision': round(t_compute * 1000 / max(1, decisions), 4),
            'overrides': switches, 'probes': probes,
        }
    finally:
        env.close()


def aggregate(rows, name, cap_s):
    if not rows:
        return {'episodes': 0}
    vals = [float(r[f'{name}_time_s']) for r in rows]
    terminal = [r[f'{name}_terminal'] for r in rows]
    return {
        'episodes': len(rows),
        'mean_capped_survival_s': statistics.mean(vals),
        'median_capped_survival_s': statistics.median(vals),
        'threshold_survival_counts': {str(t):sum(v >= t - 1e-5 for v in vals)
                                      for t in (60, 120, 300, 600, 1200) if t <= cap_s},
        'terminated': terminal.count('terminated'),
        'environment_truncated': terminal.count('truncated'),
        'step_censored': terminal.count('step_cap'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-episodes', type=int, default=20)
    p.add_argument('--holdout-episodes', type=int, default=20)
    p.add_argument('--steps', type=int, default=900)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--report-every', type=int, default=5)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--out', default='logs/v3_comparison')
    args = p.parse_args()
    if min(args.steps, args.report_every) <= 0 or min(args.reference_episodes, args.holdout_episodes) < 0:
        p.error('Invalid episode count or steps')
    if not args.reference_episodes + args.holdout_episodes:
        p.error('At least one episode is needed')
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    ref = seeds(args.reference_episodes, args.seed)
    hold = seeds(args.holdout_episodes, args.seed + 1, ref)
    tasks = [('reference', i, seed) for i, seed in enumerate(ref)] + [('holdout', i, seed) for i, seed in enumerate(hold)]
    csv_path = output / 'episode_pairs.csv'
    rows = []
    completed = set()
    if args.resume and csv_path.exists():
        with csv_path.open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        completed = {(r['split'], int(r['episode']), int(r['seed'])) for r in rows}
        expected = {(split, i, seed) for split, i, seed in tasks}
        if not completed.issubset(expected):
            raise RuntimeError('Previous output does not match current configuration/seeds')
    elif csv_path.exists():
        raise RuntimeError(f'{csv_path} exists; specify another --out or use --resume')
    simulator = ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv('map_data/object_tiles.csv'))
    v2 = ActionEvaluatorV2(simulator, robust_mode='none')
    v3 = ActionEvaluatorV3(simulator)
    for split, i, seed in tasks:
        if (split, i, seed) in completed:
            continue
        b = play(v2, seed, args.steps)
        c = play(v3, seed, args.steps, is_v3=True)
        row = {'split': split, 'episode': i, 'seed': seed,
               'v2_time_s': b['time_s'], 'v3_time_s': c['time_s'],
               'delta_s': round(c['time_s'] - b['time_s'], 6),
               'v2_state': b['state'], 'v3_state': c['state'],
               'v2_terminal': b['terminal'], 'v3_terminal': c['terminal'],
               'v2_decisions': b['decisions'], 'v3_decisions': c['decisions'],
               'v2_ms_decision': b['ms_decision'], 'v3_ms_decision': c['ms_decision'],
               'v3_overrides': c['overrides'], 'v3_recovery_probes': c['probes']}
        rows.append(row)
        with csv_path.open('a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        if len(rows) % args.report_every == 0:
            print('PROGRESS', len(rows), '/', len(tasks), split,
                  'v2', round(statistics.mean(float(x['v2_time_s']) for x in rows), 3),
                  'v3', round(statistics.mean(float(x['v3_time_s']) for x in rows), 3),
                  flush=True)
    horizon = args.steps * simulator.frame_skip / simulator.fps
    report = {
        'config': vars(args), 'max_observed_seconds': horizon,
        'reference': {}, 'holdout': {},
        'warning': 'Right-censored at step cap; game may truncate earlier. '
                   'Matching seeds do not ensure identical future spawn RNG consumption.'}
    for split in ('reference','holdout'):
        split_rows = [r for r in rows if r['split'] == split]
        report[split] = {
            'v2': aggregate(split_rows, 'v2', horizon),
            'v3': aggregate(split_rows, 'v3', horizon),
            'mean_delta_s': statistics.mean(float(r['delta_s']) for r in split_rows) if split_rows else None,
            'v3_wins': sum(float(r['delta_s']) > 1e-5 for r in split_rows),
            'v3_losses': sum(float(r['delta_s']) < -1e-5 for r in split_rows),
        }
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('FINAL', json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
