r"""Real-env random-seed robustness benchmark, no engine-private prediction input.

From project root (PowerShell):
  .\.venv\Scripts\python.exe test_forward_robust_multiseed.py --episodes 100 --steps 100 --horizon-steps 1
  .\.venv\Scripts\python.exe test_forward_robust_multiseed.py --episodes 1000 --steps 150 --horizon-steps 8 --stride 8

Each nonoverlapping/strided anchor generates its future actions BEFORE
executing them, then compares nominal and 9-trajectory robust predictions
against the real outcome. Only no-unknown-spawn/no-star-change/no-fallen-
bullet windows are eligible; censored windows are reported, not successes.
The nine trajectories are sensitivity samples, NOT calibrated probabilities.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from agent_env import create_env
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor


def visible_bullets(obs):
    return sorted(tuple(round(float(v)) for v in row[:4])
                  for row in obs['bullet_bills']
                  if float(row[2]) > float(row[0]) and float(row[3]) > float(row[1]))


def visible_stars(obs):
    return sorted(tuple(round(float(v)) for v in row[:4])
                  for row in obs.get('stars', [])
                  if float(row[2]) > float(row[0]) and float(row[3]) > float(row[1]))


def known_bullet_boxes(start, frames, simulator):
    """Independent known-bullet projection for detecting random spawns.

    Does not use any predicted Mario or collision status. Fallen transitions
    are censored separately by checking real engine states.
    """
    out = []
    width = simulator.motion.cfg.world_width
    fps = simulator.fps
    for bullet in simulator.bullets_from_observation(start):
        moved = bullet.advanced(frames=frames, fps=fps)
        if (moved.vx > 0 and moved.bbox[0] > width) or (moved.vx < 0 and moved.bbox[2] < 0):
            continue
        out.append(moved.bbox)
    return sorted(out)


def has_hidden_fallen(env):
    return any(b.state_ != 'active' for b in env.unwrapped._bullet_bills)


def run(args):
    if args.episodes < 1 or args.steps < 1 or args.horizon_steps < 1 or args.stride < 1:
        raise ValueError('episodes, steps, horizon-steps and stride must be >= 1')
    if args.stride < args.horizon_steps:
        raise ValueError('stride < horizon-steps would require overlapping windows; use stride >= horizon-steps')
    if args.horizon_steps > args.steps:
        raise ValueError('horizon-steps must not exceed steps')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    simulator = ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv(
        Path('map_data/object_tiles.csv')))
    master_rng = random.Random(args.seed)
    fieldnames = [
        'episode', 'env_seed', 'start_step', 'horizon_steps', 'actual_frames',
        'actual_terminated', 'actual_lethal_frame', 'nominal_lethal_frame',
        'robust_lethal_frames', 'robust_status', 'nominal_correct',
        'robust_includes_actual', 'actual_safe_robust_falsely_all_lethal',
        'actual_lethal_nominal_missed', 'actual_lethal_robust_missed',
        'predicted_nominal_reason', 'censor_reason', 'nominal_time_ms',
        'robust_time_ms', 'position_spread_x', 'position_spread_y',
    ]
    counters = {k: 0 for k in [
        'episodes', 'windows', 'eligible', 'censored', 'actual_lethal', 'actual_safe',
        'nominal_correct', 'robust_includes_actual', 'boundary_sensitive',
        'nominal_false_safe', 'robust_false_safe', 'nominal_false_alarm',
        'robust_falsely_all_lethal', 'nominal_time_ns', 'robust_time_ns',
        'sampled_lethal_windows',
    ]}
    censor_counts = {}
    with (out / 'windows.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for episode in range(args.episodes):
            env = create_env(render_mode='none')
            counters['episodes'] += 1
            try:
                env_seed = master_rng.randrange(2**32)
                obs, info = env.reset(seed=env_seed)
                step = 0
                terminated = truncated = False
                while step < args.steps and not terminated and not truncated:
                    # Keep exact random action generation scheme compatible with
                    # the existing 100-episode one-step smoke test.
                    anchor_step = step
                    n = min(args.horizon_steps, args.steps - step)
                    actions = [np.asarray([
                        master_rng.choice([-1., -.5, 0., .5, 1.]),
                        master_rng.choice([0., 1.])], dtype=np.float32)
                        for _ in range(n)]
                    start_obs = obs
                    pre_fallen = has_hidden_fallen(env)
                    before = time.perf_counter_ns()
                    nominal = simulator.simulate_observation(start_obs, actions)
                    mid = time.perf_counter_ns()
                    robust = simulator.simulate_observation_robust(start_obs, actions)
                    after = time.perf_counter_ns()
                    counters['windows'] += 1
                    nominal_ns, robust_ns = mid-before, after-mid
                    actual_frames = 0
                    reasons = set()
                    if pre_fallen:
                        reasons.add('initial_hidden_fallen_bullet')
                    if nominal.termination_reason == 'unknown_star_pickup':
                        reasons.add('nominal_unknown_star')
                    if any(s.termination_reason == 'unknown_star_pickup' for s in robust.samples):
                        reasons.add('sample_unknown_star')
                    base_stars = visible_stars(start_obs)
                    prior_time = float(info['time_elapsed'])
                    for action in actions:
                        next_obs, _, terminated, truncated, next_info = env.step(action)
                        dt = float(next_info['time_elapsed']) - prior_time
                        prior_time = float(next_info['time_elapsed'])
                        internal_frames = int(round(dt * simulator.fps))
                        if internal_frames < 1 or internal_frames > simulator.frame_skip or abs(
                                dt * simulator.fps - internal_frames) > 1e-4:
                            reasons.add('invalid_elapsed_frames')
                        actual_frames += internal_frames
                        if has_hidden_fallen(env):
                            reasons.add('hidden_fallen_bullet')
                        if visible_stars(next_obs) != base_stars:
                            reasons.add('star_configuration_changed')
                        if visible_bullets(next_obs) != known_bullet_boxes(
                                start_obs, actual_frames, simulator):
                            reasons.add('new_or_unmodelled_bullet')
                        if bool(next_obs['mario'][6]) != bool(start_obs['mario'][6]):
                            reasons.add('invincibility_transition')
                        obs, info = next_obs, next_info
                        step += 1
                        if terminated or truncated:
                            break
                    window_terminated, window_truncated = bool(terminated), bool(truncated)
                    window_state = info.get('game_state')
                    # --stride controls how far ahead the next anchor starts.
                    if not terminated and not truncated:
                        for _ in range(max(0, args.stride - n)):
                            if step >= args.steps:
                                break
                            extra_action = np.asarray([
                                master_rng.choice([-1., -.5, 0., .5, 1.]),
                                master_rng.choice([0., 1.])], dtype=np.float32)
                            obs, _, terminated, truncated, info = env.step(extra_action)
                            step += 1
                            if terminated or truncated:
                                break
                    if window_truncated or (window_terminated and window_state != 'game_over'):
                        reasons.add('non_death_terminal')
                    actual_lethal = actual_frames if window_terminated and not window_truncated and not reasons.intersection(
                        {'non_death_terminal'}) else None
                    nominal_frame = (nominal.first_lethal_event.frame
                                     if nominal.first_lethal_event is not None else None)
                    sampled = robust.lethal_frames
                    nominal_correct = nominal_frame == actual_lethal
                    includes = actual_lethal in sampled
                    row = dict(
                        episode=episode, env_seed=env_seed, start_step=anchor_step,
                        horizon_steps=n, actual_frames=actual_frames,
                        actual_terminated=int(window_terminated), actual_lethal_frame=actual_lethal,
                        nominal_lethal_frame=nominal_frame,
                        robust_lethal_frames=';'.join('none' if x is None else str(x) for x in sampled),
                        robust_status=robust.collision_status,
                        nominal_correct=int(nominal_correct), robust_includes_actual=int(includes),
                        actual_safe_robust_falsely_all_lethal=int(actual_lethal is None and all(x is not None for x in sampled)),
                        actual_lethal_nominal_missed=int(actual_lethal is not None and nominal_frame is None),
                        actual_lethal_robust_missed=int(actual_lethal is not None and actual_lethal not in sampled),
                        predicted_nominal_reason=nominal.termination_reason,
                        censor_reason=';'.join(sorted(reasons)),
                        nominal_time_ms=round(nominal_ns/1e6, 4),
                        robust_time_ms=round(robust_ns/1e6, 4),
                        position_spread_x=robust.position_spread_px[0],
                        position_spread_y=robust.position_spread_px[1],
                    )
                    writer.writerow(row)
                    if reasons:
                        counters['censored'] += 1
                        for reason in reasons:
                            censor_counts[reason] = censor_counts.get(reason, 0) + 1
                    else:
                        counters['eligible'] += 1
                        counters['nominal_time_ns'] += nominal_ns
                        counters['robust_time_ns'] += robust_ns
                        counters['nominal_correct'] += nominal_correct
                        counters['robust_includes_actual'] += includes
                        counters['boundary_sensitive'] += robust.boundary_sensitive
                        counters['sampled_lethal_windows'] += int(any(x is not None for x in sampled))
                        counters['actual_lethal'] += int(actual_lethal is not None)
                        counters['actual_safe'] += int(actual_lethal is None)
                        counters['nominal_false_safe'] += int(actual_lethal is not None and nominal_frame is None)
                        counters['robust_false_safe'] += int(actual_lethal is not None and all(x is None for x in sampled))
                        counters['nominal_false_alarm'] += int(actual_lethal is None and nominal_frame is not None)
                        counters['robust_falsely_all_lethal'] += int(actual_lethal is None and all(x is not None for x in sampled))
            finally:
                env.close()
            if (episode + 1) % args.report_every == 0:
                print('PROGRESS', episode+1, '/', args.episodes, 'eligible', counters['eligible'],
                      'censored', counters['censored'], 'nominal_correct', counters['nominal_correct'],
                      'robust_include', counters['robust_includes_actual'], flush=True)
    eligible = counters['eligible']
    summary = {
        'config': vars(args),
        'counts': counters,
        'censor_reasons': censor_counts,
        'rates': {
            'coverage': eligible / counters['windows'] if counters['windows'] else None,
            'nominal_exact_frame_accuracy': counters['nominal_correct'] / eligible if eligible else None,
            'robust_actual_outcome_in_sample_set': counters['robust_includes_actual'] / eligible if eligible else None,
            'boundary_sensitive_rate': counters['boundary_sensitive'] / eligible if eligible else None,
            'nominal_false_safe_given_actual_lethal': counters['nominal_false_safe'] / counters['actual_lethal'] if counters['actual_lethal'] else None,
            'robust_false_safe_given_actual_lethal': counters['robust_false_safe'] / counters['actual_lethal'] if counters['actual_lethal'] else None,
            'mean_nominal_ms': counters['nominal_time_ns'] / eligible / 1e6 if eligible else None,
            'mean_robust_ms': counters['robust_time_ns'] / eligible / 1e6 if eligible else None,
            'robust_to_nominal_time_ratio': counters['robust_time_ns'] / counters['nominal_time_ns'] if counters['nominal_time_ns'] else None,
        },
        'caveats': [
            'Nine samples are not a probability distribution or a guaranteed interval.',
            'Unknown future bullet spawns, fallen bullets and star transitions are censored.',
            'Action windows have random controls; this is prediction evaluation, not policy performance.',
            'No unseen engine state is used as prediction input; hidden bullet state is used only to censor.',
            'A reported safe sample means no lethal event inside the chosen horizon only.',
        ],
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FINAL', json.dumps(summary['rates'], ensure_ascii=False, indent=2))
    print('OUTPUT', out / 'summary.json', out / 'windows.csv')
    if eligible == 0:
        print('WARNING: no eligible windows; inspect censor_reasons in summary.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--horizon-steps', type=int, default=1)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--out', default='logs/forward_robust_multiseed')
    p.add_argument('--report-every', type=int, default=25)
    args = p.parse_args()
    if args.report_every <= 0:
        p.error('--report-every must be >0')
    run(args)


if __name__ == '__main__':
    main()
