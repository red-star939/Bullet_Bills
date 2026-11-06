r"""Gym one-step smoke test for ForwardSimulator (no prediction data leakage).

Run from project root:
  .\.venv\Scripts\python.exe test_forward_simulator_env.py --episodes 20 --steps 100

Prediction uses ONLY the current observation. The real engine's private objects
are read AFTER env.step solely to detect unobservable state/new-spawn cases,
not fed into the forward simulation. Episodes are deterministic per seed.
This smoke test does not measure multi-action open-loop accuracy.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from agent_env import create_env
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261106)
    args = parser.parse_args()

    motion = MarioMotionPredictor.from_object_tiles_csv(
        Path('map_data/object_tiles.csv'))
    forward = ForwardSimulator(motion)
    rng = random.Random(args.seed)
    checked, skipped, bad, term_checked = 0, 0, 0, 0
    first_bad = []
    for episode in range(args.episodes):
        env = create_env(render_mode='none')
        try:
            seed = rng.randrange(2**32)
            obs, info = env.reset(seed=seed)
            for step in range(args.steps):
                x = rng.choice([-1., -.5, 0., .5, 1.])
                jump = rng.choice([0., 1.])
                action = np.array([x, jump], dtype=np.float32)

                # Predict first, before the actual environment step executes.
                pred = forward.simulate_observation(obs, [action])
                prior_time = float(info['time_elapsed'])
                next_obs, _, terminated, truncated, next_info = env.step(action)
                dt = float(next_info['time_elapsed']) - prior_time
                expected_dt = motion.cfg.frame_skip / motion.cfg.fps

                # Only compare full physics steps. A new spawn or observed
                # 'fallen' bullet may invalidate the observation-only model.
                # Such cases are excluded, not silently counted as correct.
                if (abs(dt - expected_dt) > 1e-5
                        or pred.termination_reason == 'unknown_star_pickup'
                        or not pred.frames
                        or any(s != 'active' for s in
                               (b.state_ for b in env.unwrapped._bullet_bills))):
                    skipped += 1
                else:
                    # Spawn detection: the raw 5-value observation lacks stable
                    # IDs. Count changes alone are insufficient if OOB happens,
                    # so check whether every new observed projectile can be
                    # explained by some predicted final projectile bbox.
                    actual_rows = [tuple(int(round(float(v))) for v in row[:4])
                                   for row in next_obs['bullet_bills']
                                   if float(row[2]) > float(row[0]) and
                                   float(row[3]) > float(row[1])]
                    predicted_rows = [b.bbox for b in pred.bullets]
                    if sorted(actual_rows) != sorted(predicted_rows):
                        skipped += 1
                    else:
                        checked += 1
                        actual = next_obs['mario']
                        final = pred.mario
                        dx = final.left - round(float(actual[0]))
                        dy = final.top - round(float(actual[1]))
                        dvx = final.vx - float(actual[4])
                        dvy = final.vy - float(actual[5])
                        termination_match = pred.terminated == bool(terminated)
                        if terminated:
                            term_checked += 1
                        if (abs(dx) > 1 or abs(dy) > 1 or
                            abs(dvx) > 0.001 or abs(dvy) > 0.001 or
                            not termination_match):
                            bad += 1
                            if len(first_bad) < 10:
                                first_bad.append((seed, step + 1, dx, dy,
                                                  dvx, dvy, termination_match))

                obs, info = next_obs, next_info
                if terminated or truncated:
                    break
        finally:
            env.close()
    print(f'episodes={args.episodes} checked_steps={checked} '
          f'skipped_unknown={skipped} mismatches={bad} '
          f'completed_terminal_steps={term_checked}')
    for item in first_bad:
        print('MISMATCH seed,step,dx,dy,dvx,dvy,terminal_match=', item)
    if checked == 0:
        print('WARNING: no comparable steps; expand scenario or inspect skip criteria')
    elif bad == 0:
        print('Forward Simulator one-action environment smoke check passed!')
    else:
        print('Discrepancies require first-mismatch investigation')


if __name__ == '__main__':
    main()
