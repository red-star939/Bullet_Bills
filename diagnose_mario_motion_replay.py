r"""Detailed deterministic replay of previously failed episodes.

Run from project root:
  .\.venv\Scripts\python.exe diagnose_mario_motion_replay.py --episodes 9 18

Prints actual/predicted VX and both LOCAL and OPEN errors. Use this to
isolate the first mismatch after exact float32 force conversion.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np

from agent_env import create_env
from models.mario_motion_predictor import MarioMotionPredictor, MarioState
from test_mario_motion_multiseed import decide_action, TILE_CSV


def get_record(path: Path, episode: int) -> dict:
    with path.open(newline='', encoding='utf-8-sig') as f:
        for record in csv.DictReader(f):
            if int(record['episode']) == episode:
                return record
    raise ValueError(f'episode {episode} not found in {path}')


def diagnostic(record: dict, model: MarioMotionPredictor, steps: int, tol: float):
    ep = int(record['episode'])
    seed, action_seed, mode = int(record['env_seed']), int(record['action_seed']), record['mode']
    print(f'\n=== episode={ep} env_seed={seed} action_seed={action_seed} mode={mode} ===')
    print('step | action_x jump | force_x force_y | actual_x pred_x | actual_vx pred_vx | open_dx open_dvx | local_dx local_dvx | grounded | note')
    env = create_env(render_mode='none')
    rng = random.Random(action_seed)
    control = {}
    try:
        obs, info = env.reset(seed=seed)
        open_state = MarioState.from_observation(obs['mario'])
        prev_time = float(info['time_elapsed'])
        first_bad = None
        for step in range(1, steps + 1):
            action = decide_action(mode, step - 1, rng, control)
            action_array = np.asarray(action, dtype=np.float32)
            force = model.action_to_force(action_array)
            local_start = MarioState.from_observation(obs['mario'])
            local, _ = model.step(local_start, action_array)
            open_state, _ = model.step(open_state, action_array)
            obs, _, terminated, truncated, info = env.step(action_array)
            dt = float(info['time_elapsed']) - prev_time
            prev_time = float(info['time_elapsed'])
            complete = math.isclose(dt, model.cfg.frame_skip/model.cfg.fps,
                                    rel_tol=0.0, abs_tol=1e-5)
            if not complete:
                print(f'{step:>4d} | PARTIAL-STEP dt={dt:.9f}; excluded from comparison')
                break
            m = obs['mario']
            dx, dvx = open_state.left-int(m[0]), open_state.vx-float(m[4])
            ldx, ldvx = local.left-int(m[0]), local.vx-float(m[4])
            ground_ok = (open_state.grounded == (not bool(m[7])))
            bad = abs(dx)>1 or abs(open_state.top-int(m[1]))>1 or abs(dvx)>tol or abs(open_state.vy-float(m[5]))>tol or not ground_ok
            if bad and first_bad is None:
                first_bad = step
            note = 'FIRST_MISMATCH' if first_bad==step else ('mismatch' if bad else '')
            print(f'{step:>4d} | {action[0]:+5.2f} {action[1]:4.2f} | {force[0]:+7.3f} {force[1]:+7.1f} | '
                  f'{int(m[0]):>8d} {open_state.left:>6d} | {float(m[4]):+10.5f} {open_state.vx:+10.5f} | '
                  f'{dx:+7d} {dvx:+9.5f} | {ldx:+8d} {ldvx:+9.5f} | '
                  f'{int(ground_ok):>8d} | {note}')
            if terminated or truncated:
                break
        print(f'first mismatch step: {first_bad if first_bad is not None else "none"}')
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('logs/mario_motion_multiseed/episode_summary.csv'))
    parser.add_argument('--episodes', type=int, nargs='+', default=[9, 18])
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--velocity-tol', type=float, default=1e-3)
    args = parser.parse_args()
    model = MarioMotionPredictor.from_object_tiles_csv(TILE_CSV)
    for ep in args.episodes:
        diagnostic(get_record(args.source, ep), model, args.steps, args.velocity_tol)


if __name__ == '__main__':
    main()
