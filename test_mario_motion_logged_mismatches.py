"""Offline regression: check that idle_threshold=10 explains old replay errors.

Usage from project root:
  python test_mario_motion_logged_mismatches.py
  python test_mario_motion_logged_mismatches.py --source logs/mario_motion_replay/replay_mismatches.csv

This script compares against SAVED engine observations, not a new engine run.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from models.mario_motion_predictor import MarioMotionPredictor, MarioState, MotionConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('logs/mario_motion_replay/replay_mismatches.csv'))
    parser.add_argument('--tile-csv', type=Path,
                        default=Path('map_data/object_tiles.csv'))
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(f'Cannot find {args.source}')

    groups = defaultdict(list)
    with args.source.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            groups[int(row['episode'])].append(row)

    # Replay CSV stores the previous predictor's delta against the game:
    # actual_vx = predicted_old_vx - old_open_dvx.
    old = MarioMotionPredictor.from_object_tiles_csv(
        args.tile_csv, MotionConfig(idle_threshold=1.0))
    fixed = MarioMotionPredictor.from_object_tiles_csv(
        args.tile_csv, MotionConfig(idle_threshold=10.0))
    total = 0
    max_vx_residual = 0.0
    max_x_error = 0

    for ep, rows in sorted(groups.items()):
        if rows[0]['mode'] != 'random_each_step':
            raise ValueError(f'Unsupported mode for offline reconstruction: {rows[0]["mode"]}')
        rng = random.Random(int(rows[0]['action_seed']))
        s_old = MarioState(624, 720, 48, 48, 0.0, 0.0, True)
        s_fix = replace(s_old)
        saved_states = {}
        max_step = max(int(r['step']) for r in rows)
        for step in range(1, max_step + 1):
            action = (rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]),
                      rng.choice([0.0, 0.0, 1.0]))
            s_old, _ = old.step(s_old, action)
            s_fix, _ = fixed.step(s_fix, action)
            saved_states[step] = (s_old.vx, s_fix.vx, s_fix.left)

        local_max_vx = 0.0
        local_max_x = 0
        for row in rows:
            old_vx, new_vx, new_x = saved_states[int(row['step'])]
            engine_vx = old_vx - float(row['open_dvx'])
            residual = abs(new_vx - engine_vx)
            dx = abs(new_x - int(row['actual_x']))
            local_max_vx = max(local_max_vx, residual)
            local_max_x = max(local_max_x, dx)
            total += 1
        max_vx_residual = max(max_vx_residual, local_max_vx)
        max_x_error = max(max_x_error, local_max_x)
        print(f'episode={ep:3d} rows={len(rows):3d} '
              f'max_x_error={local_max_x}px max_vx_residual={local_max_vx:.8f}')

    print(f'OFFLINE SUMMARY: episodes={len(groups)} rows={total} '
          f'max_x_error={max_x_error}px max_vx_residual={max_vx_residual:.8f}px/s')
    assert max_x_error == 0, 'Offline comparison still has position mismatch'
    assert max_vx_residual < 1e-3, 'Offline comparison still has velocity mismatch'
    print('Saved-log regression passed; run real-environment replay next.')


if __name__ == '__main__':
    main()
