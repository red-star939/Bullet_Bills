r"""Replay saved multiseed failures after a predictor patch.

Windows PowerShell, from project root:
  .\.venv\Scripts\python.exe replay_mario_motion_failures.py
  .\.venv\Scripts\python.exe replay_mario_motion_failures.py --limit 100
  .\.venv\Scripts\python.exe replay_mario_motion_failures.py --all-failures

Uses episode_summary.csv's (episode, env_seed, action_seed, mode) to replay
identical random action sequences. Does not overwrite original logs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from models.mario_motion_predictor import MarioMotionPredictor
from test_mario_motion_multiseed import TILE_CSV, Trial, run_trial


def read_original(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('logs/mario_motion_multiseed/episode_summary.csv'))
    parser.add_argument('--out', type=Path,
                        default=Path('logs/mario_motion_replay'))
    parser.add_argument('--limit', type=int, default=20,
                        help='Maximum originally failed episodes to replay (default 20)')
    parser.add_argument('--all-failures', action='store_true')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--threshold-px', type=int, default=1)
    parser.add_argument('--velocity-tol', type=float, default=1e-3)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f'Missing original CSV: {args.source}')
    if args.limit < 1 or args.steps < 1:
        parser.error('--limit and --steps must be positive')

    original = read_original(args.source)
    by_episode = {int(r['episode']): r for r in original}
    failed = [r for r in original if int(r['bad_complete_steps']) > 0]
    # Always include earliest known reproductions if present, then other failures.
    selected = []
    seen = set()
    for episode in (8, 18):
        r = by_episode.get(episode)
        if r and int(r['bad_complete_steps']) > 0:
            selected.append(r)
            seen.add(episode)
    for r in failed:
        episode = int(r['episode'])
        if episode not in seen:
            selected.append(r)
            seen.add(episode)
    if not args.all_failures:
        selected = selected[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    predictor = MarioMotionPredictor.from_object_tiles_csv(TILE_CSV)
    summary_path = args.out / 'replay_summary.csv'
    mismatch_path = args.out / 'replay_mismatches.csv'
    summary_fields = ['episode', 'env_seed', 'action_seed', 'mode',
                      'old_bad_complete_steps', 'new_bad_complete_steps',
                      'old_max_x_px', 'old_max_y_px',
                      'new_max_x_px', 'new_max_y_px',
                      'new_max_vx_error', 'new_max_vy_error',
                      'new_complete_steps', 'new_end_reason']
    mismatches = []
    successes = 0
    with summary_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for i, record in enumerate(selected, start=1):
            trial = Trial(int(record['episode']), int(record['env_seed']),
                          int(record['action_seed']), record['mode'])
            result, bad_rows = run_trial(trial, predictor, args.steps,
                                         args.threshold_px, args.velocity_tol)
            mismatches.extend(bad_rows)
            successes += int(result['bad_complete_steps'] == 0)
            writer.writerow({
                'episode': trial.episode,
                'env_seed': trial.env_seed,
                'action_seed': trial.action_seed,
                'mode': trial.mode,
                'old_bad_complete_steps': record['bad_complete_steps'],
                'new_bad_complete_steps': result['bad_complete_steps'],
                'old_max_x_px': record['max_open_x_error_px'],
                'old_max_y_px': record['max_open_y_error_px'],
                'new_max_x_px': result['max_open_x_error_px'],
                'new_max_y_px': result['max_open_y_error_px'],
                'new_max_vx_error': result['max_open_vx_error'],
                'new_max_vy_error': result['max_open_vy_error'],
                'new_complete_steps': result['complete_steps'],
                'new_end_reason': result['end_reason'],
            })
            f.flush()
            print(f'[{i}/{len(selected)}] episode={trial.episode} '
                  f'mode={trial.mode} bad_steps '
                  f'{record["bad_complete_steps"]} -> '
                  f'{result["bad_complete_steps"]}, '
                  f'max_xy=({result["max_open_x_error_px"]},'
                  f'{result["max_open_y_error_px"]})px', flush=True)

    if mismatches:
        with mismatch_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(mismatches[0]))
            writer.writeheader()
            writer.writerows(mismatches)
    else:
        # Keep header-only file to make a successful replay obvious.
        with mismatch_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'env_seed', 'action_seed', 'mode', 'step',
                'time_s', 'action_x', 'action_jump', 'actual_x', 'actual_y',
                'predicted_x', 'predicted_y', 'open_dx', 'open_dy',
                'open_dvx', 'open_dvy', 'grounded_match'])
            writer.writeheader()
    print(f'REPLAY SUMMARY: cleared={successes}/{len(selected)}; '
          f'remaining_bad_steps={len(mismatches)}')
    print('Saved:', summary_path, mismatch_path, sep='\n  ')


if __name__ == '__main__':
    main()
