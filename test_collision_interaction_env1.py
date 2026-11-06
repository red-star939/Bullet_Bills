r"""Validate CollisionPredictor + InteractionSimulator against the real game.

Run from mario_pj project root:
  .\.venv\Scripts\python.exe test_collision_interaction_env.py --source logs/mario_motion_multiseed_v2/episode_summary.csv --episodes 4 --steps 140
  .\.venv\Scripts\python.exe test_collision_interaction_env.py --source logs/mario_motion_multiseed/episode_summary.csv --episodes 593 --steps 65

IMPORTANT: This is an *oracle-bullet* integration test: at the start of EACH
physics frame, bullets are copied from the real environment, including those
that spawned since the last frame. Mario's prediction is NEVER reset, so we can
measure post-contact drift. It is NOT a demonstration of future spawn prediction.
No game-physics methods or random streams are modified: the original internal
step is called exactly once per physics frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent_env import create_env
from models.mario_motion_predictor import MarioMotionPredictor, MarioState
from models.collision_predictor import BulletState
from models.interaction_simulator import InteractionSimulator
from test_mario_motion_multiseed import TILE_CSV, decide_action


def get_case(path: Path, episode: int) -> dict[str, str]:
    with path.open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            if int(row['episode']) == episode:
                return row
    raise ValueError(f'Episode {episode} not found in {path}')


def snapshot(core):
    obs = core._generate_obs()
    mario = np.asarray(obs['mario'], dtype=np.float64).copy()
    bullets = []
    for i, b in enumerate(obs['bullet_bills']):
        left, top, right, bottom, vx = map(float, b)
        if right > left and bottom > top:
            bullets.append(BulletState(i, tuple(map(lambda x: int(round(x)),
                                                       (left, top, right, bottom))), vx))
    return mario, bullets


def state_error(pred: MarioState, mario: np.ndarray):
    return dict(dx=pred.left - round(float(mario[0])),
                dy=pred.top - round(float(mario[1])),
                dvx=pred.vx - float(mario[4]),
                dvy=pred.vy - float(mario[5]),
                grounded_match=int(pred.grounded == (not bool(mario[7]))))


def write_rows(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


FIELDS = [
    'episode', 'env_seed', 'mode', 'frame', 'time_s', 'step', 'subframe',
    'action_x', 'action_jump', 'pre_mario_bbox', 'actual_mario_bbox',
    'pred_mario_bbox', 'before_vy', 'actual_vy', 'pred_vy',
    'dx', 'dy', 'dvx', 'dvy', 'grounded_match', 'actual_invincible',
    'pre_bullets', 'post_bullets', 'events', 'pred_stomp',
    'pred_unresolved', 'actual_bounce_witness', 'bullet_spawn_possible',
    'force_match', 'bad', 'game_state',
]
SUMMARY_FIELDS = [
    'episode', 'env_seed', 'action_seed', 'mode', 'frames', 'steps',
    'pred_stomps', 'actual_bounce_witnesses', 'unresolved_contacts',
    'first_pred_stomp_frame', 'first_actual_bounce_frame',
    'first_bad_frame', 'first_bad_dy', 'first_bad_dvy', 'bad_frames',
    'max_abs_dx', 'max_abs_dy', 'max_abs_dvx', 'max_abs_dvy',
    'end_reason', 'trace_file',
]


def run_case(record: dict[str, str], args, one_frame_motion: MarioMotionPredictor):
    episode = int(record['episode'])
    env_seed = int(record['env_seed'])
    action_seed = int(record['action_seed'])
    mode = record['mode']
    rng = random.Random(action_seed)
    control = {}
    sim = InteractionSimulator(one_frame_motion)
    env = create_env(render_mode='none')
    core = env.unwrapped
    if not hasattr(core, '_step_internal') or not hasattr(core, '_generate_obs'):
        env.close()
        raise RuntimeError('Installed environment lacks _step_internal or _generate_obs')
    obs, _ = env.reset(seed=env_seed)
    predicted = MarioState.from_observation(obs['mario'])
    original_step = core._step_internal
    rows = []
    current_action = np.array([0., 0.], dtype=np.float32)
    step_index = 0
    subframe = 0
    ending = 'max_steps'

    def observed_frame(force):
        nonlocal predicted, subframe
        before, before_bullets = snapshot(core)
        subframe += 1
        frame = len(rows) + 1
        expected_force = one_frame_motion.action_to_force(current_action)
        force_match = all(math.isclose(float(a), float(b), abs_tol=1e-6, rel_tol=0)
                          for a, b in zip(force, expected_force))
        if not force_match:
            raise AssertionError(f'Engine force differs at frame {frame}: '
                                 f'{force!r} vs {expected_force!r}')

        # The simulator runs precisely ONE physical frame; bullet positions
        # are fed from current real observation, never from future frames.
        result = sim.simulate(predicted, before_bullets, [current_action],
                              stop_on_unresolved=True)
        predicted = result.mario
        finished = original_step(force)  # exactly one real physics frame
        actual, after_bullets = snapshot(core)
        err = state_error(predicted, actual)
        events = [dict(type=e.event_type, id=e.bullet_id, bbox=e.bullet_bbox)
                  for e in result.events]
        stomp = any(e.event_type == 'stomp' for e in result.events)
        unresolved = any(e.event_type == 'unresolved_contact' for e in result.events)
        # Witness, not proof of stomp: other collisions could change vy.
        actual_bounce = (float(actual[5]) <= -499.9 and
                         float(before[5]) >= 0 and
                         float(before[5]) - float(actual[5]) > 100)
        bad = (abs(err['dx']) > args.px_tol or abs(err['dy']) > args.px_tol or
               abs(err['dvx']) > args.velocity_tol or
               abs(err['dvy']) > args.velocity_tol or not err['grounded_match'])
        rows.append(dict(
            episode=episode, env_seed=env_seed, mode=mode, frame=frame,
            time_s=round(frame / one_frame_motion.cfg.fps, 8),
            step=step_index, subframe=subframe,
            action_x=float(current_action[0]), action_jump=float(current_action[1]),
            pre_mario_bbox=json.dumps([int(round(v)) for v in before[:4]]),
            actual_mario_bbox=json.dumps([int(round(v)) for v in actual[:4]]),
            pred_mario_bbox=json.dumps(predicted.bbox),
            before_vy=float(before[5]), actual_vy=float(actual[5]),
            pred_vy=predicted.vy, **err,
            actual_invincible=int(bool(actual[6])),
            pre_bullets=len(before_bullets), post_bullets=len(after_bullets),
            events=json.dumps(events), pred_stomp=int(stomp),
            pred_unresolved=int(unresolved),
            actual_bounce_witness=int(actual_bounce),
            # A bullet can spawn during the frame. This is NOT a reliable
            # count-only detector if spawning and removals coincide.
            bullet_spawn_possible=int(len(after_bullets) > len(before_bullets)),
            force_match=int(force_match), bad=int(bool(bad)),
            game_state=str(getattr(core, '_game_state', '')),
        ))
        return finished

    try:
        core._step_internal = observed_frame
        for step_index in range(1, args.steps + 1):
            subframe = 0
            current_action = np.asarray(
                decide_action(mode, step_index-1, rng, control), dtype=np.float32)
            _, _, terminated, truncated, info = env.step(current_action)
            if terminated or truncated:
                ending = str(info.get('game_state', 'terminal'))
                break
    finally:
        core._step_internal = original_step
        env.close()

    out_path = args.out / f'episode_{episode}_{mode}_interaction_frames.csv'
    write_rows(out_path, rows, FIELDS)
    bad_rows = [r for r in rows if r['bad']]
    first_bad = bad_rows[0] if bad_rows else None
    actual_bounces = [r for r in rows if r['actual_bounce_witness']]
    predicted_stomps = [r for r in rows if r['pred_stomp']]
    unresolved_contacts = [r for r in rows if r['pred_unresolved']]
    summary = dict(
        episode=episode, env_seed=env_seed, action_seed=action_seed, mode=mode,
        frames=len(rows), steps=step_index,
        pred_stomps=len(predicted_stomps),
        actual_bounce_witnesses=len(actual_bounces),
        unresolved_contacts=len(unresolved_contacts),
        first_pred_stomp_frame=predicted_stomps[0]['frame'] if predicted_stomps else '',
        first_actual_bounce_frame=actual_bounces[0]['frame'] if actual_bounces else '',
        first_bad_frame=first_bad['frame'] if first_bad else '',
        first_bad_dy=first_bad['dy'] if first_bad else '',
        first_bad_dvy=first_bad['dvy'] if first_bad else '',
        bad_frames=len(bad_rows),
        max_abs_dx=max((abs(r['dx']) for r in rows), default=0),
        max_abs_dy=max((abs(r['dy']) for r in rows), default=0),
        max_abs_dvx=max((abs(r['dvx']) for r in rows), default=0),
        max_abs_dvy=max((abs(r['dvy']) for r in rows), default=0),
        end_reason=ending, trace_file=str(out_path),
    )
    interesting = sorted({int(r['frame']) for r in rows if
                          r['pred_stomp'] or r['actual_bounce_witness'] or r['pred_unresolved']})
    centers = interesting[:5] if interesting else ([int(first_bad['frame'])] if first_bad else [])
    window_rows = [r for r in rows if any(abs(int(r['frame']) - c) <= args.window
                                                for c in centers)]
    if window_rows:
        write_rows(args.out / f'episode_{episode}_{mode}_interaction_windows.csv',
                   window_rows, FIELDS)
    print(f'episode={episode} mode={mode} frames={len(rows)} '
          f'pred_stomps={len(predicted_stomps)} actual_bounces={len(actual_bounces)} '
          f'unresolved={len(unresolved_contacts)} bad_frames={len(bad_rows)}')
    print(f'  first stomp: predicted={summary["first_pred_stomp_frame"] or "-"} '
          f'actual={summary["first_actual_bounce_frame"] or "-"}; '
          f'first mismatch={summary["first_bad_frame"] or "-"}')
    if centers:
        for row in window_rows:
            if row['frame'] in interesting or (first_bad and row['frame'] == first_bad['frame']):
                print(f'  frame={row["frame"]} events={row["events"]} '
                      f'actual_vy={row["actual_vy"]:.3f} pred_vy={row["pred_vy"]:.3f} '
                      f'dy={row["dy"]:+d} dvy={row["dvy"]:+.3f}')
    print(f'  trace: {out_path}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='episode_summary.csv from the same run as the episode indices')
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=140)
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--px-tol', type=int, default=1)
    parser.add_argument('--velocity-tol', type=float, default=1e-3)
    parser.add_argument('--out', type=Path, default=Path('logs/collision_interaction_env'))
    args = parser.parse_args()
    if args.steps < 1 or args.window < 0 or args.px_tol < 0 or args.velocity_tol < 0:
        parser.error('steps>0, window>=0, px-tol>=0, velocity-tol>=0 required')
    base_motion = MarioMotionPredictor.from_object_tiles_csv(TILE_CSV)
    one_frame_motion = MarioMotionPredictor(
        base_motion.platforms, config=replace(base_motion.cfg, frame_skip=1))
    summaries = [run_case(get_case(args.source, ep), args, one_frame_motion)
                 for ep in args.episodes]
    path = args.out / 'interaction_summary.csv'
    write_rows(path, summaries, SUMMARY_FIELDS)
    print(f'Integration summary saved: {path}')
    print('NOTE: bullet observations are synchronized each physics frame; '
          'future unobserved bullet spawns are NOT predicted.')


if __name__ == '__main__':
    main()
