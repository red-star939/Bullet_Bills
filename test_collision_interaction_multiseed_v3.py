"""Randomized, reproducible 60-FPS collision/interaction validation (v3).

Run at mario_pj root, alongside agent_env.py and test_mario_motion_multiseed.py.
The simulator sees REAL bullets at the START of each physics frame (oracle
input); it never sees future frames. New bullets spawned during a frame and
star pickups are censored/re-synchronized rather than scored as failures.

Examples:
  python test_collision_interaction_multiseed.py --episodes 10000 --steps 300
  python test_collision_interaction_multiseed.py --source logs/mario_motion_multiseed_v2/episode_summary.csv --episodes 4 593
  python test_collision_interaction_multiseed.py --forever --report-every 100

Collision audit samples actual pre-impact vy inside the engine call.
Does not change engine physics or implement future spawn prediction.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import secrets
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from agent_env import create_env
from models.mario_motion_predictor import MarioMotionPredictor, MarioState
from models.collision_predictor import BulletState
from models.interaction_simulator import InteractionSimulator
from models.collision_event_audit import capture_collision_witness, observed_collision_events
from test_mario_motion_multiseed import TILE_CSV, MODES, decide_action

SEED_MAX = 2**32
SUMMARY_COLUMNS = (
    'episode env_seed action_seed mode frames steps endpoint events_scored '
    'spawn_censored star_censored other_censored pred_stomps actual_stomps '
    'pred_invincible_hits actual_invincible_hits pred_deaths actual_deaths '
    'event_errors motion_errors resyncs first_error_frame max_dx max_dy '
    'max_dvx max_dvy'
).split()
EVENT_COLUMNS = (
    'episode env_seed action_seed mode frame step subframe kind reason '
    'pred_events actual_events pre_mario actual_mario pred_mario '
    'pre_bullets post_bullets dx dy dvx dvy pred_dead actual_dead'
).split()


@dataclass(frozen=True)
class Trial:
    episode: int
    env_seed: int
    action_seed: int
    mode: str


def record_trials(path: Path, episode_ids: list[int] | None):
    found = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            ep = int(row['episode'])
            if episode_ids is None or ep in episode_ids:
                found[ep] = Trial(ep, int(row['env_seed']),
                                  int(row['action_seed']), row['mode'])
    if episode_ids is not None:
        missing = set(episode_ids) - found.keys()
        if missing:
            raise ValueError(f'Missing episodes in {path}: {sorted(missing)}')
    return [found[k] for k in sorted(found)]


def mario_observation(core):
    # Use engine observation semantics for positions, velocities and flags.
    mario = core._mario
    r = mario.rect
    v = mario._velocity
    return np.array((r.left, r.top, r.right, r.bottom,
                     float(v.x), float(v.y), int(bool(mario.invincible_)),
                     int(not bool(mario._is_grounded))), dtype=np.float64)


def bullet_snapshot(core):
    """Capture object identity, state and LTRB bbox (NOT pygame Rect XYWH)."""
    return {
        id(b): (b, b.state_, (b.rect.left, b.rect.top, b.rect.right, b.rect.bottom), float(b._velocity))
        for b in core._bullet_bills
    }


def error(pred, actual):
    return (pred.left - int(actual[0]), pred.top - int(actual[1]),
            pred.vx - float(actual[4]), pred.vy - float(actual[5]))



def run_trial(trial: Trial, simulator: InteractionSimulator, args, writers):
    env = create_env(render_mode='none')
    core = env.unwrapped
    rng = random.Random(trial.action_seed)
    control = {}
    counts = {k: 0 for k in SUMMARY_COLUMNS if k not in (
        'episode', 'env_seed', 'action_seed', 'mode', 'endpoint',
        'first_error_frame')}
    counts['first_error_frame'] = ''
    for k in ('max_dx', 'max_dy', 'max_dvx', 'max_dvy'):
        counts[k] = 0.0
    counts['endpoint'] = 'max_steps'
    step_idx = 0
    subframe = 0
    current_action = np.zeros(2, dtype=np.float32)
    events_written = 0
    max_events = args.max_events_per_episode
    collision_witness = None
    try:
        obs, _ = env.reset(seed=trial.env_seed)
        pred = MarioState.from_observation(obs['mario'])
        original_step = core._step_internal
        original_collision = core._mario.check_bullet_collision

        def witnessed_collision(bullets):
            nonlocal collision_witness
            collision_witness = capture_collision_witness(core._mario, bullets)
            return original_collision(bullets)

        core._mario.check_bullet_collision = witnessed_collision

        def observed_frame(force):
            nonlocal pred, subframe, events_written, collision_witness
            collision_witness = None
            subframe += 1
            frame = counts['frames'] + 1
            counts['frames'] = frame
            before_m = mario_observation(core)
            before_b = bullet_snapshot(core)
            # Input to simulator contains only bullets known BEFORE frame tick.
            bullets = [BulletState(i, bbox, vx, state)
                       for i, (_obj, state, bbox, vx)
                       in enumerate(before_b.values())]
            force_expected = simulator.motion.action_to_force(current_action)
            if not all(math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-5)
                       for a, b in zip(force, force_expected)):
                raise RuntimeError(f'Action force mismatch frame={frame}: {force} vs {force_expected}')
            result = simulator.simulate(pred, bullets, [current_action])
            predicted = result.mario
            finished = original_step(force)
            after_m = mario_observation(core)
            after_b = bullet_snapshot(core)
            actual_alive = bool(core._mario.active_)
            game_state = str(core._game_state)
            # A brand-new bullet may have been generated AFTER the simulation
            # took its snapshot, or a star may have changed invincibility.
            new_bullets = set(after_b) - set(before_b)
            invincibility_changed = bool(after_m[6]) != bool(before_m[6])
            terminal_without_motion = (game_state == 'cleared' and
                                       all(abs(float(a-b)) < 1e-8 for a, b in
                                           zip(after_m[:6], before_m[:6])))
            reason = ''
            if new_bullets:
                reason = 'spawn'; counts['spawn_censored'] += 1
            elif invincibility_changed:
                reason = 'invincibility_change'; counts['star_censored'] += 1
            elif terminal_without_motion:
                reason = 'clear_before_physics'; counts['other_censored'] += 1
            else:
                actual = observed_collision_events(
                    collision_witness,
                    {ident: state for ident, (_obj, state, _bbox, _vx) in after_b.items()},
                    actual_alive, float(after_m[3]),
                    world_height=simulator.motion.cfg.world_height,
                )
                if any(label in actual for label in ('multiple_hit', 'uncertain_hit', 'missing_witness')):
                    reason = actual[0]; counts['other_censored'] += 1
            observed = observed_collision_events(
                collision_witness,
                {ident: state for ident, (_obj, state, _bbox, _vx) in after_b.items()},
                actual_alive, float(after_m[3]),
                world_height=simulator.motion.cfg.world_height,
            )
            predicted_events = [e.event_type for e in result.events
                                if e.event_type in ('stomp', 'invincible_hit', 'death')]
            # Death is one outcome per frame, irrespective of bullet count.
            if 'death' in predicted_events:
                predicted_events = ['death']
            if not reason:
                counts['events_scored'] += 1
                for key, flag in (
                    ('pred_stomps', 'stomp'), ('actual_stomps', 'stomp'),
                    ('pred_invincible_hits', 'invincible_hit'),
                    ('actual_invincible_hits', 'invincible_hit'),
                    ('pred_deaths', 'death'), ('actual_deaths', 'death'),
                ):
                    target = predicted_events if key.startswith('pred') else observed
                    counts[key] += int(flag in target)
            dx, dy, dvx, dvy = error(predicted, after_m)
            for key, value in zip(('max_dx','max_dy','max_dvx','max_dvy'),
                                  (dx,dy,dvx,dvy)):
                if not reason:
                    counts[key] = max(counts[key], abs(value))
            motion_bad = not reason and (
                abs(dx) > args.px_tol or abs(dy) > args.px_tol or
                abs(dvx) > args.velocity_tol or abs(dvy) > args.velocity_tol or
                bool(predicted.grounded) != bool(core._mario._is_grounded) or
                bool(predicted.invincible) != bool(after_m[6]))
            event_bad = not reason and (sorted(set(predicted_events)) != sorted(set(observed)))
            if motion_bad:
                counts['motion_errors'] += 1
            if event_bad:
                counts['event_errors'] += 1
            if (event_bad or motion_bad) and not counts['first_error_frame']:
                counts['first_error_frame'] = frame
            important = bool(event_bad or motion_bad or reason or observed or predicted_events)
            if important and events_written < max_events:
                writers[1].writerow(dict(
                    episode=trial.episode, env_seed=trial.env_seed,
                    action_seed=trial.action_seed, mode=trial.mode,
                    frame=frame, step=step_idx, subframe=subframe,
                    kind='mismatch' if (event_bad or motion_bad) else
                         ('censored' if reason else 'event'), reason=reason,
                    pred_events=';'.join(predicted_events),
                    actual_events=';'.join(observed),
                    pre_mario=str(tuple(before_m[:6])),
                    actual_mario=str(tuple(after_m[:6])),
                    pred_mario=str((predicted.bbox, predicted.vx, predicted.vy)),
                    pre_bullets=len(before_b), post_bullets=len(after_b),
                    dx=dx, dy=dy, dvx=dvx, dvy=dvy,
                    pred_dead=int(result.terminated), actual_dead=int(not actual_alive),
                ))
                events_written += 1
            # Exogenous transitions and local divergences are explicitly
            # re-anchored. This is event-by-event validation, NOT open-loop.
            if reason or motion_bad or event_bad:
                pred = MarioState.from_observation(after_m)
                counts['resyncs'] += 1
            else:
                pred = predicted
            return finished

        core._step_internal = observed_frame
        try:
            for step_idx in range(1, args.steps + 1):
                counts['steps'] = step_idx
                subframe = 0
                current_action = np.asarray(
                    decide_action(trial.mode, step_idx - 1, rng, control),
                    dtype=np.float32)
                _obs, _reward, terminated, truncated, info = env.step(current_action)
                if terminated or truncated:
                    counts['endpoint'] = str(info.get('game_state', 'terminated'))
                    break
        finally:
            core._step_internal = original_step
            core._mario.check_bullet_collision = original_collision
    finally:
        env.close()
    return dict(episode=trial.episode, env_seed=trial.env_seed,
                action_seed=trial.action_seed, mode=trial.mode, **counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    q = parser.add_mutually_exclusive_group()
    q.add_argument('--episodes', nargs='+', type=int,
                   help='With --source: episode indices; without --source: count (one number)')
    q.add_argument('--forever', action='store_true')
    parser.add_argument('--source', type=Path,
                        help='Replay original (env_seed, action_seed, mode) from summary CSV')
    parser.add_argument('--mismatches-from', type=Path,
                        help='Select all episode IDs with kind=mismatch from previous event_details.csv (requires --source)')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--report-every', type=int, default=100)
    parser.add_argument('--px-tol', type=int, default=1)
    parser.add_argument('--velocity-tol', type=float, default=1e-3)
    parser.add_argument('--max-events-per-episode', type=int, default=40)
    parser.add_argument('--rng-seed', type=int, default=None)
    parser.add_argument('--out', type=Path, default=Path('logs/collision_interaction_multiseed'))
    args = parser.parse_args()
    if args.steps < 1 or args.report_every < 1 or args.max_events_per_episode < 0:
        parser.error('steps and report-every must be positive, max-events must be >=0')
    if args.px_tol < 0 or args.velocity_tol < 0:
        parser.error('tolerances must be nonnegative')
    if args.mismatches_from:
        if not args.source or args.episodes is not None:
            parser.error('--mismatches-from requires --source, and cannot combine with --episodes')
        with args.mismatches_from.open(encoding='utf-8-sig', newline='') as f:
            args.episodes = sorted({int(r['episode']) for r in csv.DictReader(f)
                                    if r['kind'] == 'mismatch'})
        if not args.episodes:
            parser.error(f'No mismatch episodes in {args.mismatches_from}')
        print(f'Replaying {len(args.episodes)} prior mismatch episodes', flush=True)
    if args.source:
        if args.forever:
            parser.error('--source cannot be combined with --forever')
        trial_list = record_trials(args.source, args.episodes)
        if not trial_list:
            parser.error('No trials found in --source')
    else:
        if args.episodes and (len(args.episodes) != 1 or args.episodes[0] < 1):
            parser.error('Without --source, --episodes must be a single positive count')
        n = args.episodes[0] if args.episodes else 10000
        rng = random.Random(args.rng_seed) if args.rng_seed is not None else random.SystemRandom()
        def trials():
            ep = 0
            while args.forever or ep < n:
                ep += 1
                yield Trial(ep, rng.randrange(SEED_MAX), rng.randrange(SEED_MAX),
                            MODES[(ep - 1) % len(MODES)])
        trial_list = trials()
    if not TILE_CSV.exists():
        raise FileNotFoundError(f'Expected {TILE_CSV} in project root')
    base = MarioMotionPredictor.from_object_tiles_csv(TILE_CSV)
    sim = InteractionSimulator(MarioMotionPredictor(base.platforms,
          replace(base.cfg, frame_skip=1)))
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / 'episode_summary.csv'
    event_path = args.out / 'event_details.csv'
    totals = dict(episodes=0, frames=0, events_scored=0, event_errors=0,
                  motion_errors=0, spawn_censored=0, star_censored=0,
                  other_censored=0, pred_stomps=0, actual_stomps=0,
                  pred_deaths=0, actual_deaths=0)
    try:
        with summary_path.open('w', encoding='utf-8-sig', newline='') as sf, \
             event_path.open('w', encoding='utf-8-sig', newline='') as ef:
            sw = csv.DictWriter(sf, SUMMARY_COLUMNS, extrasaction='ignore')
            ew = csv.DictWriter(ef, EVENT_COLUMNS, extrasaction='ignore')
            sw.writeheader(); ew.writeheader()
            for trial in trial_list:
                row = run_trial(trial, sim, args, (sw, ew))
                sw.writerow(row)
                for k in totals:
                    totals[k] += row.get(k, 0) if k != 'episodes' else 1
                if totals['episodes'] % args.report_every == 0:
                    sf.flush(); ef.flush()
                    print('PROGRESS', totals, flush=True)
    except KeyboardInterrupt:
        print('Interrupted; completed rows are retained.', flush=True)
    print('FINAL SUMMARY:', totals)
    print(f'Saved: {summary_path} | {event_path}')
    print('NOTE: Oracle-fed bullet state at frame START; spawn/star frames censored,')
    print('      mismatches resynchronized; NOT open-loop future-spawn validation.')


if __name__ == '__main__':
    main()
