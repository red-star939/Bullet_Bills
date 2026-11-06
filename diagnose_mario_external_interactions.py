"""Frame-accurate *read-only* instrumentation of BulletBill Mario physics.

Run from mario_pj project root, with agent_env.py and models/ installed.

Examples:
  .\\.venv\\Scripts\\python.exe diagnose_mario_external_interactions.py --source logs/mario_motion_multiseed_v2/episode_summary.csv --episodes 4 --steps 140
  .\\.venv\\Scripts\\python.exe diagnose_mario_external_interactions.py --source logs/mario_motion_multiseed/episode_summary.csv --episodes 593 --steps 65

This wraps env.unwrapped._step_internal temporarily, collects the actual Mario
and BulletBill observations BEFORE and AFTER each physical frame, and advances
an independent MarioMotionPredictor in parallel. It does not replace, skip,
or change any physics frame, action, or random-number generator call.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np

from agent_env import create_env
from models.mario_motion_predictor import MarioMotionPredictor, MarioState
from test_mario_motion_multiseed import decide_action, TILE_CSV


def find_record(source: Path, episode: int) -> dict:
    with source.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if int(row['episode']) == episode:
                return row
    raise ValueError(f'Episode {episode} not found in {source}')


def capture(core) -> tuple[np.ndarray, list[tuple[float, float, float, float, float]]]:
    """Capture game's official observation without advancing game state.

    Observation bullet rows are anonymous (no persistent IDs); this is a
    *geometry* diagnostic, not a bullet identity tracker.
    """
    obs = core._generate_obs()
    mario = np.asarray(obs['mario'], dtype=np.float64).copy()
    bullets = []
    for entry in obs['bullet_bills']:
        values = tuple(float(v) for v in entry)
        if len(values) >= 5 and values[2] > values[0] and values[3] > values[1]:
            bullets.append(values[:5])
    return mario, bullets


def overlaps(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def horizontal_gap(a, b) -> float:
    return max(0.0, b[0] - a[2], a[0] - b[2])


def vertical_gap(a, b) -> float:
    return max(0.0, b[1] - a[3], a[1] - b[3])


def nearest_bullet(mario, bullets):
    if not bullets:
        return None, None, None, None
    a = mario[:4]
    b = min(bullets, key=lambda bb: math.hypot(horizontal_gap(a, bb), vertical_gap(a, bb)))
    return b, horizontal_gap(a, b), vertical_gap(a, b), overlaps(a, b)


def extract_frame(before, after, predicted, pre_bullets, post_bullets,
                  frame_index, action_step, subframe, force,
                  px_tol, velocity_tol, game_state):
    actual = after
    dx = int(predicted.left - int(round(float(actual[0]))))
    dy = int(predicted.top - int(round(float(actual[1]))))
    dvx = float(predicted.vx - actual[4])
    dvy = float(predicted.vy - actual[5])
    match_ground = bool(predicted.grounded == (not bool(actual[7])))
    bad = (abs(dx) > px_tol or abs(dy) > px_tol or
           abs(dvx) > velocity_tol or abs(dvy) > velocity_tol or not match_ground)
    closest, gap_x, gap_y, overlapping = nearest_bullet(after, post_bullets)
    before_closest, before_gap_x, before_gap_y, before_overlap = nearest_bullet(before, pre_bullets)
    # Direct near-contact witness, independent of array-row ordering.
    near_contact = (closest is not None and gap_x <= 5 and gap_y <= 5)
    pre_contact = (before_closest is not None and before_gap_x <= 5 and before_gap_y <= 5)
    # Jump/stomp-like change: actual vertical impulse differs from the free
    # motion model. Label as a *candidate*, not proven collision causation.
    actual_delta_vy = float(after[5] - before[5])
    jump_like = actual_delta_vy < -100 and float(before[5]) >= 0
    pbb = list(predicted.bbox)
    return {
        'frame': frame_index, 'time_s': round(frame_index/60, 8),
        'step': action_step, 'subframe': subframe,
        'action_force_x': force[0], 'action_force_y': force[1],
        'actual_bbox': json.dumps([int(round(x)) for x in actual[:4]]),
        'pred_bbox': json.dumps(pbb),
        'actual_vx': float(actual[4]), 'actual_vy': float(actual[5]),
        'before_vy': float(before[5]), 'pred_vx': predicted.vx, 'pred_vy': predicted.vy,
        'dx': dx, 'dy': dy, 'dvx': dvx, 'dvy': dvy,
        'actual_grounded': int(not bool(actual[7])),
        'pred_grounded': int(predicted.grounded),
        'invincible': int(bool(actual[6])),
        'bullets_before': len(pre_bullets), 'bullets_after': len(post_bullets),
        'nearest_bullet_bbox': json.dumps([round(x,3) for x in closest[:4]]) if closest else '',
        'nearest_bullet_vx': closest[4] if closest else '',
        'nearest_gap_x': gap_x if closest else '',
        'nearest_gap_y': gap_y if closest else '',
        'bbox_overlap': int(bool(overlapping)),
        'near_contact': int(bool(near_contact or pre_contact)),
        'before_overlap': int(bool(before_overlap)),
        'vertical_impulse_candidate': int(jump_like),
        'bad': int(bad), 'game_state': game_state,
    }


def diagnose(record, predictor, max_steps, px_tol, velocity_tol, window, output: Path):
    seed = int(record['env_seed'])
    action_seed = int(record['action_seed'])
    mode = record['mode']
    episode = int(record['episode'])
    rng = random.Random(action_seed)
    control = {}
    env = create_env(render_mode='none')
    core = env.unwrapped
    if not hasattr(core, '_step_internal') or not hasattr(core, '_generate_obs'):
        env.close()
        raise RuntimeError('Unsupported kymnasium version: _step_internal/_generate_obs missing')
    obs, info = env.reset(seed=seed)
    predicted = MarioState.from_observation(obs['mario'])
    original = core._step_internal
    trace = []
    step_counter = 0
    frame_counter = 0
    subframe = 0
    first_bad = None
    try:
        def instrumented(action_force):
            nonlocal predicted, frame_counter, subframe, first_bad
            before, pre_bullets = capture(core)
            # A predicted frame must see the SAME force as the real frame.
            # In particular, do not recompute from a normalized action here.
            expected = predictor._internal_step_force(predicted, action_force)
            finished = original(action_force)
            after, post_bullets = capture(core)
            frame_counter += 1
            subframe += 1
            row = extract_frame(before, after, expected, pre_bullets, post_bullets,
                                frame_counter, step_counter, subframe, action_force,
                                px_tol, velocity_tol, getattr(core, '_game_state', ''))
            trace.append(row)
            if row['bad'] and first_bad is None:
                first_bad = len(trace) - 1
            predicted = expected
            return finished

        core._step_internal = instrumented
        for step_counter in range(1, max_steps+1):
            subframe = 0
            action = np.asarray(decide_action(mode, step_counter-1, rng, control), dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        core._step_internal = original
        env.close()

    output.mkdir(parents=True, exist_ok=True)
    prefix = f'episode_{episode}_{mode}'
    all_path = output / f'{prefix}_frames.csv'
    with all_path.open('w', newline='', encoding='utf-8-sig') as f:
        fields = list(trace[0]) if trace else ['frame', 'step', 'bad']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trace)

    if first_bad is None:
        print(f'episode={episode} mode={mode}: no frame mismatch across {len(trace)} internal frames')
        return
    lo = max(0, first_bad-window)
    hi = min(len(trace), first_bad+window+1)
    nearby_path = output / f'{prefix}_first_mismatch_window.csv'
    with nearby_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace[lo:hi])

    key = trace[first_bad]
    print(f'episode={episode} mode={mode} seed={seed} action_seed={action_seed}')
    print(f'  FIRST BAD physical frame={key["frame"]}, env.step={key["step"]}, '
          f'subframe={key["subframe"]}, t={key["time_s"]:.6f}s')
    print(f'  position error(dx,dy)=({key["dx"]},{key["dy"]}) px; '
          f'velocity error(dvx,dvy)=({key["dvx"]:+.6f},{key["dvy"]:+.6f}) px/s')
    print(f'  Mario before vy={key["before_vy"]:.3f}, actual vy={key["actual_vy"]:.3f}, '
          f'predicted vy={key["pred_vy"]:.3f}')
    print(f'  bullets={key["bullets_after"]}, nearest={key["nearest_bullet_bbox"]}, '
          f'gap=({key["nearest_gap_x"]},{key["nearest_gap_y"]}), '
          f'contact={key["near_contact"]}, impulse_candidate={key["vertical_impulse_candidate"]}')
    print('  Nearby internal frames:')
    print('  frame step sub | dy    dvy       actual_vy   pred_vy  near  impulse')
    for r in trace[lo:hi]:
        print(f'  {r["frame"]:5d} {r["step"]:4d} {r["subframe"]:3d} | '
              f'{r["dy"]:+4d} {r["dvy"]:+9.3f} '
              f'{r["actual_vy"]:+10.3f} {r["pred_vy"]:+9.3f} '
              f'{r["near_contact"]:5d} {r["vertical_impulse_candidate"]:7d}')
    print(f'  Full trace: {all_path}')
    print(f'  First mismatch window: {nearby_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True,
                   help='episode_summary.csv from the relevant trial run')
    p.add_argument('--episodes', type=int, nargs='+', required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--window', type=int, default=12, help='frames before/after first mismatch')
    p.add_argument('--threshold-px', type=int, default=1)
    p.add_argument('--velocity-tol', type=float, default=1e-3)
    p.add_argument('--out', type=Path, default=Path('logs/mario_external_diagnosis'))
    a = p.parse_args()
    if a.steps < 1 or a.window < 0:
        p.error('--steps must be >=1 and --window >=0')
    predictor = MarioMotionPredictor.from_object_tiles_csv(TILE_CSV)
    for ep in a.episodes:
        rec = find_record(a.source, ep)
        diagnose(rec, predictor, a.steps, a.threshold_px, a.velocity_tol, a.window, a.out)


if __name__ == '__main__':
    main()
