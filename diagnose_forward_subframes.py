"""Compare REAL internal Mario/Bullet states with ForwardSimulator at each subframe.

Replays the same seed/action stream as test_forward_simulator_env.py.
No physics mutation: temporarily wraps _step_internal solely to capture its outputs.
Run: .\\.venv\\Scripts\\python.exe diagnose_forward_subframes.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from agent_env import create_env
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor


def bbox(rect):
    return [int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)]


def snapshot(core):
    m = core._mario
    return {
        "time": float(core._time_elapsed),
        "game_state": str(core._game_state),
        "mario_bbox": bbox(m.rect),
        "mario_vx_internal": float(m._velocity.x),
        "mario_vy_internal": float(m._velocity.y),
        "mario_grounded": bool(m._is_grounded),
        "mario_active": bool(m.active_),
        "mario_invincible": bool(m.invincible_),
        "bullets": [
            {"index": i, "bbox": bbox(b.rect), "state": str(b.state_),
             "vx": float(b._velocity)}
            for i, b in enumerate(core._bullet_bills)
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target-seed', type=int, default=726388407)
    p.add_argument('--target-step', type=int, default=53)
    p.add_argument('--window', type=int, default=2)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--rng-seed', type=int, default=20261106)
    p.add_argument('--out', default='logs/forward_diagnosis/seed_726388407_subframes.json')
    args = p.parse_args()

    motion = MarioMotionPredictor.from_object_tiles_csv('map_data/object_tiles.csv')
    simulator = ForwardSimulator(motion)
    rng = random.Random(args.rng_seed)
    records = []
    found = False
    for episode in range(args.episodes):
        env = create_env(render_mode='none')
        try:
            seed = rng.randrange(2**32)
            obs, info = env.reset(seed=seed)
            if seed == args.target_seed:
                found = True
                print(f'TARGET episode={episode} seed={seed}')
            for step in range(1, args.steps + 1):
                action = np.array([rng.choice([-1., -.5, 0., .5, 1.]),
                                   rng.choice([0., 1.])], dtype=np.float32)
                is_target_window = (seed == args.target_seed and
                                    args.target_step - args.window <= step <= args.target_step + args.window)
                if is_target_window:
                    core = env.unwrapped
                    before = snapshot(core)
                    prediction = simulator.simulate_observation(obs, [action])
                    real_frames = []
                    original = core._step_internal

                    def traced_internal(force):
                        stopped = original(force)
                        s = snapshot(core)
                        s['ended_internal_step'] = bool(stopped)
                        real_frames.append(s)
                        return stopped

                    core._step_internal = traced_internal
                    try:
                        new_obs, _, terminated, truncated, new_info = env.step(action)
                    finally:
                        core._step_internal = original

                    pred_frames = []
                    for f in prediction.frames:
                        pred_frames.append({
                            'subframe': int(f.subframe),
                            'mario_bbox': list(f.mario_bbox),
                            'mario_vx': float(f.vx), 'mario_vy': float(f.vy),
                            'grounded': bool(f.grounded),
                            'events': [
                                {'kind': e.kind, 'bullet_id': e.bullet_id,
                                 'bullet_bbox': list(e.bullet_bbox) if e.bullet_bbox else None}
                                for e in f.events
                            ],
                            'bullets': [
                                {'id': b.bullet_id, 'bbox': list(b.bbox), 'state': b.state}
                                for b in f.bullets
                            ],
                        })
                    record = {'step': step, 'action': action.tolist(),
                              'before': before, 'real_frames': real_frames,
                              'pred_frames': pred_frames,
                              'actual_terminated': bool(terminated),
                              'predicted_terminated': bool(prediction.terminated),
                              'predicted_reason': prediction.termination_reason}
                    records.append(record)
                    print(f'\nSTEP {step}, action={action.tolist()}')
                    print('sub | real (x,y,bottom) / pred (x,y,bottom) | real vy_internal / pred vy | events')
                    for idx in range(max(len(real_frames), len(pred_frames))):
                        r = real_frames[idx] if idx < len(real_frames) else None
                        f = pred_frames[idx] if idx < len(pred_frames) else None
                        rb = r['mario_bbox'] if r else None
                        pb = f['mario_bbox'] if f else None
                        print(f'{idx + 1:>3} | '
                              f'{(rb[0],rb[1],rb[3]) if rb else None} / '
                              f'{(pb[0],pb[1],pb[3]) if pb else None} | '
                              f'{r["mario_vy_internal"] if r else None} / '
                              f'{f["mario_vy"] if f else None} | '
                              f'{f["events"] if f else None}')
                else:
                    new_obs, _, terminated, truncated, new_info = env.step(action)
                obs, info = new_obs, new_info
                if terminated or truncated:
                    break
            if found:
                break
        finally:
            env.close()
    if not found:
        raise SystemExit('Seed not found: check RNG seed, episodes and steps.')
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSaved:', output)


if __name__ == '__main__':
    main()
