"""Reproduce the exact RNG stream used by test_forward_simulator_env.py.

Usage:
  .\\.venv\\Scripts\\python.exe diagnose_forward_terminal.py --target-seed 726388407 --target-step 53

This script DOES NOT change environment or simulator physics.
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


def mario_dict(state):
    return {
        'bbox': list(state.bbox), 'bottom': int(state.bottom),
        'vx': float(state.vx), 'vy': float(state.vy),
        'grounded': bool(state.grounded), 'invincible': bool(state.invincible),
    }


def obs_dict(obs):
    m = obs['mario']
    return {
        'bbox': [float(v) for v in m[:4]],
        'bottom': float(m[3]), 'vx': float(m[4]),
        'vy': float(m[5]), 'invincible': bool(m[6]),
        'jumping': bool(m[7]),
    }


def run(args):
    motion = MarioMotionPredictor.from_object_tiles_csv(Path('map_data/object_tiles.csv'))
    forward = ForwardSimulator(motion)
    rng = random.Random(args.seed)
    out = []
    target_seen = False

    for ep in range(args.episodes):
        env = create_env(render_mode='none')
        try:
            env_seed = rng.randrange(2**32)
            obs, info = env.reset(seed=env_seed)
            track = env_seed == args.target_seed
            if track:
                target_seen = True
                print(f'TARGET episode_index={ep} env_seed={env_seed}')
            for step_idx in range(1, args.steps + 1):
                x = rng.choice([-1., -.5, 0., .5, 1.])
                jump = rng.choice([0., 1.])
                action = np.array([x, jump], dtype=np.float32)
                pred = forward.simulate_observation(obs, [action])
                before_time = float(info['time_elapsed'])
                next_obs, _, terminated, truncated, next_info = env.step(action)
                after_time = float(next_info['time_elapsed'])
                if track and step_idx >= args.target_step - args.window and step_idx <= args.target_step + args.window:
                    actual = obs_dict(next_obs)
                    predicted = mario_dict(pred.mario)
                    actual_bottom = actual['bottom']
                    predicted_bottom = predicted['bottom']
                    frames = [
                        {
                            'subframe': f.subframe, 'mario_bbox': list(f.mario_bbox),
                            'vx': float(f.vx), 'vy': float(f.vy),
                            'grounded': bool(f.grounded),
                            'events': [e.kind for e in f.events],
                        } for f in pred.frames
                    ]
                    entry = {
                        'episode_index': ep, 'env_seed': env_seed, 'step': step_idx,
                        'action': action.tolist(), 'time_before': before_time,
                        'time_after': after_time, 'dt': after_time - before_time,
                        'actual': actual, 'predicted': predicted,
                        'dy': predicted['bbox'][1] - actual['bbox'][1],
                        'actual_terminated': bool(terminated),
                        'actual_truncated': bool(truncated),
                        'actual_game_state': str(next_info.get('game_state')),
                        'predicted_terminated': bool(pred.terminated),
                        'predicted_reason': pred.termination_reason,
                        'predicted_events': [
                            {'frame': e.frame, 'kind': e.kind, 'bullet_id': e.bullet_id}
                            for e in pred.events
                        ],
                        'actual_bottom': actual_bottom,
                        'predicted_bottom': predicted_bottom,
                        'world_height': motion.cfg.world_height,
                        'actual_fall_condition': actual_bottom > motion.cfg.world_height,
                        'predicted_fall_condition': predicted_bottom > motion.cfg.world_height,
                        'bullet_count_before': sum(float(r[2]) > float(r[0]) for r in obs['bullet_bills']),
                        'bullet_count_after': sum(float(r[2]) > float(r[0]) for r in next_obs['bullet_bills']),
                        'assumptions': list(pred.assumptions), 'predicted_frames': frames,
                    }
                    out.append(entry)
                    print(f"step={step_idx:3d} action={entry['action']} "
                          f"actual={actual['bbox']} pred={predicted['bbox']} dy={entry['dy']:+.0f} "
                          f"actual_term={terminated} game_state={entry['actual_game_state']} "
                          f"pred_term={pred.terminated} pred_reason={pred.termination_reason} "
                          f"actual_bottom={actual_bottom} pred_bottom={predicted_bottom} "
                          f"height={motion.cfg.world_height}")
                    if step_idx == args.target_step:
                        print('target predicted events:', entry['predicted_events'])
                        print('target subframes:', frames)
                obs, info = next_obs, next_info
                if terminated or truncated:
                    break
            if track:
                break
        finally:
            env.close()

    if not target_seen:
        raise SystemExit('Target seed not encountered. Use the same --seed/--episodes/--steps as the original smoke test.')
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--target-seed', type=int, default=726388407)
    p.add_argument('--target-step', type=int, default=53)
    p.add_argument('--seed', type=int, default=20261106, help='Same RNG seed as original smoke test')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--out', default='logs/forward_diagnosis/seed_726388407_step_53.json')
    run(p.parse_args())
