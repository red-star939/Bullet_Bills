"""Compare one-step Mario predictions with the real Bullet Bills environment.

Run in the project root after copying the predictor to models/:
    python test_mario_motion_env.py
"""
from pathlib import Path

import numpy as np

from agent_env import create_env
from models.mario_motion_predictor import MarioMotionPredictor, MarioState


def run_trial(label, action, steps=8):
    predictor = MarioMotionPredictor.from_object_tiles_csv(
        Path('map_data/object_tiles.csv')
    )
    env = create_env(render_mode='none')
    try:
        obs, info = env.reset(seed=42)
        print(f'\n[{label}] action={action}')
        print(' step | predicted x,y | actual x,y | dx,dy | predicted vx,vy | actual vx,vy')
        for step in range(1, steps + 1):
            # Reinitialize from actual observation on every step to isolate local model error.
            state = MarioState.from_observation(obs['mario'])
            pred, _ = predictor.step(state, action)
            obs, reward, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32)
            )
            m = obs['mario']
            dx = pred.left - int(m[0])
            dy = pred.top - int(m[1])
            print(
                f'{step:5d} | {pred.left:4d},{pred.top:4d} '
                f'| {m[0]:4.0f},{m[1]:4.0f} | {dx:+3d},{dy:+3d} '
                f'| {pred.vx:8.2f},{pred.vy:8.2f} '
                f'| {m[4]:8.2f},{m[5]:8.2f}'
            )
            if terminated or truncated:
                break
    finally:
        env.close()


def main():
    for label, action in [
        ('STILL', [0.0, 0.0]),
        ('RIGHT', [1.0, 0.0]),
        ('LEFT', [-1.0, 0.0]),
        ('JUMP', [0.0, 1.0]),
    ]:
        run_trial(label, action)


if __name__ == '__main__':
    main()
