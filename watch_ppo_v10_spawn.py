"""Watch original PPO v10 checkpoint with optional our spawn-aware safety shield.

Original Pygame renderer; no re-training; original 148-dimensional v10 feature
extractor and continuous actions are unchanged. Requires agent_ppo_v10.py.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import gymnasium as gym
import kymnasium  # registers environment
import numpy as np
import pygame
import torch
import agent_ppo_v10 as v10
from models.v10_spawn_guidance import V10SpawnGuidance


def run(args):
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    torch.set_num_threads(args.torch_threads)
    agent = v10.BulletBillAgent.load(args.model)
    agent.deterministic = not args.stochastic
    guide = (None if args.guidance == 'ppo' else V10SpawnGuidance(
        map_path=args.map, caution_px=args.caution_px, critical_px=args.critical_px,
        safety_horizon_steps=args.safety_horizon_steps,
        spawn_horizon_steps=args.spawn_horizon_steps))
    env = gym.make(id=v10.ENV_ID, render_mode='human', bgm=False)
    clock = pygame.time.Clock()
    print(f'MODEL {args.model} | mode={args.guidance} | numba={v10.HAS_NUMBA} | '
          f'fps={args.fps} | deterministic={not args.stochastic}', flush=True)
    if not v10.HAS_NUMBA:
        print('WARNING: numba missing; v10 feature extraction may be very slow.', flush=True)
    quit_requested = False
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep if args.seed is not None else None
            obs, info = env.reset(seed=seed)
            agent.fe.reset()
            previous_action = None
            overrides = spawn_overrides = known_overrides = wall_steps = 0
            agent_ms = guide_ms = 0.0
            recorded = 0
            start = time.perf_counter()
            terminated = truncated = False
            steps = 0
            for step in range(args.steps):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        quit_requested = True
                if quit_requested:
                    break
                t0 = time.perf_counter()
                original_action = agent.act(obs, info)
                t1 = time.perf_counter()
                if guide is None:
                    action = np.asarray(original_action, dtype=np.float32)
                    meta = {'overridden': False, 'spawn_overridden': False,
                            'known_world_overridden': False}
                else:
                    action, meta = guide.decide(obs, original_action, previous_action=previous_action,
                                                mode=args.guidance)
                t2 = time.perf_counter()
                if step > 0:  # avoid initial numba JIT compilation in profiling
                    agent_ms += (t1 - t0) * 1000
                    guide_ms += (t2 - t1) * 1000
                    recorded += 1
                overrides += int(meta['overridden'])
                spawn_overrides += int(meta['spawn_overridden'])
                known_overrides += int(meta['known_world_overridden'])
                if guide is not None and 'wall_clearance_px' in meta:
                    wall_steps += int(meta['wall_clearance_px'] < args.caution_px)
                previous_action = tuple(float(x) for x in action)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if args.profile_every and steps % args.profile_every == 0:
                    print(f'PERF ep={ep+1} steps={steps} '
                          f'v10_act_ms={agent_ms/max(1,recorded):.1f} '
                          f'guidance_ms={guide_ms/max(1,recorded):.1f} '
                          f'overrides={overrides} spawn_overrides={spawn_overrides}', flush=True)
                if args.fps > 0:
                    clock.tick(args.fps)
                if terminated or truncated:
                    break
            elapsed = float(info.get('time_elapsed', steps * v10.DT))
            status = 'user_quit' if quit_requested else 'terminated' if terminated else 'truncated' if truncated else 'step_limit'
            print(f'EPISODE {ep+1} seed={seed} survival={elapsed:.3f}s steps={steps} '
                  f'overrides={overrides} spawn_overrides={spawn_overrides} '
                  f'known_overrides={known_overrides} wall_steps={wall_steps} '
                  f'v10_act_ms={agent_ms/max(1,recorded):.1f} '
                  f'guidance_ms={guide_ms/max(1,recorded):.1f} '
                  f'result={status} wall={time.perf_counter()-start:.1f}s', flush=True)
            if quit_requested:
                break
    finally:
        env.close()
        pygame.quit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='models/bullet_bill_ppo_v10_best.pt')
    p.add_argument('--guidance', choices=['ppo', 'shield', 'spawn_shield', 'teacher'], default='spawn_shield')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--steps', type=int, default=900)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--fps', type=float, default=0, help='0=no extra FPS wait; 15=real-time target')
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--profile-every', type=int, default=100)
    p.add_argument('--map', default='map_data/object_tiles.csv')
    p.add_argument('--caution-px', type=int, default=180)
    p.add_argument('--critical-px', type=int, default=96)
    p.add_argument('--safety-horizon-steps', type=int, default=12)
    p.add_argument('--spawn-horizon-steps', type=int, default=12)
    p.add_argument('--stochastic', action='store_true')
    a = p.parse_args()
    if a.episodes < 1 or a.steps < 1 or a.fps < 0 or a.torch_threads < 1:
        p.error('Invalid episodes, steps, fps or torch threads')
    run(a)

if __name__ == '__main__':
    main()
