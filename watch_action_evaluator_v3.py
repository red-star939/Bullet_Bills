"""Run ActionEvaluator v3 in the ORIGINAL Pygame game window, no PPO or .pt required."""
import argparse
import random
import time

import numpy as np

from models.action_evaluator_v3 import ActionEvaluatorV3
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--steps', type=int, default=18000)
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args()
    if args.fps <= 0 or args.steps <= 0 or args.episodes <= 0:
        parser.error('positive episodes, steps, fps required')
    import gymnasium as gym
    import kymnasium  # noqa: F401
    try:
        import pygame
    except ImportError:
        pygame = None
    env = gym.make('kymnasium/BulletBill-Continuous-Hard-Stage-2', render_mode='human', bgm=False)
    sim = ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv('map_data/object_tiles.csv'))
    controller = ActionEvaluatorV3(sim)
    rng = random.Random(args.seed)
    overrides = 0
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=rng.randrange(2**32))
            prev = None
            for step in range(args.steps):
                start = time.perf_counter()
                if pygame is not None:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            return
                decision = controller.evaluate(obs, current_action=prev)
                action = np.asarray(decision.best_action, dtype=np.float32)
                prev = decision.best_action
                overrides += int(decision.changed_from_baseline)
                obs, _, terminated, truncated, info = env.step(action)
                if decision.changed_from_baseline:
                    print('RECOVERY', 'episode', ep+1, 'step', step+1,
                          'reason', decision.change_reason, 'action', decision.best_action, flush=True)
                if terminated or truncated:
                    print('EPISODE', ep+1, 'time_s', info.get('time_elapsed'),
                          'state', info.get('game_state'), 'overrides', overrides, flush=True)
                    break
                delay = 1.0 / args.fps - (time.perf_counter() - start)
                if delay > 0:
                    time.sleep(delay)
            else:
                print('EPISODE', ep+1, 'STEP CAP', 'steps', args.steps,
                      'time_s', info.get('time_elapsed'), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
