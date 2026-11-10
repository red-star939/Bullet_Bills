"""Render a collaborator's PPO v10 checkpoint in the original BulletBill game.

Requires agent_ppo_v10.py and its original dependencies in the project root.
Example:
  .\\.venv\\Scripts\\python.exe watch_ppo_v10.py --model models/bullet_bill_ppo_v10_best.pt --episodes 5
"""
import argparse
from pathlib import Path
import time

import gymnasium as gym
import kymnasium  # noqa: F401  (registers the BulletBill environment)
import pygame

from agent_ppo_v10 import BulletBillAgent, ENV_ID, DT


def run(args):
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {model_path}")

    agent = BulletBillAgent.load(str(model_path))
    agent.deterministic = not args.stochastic
    env = gym.make(id=ENV_ID, render_mode="human", bgm=False)
    clock = pygame.time.Clock()

    print(f"모델: {model_path}")
    print(f"행동: {'확률적 샘플링' if args.stochastic else '결정론적 선택'} | 원본 게임 화면 | ESC / 창 닫기: 종료")
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep if args.seed is not None else None
            observation, info = env.reset(seed=seed)
            agent.fe.reset()  # stateful predictor: reset once per episode
            start = time.perf_counter()
            steps = 0
            stopped = False
            while steps < args.max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        stopped = True
                if stopped:
                    break
                action = agent.act(observation, info)
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
                if args.fps > 0:
                    clock.tick(args.fps)
            elapsed = float(info.get("time_elapsed", steps * DT))
            result = "user_quit" if stopped else ("game_over" if terminated else "truncated" if truncated else "step_limit")
            print(f"EPISODE {ep+1} seed={seed} survival={elapsed:.3f}s steps={steps} result={result} wall={time.perf_counter()-start:.1f}s", flush=True)
            if stopped:
                break
    finally:
        env.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Watch collaborator PPO v10 in original BulletBill Pygame renderer")
    parser.add_argument("--model", default="models/bullet_bill_ppo_v10_best.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15, help="target decisions per second; 0=unlimited")
    parser.add_argument("--max-steps", type=int, default=18000)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.fps < 0:
        parser.error("episodes/max-steps must be positive and fps must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
