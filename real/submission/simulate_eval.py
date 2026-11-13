"""콘솔/화면 없는 시뮬레이션 평가 스크립트 (다중 시드 성능 검증용)"""
import os
import sys
import gymnasium as gym
import numpy as np

submission_dir = os.path.dirname(os.path.abspath(__file__))
if submission_dir not in sys.path:
    sys.path.insert(0, submission_dir)

from agent import YourAgent

def evaluate_headless(seeds=(42, 100, 2026, 731209, 9999), max_steps=9000):
    model_path = os.path.join(submission_dir, 'best_model.pt')
    agent = YourAgent.load(model_path)
    env = gym.make('kymnasium/BulletBill-Continuous-Hard-Stage-2', render_mode='rgb_array')

    results = []
    print(f"=== YourAgent Multi-seed Evaluation ({len(seeds)} episodes) ===")

    for i, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        step = 0
        while step < max_steps:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if terminated or truncated:
                break

        t = float(info.get('time_elapsed', 0) or 0)
        state = info.get('game_state', 'unknown')
        print(f"[{i+1}/{len(seeds)}] Seed {seed:8d} | Survival Time: {t:6.2f}s | Steps: {step:5d} | State: {state}")
        results.append(t)

    env.close()
    mean_t = np.mean(results)
    max_t = np.max(results)
    min_t = np.min(results)
    print("=" * 60)
    print(f"Summary: Mean={mean_t:.2f}s | Min={min_t:.2f}s | Max={max_t:.2f}s")

if __name__ == '__main__':
    evaluate_headless()

