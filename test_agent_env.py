
import numpy as np
from agent_env import create_env


def main():
    env = create_env(render_mode="none")

    try:
        obs, info = env.reset(seed=42)

        print("=== Agent 환경 시작 ===")
        print("관측 공간:", env.observation_space)
        print("행동 공간:", env.action_space)

        for step in range(3000):
            # 임시 무작위 에이전트
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            if step % 60 == 0:
                print(
                    f"Step: {step} | "
                    f"Time: {info['time_elapsed']:.2f}s | "
                    f"State: {info['game_state']}"
                )

            if terminated or truncated:
                print("=== 에피소드 종료 ===")
                print("생존 시간:", info["time_elapsed"])
                print("종료 상태:", info["game_state"])
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()
