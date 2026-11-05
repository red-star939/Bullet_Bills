
import gymnasium as gym
import kymnasium
import numpy as np

ENV_ID = "kymnasium/BulletBill-Continuous-Hard-Stage-2"

def main():
    env = gym.make(
        id=ENV_ID,
        render_mode="human",
        bgm=False
    )

    try:
        obs, info = env.reset(seed=42)

        print("=== 환경 생성 성공 ===")
        print("Observation Space:", env.observation_space)
        print("Action Space:", env.action_space)

        for step in range(1000):
            # 이동 및 점프를 하지 않는 기본 행동
            action = np.array([0.0, 0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if step % 60 == 0:
                print(
                    f"Step={step}, "
                    f"Elapsed={info.get('time_elapsed')}, "
                    f"State={info.get('game_state')}"
                )

            if terminated or truncated:
                print("=== 게임 종료 ===")
                print(info)
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()
