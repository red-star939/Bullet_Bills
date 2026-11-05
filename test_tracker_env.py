
import numpy as np

from agent_env import create_env
from models.bullet_tracker import BulletTracker


def main():
    env = create_env(render_mode="none")

    tracker = BulletTracker(
        fps=60,
        frame_skip=4
    )

    try:
        obs, info = env.reset(seed=42)
        tracker.reset()

        # 포탄별 이전 위치와 관측 시간
        previous_positions = {}

        for step in range(1800):
            action = np.array(
                [0.0, 0.0],
                dtype=np.float32
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            current_time = info["time_elapsed"]

            # 포탄 추적기 업데이트
            bullets = tracker.update(obs)

            # 1. 새로운 포탄 탐지
            for bullet in bullets:
                if bullet.is_new:
                    print(
                        f"[NEW] ID={bullet.bullet_id} | "
                        f"Lane={bullet.lane} | "
                        f"Y={bullet.center_y:.1f} | "
                        f"X={bullet.center_x:.1f} | "
                        f"VX={bullet.vx:.1f} | "
                        f"Direction={bullet.direction}"
                    )

            # 2. 실제 이동 속도 측정
            for bullet in bullets:
                bullet_id = bullet.bullet_id

                if bullet_id in previous_positions:
                    prev_x, prev_time = (
                        previous_positions[bullet_id]
                    )

                    dt = current_time - prev_time

                    if dt > 0:
                        measured_vx = (
                            bullet.center_x - prev_x
                        ) / dt

                        if step % 30 == 0:
                            print(
                                f"[VELOCITY] ID={bullet_id} | "
                                f"Measured={measured_vx:.2f} | "
                                f"Reported={bullet.vx:.2f} | "
                                f"Error={measured_vx - bullet.vx:.2f}"
                            )

                previous_positions[bullet_id] = (
                    bullet.center_x,
                    current_time
                )

            # 3. 추적 종료된 포탄 처리
            for bullet_id in tracker.lost_ids:
                print(f"[LOST] ID={bullet_id}")

                previous_positions.pop(
                    bullet_id, None
                )

            # 4. 30스텝마다 추적 정보 출력
            if step % 30 == 0:
                print(
                    f"\nTime={current_time:.2f}s | "
                    f"Visible bullets={len(bullets)}"
                )

                for bullet in bullets:
                    print(
                        f"  [TRACK] ID={bullet.bullet_id} | "
                        f"X={bullet.center_x:.1f} | "
                        f"VX={bullet.vx:.1f} | "
                        f"EffectiveVX={bullet.effective_vx:.1f} | "
                        f"Lane={bullet.lane} | "
                        f"Age={bullet.age_frames}"
                    )

            # 5. 게임 종료 확인
            if terminated or truncated:
                print("\nEpisode finished:", info)
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()
