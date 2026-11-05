
import math
import numpy as np

from agent_env import create_env
from models.bullet_tracker import BulletTracker
from models.risk_estimator import (
    analyze_bullets,
    get_most_dangerous,
    get_total_risk,
)


def format_ttc(ttc):
    if ttc is None:
        return "N/A"

    return (
        f"{ttc:.3f}s"
        if math.isfinite(ttc)
        else "inf"
    )


def main():
    env = create_env(render_mode="none")

    tracker = BulletTracker(
        fps=60,
        frame_skip=4
    )

    try:
        obs, info = env.reset(seed=42)
        tracker.reset()

        for step in range(1800):
            # 마리오 정지: 충돌 예측 검증용
            action = np.array(
                [0.0, 0.0],
                dtype=np.float32
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            current_time = float(info["time_elapsed"])

            # 1. 포탄 추적
            bullets = tracker.update(
                obs,
                time_elapsed=current_time
            )

            # 2. 위험도 분석
            risks = analyze_bullets(
                obs,
                tracked_bullets=bullets
            )

            total_risk = get_total_risk(risks)
            dangerous = get_most_dangerous(risks)

            # 3. 신규 포탄의 최초 상태 출력
            for bullet in bullets:
                if not bullet.is_new:
                    continue

                print(
                    f"\n[INITIAL BULLET] "
                    f"ID={bullet.bullet_id} | "
                    f"Time={bullet.first_seen_time:.3f}s"
                )

                print(
                    f"  BulletX0={bullet.initial_bullet_x:.1f} | "
                    f"MarioX0={bullet.initial_mario_x:.1f}"
                )

                print(
                    f"  D0={bullet.initial_distance:.1f}px | "
                    f"CollisionGap0="
                    f"{bullet.initial_collision_gap:.1f}px"
                )

                print(
                    f"  RelativeVX0="
                    f"{bullet.initial_relative_vx:.1f}px/s | "
                    f"ClosingSpeed0="
                    f"{bullet.initial_closing_speed:.1f}px/s | "
                    f"TTC0={format_ttc(bullet.initial_ttc)}"
                )

            # 4. 5스텝마다 위험도 출력
            # 5 * (4 / 60) = 약 0.333초
            if step % 5 == 0 or terminated or truncated:
                mario = obs["mario"]

                print(
                    f"\nTime={current_time:.3f}s | "
                    f"MarioX={mario[0]:.0f}~{mario[2]:.0f} | "
                    f"TotalRisk={total_risk:.3f}"
                )

                for risk in risks:
                    print(
                        f"  ID={risk['bullet_id']} | "
                        f"Lane={risk['lane']} | "
                        f"Gap={risk['horizontal_gap']:.1f}px | "
                        f"Closing={risk['closing_speed']:.1f}px/s | "
                        f"TTC={format_ttc(risk['ttc'])} | "
                        f"Risk={risk['risk']:.3f} | "
                        f"Level={risk['risk_level']}"
                    )

                    # 최초 TTC와 현재 TTC 비교
                    if risk["initial_ttc"] is not None:
                        print(
                            f"    InitialTTC="
                            f"{format_ttc(risk['initial_ttc'])} | "
                            f"CurrentTTC="
                            f"{format_ttc(risk['ttc'])}"
                        )

                if dangerous is not None:
                    print(
                        f"  [DANGER] "
                        f"ID={dangerous['bullet_id']} | "
                        f"TTC={format_ttc(dangerous['ttc'])}"
                    )

            # 5. 게임 종료
            if terminated or truncated:
                print(
                    "\nEpisode finished:",
                    info
                )
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()
