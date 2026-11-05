
import math

import numpy as np

from models.bullet_tracker import BulletTracker
from models.risk_estimator import analyze_bullets


def make_obs(bullet_box, mario_vx=0.0):
    mario = np.array(
        [624, 720, 672, 768, mario_vx, 0, 0, 0],
        dtype=np.float32
    )

    bullets = np.zeros((30, 5), dtype=np.float32)
    bullets[0] = bullet_box

    return {
        "mario": mario,
        "bullet_bills": bullets,
    }


def main():
    tracker = BulletTracker(
        fps=60,
        frame_skip=4
    )

    # 초기 관측: 왼쪽에서 접근하는 포탄
    obs1 = make_obs(
        [384, 720, 432, 768, 285.86]
    )

    bullets1 = tracker.update(
        obs1,
        time_elapsed=2.0
    )

    bullet1 = bullets1[0]

    assert bullet1.is_new
    assert bullet1.first_seen_time == 2.0
    assert bullet1.initial_bullet_x == 408.0
    assert bullet1.initial_mario_x == 648.0

    # 중심 사이 거리: 648 - 408 = 240
    assert bullet1.initial_distance == 240.0

    # 경계 사이 거리: 624 - 432 = 192
    assert bullet1.initial_collision_gap == 192.0

    # effective_vx = int(285.86 / 60) * 60 = 240
    assert bullet1.effective_vx == 240.0
    assert bullet1.initial_relative_vx == 240.0
    assert bullet1.initial_closing_speed == 240.0

    # 초기 TTC = 192 / 240 = 0.8초
    assert math.isclose(bullet1.initial_ttc, 0.8)

    print("[INITIAL]")
    print(
        f"D0={bullet1.initial_distance:.1f}px | "
        f"Gap0={bullet1.initial_collision_gap:.1f}px | "
        f"RelativeVX={bullet1.initial_relative_vx:.1f}px/s | "
        f"TTC0={bullet1.initial_ttc:.3f}s"
    )

    # 다음 관측: 1 step = 4/60초
    # 포탄은 이 시간 동안 16px 이동
    obs2 = make_obs(
        [400, 720, 448, 768, 285.86]
    )

    bullets2 = tracker.update(
        obs2,
        time_elapsed=2.0 + 4.0 / 60.0
    )

    bullet2 = bullets2[0]

    # 동일한 포탄 ID 유지
    assert bullet2.bullet_id == bullet1.bullet_id
    assert not bullet2.is_new

    # 최초 상태는 수정되지 않아야 함
    assert bullet2.initial_distance == 240.0
    assert bullet2.initial_collision_gap == 192.0
    assert math.isclose(bullet2.initial_ttc, 0.8)

    # 현재 TTC = (624 - 448) / 240
    risks = analyze_bullets(obs2, tracked_bullets=bullets2)
    risk = risks[0]

    assert math.isclose(risk["ttc"], 176.0 / 240.0)
    assert math.isclose(risk["initial_ttc"], 0.8)

    print("\n[CURRENT]")
    print(
        f"InitialTTC={risk['initial_ttc']:.3f}s | "
        f"CurrentTTC={risk['ttc']:.3f}s"
    )

    # 오른쪽에서 접근하는 경우도 검증
    tracker.reset()

    obs_right = make_obs(
        [864, 720, 912, 768, -285.86]
    )

    right = tracker.update(
        obs_right,
        time_elapsed=1.0
    )[0]

    assert right.initial_distance == 240.0
    assert right.initial_collision_gap == 192.0
    assert right.initial_relative_vx == -240.0
    assert right.initial_closing_speed == 240.0
    assert math.isclose(right.initial_ttc, 0.8)

    print("\n[RIGHT]")
    print(
        f"RelativeVX={right.initial_relative_vx:.1f} | "
        f"TTC0={right.initial_ttc:.3f}s"
    )

    print("\nAll initial range tests passed!")


if __name__ == "__main__":
    main()
