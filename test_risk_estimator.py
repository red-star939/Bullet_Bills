
import math

import numpy as np

from models.bullet_tracker import BulletTracker
from models.risk_estimator import RiskEstimator


def make_observation(
    bullet_rows,
    mario_vx=0.0,
    mario_top=720.0,
    mario_bottom=768.0,
    invincible=False,
):
    mario = np.array(
        [
            624, mario_top,
            672, mario_bottom,
            mario_vx, 0,
            float(invincible), 0
        ],
        dtype=np.float32
    )

    bullets = np.zeros((30, 5), dtype=np.float32)

    for i, row in enumerate(bullet_rows):
        bullets[i] = row

    return {
        "mario": mario,
        "bullet_bills": bullets,
    }


def estimate(observation):
    tracker = BulletTracker(fps=60, frame_skip=4)
    estimator = RiskEstimator(tau=1.0)

    tracked = tracker.update(observation)
    return estimator.analyze(observation, tracked)


def main():
    # 1. 왼쪽에서 오른쪽으로 접근
    obs = make_observation([
        [384, 720, 432, 768, 285.86]
    ])

    result = estimate(obs)[0]

    assert result.vertical_overlap
    assert result.horizontal_gap == 192.0
    assert result.closing_speed == 240.0
    assert math.isclose(result.ttc, 0.8)
    assert math.isclose(
        result.risk_score,
        math.exp(-0.8)
    )

    print(
        f"[LEFT] TTC={result.ttc:.3f}s "
        f"Risk={result.risk_score:.3f}"
    )

    # 2. 오른쪽에서 왼쪽으로 접근
    obs = make_observation([
        [864, 720, 912, 768, -285.86]
    ])

    result = estimate(obs)[0]

    assert math.isclose(result.ttc, 0.8)

    print(
        f"[RIGHT] TTC={result.ttc:.3f}s"
    )

    # 3. 마리오에게서 멀어지는 포탄
    obs = make_observation([
        [384, 720, 432, 768, -285.86]
    ])

    result = estimate(obs)[0]

    assert math.isinf(result.ttc)
    assert result.risk_score == 0.0

    print("[AWAY] TTC=inf Risk=0")

    # 4. Y축 범위가 겹치지 않는 포탄
    obs = make_observation([
        [384, 600, 432, 648, 285.86]
    ])

    result = estimate(obs)[0]

    assert not result.vertical_overlap
    assert math.isinf(result.ttc)
    assert math.isclose(
        result.horizontal_ttc, 0.8
    )

    print("[VERTICAL] No current Y overlap")

    # 5. 무적 상태
    obs = make_observation(
        [[384, 720, 432, 768, 285.86]],
        invincible=True
    )

    result = estimate(obs)[0]

    assert math.isclose(result.ttc, 0.8)
    assert result.risk_score == 0.0

    print("[INVINCIBLE] Risk=0")

    # 6. 이미 X축과 Y축에서 겹쳐 있음
    obs = make_observation([
        [600, 720, 648, 768, 285.86]
    ])

    result = estimate(obs)[0]

    assert result.ttc == 0.0
    assert result.risk_score == 1.0

    print("[OVERLAP] TTC=0 Risk=1")

    # 7. 다중 포탄 중 가장 위험한 포탄
    obs = make_observation([
        [384, 720, 432, 768, 285.86],
        [504, 720, 552, 768, 285.86],
    ])

    results = estimate(obs)

    estimator = RiskEstimator()
    dangerous = estimator.get_most_dangerous(results)

    assert dangerous is not None
    assert math.isclose(dangerous.ttc, 0.3)
    assert dangerous.bullet_id == 2

    print(
        f"[MULTI] Most dangerous ID="
        f"{dangerous.bullet_id}"
    )

    print("\nAll RiskEstimator tests passed!")


if __name__ == "__main__":
    main()
