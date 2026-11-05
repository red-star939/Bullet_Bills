
import numpy as np

from models.bullet_tracker import BulletTracker


def make_observation(*bullets):
    array = np.zeros((30, 5), dtype=np.float32)

    for i, bullet in enumerate(bullets):
        array[i] = bullet

    return {"bullet_bills": array}


def main():
    tracker = BulletTracker(
        fps=60,
        frame_skip=1
    )

    # Frame 1: 양쪽에서 두 개의 포탄 탐지
    obs1 = make_observation(
        [-48, 600, 0, 648, 240],
        [1296, 700, 1344, 748, -270],
    )

    result1 = tracker.update(obs1)

    assert len(result1) == 2
    assert tracker.new_ids == [1, 2]

    print("Frame 1")
    for bullet in result1:
        print(bullet.to_dict())

    # Frame 2: 포탄 이동, 배열 순서 변경
    obs2 = make_observation(
        [1291, 700, 1339, 748, -270],
        [-44, 600, 4, 648, 240],
    )

    result2 = tracker.update(obs2)

    assert len(result2) == 2
    assert tracker.new_ids == []
    assert [b.bullet_id for b in result2] == [1, 2]

    assert result2[0].center_x == -20.0
    assert result2[1].center_x == 1315.0

    print("\nFrame 2")
    for bullet in result2:
        print(
            f"ID={bullet.bullet_id} "
            f"X={bullet.center_x:.1f} "
            f"Lane={bullet.lane} "
            f"Direction={bullet.direction}"
        )

    # Frame 3: 새로운 포탄 추가
    obs3 = make_observation(
        [-40, 600, 8, 648, 240],
        [1286, 700, 1334, 748, -270],
        [-48, 300, 0, 348, 220],
    )

    result3 = tracker.update(obs3)

    assert len(result3) == 3
    assert tracker.new_ids == [3]

    print("\nFrame 3")
    print("New IDs:", tracker.new_ids)

    # Frame 4~6: 모든 포탄이 사라짐
    empty = make_observation()

    for _ in range(3):
        tracker.update(empty)

    assert len(tracker.tracks) == 0
    assert tracker.lost_ids == [1, 2, 3]

    print("\nLost IDs:", tracker.lost_ids)
    print("\nAll BulletTracker tests passed!")


if __name__ == "__main__":
    main()
