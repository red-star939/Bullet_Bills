
from models.lane_manager import LaneManager


def main():
    manager = LaneManager()

    manager.print_lanes()

    print("\n===== POSITION TEST =====")

    positions = [0, 47, 48, 143, 600, 624, 863, 864]

    for y in positions:
        lane = manager.get_lane(y)
        print(f"Y={y:4d} -> Lane={lane}")

    print("\n===== OBJECT TEST =====")

    # 실제 초기 마리오 좌표
    mario_top = 720
    mario_bottom = 768

    print(
        "Mario center lane:",
        manager.get_center_lane(
            mario_top,
            mario_bottom
        )
    )

    print(
        "Mario occupied lanes:",
        manager.get_occupied_lanes(
            mario_top,
            mario_bottom
        )
    )

    # 포탄의 가상 바운딩 박스
    bullet_top = 700
    bullet_bottom = 748

    print(
        "Bullet center lane:",
        manager.get_center_lane(
            bullet_top,
            bullet_bottom
        )
    )

    print(
        "Bullet occupied lanes:",
        manager.get_occupied_lanes(
            bullet_top,
            bullet_bottom
        )
    )

    print(
        "Vertical collision possible:",
        manager.vertical_overlap(
            mario_top,
            mario_bottom,
            bullet_top,
            bullet_bottom
        )
    )

    # 화면 밖 객체 처리 테스트
    assert manager.get_lane(-1) is None
    assert manager.get_lane(864) is None

    assert manager.get_occupied_lanes(
        -20, 20
    ) == [0]

    assert manager.get_occupied_lanes(
        864, 912
    ) == []

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
