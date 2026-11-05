
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Lane:
    index: int
    y_min: int
    y_max: int
    y_center: float


class LaneManager:
    """
    Bullet Bills 환경의 수평 레인 관리 클래스.

    좌표계:
        원점: 좌측 상단 (0, 0)
        Y축: 아래 방향이 양수

    레인:
        0 ~ 17
        각 레인 높이: 48px
    """

    def __init__(
        self,
        screen_height: int = 864,
        lane_height: int = 48
    ):
        if screen_height <= 0 or lane_height <= 0:
            raise ValueError("크기는 양수여야 합니다.")

        if screen_height % lane_height != 0:
            raise ValueError(
                "화면 높이는 레인 높이로 나누어떨어져야 합니다."
            )

        self.screen_height = screen_height
        self.lane_height = lane_height
        self.num_lanes = screen_height // lane_height

        self.lanes = [
            Lane(
                index=i,
                y_min=i * lane_height,
                y_max=(i + 1) * lane_height,
                y_center=(i + 0.5) * lane_height
            )
            for i in range(self.num_lanes)
        ]

    def get_lane(self, y: float) -> Optional[int]:
        """
        Y좌표가 속한 레인 번호를 반환합니다.

        화면 밖 좌표는 None을 반환합니다.
        """
        if not 0 <= y < self.screen_height:
            return None

        return int(y // self.lane_height)

    def get_occupied_lanes(
        self,
        top: float,
        bottom: float
    ) -> list[int]:
        """
        바운딩 박스가 실제로 점유하는 레인 목록.

        범위는 [top, bottom)으로 처리합니다.
        """
        if bottom <= top:
            return []

        # 화면 내부와 교차하는 구간만 사용
        clipped_top = max(0.0, top)
        clipped_bottom = min(
            float(self.screen_height),
            bottom
        )

        if clipped_bottom <= clipped_top:
            return []

        first = int(clipped_top // self.lane_height)

        # 끝 경계가 정확히 레인 경계에 있는 경우
        # 다음 레인을 잘못 포함하지 않도록 처리
        import math
        last = math.ceil(
            clipped_bottom / self.lane_height
        ) - 1

        return list(range(first, last + 1))

    def get_center_lane(
        self,
        top: float,
        bottom: float
    ) -> Optional[int]:
        """객체 중심 Y좌표의 대표 레인."""
        return self.get_lane((top + bottom) / 2)

    def get_lane_info(self, index: int) -> Lane:
        """레인 상세 정보."""
        if not 0 <= index < self.num_lanes:
            raise ValueError("존재하지 않는 레인입니다.")

        return self.lanes[index]

    def vertical_overlap(
        self,
        top_a: float,
        bottom_a: float,
        top_b: float,
        bottom_b: float
    ) -> bool:
        """두 객체의 Y축 충돌 범위 비교."""
        return top_a < bottom_b and bottom_a > top_b

    def print_lanes(self):
        print("===== LANE INFORMATION =====")

        for lane in self.lanes:
            print(
                f"Lane {lane.index:02d}: "
                f"Y=[{lane.y_min:3d}, {lane.y_max:3d}) "
                f"Center={lane.y_center:.1f}"
            )
