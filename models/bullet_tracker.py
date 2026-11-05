
from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
from typing import Optional
from .lane_manager import LaneManager


@dataclass
class TrackedBullet:
    bullet_id: int

    # 포탄 Bounding Box
    left: float
    top: float
    right: float
    bottom: float

    # 포탄 수평 속도
    vx: float

    # 레인 정보
    lane: Optional[int]
    occupied_lanes: list[int]

    # 추적 상태
    first_seen_frame: int
    age_frames: int = 1
    missed_frames: int = 0
    is_new: bool = True

    # 물리 시뮬레이션 FPS
    fps: int = 60

    # 최초 관측 시점 (게임 시간)
    first_seen_time: Optional[float] = None

    # 최초 관측된 포탄 및 마리오 중심 좌표
    initial_bullet_x: Optional[float] = None
    initial_mario_x: Optional[float] = None

    # 최초 상대 위치 및 속도
    initial_distance: Optional[float] = None
    initial_collision_gap: Optional[float] = None
    initial_relative_vx: Optional[float] = None
    initial_closing_speed: Optional[float] = None

    # 최초 충돌 예측
    initial_vertical_overlap: Optional[bool] = None
    initial_ttc: Optional[float] = None


    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def direction(self) -> str:
        if self.vx > 0:
            return "right"
        if self.vx < 0:
            return "left"
        return "stationary"
    
    @property
    def effective_vx(self) -> float:
        return float(
            math.trunc(self.vx / self.fps) * self.fps
        )

    @property
    def bbox(self) -> tuple:
        return (
            self.left,
            self.top,
            self.right,
            self.bottom
        )

    def to_dict(self) -> dict:
        return {
            "bullet_id": self.bullet_id,
            "bbox": list(self.bbox),
            "center_x": self.center_x,
            "center_y": self.center_y,
            "vx": self.vx,
            "direction": self.direction,
            "lane": self.lane,
            "occupied_lanes": self.occupied_lanes,
            "first_seen_frame": self.first_seen_frame,
            "age_frames": self.age_frames,
            "missed_frames": self.missed_frames,
            "is_new": self.is_new,
            "effective_vx": self.effective_vx,
            "first_seen_time": self.first_seen_time,
            "initial_bullet_x": self.initial_bullet_x,
            "initial_mario_x": self.initial_mario_x,
            "initial_distance": self.initial_distance,
            "initial_collision_gap": self.initial_collision_gap,
            "initial_relative_vx": self.initial_relative_vx,
            "initial_closing_speed": self.initial_closing_speed,
            "initial_vertical_overlap": self.initial_vertical_overlap,
            "initial_ttc": self.initial_ttc,
        }


class BulletTracker:
    """
    Bullet Bills 다중 포탄 추적기.

    기능:
        1. 신규 포탄 탐지
        2. 포탄별 고유 ID 발급
        3. 위치 및 속도 추적
        4. 레인 정보 갱신
        5. 사라진 포탄 추적 종료

    주의:
        입력은 observation["bullet_bills"] 사용.
        한 번의 update()는 게임 1프레임에 대응.
    """

    def __init__(
        self,
        lane_manager: Optional[LaneManager] = None,
        fps: int = 60,
        frame_skip: int = 4,
        max_missed_frames: int = 2,
        position_tolerance: float = 15.0,
        velocity_tolerance: float = 60.0,
        y_tolerance: float = 4.0,
    ):
        self.lane_manager = (
            lane_manager
            if lane_manager is not None
            else LaneManager()
        )

        if fps <= 0 or frame_skip <= 0:
            raise ValueError("fps와 frame_skip은 양수여야 합니다.")

        self.fps = fps
        self.frame_skip = frame_skip

        # env.step() 한 번에 경과하는 시뮬레이션 시간
        self.dt = frame_skip / fps
        self.max_missed_frames = max_missed_frames

        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.y_tolerance = y_tolerance

        self.reset()

    def reset(self):
        """에피소드 시작 시 추적 상태 초기화."""
        self.tracks: dict[int, TrackedBullet] = {}
        self.next_id = 1
        self.frame_index = 0

        self.new_ids: list[int] = []
        self.lost_ids: list[int] = []

    def _parse_detections(
        self,
        bullet_array: np.ndarray
    ) -> list[dict]:

        detections = []

        for row in bullet_array:
            if not np.any(row):
                continue

            left, top, right, bottom, vx = map(
                float, row
            )

            center_y = (top + bottom) / 2

            detections.append({
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "vx": vx,
                "center_x": (left + right) / 2,
                "center_y": center_y,
                "lane": self.lane_manager.get_lane(
                    center_y
                ),
                "occupied_lanes":
                    self.lane_manager.get_occupied_lanes(
                        top, bottom
                    ),
            })

        return detections

    def _match_cost(
        self,
        track: TrackedBullet,
        det: dict
    ) -> Optional[float]:

        # 기존 포탄과 신규 관측의 매칭 비용.
        # 매칭할 수 없으면 None.

        # 이동 방향이 다른 포탄은 매칭하지 않음
        if track.vx * det["vx"] < 0:
            return None

        velocity_error = abs(track.vx - det["vx"])

        if velocity_error > self.velocity_tolerance:
            return None

        y_error = abs(
            track.center_y - det["center_y"]
        )

        if y_error > self.y_tolerance:
            return None

        # 누락 프레임을 포함한 예상 이동 시간
        elapsed = (
            track.missed_frames + 1
        ) * self.dt

        predicted_x = (
                track.center_x + track.effective_vx * elapsed
        )

        x_error = abs(
            predicted_x - det["center_x"]
        )

        if x_error > self.position_tolerance:
            return None

        # 좌표·속도 차이를 함께 고려
        return (
            x_error
            + 2.0 * y_error
            + 0.05 * velocity_error
        )

    
    def _create_track(
        self,
        det: dict,
        mario: Optional[np.ndarray] = None,
        time_elapsed: Optional[float] = None,
    ):
        bullet_id = self.next_id
        self.next_id += 1

        track = TrackedBullet(
            bullet_id=bullet_id,
            left=det["left"],
            top=det["top"],
            right=det["right"],
            bottom=det["bottom"],
            vx=det["vx"],
            lane=det["lane"],
            occupied_lanes=det["occupied_lanes"],
            first_seen_frame=self.frame_index,
            fps=self.fps,
        )

        # 신규 포탄의 최초 관측 상태 기록
        if mario is not None:
            self._record_initial_state(
                track,
                mario,
                time_elapsed
            )

        self.tracks[bullet_id] = track
        self.new_ids.append(bullet_id)



    def _record_initial_state(
        self,
        track: TrackedBullet,
        mario: np.ndarray,
        time_elapsed: Optional[float],
    ) -> None:
        """
        포탄 최초 관측 시점의 마리오-포탄
        상대 위치와 초기 TTC를 기록한다.

        이 함수는 포탄 ID가 새로 만들어질 때
        단 한 번 호출한다.
        """

        ml = float(mario[0])
        mt = float(mario[1])
        mr = float(mario[2])
        mb = float(mario[3])
        vm = float(mario[4])

        xb = track.center_x
        xm = (ml + mr) / 2.0

        # 1. 최초 관측 순간 고정값
        track.first_seen_time = time_elapsed
        track.initial_bullet_x = xb
        track.initial_mario_x = xm

        # 2. 중심 간 초기 사거리
        track.initial_distance = abs(xm - xb)

        # 3. 부호를 포함한 상대속도
        vb = track.effective_vx
        track.initial_relative_vx = vb - vm

        # 4. 수직 충돌 범위
        vertical = track.top < mb and track.bottom > mt
        track.initial_vertical_overlap = vertical

        # 5. 바운딩 박스 기준 초기 간격
        if track.right <= ml:
            gap = ml - track.right
            closing_speed = vb - vm

        elif track.left >= mr:
            gap = track.left - mr
            closing_speed = vm - vb

        else:
            # 이미 수평으로 겹쳐 있음
            gap = 0.0
            closing_speed = 0.0

        track.initial_collision_gap = float(gap)
        track.initial_closing_speed = float(closing_speed)

        # 6. 최초 TTC
        if not vertical:
            ttc = math.inf
        elif gap == 0:
            ttc = 0.0
        elif closing_speed <= 0:
            ttc = math.inf
        else:
            ttc = gap / closing_speed

        track.initial_ttc = float(ttc)

    def _update_track(
        self,
        track: TrackedBullet,
        det: dict
    ):
        track.left = det["left"]
        track.top = det["top"]
        track.right = det["right"]
        track.bottom = det["bottom"]
        track.vx = det["vx"]

        track.lane = det["lane"]
        track.occupied_lanes = det["occupied_lanes"]

        track.missed_frames = 0
        track.is_new = False

    def update(
        self,
        observation: dict,
        time_elapsed: Optional[float] = None
    ) -> list[TrackedBullet]:
        """관측 1프레임을 입력받아 추적 갱신."""

        self.frame_index += 1
        self.new_ids = []
        self.lost_ids = []

        detections = self._parse_detections(
            observation["bullet_bills"]
        )

        # 기존 추적 객체의 경과 시간 갱신
        for track in self.tracks.values():
            track.age_frames += 1
            track.is_new = False

        # 가능한 모든 매칭 조합의 비용 계산
        candidates = []

        for bullet_id, track in self.tracks.items():
            for det_idx, det in enumerate(detections):
                cost = self._match_cost(track, det)

                if cost is not None:
                    candidates.append(
                        (cost, bullet_id, det_idx)
                    )

        # 가장 비용이 작은 조합부터 매칭
        candidates.sort()

        matched_tracks = set()
        matched_detections = set()

        for cost, bullet_id, det_idx in candidates:
            if bullet_id in matched_tracks:
                continue

            if det_idx in matched_detections:
                continue

            self._update_track(
                self.tracks[bullet_id],
                detections[det_idx]
            )

            matched_tracks.add(bullet_id)
            matched_detections.add(det_idx)

        # 매칭되지 않은 기존 포탄 처리
        for bullet_id in list(self.tracks.keys()):
            if bullet_id in matched_tracks:
                continue

            track = self.tracks[bullet_id]
            track.missed_frames += 1

            if (
                track.missed_frames
                > self.max_missed_frames
            ):
                self.lost_ids.append(bullet_id)
                del self.tracks[bullet_id]

        # 새로 등장한 포탄에 ID 부여
        mario = observation.get("mario")

        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                self._create_track(
                    det,
                    mario=mario,
                    time_elapsed=time_elapsed,
                )

        # 이번 프레임에서 실제 관측된 포탄만 반환
        visible_tracks = [
            track
            for track in self.tracks.values()
            if track.missed_frames == 0
        ]

        return sorted(
            visible_tracks,
            key=lambda track: track.bullet_id
        )
