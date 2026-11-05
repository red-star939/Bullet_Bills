
import math
from typing import Optional

import numpy as np

from .lane_manager import LaneManager


FPS = 60
RISK_TAU = 1.0

lane_manager = LaneManager()


def effective_velocity(vx: float) -> float:
    """
    실제 게임 엔진의 정수 픽셀 이동을 반영.
    """
    return float(math.trunc(float(vx) / FPS) * FPS)


def get_lane(y: float) -> Optional[int]:
    return lane_manager.get_lane(y)


def get_occupied_lanes(top: float, bottom: float) -> list[int]:
    return lane_manager.get_occupied_lanes(top, bottom)


def calculate_collision_info(mario, bullet):
    """
    mario:
        [left, top, right, bottom, vx, vy,
         invincible, jumping]

    bullet:
        [left, top, right, bottom, vx]

    반환값:
        horizontal_gap
        relative_vx
        closing_speed
        vertical_overlap
        ttc
    """

    ml, mt, mr, mb, vm, _, _, _ = map(float, mario)
    bl, bt, br, bb, vb = map(float, bullet)

    # 게임에서 실제 적용되는 포탄 이동 속도
    vb_effective = effective_velocity(vb)

    # 부호를 유지한 상대속도
    relative_vx = vb_effective - vm

    # Y축 충돌 범위
    vertical_overlap = (
        bt < mb and bb > mt
    )

    # X축 간격 및 접근 상대속도
    if br <= ml:
        # 포탄이 마리오 왼쪽에 있는 경우
        gap = ml - br
        closing_speed = relative_vx

    elif bl >= mr:
        # 포탄이 마리오 오른쪽에 있는 경우
        gap = bl - mr
        closing_speed = -relative_vx

    else:
        # 이미 수평으로 겹침
        gap = 0.0
        closing_speed = 0.0

    # 현재 수직 범위가 겹치지 않음
    if not vertical_overlap:
        ttc = math.inf

    # 현재 바운딩 박스가 겹침
    elif gap == 0:
        ttc = 0.0

    # 수평 간격이 줄어들지 않음
    elif closing_speed <= 0:
        ttc = math.inf

    else:
        ttc = gap / closing_speed

    return {
        "horizontal_gap": float(gap),
        "relative_vx": float(relative_vx),
        "closing_speed": float(closing_speed),
        "vertical_overlap": vertical_overlap,
        "ttc": float(ttc),
        "effective_vx": vb_effective,
    }


def calculate_ttc(mario, bullet):
    """
    기존 함수와 호환되는 TTC 계산.
    """
    result = calculate_collision_info(mario, bullet)
    return result["ttc"]


def calculate_risk(ttc, invincible=False, tau=RISK_TAU):
    """
    TTC 기반 위험 점수: 0~1.

    무적이면 0을 반환.
    """
    if invincible or not math.isfinite(ttc):
        return 0.0

    return float(math.exp(-max(0.0, ttc) / tau))


def analyze_bullets(
    observation,
    tracked_bullets=None,
):
    """
    포탄 위험도를 계산한다.

    tracked_bullets:
        None -> 기존 observation 배열 사용
        list -> BulletTracker 결과 사용

    반환:
        위험도가 높은 포탄부터 정렬된 dict 목록.
    """

    mario = observation["mario"]
    invincible = bool(mario[6])

    results = []

    if tracked_bullets is None:
        # 기존 배열 기반 분석
        sources = []

        for idx, row in enumerate(
            observation["bullet_bills"]
        ):
            if not np.any(row):
                continue

            sources.append((idx, row, None))

    else:
        # 추적기가 부여한 ID와 초기 정보 사용
        sources = []

        for track in tracked_bullets:
            row = [
                track.left,
                track.top,
                track.right,
                track.bottom,
                track.vx,
            ]
            sources.append(
                (track.bullet_id, row, track)
            )

    for bullet_id, row, track in sources:
        bl, bt, br, bb, vx = map(float, row)

        center_x = (bl + br) / 2.0
        center_y = (bt + bb) / 2.0

        lane = get_lane(center_y)
        occupied = get_occupied_lanes(bt, bb)

        collision = calculate_collision_info(
            mario, row
        )

        ttc = collision["ttc"]

        risk = calculate_risk(
            ttc,
            invincible=invincible
        )

        # 초기 상태는 BulletTracker에서 가져온다.
        # 추적 정보가 없는 경우 None으로 유지.
        initial_distance = (
            getattr(track, "initial_distance", None)
            if track is not None else None
        )

        initial_gap = (
            getattr(track, "initial_collision_gap", None)
            if track is not None else None
        )

        initial_relative_vx = (
            getattr(track, "initial_relative_vx", None)
            if track is not None else None
        )

        initial_ttc = (
            getattr(track, "initial_ttc", None)
            if track is not None else None
        )

        first_seen_time = (
            getattr(track, "first_seen_time", None)
            if track is not None else None
        )

        if risk >= 0.7:
            risk_level = "HIGH"
        elif risk >= 0.35:
            risk_level = "MEDIUM"
        elif risk > 0:
            risk_level = "LOW"
        else:
            risk_level = "NONE"

        results.append({
            "bullet_id": int(bullet_id),
            "lane": lane,
            "occupied_lanes": occupied,

            "x_left": bl,
            "x_right": br,
            "center_x": center_x,
            "vx": vx,
            "effective_vx": collision["effective_vx"],

            "horizontal_gap": collision["horizontal_gap"],
            "relative_vx": collision["relative_vx"],
            "closing_speed": collision["closing_speed"],
            "vertical_overlap": collision["vertical_overlap"],

            "first_seen_time": first_seen_time,
            "initial_distance": initial_distance,
            "initial_collision_gap": initial_gap,
            "initial_relative_vx": initial_relative_vx,
            "initial_ttc": initial_ttc,

            "ttc": ttc,
            "risk": risk,
            "risk_level": risk_level,
            "invincible": invincible,
        })

    # 위험도가 높은 순으로 정렬
    results.sort(
        key=lambda item: (
            -item["risk"],
            item["ttc"],
            item["bullet_id"],
        )
    )

    return results


def get_most_dangerous(results):
    """위험 점수가 가장 높은 포탄."""
    if not results:
        return None

    most_dangerous = max(
        results,
        key=lambda item: item["risk"]
    )

    if most_dangerous["risk"] <= 0:
        return None

    return most_dangerous


def get_total_risk(results):
    """포탄 중 최대 위험 점수."""
    return max(
        (item["risk"] for item in results),
        default=0.0
    )
