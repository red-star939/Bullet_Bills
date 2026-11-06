"""Frame-level AABB geometry for BulletBill-Continuous-Hard-Stage-2.

Geometry is separated from response mechanics. Predicted contacts are hypotheses,
not a guarantee that the game engine will resolve them as stomps or deaths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math

BBox = tuple[int, int, int, int]


def overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def horizontal_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and a[2] > b[0]


def effective_bullet_vx(vx: float, fps: int = 60) -> int:
    """Actual integer-pixel translation per internal physics frame."""
    return math.trunc(float(vx) / fps)


@dataclass(frozen=True)
class BulletState:
    bullet_id: int
    bbox: BBox
    vx: float  # nominal velocity (px/s), not already quantized

    @classmethod
    def from_tracked(cls, bullet: object) -> 'BulletState':
        return cls(
            bullet_id=int(getattr(bullet, 'bullet_id', getattr(bullet, 'id', -1))),
            bbox=(int(round(bullet.left)), int(round(bullet.top)),
                  int(round(bullet.right)), int(round(bullet.bottom))),
            vx=float(bullet.vx),
        )

    def advanced(self, frames: int = 1, fps: int = 60) -> 'BulletState':
        dx = effective_bullet_vx(self.vx, fps) * frames
        l, t, r, b = self.bbox
        return BulletState(self.bullet_id, (l + dx, t, r + dx, b), self.vx)


@dataclass(frozen=True)
class Contact:
    frame: int
    time_s: float
    bullet_id: int
    mario_bbox: BBox
    bullet_bbox: BBox
    kind: str  # stomp_candidate, side_or_bottom, or overlap
    previous_mario_bbox: BBox
    previous_bullet_bbox: BBox


def classify_contact(previous_mario: Sequence[float], mario: Sequence[float],
                     previous_bullet: Sequence[float], bullet: Sequence[float],
                     previous_mario_vy: float) -> str:
    """Classify observed overlap as a plausible top-down contact.

    Stomp criterion: falling before motion, horizontal overlap in the current
    frame, and Mario's previous feet at or above the bullet's previous top.
    This is an *inferred* criterion and must be compared with engine source.
    """
    if not overlaps(mario, bullet):
        return 'none'
    if (previous_mario_vy > 0 and
            previous_mario[3] <= previous_bullet[1] and
            horizontal_overlap(mario, bullet)):
        return 'stomp_candidate'
    return 'side_or_bottom'


def detect_contacts(previous_mario: Sequence[float], mario: Sequence[float],
                    previous_mario_vy: float, previous_bullets: Iterable[BulletState],
                    frame: int, fps: int = 60) -> tuple[list[Contact], list[BulletState]]:
    """Advance each bullet once, return overlapping contacts and advanced bullets.

    The caller advances Mario with MarioMotionPredictor first. Contacts use
    the post-motion AABBs while the previous boxes distinguish approach.
    """
    previous_bullets = list(previous_bullets)
    current_bullets = [b.advanced(fps=fps) for b in previous_bullets]
    contacts = []
    for prev, current in zip(previous_bullets, current_bullets):
        kind = classify_contact(previous_mario, mario, prev.bbox, current.bbox,
                                previous_mario_vy)
        if kind != 'none':
            contacts.append(Contact(frame=frame, time_s=frame / fps,
                                    bullet_id=current.bullet_id,
                                    mario_bbox=tuple(map(int, mario)),
                                    bullet_bbox=current.bbox, kind=kind,
                                    previous_mario_bbox=tuple(map(int, previous_mario)),
                                    previous_bullet_bbox=prev.bbox))
    return contacts, current_bullets
