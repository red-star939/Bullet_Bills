"""Frame-resolution Bullet Bill collision geometry (v2).

This module predicts *geometric contact candidates*, NOT an authoritative
engine outcome.  Stomps are validated against the logged physical frames;
side/bottom/death/invincibility require exact local engine-source verification.

All rectangles use pygame's half-open convention: touching edges alone is
NOT collision.  A bullet moves by int(nominal_vx / fps) each internal frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math

BBox = tuple[int, int, int, int]


def overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    """Pygame Rect-style strict area intersection (not edge touch)."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def horizontal_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and a[2] > b[0]


def effective_bullet_vx(vx: float, fps: int = 60) -> int:
    """Actual signed integer-pixel translation per internal physics frame."""
    if fps <= 0 or not math.isfinite(float(vx)):
        raise ValueError('fps must be positive and velocity must be finite')
    return math.trunc(float(vx) / fps)


@dataclass(frozen=True)
class BulletState:
    bullet_id: int
    bbox: BBox
    vx: float  # nominal velocity, px/sec; NOT effective/quantized velocity
    state: str = "active"  # engine: active, fallen, oob

    @classmethod
    def from_tracked(cls, bullet: object) -> 'BulletState':
        return cls(
            bullet_id=int(getattr(bullet, 'bullet_id', getattr(bullet, 'id', -1))),
            bbox=(int(round(bullet.left)), int(round(bullet.top)),
                  int(round(bullet.right)), int(round(bullet.bottom))),
            vx=float(bullet.vx),
            state=str(getattr(bullet, "state", getattr(bullet, "state_", "active"))),
        )

    def advanced(self, frames: int = 1, fps: int = 60) -> 'BulletState':
        if frames < 0:
            raise ValueError('frames must be >= 0')
        l, t, r, b = self.bbox
        if self.state == 'active':
            dx = effective_bullet_vx(self.vx, fps) * frames
            return BulletState(self.bullet_id, (l + dx, t, r + dx, b), self.vx, self.state)
        if self.state == 'fallen':
            dy = math.trunc(700.0 / fps) * frames
            return BulletState(self.bullet_id, (l, t + dy, r, b + dy), self.vx, self.state)
        return self


@dataclass(frozen=True)
class ContactClassification:
    kind: str  # none, stomp_candidate, side_or_bottom
    entry_face: str  # none, top, bottom, left, right, persistent, ambiguous
    pre_existing_overlap: bool


@dataclass(frozen=True)
class Contact:
    frame: int
    time_s: float
    bullet_id: int
    mario_bbox: BBox
    bullet_bbox: BBox
    kind: str
    previous_mario_bbox: BBox
    previous_bullet_bbox: BBox
    entry_face: str = 'ambiguous'
    pre_existing_overlap: bool = False


def classify_contact_detail(previous_mario: Sequence[float], mario: Sequence[float],
                            previous_bullet: Sequence[float], bullet: Sequence[float],
                            previous_mario_vy: float) -> ContactClassification:
    """Classify new overlap using pre/post boxes at one physics frame.

    A *top entry* (potential stomp) requires falling at start, previous feet
    at/above previous bullet top, AND actually crossing the bullet's top.
    This correctly excludes the ascending underside collision observed on
    episode 4/frame 539. 'side_or_bottom' never asserts death.
    """
    if not overlaps(mario, bullet):
        return ContactClassification('none', 'none', False)
    existing = overlaps(previous_mario, previous_bullet)
    if existing:
        return ContactClassification('side_or_bottom', 'persistent', True)
    if (previous_mario_vy > 0 and
            previous_mario[3] <= previous_bullet[1] and
            mario[3] > bullet[1]):
        return ContactClassification('stomp_candidate', 'top', False)
    if previous_mario[1] >= previous_bullet[3] and mario[1] < bullet[3]:
        return ContactClassification('side_or_bottom', 'bottom', False)
    if previous_mario[2] <= previous_bullet[0] and mario[2] > bullet[0]:
        return ContactClassification('side_or_bottom', 'left', False)
    if previous_mario[0] >= previous_bullet[2] and mario[0] < bullet[2]:
        return ContactClassification('side_or_bottom', 'right', False)
    return ContactClassification('side_or_bottom', 'ambiguous', False)


def classify_contact(previous_mario: Sequence[float], mario: Sequence[float],
                     previous_bullet: Sequence[float], bullet: Sequence[float],
                     previous_mario_vy: float) -> str:
    """Backward-compatible class label used by existing simulator/tests."""
    return classify_contact_detail(previous_mario, mario, previous_bullet,
                                   bullet, previous_mario_vy).kind


def detect_contacts(previous_mario: Sequence[float], mario: Sequence[float],
                    previous_mario_vy: float, previous_bullets: Iterable[BulletState],
                    frame: int, fps: int = 60) -> tuple[list[Contact], list[BulletState]]:
    """Advance bullets exactly once and return all current overlapping contacts.

    Does NOT remove projectiles, decide lethal outcome, or spawn future bullets.
    Input bullets must be positioned at the *beginning* of this frame.
    """
    current_bullets = []
    contacts: list[Contact] = []
    for previous in previous_bullets:
        current = previous.advanced(fps=fps)
        current_bullets.append(current)
        classification = classify_contact_detail(previous_mario, mario,
                                                 previous.bbox, current.bbox,
                                                 previous_mario_vy)
        if classification.kind != 'none':
            contacts.append(Contact(
                frame=frame, time_s=frame / fps,
                bullet_id=current.bullet_id,
                mario_bbox=tuple(map(int, mario)),
                bullet_bbox=current.bbox,
                kind=classification.kind,
                previous_mario_bbox=tuple(map(int, previous_mario)),
                previous_bullet_bbox=previous.bbox,
                entry_face=classification.entry_face,
                pre_existing_overlap=classification.pre_existing_overlap,
            ))
    return contacts, current_bullets


@dataclass(frozen=True)
class FrameCollisionResult:
    contacts: tuple[Contact, ...]
    bullets: tuple[BulletState, ...]
    has_stomp_candidate: bool
    has_unresolved_contact: bool


class CollisionPredictor:
    """Stateless collision geometry adapter for the future action evaluator.

    For a multi-frame trajectory the caller must feed the *previous* Mario
    rectangle/velocity each time. Any predicted stomp/bullet removal is the
    responsibility of InteractionSimulator, not this geometry module.
    """
    def __init__(self, fps: int = 60):
        if fps <= 0:
            raise ValueError('fps must be positive')
        self.fps = fps

    def predict_frame(self, previous_mario: Sequence[float], mario: Sequence[float],
                      previous_mario_vy: float, bullets: Iterable[BulletState],
                      frame: int) -> FrameCollisionResult:
        contacts, moved = detect_contacts(previous_mario, mario,
                                          previous_mario_vy, bullets,
                                          frame, self.fps)
        return FrameCollisionResult(
            contacts=tuple(contacts), bullets=tuple(moved),
            has_stomp_candidate=any(c.kind == 'stomp_candidate' for c in contacts),
            has_unresolved_contact=any(c.kind != 'stomp_candidate' for c in contacts),
        )


# Authoritative engine resolution derived from installed kymnasium objs.py
# Mario.check_bullet_collision, called after Mario.tick_frame and Bullet.tick_frame.
@dataclass(frozen=True)
class EngineContact:
    bullet_id: int
    outcome: str  # stomp, invincible_hit, death, ignored, none
    mario_bbox: BBox
    bullet_bbox: BBox


def engine_contact_outcome(mario: Sequence[float], bullet: Sequence[float],
                           mario_vy: float, invincible: bool = False,
                           bullet_state: str = 'active') -> str:
    """Exact priority of the engine's check_bullet_collision branch.

    Post-movement Mario velocity/rectangle and post-movement bullet rectangle
    are mandatory. No previous-frame entry-face heuristic is used by the engine.
    """
    if not overlaps(mario, bullet):
        return 'none'
    if bullet_state != 'active':
        return 'ignored'
    # pygame Rect.centery uses integer floor for odd rect heights.
    center_y = int(bullet[1]) + (int(bullet[3]) - int(bullet[1])) // 2
    if mario_vy > 0 and mario[3] < center_y:
        return 'stomp'
    if invincible:
        return 'invincible_hit'
    return 'death'


def resolve_engine_contacts(mario: Sequence[float], mario_vy: float,
                            bullets: Iterable[BulletState],
                            invincible: bool = False) -> list[EngineContact]:
    """Resolve bullets in supplied order, matching the engine's loop order.

    When a stomp changes vy to -500, later bullets in this SAME frame see
    that updated vy. Fallen bullets remain present but are ignored on contact.
    """
    result: list[EngineContact] = []
    current_vy = float(mario_vy)
    for bullet in bullets:
        outcome = engine_contact_outcome(mario, bullet.bbox, current_vy,
                                         invincible, bullet.state)
        if outcome != 'none':
            result.append(EngineContact(bullet.bullet_id, outcome,
                                        tuple(map(int, mario)), bullet.bbox))
        if outcome == 'stomp':
            current_vy = -500.0
    return result


def predict_engine_frame(mario_bbox: Sequence[float], mario_post_vy: float,
                         bullets_before: Iterable[BulletState],
                         invincible: bool = False, fps: int = 60
                         ) -> tuple[list[EngineContact], list[BulletState]]:
    """Move bullets, then evaluate the exact engine contact branch.

    Does not apply the resulting changes. InteractionSimulator is responsible
    for velocity updates, fallen states, cleanup and termination.
    """
    moved = [b.advanced(fps=fps) for b in bullets_before]
    return resolve_engine_contacts(mario_bbox, mario_post_vy, moved,
                                   invincible=invincible), moved
