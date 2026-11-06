"""Observed bullet-event labels from the engine's pre-collision witness.

Unlike a final-vy == -500 heuristic, this records the Mario state *at the
entry* to check_bullet_collision, after movement and star pickup.  The
active -> fallen state change is used as independent outcome evidence.
All rectangle coordinates use (left, top, right, bottom).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class WitnessBullet:
    identity: int
    bbox: tuple[int, int, int, int]
    state: str


@dataclass(frozen=True)
class CollisionWitness:
    mario_bbox: tuple[int, int, int, int]
    mario_vy: float
    invincible: bool
    bullets: tuple[WitnessBullet, ...]


def capture_collision_witness(mario, bullets: Sequence[object]) -> CollisionWitness:
    """Call immediately before the installed engine's bullet collision method."""
    r = mario.rect
    return CollisionWitness(
        (int(r.left), int(r.top), int(r.right), int(r.bottom)),
        float(mario._velocity.y), bool(mario.invincible_),
        tuple(WitnessBullet(id(b),
              (int(b.rect.left), int(b.rect.top),
               int(b.rect.right), int(b.rect.bottom)),
              str(b.state_)) for b in bullets),
    )


def observed_collision_events(
    witness: CollisionWitness | None,
    after_states: Mapping[int, str],
    actual_alive: bool,
    actual_bottom: float,
    world_height: int = 864,
) -> list[str]:
    """Return observed outcomes or an explicit ambiguous label.

    A single active->fallen transition is classified using the *pre-impact*
    velocity, contact geometry, and invincibility. Multiple transitions are
    censored because the current evaluator counts one event type per frame.
    """
    if witness is None:
        return ['missing_witness']

    changed = [b for b in witness.bullets if b.state == 'active'
               and after_states.get(b.identity) == 'fallen']
    if len(changed) > 1:
        return ['multiple_hit']
    if len(changed) == 1:
        b = changed[0]
        m = witness.mario_bbox
        bb = b.bbox
        overlap = m[0] < bb[2] and m[2] > bb[0] and m[1] < bb[3] and m[3] > bb[1]
        if not overlap:
            return ['uncertain_hit']
        center_y = bb[1] + (bb[3] - bb[1]) // 2
        if witness.mario_vy > 0 and m[3] < center_y:
            return ['stomp']
        if witness.invincible:
            return ['invincible_hit']
        return ['uncertain_hit']

    if not actual_alive and float(actual_bottom) <= float(world_height):
        return ['death']
    return []
