"""Conservative Mario/bullet interaction simulator (version 1).

Confirmed by logged frames: a descending Mario stomping a Bullet Bill has
velocity set to -500 px/s *after* movement in that physics frame, and the
stomped bullet disappears. Other contact outcomes are explicitly uncertain.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .mario_motion_predictor import MarioMotionPredictor, MarioState
from .collision_predictor import BulletState, Contact, detect_contacts


@dataclass(frozen=True)
class InteractionConfig:
    stomp_bounce_vy: float = -500.0
    fps: int = 60


@dataclass(frozen=True)
class InteractionEvent:
    frame: int
    time_s: float
    bullet_id: int
    event_type: str  # stomp or unresolved_contact
    mario_bbox: tuple[int, int, int, int]
    bullet_bbox: tuple[int, int, int, int]


@dataclass
class InteractionResult:
    mario: MarioState
    bullets: list[BulletState]
    events: list[InteractionEvent]
    frames_simulated: int
    unresolved: bool


class InteractionSimulator:
    """Advances Mario and currently observed bullets at internal frame rate.

    Does not predict bullets that have not spawned yet or model time-varying
    invincibility. A contact other than a stomp is recorded as unresolved
    (not automatically labelled death) to avoid overclaiming engine parity.
    """
    def __init__(self, motion: MarioMotionPredictor,
                 config: InteractionConfig | None = None):
        self.motion = motion
        self.config = config or InteractionConfig(fps=motion.cfg.fps)
        if self.config.fps != motion.cfg.fps:
            raise ValueError('Simulation FPS must match motion predictor FPS')

    def simulate(self, mario: MarioState, bullets: Sequence[BulletState],
                 actions: Sequence[Sequence[float]],
                 stop_on_unresolved: bool = True) -> InteractionResult:
        state = replace(mario)
        active = list(bullets)
        events: list[InteractionEvent] = []
        frame = 0
        unresolved = False
        for action in actions:
            force = self.motion.action_to_force(action)
            for _ in range(self.motion.cfg.frame_skip):
                before = replace(state)
                state = self.motion._internal_step_force(state, force)
                frame += 1
                contacts, active = detect_contacts(before.bbox, state.bbox,
                                                    before.vy, active, frame,
                                                    self.config.fps)
                stomped_ids: set[int] = set()
                for contact in contacts:
                    event_type = 'stomp' if contact.kind == 'stomp_candidate' else 'unresolved_contact'
                    events.append(InteractionEvent(
                        frame, frame / self.config.fps, contact.bullet_id,
                        event_type, contact.mario_bbox, contact.bullet_bbox))
                    if event_type == 'stomp':
                        stomped_ids.add(contact.bullet_id)
                    else:
                        unresolved = True
                if stomped_ids:
                    state.vy = self.config.stomp_bounce_vy
                    state.grounded = False
                    active = [b for b in active if b.bullet_id not in stomped_ids]
                if unresolved and stop_on_unresolved:
                    return InteractionResult(state, active, events, frame, True)
        return InteractionResult(state, active, events, frame, unresolved)
