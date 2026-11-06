"""60-FPS engine-ordered bullet/mario interactions for existing bullets.

Engine source: _step_internal() updates Mario, then Bullet Bill positions,
then stars, then collisions; stomp has priority over invincibility; fallen
bullets persist and fall at int(700/60) px/frame until offscreen.

Future star pickup/spawns cannot be predicted from bullets alone. For such
situations pass a current invincibility state or stop at the unknown event.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .mario_motion_predictor import MarioMotionPredictor, MarioState
from .collision_predictor import BulletState, engine_contact_outcome


@dataclass(frozen=True)
class InteractionConfig:
    stomp_bounce_vy: float = -500.0
    fps: int = 60
    fallen_speed: float = 700.0


@dataclass(frozen=True)
class InteractionEvent:
    frame: int
    time_s: float
    bullet_id: int
    event_type: str  # stomp, invincible_hit, death, ignored_contact
    mario_bbox: tuple[int, int, int, int]
    bullet_bbox: tuple[int, int, int, int]


@dataclass
class InteractionResult:
    mario: MarioState
    bullets: list[BulletState]
    events: list[InteractionEvent]
    frames_simulated: int
    unresolved: bool
    terminated: bool = False
    termination_reason: str | None = None


class InteractionSimulator:
    def __init__(self, motion: MarioMotionPredictor,
                 config: InteractionConfig | None = None):
        self.motion = motion
        self.config = config or InteractionConfig(fps=motion.cfg.fps)
        if self.config.fps != motion.cfg.fps:
            raise ValueError('Simulation FPS must match motion predictor FPS')

    def simulate(self, mario: MarioState, bullets: Sequence[BulletState],
                 actions: Sequence[Sequence[float]],
                 stop_on_unresolved: bool = True) -> InteractionResult:
        """Run full action-steps; stop immediately after a predicted death.

        `stop_on_unresolved` retained for API compatibility. No contact
        categories are unresolved under known invincibility state; future
        star pickups and new bullet spawns remain outside this model.
        """
        state = replace(mario)
        active = list(bullets)  # includes fallen bullets
        events: list[InteractionEvent] = []
        frame = 0
        for action in actions:
            force = self.motion.action_to_force(action)
            for _ in range(self.motion.cfg.frame_skip):
                state = self.motion._internal_step_force(state, force)
                frame += 1
                # Engine updates bullets BEFORE checking collisions.
                active = [b.advanced(fps=self.config.fps) for b in active]
                # The engine resolves star pickups here, if any; not modelled.
                # Bullet check uses *post-motion* vy and bbox; loop order matters.
                current_vy = state.vy
                dead = False
                next_bullets = []
                for b in active:
                    outcome = engine_contact_outcome(state.bbox, b.bbox,
                                                     current_vy, state.invincible,
                                                     b.state)
                    if outcome != 'none':
                        events.append(InteractionEvent(frame, frame / self.config.fps,
                            b.bullet_id, 'ignored_contact' if outcome == 'ignored' else outcome,
                            state.bbox, b.bbox))
                    if outcome == 'stomp':
                        current_vy = self.config.stomp_bounce_vy
                        state.vy = current_vy
                        b = replace(b, state='fallen')
                    elif outcome == 'invincible_hit':
                        b = replace(b, state='fallen')
                    elif outcome == 'death':
                        dead = True
                    # Engine oob filtering is after the entire collision pass.
                    if b.state == 'fallen':
                        if b.bbox[1] <= self.motion.cfg.world_height:
                            next_bullets.append(b)
                    elif b.state == 'active':
                        if not ((b.vx > 0 and b.bbox[0] > self.motion.cfg.world_width)
                                or (b.vx < 0 and b.bbox[2] < 0)):
                            next_bullets.append(b)
                active = next_bullets
                if dead or state.bottom > self.motion.cfg.world_height:
                    return InteractionResult(state, active, events, frame, False,
                        terminated=True,
                        termination_reason='bullet' if dead else 'fall')
        return InteractionResult(state, active, events, frame, False)
