"""Known-object, 60-FPS forward simulator for BulletBill Continuous Stage 2.

Inputs: current observable Mario state, visible bullet bounding boxes/nominal
velocities, candidate continuous actions, and optional visible stars.

Models the *known* world's deterministic Mario/platform physics, bullet motion,
engine-ordered stomp/invincibility/death contacts and fallen-bullet cleanup.
It DOES NOT predict random future spawns, or infer remaining invincibility time
or the engine state of fallen bullets from a 5-value bullet observation.

Coordinates are (left, top, right, bottom), not pygame.Rect (x,y,w,h).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence
import math

from .mario_motion_predictor import MarioMotionPredictor, MarioState
from .collision_predictor import BulletState, engine_contact_outcome

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ForwardEvent:
    frame: int
    time_s: float
    action_step: int
    subframe: int
    kind: str  # stomp, invincible_hit, death, fall, star_unknown
    bullet_id: int | None
    mario_bbox: BBox
    bullet_bbox: BBox | None = None


@dataclass(frozen=True)
class ForwardFrame:
    frame: int
    time_s: float
    action_step: int
    subframe: int
    mario_bbox: BBox
    vx: float
    vy: float
    grounded: bool
    bullets: tuple[BulletState, ...]
    events: tuple[ForwardEvent, ...]


@dataclass
class ForwardResult:
    mario: MarioState
    bullets: list[BulletState]
    frames: list[ForwardFrame]
    events: list[ForwardEvent]
    frames_simulated: int
    actions_completed: int
    terminated: bool
    termination_reason: str | None
    # Every result is conditional on no unseen future spawns. Other caveats
    # appear here if relevant to the initial observation.
    assumptions: tuple[str, ...]

    @property
    def first_lethal_event(self) -> ForwardEvent | None:
        return next((e for e in self.events if e.kind in ('death', 'fall')), None)

    @property
    def first_collision_event(self) -> ForwardEvent | None:
        return next((e for e in self.events if e.kind in
                     ('stomp', 'invincible_hit', 'death')), None)

    @property
    def simulated_seconds(self) -> float:
        return self.frames[-1].time_s if self.frames else 0.0


class ForwardSimulator:
    """Frame-accurate prediction conditional on the *currently known* world.

    Accepts the pre-existing MarioMotionPredictor, so the level's static
    platforms and all calibrated physics constants are reused unchanged.
    """

    def __init__(self, motion: MarioMotionPredictor):
        self.motion = motion
        self.fps = motion.cfg.fps
        self.frame_skip = motion.cfg.frame_skip

    @staticmethod
    def bullets_from_observation(observation: Mapping) -> list[BulletState]:
        """Parse Gym's (N,5) bullet data. State is not observable; assume active.

        For a more accurate rollout when fallen bullets are present, supply
        BulletState objects with explicit states via `simulate(..., bullets=...)`.
        """
        out: list[BulletState] = []
        for i, row in enumerate(observation['bullet_bills']):
            if len(row) < 5:
                raise ValueError('bullet_bills rows require 5 values')
            left, top, right, bottom, vx = map(float, row[:5])
            if right <= left or bottom <= top:
                continue
            if not all(math.isfinite(x) for x in (left, top, right, bottom, vx)):
                raise ValueError('Bullet contains nonfinite values')
            out.append(BulletState(
                i, (round(left), round(top), round(right), round(bottom)),
                vx, 'active'))
        return out

    @staticmethod
    def stars_from_observation(observation: Mapping) -> tuple[BBox, ...]:
        result = []
        for row in observation.get('stars', []):
            if len(row) < 4:
                continue
            l, t, r, b = map(float, row[:4])
            if r > l and b > t:
                result.append((round(l), round(t), round(r), round(b)))
        return tuple(result)

    def simulate_observation(
        self, observation: Mapping, actions: Sequence[Sequence[float]],
        *, bullets: Iterable[BulletState] | None = None,
        stop_on_star: bool = True,
    ) -> ForwardResult:
        """Create a rollout from Gym observation, optionally with tracked states.

        `bullets` overrides 5-value observation rows, which cannot distinguish
        active from fallen. `stop_on_star=True` stops BEFORE assuming a future
        pickup's unknown invincibility effect.
        """
        mario = MarioState.from_observation(observation['mario'])
        known = list(bullets) if bullets is not None else self.bullets_from_observation(observation)
        stars = self.stars_from_observation(observation)
        caveats = []
        if bullets is None and known:
            caveats.append('bullet_states_assumed_active')
        return self.simulate(mario, known, actions, stars=stars,
                             stop_on_star=stop_on_star, extra_assumptions=caveats)

    def simulate(
        self, mario: MarioState, bullets: Iterable[BulletState],
        actions: Sequence[Sequence[float]], *, stars: Iterable[BBox] = (),
        stop_on_star: bool = True,
        extra_assumptions: Iterable[str] = (),
    ) -> ForwardResult:
        """Simulate action-step controls, returning every internal frame.

        Engine order: precomputed force -> Mario physics -> bullet physics ->
        star check -> bullet contact in original bullet order -> OOB cleanup ->
        termination. A lethal collision ends at its exact internal frame.
        A star encountered without a modelled pickup changes outcome to
        `star_unknown` if stop_on_star=True.
        """
        state = replace(mario)
        active = list(bullets)
        star_boxes = tuple(stars)
        assumptions = ['no_future_random_bullet_spawns']
        if state.invincible:
            assumptions.append('invincibility_remaining_time_unknown')
        if star_boxes and not stop_on_star:
            assumptions.append('future_star_pickups_ignored')
        assumptions.extend(extra_assumptions)
        frames: list[ForwardFrame] = []
        events: list[ForwardEvent] = []
        clock = 0
        actions_completed = 0
        terminated = False
        reason: str | None = None
        for step_idx, action in enumerate(actions, 1):
            # The source engine converts action to force ONCE per env.step.
            force = self.motion.action_to_force(action)
            for sub_idx in range(1, self.frame_skip + 1):
                clock += 1
                state = self.motion._internal_step_force(state, force)
                active = [b.advanced(fps=self.fps) for b in active]
                current_events: list[ForwardEvent] = []

                def emit(kind: str, bullet: BulletState | None = None):
                    e = ForwardEvent(
                        frame=clock, time_s=clock / self.fps,
                        action_step=step_idx, subframe=sub_idx, kind=kind,
                        bullet_id=None if bullet is None else bullet.bullet_id,
                        mario_bbox=state.bbox,
                        bullet_bbox=None if bullet is None else bullet.bbox)
                    current_events.append(e)
                    events.append(e)

                # Source checks stars before bullets; our future star/invuln
                # transition is deliberately left unknown rather than guessed.
                star_touched = any(self.motion.overlaps(state.bbox, s)
                                   for s in star_boxes)
                if star_touched and stop_on_star:
                    emit('star_unknown')
                    reason = 'unknown_star_pickup'
                    terminated = True
                if not terminated:
                    for i, bullet in enumerate(active):
                        outcome = engine_contact_outcome(
                            state.bbox, bullet.bbox, state.vy,
                            state.invincible, bullet.state)
                        if outcome == 'stomp':
                            emit('stomp', bullet)
                            state.vy = -500.0
                            active[i] = replace(bullet, state='fallen')
                        elif outcome == 'invincible_hit':
                            emit('invincible_hit', bullet)
                            active[i] = replace(bullet, state='fallen')
                        elif outcome == 'death':
                            emit('death', bullet)
                            terminated = True
                            reason = 'bullet'
                        # Engine continues checking later bullets within frame,
                        # even after setting Mario.active=False.
                    # OOB filtering happens AFTER all collision checks.
                active = [b for b in active if not (
                    (b.state == 'active' and
                     ((b.vx > 0 and b.bbox[0] > self.motion.cfg.world_width)
                      or (b.vx < 0 and b.bbox[2] < 0)))
                    or (b.state == 'fallen' and b.bbox[1] > self.motion.cfg.world_height)
                    or b.state == 'oob'
                )]
                if state.bottom > self.motion.cfg.world_height:
                    if not terminated:
                        emit('fall')
                    terminated = True
                    if reason is None:
                        reason = 'fall'
                frames.append(ForwardFrame(
                    frame=clock, time_s=clock / self.fps,
                    action_step=step_idx, subframe=sub_idx,
                    mario_bbox=state.bbox, vx=state.vx, vy=state.vy,
                    grounded=state.grounded, bullets=tuple(active),
                    events=tuple(current_events)))
                if terminated:
                    break
            if terminated:
                break
            actions_completed += 1
        return ForwardResult(
            mario=state, bullets=active, frames=frames, events=events,
            frames_simulated=clock, actions_completed=actions_completed,
            terminated=terminated, termination_reason=reason,
            assumptions=tuple(dict.fromkeys(assumptions)))
