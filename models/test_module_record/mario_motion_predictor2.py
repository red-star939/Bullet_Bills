"""Mario horizontal/vertical trajectory predictor for Bullet Bills Stage 2.

Horizontal force uses the engine-style input scaling followed by a
ground/air force cap. Does not mutate the Gym environment.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence



@dataclass(frozen=True)
class MotionConfig:
    fps: int = 60
    frame_skip: int = 4
    world_width: int = 1296
    world_height: int = 864
    tile_size: int = 48
    gravity: float = 2000.0
    jump_speed: float = 1000.0
    ground_speed_cap: float = 750.0
    air_speed_cap: float = 500.0
    friction: float = 0.90
    # Horizontal input produces an unbounded requested force of x * 100.
    # The engine clamps that request to +/-100 on ground or +/-75 in air.
    # IMPORTANT: x * min(100, 75) is NOT equivalent for fractional x.
    ground_force: float = 100.0
    air_force: float = 75.0
    idle_threshold: float = 1.0

    def __post_init__(self):
        if self.fps <= 0 or self.frame_skip <= 0:
            raise ValueError('fps and frame_skip must be positive')
        if self.tile_size <= 0 or not 0 < self.friction <= 1:
            raise ValueError('Invalid tile size or friction')


@dataclass
class MarioState:
    left: int
    top: int
    width: int
    height: int
    vx: float
    vy: float
    grounded: bool
    invincible: bool = False

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @classmethod
    def from_observation(
        cls, mario: Sequence[float]
    ) -> 'MarioState':
        ml, mt, mr, mb, vx, vy, invincible, jumping = map(float, mario)
        state = cls(
            left=round(ml), top=round(mt),
            width=round(mr - ml), height=round(mb - mt),
            vx=vx, vy=vy, grounded=not bool(jumping),
            invincible=bool(invincible),
        )
        # In observations, jumping is a not-grounded flag; trust it when supplied.
        return state


@dataclass(frozen=True)
class MotionSample:
    action_step: int
    time_s: float
    bbox: tuple[int, int, int, int]
    vx: float
    vy: float
    grounded: bool


class MarioMotionPredictor:
    """Predict future bounding boxes at 60-Hz internal physics resolution.

    ``step()`` repeats a continuous action for ``frame_skip`` internal frames.
    ``predict()`` accepts a sequence of such actions. Individual internal
    snapshots are returned for collision checking, not just step endpoints.
    """

    def __init__(self, platforms: Iterable[tuple[int, int, int, int]] = (),
                 config: MotionConfig | None = None):
        self.cfg = config or MotionConfig()
        self.platforms = tuple(tuple(map(int, p)) for p in platforms)

    @classmethod
    def from_object_tiles_csv(
        cls, csv_path: str | Path, config: MotionConfig | None = None
    ) -> 'MarioMotionPredictor':
        cfg = config or MotionConfig()
        platforms = []
        with Path(csv_path).open('r', newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                tile_id = int(row['tile_id'])
                if tile_id in (0, 26):  # Mario spawn / star: not solid
                    continue
                x, y = int(row['x_px']), int(row['y_px'])
                platforms.append((x, y, x + cfg.tile_size, y + cfg.tile_size))
        return cls(platforms, cfg)

    @staticmethod
    def overlaps(a: tuple[int, int, int, int],
                 b: tuple[int, int, int, int]) -> bool:
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    def _supported(self, s: MarioState) -> bool:
        return any(s.bottom == p[1] and s.right > p[0] and s.left < p[2]
                   for p in self.platforms)

    def _internal_step(self, s: MarioState, action: Sequence[float]) -> MarioState:
        c = self.cfg
        horizontal, jump = map(float, action)
        horizontal = max(-1.0, min(1.0, horizontal))
        jump = max(0.0, min(1.0, jump))
        dt = 1.0 / c.fps
        n = replace(s)

        # Engine order: scale normalized action into requested force first,
        # then clamp the force according to the pre-jump grounded state.
        # Example: in air, x=0.5 -> min(100*0.5, 75) = 50, NOT 37.5.
        requested_fx = horizontal * c.ground_force
        force_cap = c.ground_force if n.grounded else c.air_force
        n.vx += max(-force_cap, min(force_cap, requested_fx))
        if n.grounded and jump > 0:
            n.vy = -c.jump_speed * jump
            n.grounded = False

        speed_cap = c.ground_speed_cap if n.grounded else c.air_speed_cap
        n.vx = max(-speed_cap, min(speed_cap, n.vx))
        n.vy += c.gravity * dt
        if abs(n.vx) < c.idle_threshold:
            n.vx = 0.0

        n.left += int(n.vx * dt)  # truncation each internal frame
        n.left = max(0, min(c.world_width - n.width, n.left))
        n.vx *= c.friction

        # Resolve X collisions before Y, matching observed engine ordering.
        for p in self.platforms:
            if self.overlaps(n.bbox, p):
                if n.vx > 0:
                    n.left = p[0] - n.width
                elif n.vx < 0:
                    n.left = p[2]

        n.top += int(n.vy * dt)
        n.grounded = False
        for p in self.platforms:
            if self.overlaps(n.bbox, p):
                if n.vy > 0:
                    n.top = p[1] - n.height
                    n.vy = 0.0
                    n.grounded = True
                elif n.vy < 0:
                    n.top = p[3]
                    n.vy = 0.0
        if not n.grounded and n.vy >= 0 and self._supported(n):
            n.grounded = True
            n.vy = 0.0
        return n

    def step(self, state: MarioState, action: Sequence[float]) -> tuple[MarioState, list[MarioState]]:
        """Return final state and snapshots after every internal physics frame."""
        current = replace(state)
        subframes = []
        for _ in range(self.cfg.frame_skip):
            current = self._internal_step(current, action)
            subframes.append(replace(current))
        return current, subframes

    def predict(self, state: MarioState, actions: Sequence[Sequence[float]]) -> list[MotionSample]:
        """All subframe samples; does not include initial state."""
        samples = []
        current = replace(state)
        internal_index = 0
        for step_i, action in enumerate(actions, start=1):
            current, snapshots = self.step(current, action)
            for s in snapshots:
                internal_index += 1
                samples.append(MotionSample(
                    action_step=step_i, time_s=internal_index / self.cfg.fps,
                    bbox=s.bbox, vx=s.vx, vy=s.vy, grounded=s.grounded,
                ))
        return samples

    def predict_constant_action(self, mario: Sequence[float], action: Sequence[float],
                                steps: int) -> list[MotionSample]:
        if steps < 0:
            raise ValueError('steps must be non-negative')
        return self.predict(MarioState.from_observation(mario), [action] * steps)

    def first_collision(self, samples: Sequence[MotionSample], bullets: Sequence[object]) -> dict | None:
        """First geometry overlap with a tracked active bullet, on internal-frame grid.

        Bullets must provide left/top/right/bottom and effective_vx.
        Treat bullet Y as constant and speed as constant. Does not simulate
        bullet state changes or Mario stomp/invincibility mechanics.
        """
        fps = self.cfg.fps
        for sample in samples:
            frame = round(sample.time_s * fps)
            for bullet in bullets:
                dx = frame * int(float(bullet.effective_vx) / fps)
                bbox = (round(bullet.left) + dx, round(bullet.top),
                        round(bullet.right) + dx, round(bullet.bottom))
                if self.overlaps(sample.bbox, bbox):
                    return {
                        'bullet_id': bullet.bullet_id,
                        'time_s': sample.time_s,
                        'action_step': sample.action_step,
                        'mario_bbox': sample.bbox,
                        'bullet_bbox': bbox,
                    }
        return None
