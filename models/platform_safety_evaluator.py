"""Platform-support and landing diagnostics for known-world ForwardSimulator rollouts.

This is an *analysis utility*, not a proof of safe recovery. It uses only static
solid platform AABBs and the predicted Mario path; future random bullets and
unmodeled gameplay events remain unknown. No changes to ActionEvaluator v1/v2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .forward_simulator import ForwardResult, ForwardSimulator

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Landing:
    frame: int
    platform_index: int
    platform_bbox: BBox
    mario_bbox: BBox
    overlap_px: int
    edge_clearance_px: int


@dataclass(frozen=True)
class PlatformSafetyReport:
    frames_simulated: int
    horizon_frames: int
    termination_reason: str | None
    predicted_fall_frame: int | None
    predicted_bullet_death_frame: int | None
    completed_known_world: bool
    initially_supported: bool
    final_supported: bool
    final_grounded: bool
    landings: tuple[Landing, ...]
    # Minimum overlap width on frames with engine-reported grounded state;
    # zero if engine thinks grounded but no solid platform overlaps the feet.
    min_supported_overlap_px: int | None
    # Symmetric horizontal room before leaving the *current* supporting platform.
    # Negative value means the mario rectangle extends over an edge.
    min_edge_clearance_px: int | None
    unsupported_grounded_frames: tuple[int, ...]
    final_support_indices: tuple[int, ...]
    risk_flags: tuple[str, ...]
    assumptions: tuple[str, ...]


class PlatformSafetyEvaluator:
    """Read-only geometric support audit of a forward prediction.

    Edge clearance is diagnostic: overhang can be engine-valid and safe. A
    support candidate requires exact foot/top equality and strict horizontal
    overlap, mirroring MarioMotionPredictor._supported().
    """

    def __init__(self, simulator: ForwardSimulator, *, min_overlap_px: int = 12,
                 min_edge_clearance_px: int = 8):
        if min_overlap_px < 1 or min_edge_clearance_px < 0:
            raise ValueError('Invalid platform safety thresholds')
        self.simulator = simulator
        self.platforms: tuple[BBox, ...] = tuple(
            tuple(map(int, p)) for p in simulator.motion.platforms)
        self.min_overlap_px = int(min_overlap_px)
        self.edge_threshold_px = int(min_edge_clearance_px)

    def supports(self, bbox: BBox) -> tuple[tuple[int, int, int], ...]:
        l, _t, r, b = bbox
        hits = []
        for i, p in enumerate(self.platforms):
            if b != p[1]:
                continue
            overlap = min(r, p[2]) - max(l, p[0])
            if overlap > 0:
                # Signed edge room relative to both sides of the platform;
                # negative if Mario overhangs at either side.
                margin = min(l - p[0], p[2] - r)
                hits.append((i, overlap, margin))
        return tuple(hits)

    def analyze(self, result: ForwardResult, *, initial_bbox: BBox | None = None,
                initial_grounded: bool | None = None,
                horizon_steps: int | None = None) -> PlatformSafetyReport:
        if horizon_steps is None:
            horizon_frames = len(result.frames)
        else:
            if horizon_steps < 0:
                raise ValueError('horizon_steps must be nonnegative')
            horizon_frames = horizon_steps * self.simulator.frame_skip
        initial_support = bool(self.supports(initial_bbox)) if initial_bbox is not None else False
        if initial_grounded is not None:
            initial_support = initial_support and bool(initial_grounded)
        prev_grounded = bool(initial_grounded) if initial_grounded is not None else False
        landings: list[Landing] = []
        min_overlap = None
        min_margin = None
        unsupported = []
        for f in result.frames:
            hits = self.supports(f.mario_bbox)
            if f.grounded:
                if not hits:
                    unsupported.append(f.frame)
                else:
                    best = max(hits, key=lambda x: (x[1], x[2]))
                    min_overlap = best[1] if min_overlap is None else min(min_overlap, best[1])
                    min_margin = best[2] if min_margin is None else min(min_margin, best[2])
                    if not prev_grounded:
                        i, overlap, margin = best
                        landings.append(Landing(f.frame, i, self.platforms[i],
                                                f.mario_bbox, overlap, margin))
            prev_grounded = f.grounded
        final_supports = self.supports(result.mario.bbox)
        final_supported = bool(final_supports and result.mario.grounded)
        fall = next((e.frame for e in result.events if e.kind == 'fall'), None)
        bullet_death = next((e.frame for e in result.events if e.kind == 'death'), None)
        complete = (result.frames_simulated == horizon_frames
                    and result.termination_reason is None)
        flags = []
        if fall is not None:
            flags.append('predicted_fall')
        if bullet_death is not None:
            flags.append('predicted_bullet_death')
        if result.termination_reason == 'unknown_star_pickup':
            flags.append('unknown_star_pickup')
        if unsupported:
            flags.append('grounded_without_static_support')
        if min_overlap is not None and min_overlap < self.min_overlap_px:
            flags.append('narrow_supported_overlap')
        if min_margin is not None and min_margin < self.edge_threshold_px:
            flags.append('near_platform_edge')
        if complete and result.mario.grounded and not final_supported:
            flags.append('final_grounded_unverified')
        if complete and not result.mario.grounded:
            flags.append('airborne_at_horizon')
        return PlatformSafetyReport(
            frames_simulated=result.frames_simulated, horizon_frames=horizon_frames,
            termination_reason=result.termination_reason,
            predicted_fall_frame=fall, predicted_bullet_death_frame=bullet_death,
            completed_known_world=complete, initially_supported=initial_support,
            final_supported=final_supported, final_grounded=result.mario.grounded,
            landings=tuple(landings), min_supported_overlap_px=min_overlap,
            min_edge_clearance_px=min_margin,
            unsupported_grounded_frames=tuple(unsupported),
            final_support_indices=tuple(h[0] for h in final_supports),
            risk_flags=tuple(flags),
            assumptions=tuple(dict.fromkeys(result.assumptions + (
                'geometry_only_not_reachability_proof',
                'future_random_bullet_spawns_unknown',
                'constant_action_rollout_not_replanning',
            ))),
        )

    def evaluate_observation(self, observation: Mapping,
                             actions: Sequence[Sequence[float]], *, bullets=None,
                             stop_on_star: bool = True) -> PlatformSafetyReport:
        result = self.simulator.simulate_observation(
            observation, actions, bullets=bullets, stop_on_star=stop_on_star)
        mario = observation['mario']
        bbox = tuple(round(float(v)) for v in mario[:4])
        grounded = not bool(float(mario[7]))
        return self.analyze(result, initial_bbox=bbox,
                            initial_grounded=grounded, horizon_steps=len(actions))
