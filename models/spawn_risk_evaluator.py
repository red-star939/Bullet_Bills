"""Conservative, geometry-only risk of staying close to unseen bullet spawns.

This is NOT a calibrated bullet spawn probability. Both wall endpoints are
potential launch boundaries. Candidate motion is projected using the project's
validated MarioMotionPredictor and compared at identical horizon lengths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .mario_motion_predictor import MarioMotionPredictor, MarioState


@dataclass(frozen=True)
class SpawnRiskReport:
    initial_clearance_px: int
    min_clearance_px: int
    final_clearance_px: int
    near_wall_frames: int
    horizon_frames: int
    nearest_wall: str
    risk_score: float
    approaching_wall: bool
    assumptions: tuple[str, ...] = (
        'wall_proximity_is_exposure_proxy_not_spawn_probability',
        'future_bullet_spawn_height_speed_and_timing_unknown',
        'constant_action_prediction_only',
        'wall_risk_alone_does_not_prove_bullet_collision',
    )


class SpawnRiskEvaluator:
    def __init__(self, motion: MarioMotionPredictor, *,
                 caution_px: int = 180, critical_px: int = 96,
                 horizon_steps: int = 12):
        if not 0 < critical_px < caution_px:
            raise ValueError('Require 0 < critical_px < caution_px')
        if horizon_steps < 1:
            raise ValueError('horizon_steps must be positive')
        self.motion = motion
        self.caution_px = int(caution_px)
        self.critical_px = int(critical_px)
        self.horizon_steps = int(horizon_steps)

    def clearance(self, bbox: Sequence[int]) -> int:
        # The position is the Mario AABB, not the center; use edge clearance.
        return max(0, min(int(bbox[0]), self.motion.cfg.world_width - int(bbox[2])))

    def evaluate(self, observation: Mapping, action: Sequence[float]) -> SpawnRiskReport:
        mario = MarioState.from_observation(observation['mario'])
        initial = self.clearance(mario.bbox)
        samples = self.motion.predict(mario, [action] * self.horizon_steps)
        clearances = [self.clearance(f.bbox) for f in samples]
        minimum = min(clearances) if clearances else initial
        final = clearances[-1] if clearances else initial
        near_frames = sum(d < self.caution_px for d in clearances)
        # Sum exposure over the entire trajectory; near-critical exposure has
        # higher weight. Not probability, and should not be used as PPO reward.
        exposure = sum(max(0., (self.caution_px - d) / self.caution_px) for d in clearances)
        critical_exposure = sum(max(0., (self.critical_px - d) / self.critical_px)
                                for d in clearances)
        horizon = max(1, len(clearances))
        score = (exposure + critical_exposure) / horizon
        l, _, r, _ = mario.bbox
        nearest = 'left' if l <= self.motion.cfg.world_width - r else 'right'
        return SpawnRiskReport(
            initial_clearance_px=initial,
            min_clearance_px=minimum,
            final_clearance_px=final,
            near_wall_frames=near_frames,
            horizon_frames=len(clearances),
            nearest_wall=nearest,
            risk_score=score,
            approaching_wall=final < initial - 4,
        )
