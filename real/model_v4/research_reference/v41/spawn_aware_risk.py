"""Public-rule spawn scenario stress test, NOT an exact spawn predictor.

Uses the *published/inferred* spawn cadence from collaborator PPO v10, extended
past its 180 s feature cache, and sweeps several plausible bullet top positions,
sides and integer px/frame speeds. No hidden random seed or private state.
The resulting exposure_fraction is NOT a calibrated probability of death.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SpawnScenarioReport:
    spawn_count: int
    scenario_count: int
    contacts: int
    exposure_fraction: float
    first_contact_frame: int | None
    timing_model_validated_through_s: float = 180.0


class SpawnAwareRisk:
    def __init__(self, physics, *, max_duration_s=180., screen_width=1296,
                 offsets=(-144., -96., -48., 0.), pixels_per_frame=(3., 4., 5.),
                 overlap_margin_px=9.0):
        self.physics = physics
        self.screen_width = float(screen_width)
        self.offsets = tuple(float(x) for x in offsets)
        self.pixels_per_frame = tuple(float(x) for x in pixels_per_frame)
        self.margin = float(overlap_margin_px)
        if not self.offsets or not self.pixels_per_frame or max_duration_s <= 0:
            raise ValueError('invalid spawn scenarios')
        # The collaborator's schedule is sourced/validated only through 180 s.
        # DO NOT silently extrapolate its accelerating cadence to 600 s.
        # A future version can opt into a verified extended engine schedule.
        self.validated_through_s = min(float(max_duration_s), 180.)
        self.frames = np.asarray(physics.spawn_frames(game_duration=self.validated_through_s),
                                 dtype=np.int64)

    def upcoming_frames(self, time_s, horizon_steps):
        base_frame = int(round(float(time_s) * 60.))
        stop_frame = base_frame + int(horizon_steps) * 4
        a = np.searchsorted(self.frames, base_frame, side='right')
        b = np.searchsorted(self.frames, stop_frame, side='right')
        return tuple(int(x - base_frame) for x in self.frames[a:b])

    @staticmethod
    def _bbox_at(bboxes, frame):
        # Linear interpolation between observed 4-frame endpoints: approximate.
        k = max(0, int(frame) // 4)
        frac = (int(frame) % 4) / 4.0
        if k >= len(bboxes)-1:
            return bboxes[-1]
        a, b = bboxes[k], bboxes[k+1]
        return tuple(a[i] * (1. - frac) + b[i] * frac for i in range(4))

    def assess(self, rollout, start_time_s=0.) -> SpawnScenarioReport:
        horizon_frames = 4 * int(rollout.steps_survived)
        spawns = self.upcoming_frames(start_time_s, rollout.steps_survived)
        if horizon_frames <= 0 or not spawns:
            return SpawnScenarioReport(len(spawns), 0, 0, 0.0, None)
        n, contacts, first = 0, 0, None
        for spawn_frame in spawns:
            if spawn_frame > horizon_frames:
                continue
            m_at_spawn = self._bbox_at(rollout.bboxes, spawn_frame)
            # Per collaborator comments the spawned bullet can appear between
            # mario.top - 144 and mario.top; scenario top is a vertical sample.
            for dy in self.offsets:
                bullet_top = max(0., float(m_at_spawn[1]) + dy)
                for side in ('left', 'right'):
                    for speed in self.pixels_per_frame:
                        n += 1
                        start_left = -48. if side == 'left' else self.screen_width
                        sign = 1. if side == 'left' else -1.
                        contact = False
                        for frame in range(spawn_frame, horizon_frames+1):
                            box = self._bbox_at(rollout.bboxes, frame)
                            bl = start_left + sign * speed * (frame - spawn_frame)
                            br = bl + 48.
                            # Bounding box intersection, inflated by small
                            # uncertainty margin for substep reconstruction.
                            if (box[0] < br + self.margin and box[2] > bl - self.margin and
                                box[1] < bullet_top + 48. + self.margin and
                                box[3] > bullet_top - self.margin):
                                contact = True
                                if first is None or frame < first:
                                    first = int(frame)
                                break
                        contacts += int(contact)
        return SpawnScenarioReport(len(spawns), n, contacts,
                                   contacts / max(1, n), first)
