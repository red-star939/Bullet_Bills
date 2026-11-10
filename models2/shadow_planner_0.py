"""Non-invasive diagnostic comparison of the inherited v1 and v2 decisions.

The caller supplies the v1 action and metadata; the shadow planner evaluates the
v2 proposal without changing that action or reading engine-private state.
"""
from __future__ import annotations
import numpy as np
from .altitude_v2 import AdaptiveAltitudeV2, V2Config


class ShadowPlanner:
    def __init__(self, simulator, config: V2Config | None = None):
        self.v2 = AdaptiveAltitudeV2(simulator, config)
        self.reset()

    def reset(self):
        self.v2.reset()
        self.proposals = 0
        self.differences = 0
        self.first_difference_step = None

    def inspect(self, observation, inherited_action, *, step: int):
        baseline = np.asarray(inherited_action, dtype=np.float32).reshape(2).copy()
        candidate, diagnostics = self.v2.decide(observation, baseline)
        different = not np.array_equal(baseline, candidate)
        self.proposals += int(bool(diagnostics.get('altitude_v2_overridden')))
        self.differences += int(different)
        if different and self.first_difference_step is None:
            self.first_difference_step = step
        return baseline, candidate.copy(), dict(diagnostics, shadow_differs=different)
