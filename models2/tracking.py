"""Risk observation adapter. Does not alter policy actions or use engine-private data."""
from __future__ import annotations
import math
import numpy as np
from models.v10_risk_integration import V10RiskIntegration

class RiskObserverV2:
    def __init__(self, simulator):
        self.backend = V10RiskIntegration(simulator)
        self.reset()

    def reset(self):
        self.backend.reset()
        self.samples = 0
        self.bad_padding_rows = 0

    def observe(self, obs, time_elapsed=None):
        bullets = np.asarray(obs['bullet_bills'], dtype=np.float32).reshape(-1, 5)
        valid = np.isfinite(bullets).all(axis=1) & (bullets[:, 2] > bullets[:, 0]) & (bullets[:, 3] > bullets[:, 1])
        self.bad_padding_rows += int((~valid).sum())
        # Exclude padding before the legacy tracker and risk estimator see it.
        filtered = dict(obs)
        filtered['bullet_bills'] = np.ascontiguousarray(bullets[valid])
        summary = self.backend.observe(filtered, time_elapsed=time_elapsed)
        self.samples += 1
        return summary
