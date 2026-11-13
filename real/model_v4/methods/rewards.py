"""Configurable survival reward. No game-specific physics assumptions."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RewardConfig:
    per_step: float = 0.01
    death: float = -1.0
    success: float = 2.0
    potential_scale: float = 0.0
    gamma: float = 0.995
    action_cost_scale: float = 0.0

@dataclass
class SurvivalReward:
    config: RewardConfig = RewardConfig()

    def calculate(self, *, terminated=False, success=False, potential_before=0.0,
                  potential_after=0.0, action_cost=0.0):
        c = self.config
        r = c.per_step
        if terminated:
            r += c.success if success else c.death
        r += c.potential_scale * (c.gamma * potential_after - potential_before)
        r -= c.action_cost_scale * action_cost
        return float(r)
