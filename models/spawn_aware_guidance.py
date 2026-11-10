"""Add a conservative spawn-wall guard on top of the existing ModelGuidance.

No neural-network architecture changes. Uses PPO's existing ACTION_TABLE and
our actual ForwardSimulator / PlatformSafetyEvaluator for candidate vetting.
Inference-only; never pass shield-substituted actions as if PPO sampled them in
on-policy training.
"""
from __future__ import annotations

from typing import Mapping

from .model_guidance import ModelGuidance
from .ppo_policy import ACTION_TABLE
from .spawn_risk_evaluator import SpawnRiskEvaluator


class SpawnAwareGuidance(ModelGuidance):
    def __init__(self, *, caution_px=180, critical_px=96,
                 min_improvement=0.14, min_clearance_gain=24,
                 spawn_horizon_steps=12, **kwargs):
        super().__init__(**kwargs)
        if min_improvement < 0 or min_clearance_gain < 0:
            raise ValueError('Invalid minimum improvement')
        self.spawn = SpawnRiskEvaluator(
            self.simulator.motion, caution_px=caution_px,
            critical_px=critical_px, horizon_steps=spawn_horizon_steps)
        self.min_improvement = float(min_improvement)
        self.min_clearance_gain = int(min_clearance_gain)

    def decide(self, observation: Mapping, ppo_action_id: int,
               previous_action=None, mode='shield'):
        if mode != 'spawn_shield':
            return super().decide(observation, ppo_action_id,
                                  previous_action=previous_action, mode=mode)

        # Preserve existing known-world safety shield as the first layer.
        chosen, meta = super().decide(observation, ppo_action_id,
                                      previous_action=previous_action, mode='shield')
        base = self.spawn.evaluate(observation, ACTION_TABLE[chosen])
        info = dict(meta)
        info.update(spawn_score_before=round(base.risk_score, 4),
                    spawn_score_after=round(base.risk_score, 4),
                    spawn_overridden=False,
                    wall_clearance_px=base.initial_clearance_px)

        # Avoid intervening in mid-field movement. A single wall coordinate is
        # not a guaranteed hazard: only act on substantial *predicted* exposure.
        if (base.min_clearance_px >= self.spawn.caution_px
                or base.risk_score < 0.10
                or (not base.approaching_wall
                    and base.initial_clearance_px >= self.spawn.critical_px)):
            return chosen, info

        best_id = chosen
        best = base
        # Only an approved, completed, nonlethal predicted trajectory can be
        # an alternative. Keep the action table identical to the PPO model.
        for i, action in enumerate(ACTION_TABLE):
            if i == chosen:
                continue
            candidate = self.spawn.evaluate(observation, action)
            if candidate.risk_score >= best.risk_score:
                continue
            safe, _, report = self.check_action(observation, i)
            if not safe or not report.completed_known_world:
                continue
            # Reject airborne-at-horizon? No: jump trajectories can safely be
            # airborne. Only explicitly known lethal events are vetoed.
            best_id, best = i, candidate

        # Hysteresis: prefer original policy unless escape is meaningful.
        improved = (base.risk_score - best.risk_score >= self.min_improvement
                    and best.final_clearance_px >= base.final_clearance_px + self.min_clearance_gain)
        if best_id != chosen and improved:
            info.update(source='spawn_wall_guard', overridden=(best_id != int(ppo_action_id)),
                        spawn_overridden=True, reason='reduced_wall_spawn_exposure',
                        spawn_score_after=round(best.risk_score, 4),
                        wall_clearance_after_px=best.final_clearance_px,
                        original_shield_action=chosen)
            return best_id, info
        return chosen, info
