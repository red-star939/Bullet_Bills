"""Inference-only bridge: collaborator continuous-action PPO v10 + our safety modules.

Preserves the original continuous move and five jump levels. Replacements are
conservatively vetted by our known-world simulator. This is NOT a probability
model of future bullet spawn and should not be used as on-policy PPO training.
"""
from __future__ import annotations

from typing import Mapping
import numpy as np

from .model_guidance import ModelGuidance
from .spawn_risk_evaluator import SpawnRiskEvaluator


class V10SpawnGuidance:
    def __init__(self, *, map_path='map_data/object_tiles.csv', safety_horizon_steps=12,
                 spawn_horizon_steps=12, caution_px=180, critical_px=96,
                 min_improvement=0.14, min_clearance_gain=24):
        self.guide = ModelGuidance(map_path=map_path, safety_horizon_steps=safety_horizon_steps)
        self.spawn = SpawnRiskEvaluator(self.guide.simulator.motion,
                                        horizon_steps=spawn_horizon_steps,
                                        caution_px=caution_px, critical_px=critical_px)
        self.min_improvement = float(min_improvement)
        self.min_clearance_gain = float(min_clearance_gain)

    def checked(self, observation: Mapping, action):
        """Known-world check of ANY valid continuous action (no action-table quantization)."""
        action = self._action(action)
        report = self.guide.platform.evaluate_observation(
            observation, [action] * self.guide.safety_horizon_steps)
        reason = ('fall' if report.predicted_fall_frame is not None
                  else 'bullet_death' if report.predicted_bullet_death_frame is not None
                  else None)
        return (reason is None and report.completed_known_world), reason, report

    @staticmethod
    def _action(action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != 2 or not np.all(np.isfinite(a)):
            raise ValueError('Action must be two finite floats [move, jump]')
        if not (-1.0001 <= a[0] <= 1.0001 and -0.0001 <= a[1] <= 1.0001):
            raise ValueError('Continuous action outside environment bounds')
        return a.copy()

    def decide(self, observation, v10_action, *, previous_action=None, mode='spawn_shield'):
        if mode not in ('ppo', 'shield', 'spawn_shield', 'teacher'):
            raise ValueError('mode must be ppo/shield/spawn_shield/teacher')
        proposed = self._action(v10_action)
        chosen = proposed.copy()
        info = {'mode': mode, 'source': 'v10', 'overridden': False,
                'spawn_overridden': False, 'known_world_overridden': False}
        if mode == 'ppo':
            return chosen, info
        if mode == 'teacher':
            teacher_id, _ = self.guide.teach(observation, previous_action)
            from .ppo_policy import ACTION_TABLE
            chosen = ACTION_TABLE[teacher_id].copy()
            info.update(source='teacher_v2', overridden=not np.array_equal(chosen, proposed))
            return chosen, info

        safe, reason, _ = self.checked(observation, proposed)
        info['raw_known_world_safe'] = bool(safe)
        info['raw_known_world_reason'] = reason
        if not safe:
            teacher_id, _ = self.guide.teach(observation, previous_action)
            from .ppo_policy import ACTION_TABLE
            alternative = ACTION_TABLE[teacher_id].copy()
            teacher_safe, _, _ = self.checked(observation, alternative)
            if teacher_safe:
                chosen = alternative
                info.update(source='teacher_known_world_shield', known_world_overridden=True)
            else:
                info['source'] = 'v10_no_verified_alternative'

        if mode == 'spawn_shield':
            base = self.spawn.evaluate(observation, chosen)
            info.update(wall_clearance_px=base.initial_clearance_px,
                        spawn_score_before=round(base.risk_score, 4),
                        spawn_score_after=round(base.risk_score, 4))
            if (base.min_clearance_px < self.spawn.caution_px and base.risk_score >= 0.10
                    and (base.approaching_wall or base.initial_clearance_px < self.spawn.critical_px)):
                best_action = chosen
                best_risk = base
                # Preserve collaborator's jump amplitude; adjust only horizontal motion.
                # A gentle centering alternative can be superior to a hard ±1 correction.
                for move in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    candidate = np.asarray([move, chosen[1]], dtype=np.float32)
                    if np.array_equal(candidate, chosen):
                        continue
                    risk = self.spawn.evaluate(observation, candidate)
                    if risk.risk_score >= best_risk.risk_score:
                        continue
                    accepted, _, _ = self.checked(observation, candidate)
                    if accepted:
                        best_action, best_risk = candidate, risk
                if (base.risk_score - best_risk.risk_score >= self.min_improvement
                        and best_risk.final_clearance_px - base.final_clearance_px >= self.min_clearance_gain):
                    chosen = best_action.copy()
                    info.update(source='spawn_wall_guard', spawn_overridden=True,
                                spawn_score_after=round(best_risk.risk_score, 4),
                                wall_clearance_after_px=best_risk.final_clearance_px)

        info['overridden'] = not np.array_equal(proposed, chosen)
        return chosen, info
