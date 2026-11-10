"""Inference-only strategic altitude overlay for BulletBill Stage 2.

Uses only public observations, observed bullet vertical distribution and the
existing known-world ForwardSimulator. Does NOT model unseen random spawns,
claim upper area always safe, or use engine-private information.

Controller acts after an existing PPO/Spawn Shield proposal. When there is an
observed lower-half bullet cluster and a comparatively quiet upper half, it
searches for a validated upward route. Otherwise it preserves the action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

from .forward_simulator import ForwardSimulator
from .mario_motion_predictor import MarioState


@dataclass(frozen=True)
class AltitudeConfig:
    midline_y: float = 432.0
    crowd_lower_min: int = 3
    crowd_advantage: int = 2
    trigger_confirm_steps: int = 3
    evaluate_every_steps: int = 5
    rollout_steps: int = 22  # 1.47 simulated seconds at frame_skip=4
    minimum_ascent_px: float = 45.0
    minimum_advantage_px: float = 24.0
    min_final_wall_clearance_px: float = 65.0
    ascent_target_y: float = 415.0
    cooldown_steps: int = 25
    # Counts active-looking OBSERVED bullets; engine-hidden bullet states unknown.


class AdaptiveAltitudeController:
    def __init__(self, simulator: ForwardSimulator, *, config: AltitudeConfig | None = None,
                 candidate_actions: Sequence[Sequence[float]] | None = None):
        self.simulator = simulator
        self.config = config or AltitudeConfig()
        if self.config.rollout_steps < 1 or self.config.trigger_confirm_steps < 1:
            raise ValueError('Invalid altitude controller configuration')
        # None means continuous action control (v10): x variants with full jump.
        # For discrete PPO, use the trained policy's exact 8-action table.
        self.candidate_actions = None if candidate_actions is None else tuple(
            tuple(float(x) for x in a) for a in candidate_actions)
        self.reset()

    def reset(self):
        self.step_index = 0
        self.crowd_streak = 0
        self.cooldown = 0
        self.last_decision_step = -10**9
        self.escapes_attempted = 0
        self.escapes_completed = 0
        self.last_escaped = False

    def _distribution(self, observation: Mapping):
        bullets = np.asarray(observation.get('bullet_bills', []), dtype=np.float32)
        if bullets.size == 0:
            return 0, 0
        bullets = bullets.reshape(-1, bullets.shape[-1])
        upper = lower = 0
        for b in bullets:
            if len(b) < 5 or not np.all(np.isfinite(b[:5])) or b[2] <= b[0] or b[3] <= b[1]:
                continue
            # Assign to midpoint for non-duplicated crowd counts.
            if (float(b[1]) + float(b[3])) / 2 < self.config.midline_y:
                upper += 1
            else:
                lower += 1
        return upper, lower

    def _clearance(self, bbox):
        return min(float(bbox[0]), float(self.simulator.motion.cfg.world_width - bbox[2]))

    def _candidates(self, proposed):
        if self.candidate_actions is None:
            raw = [(float(proposed[0]), 1.0), (-1., 1.), (-.5, 1.), (0., 1.), (.5, 1.), (1., 1.)]
        else:
            raw = [a for a in self.candidate_actions if a[1] > 0.0]
        seen = set()
        for a in raw:
            key = (float(a[0]), float(a[1]))
            if key not in seen:
                seen.add(key)
                yield np.asarray(key, dtype=np.float32)

    def _rollout(self, observation, action):
        result = self.simulator.simulate_observation(
            observation, [action] * self.config.rollout_steps, stop_on_star=True)
        # Reject incomplete or unknown rollouts; unknown star behavior is NOT safe.
        if (result.terminated or result.frames_simulated !=
                self.config.rollout_steps * self.simulator.frame_skip):
            return None
        return result

    def decide(self, observation: Mapping, proposed_action: Sequence[float]):
        """Return (continuous action, diagnostic metadata), resetting per episode."""
        self.step_index += 1
        selected = np.asarray(proposed_action, dtype=np.float32).reshape(2).copy()
        upper, lower = self._distribution(observation)
        m = MarioState.from_observation(observation['mario'])
        center_y = 0.5 * (m.top + m.bottom)
        crowded = (lower >= self.config.crowd_lower_min and
                   lower - upper >= self.config.crowd_advantage)
        self.crowd_streak = self.crowd_streak + 1 if crowded else 0
        meta = dict(altitude_mode='acrobat', altitude_overridden=False,
                    upper_bullets=upper, lower_bullets=lower,
                    altitude_reason='no_escape_trigger', altitude_plans=0)
        if self.cooldown:
            self.cooldown -= 1
            meta['altitude_reason'] = 'cooldown'
            return selected, meta
        if center_y <= self.config.ascent_target_y:
            if self.last_escaped:
                self.escapes_completed += 1
                self.last_escaped = False
            meta['altitude_mode'] = 'upper_hold'
            meta['altitude_reason'] = 'already_upper'
            return selected, meta
        if not crowded or self.crowd_streak < self.config.trigger_confirm_steps:
            return selected, meta
        if self.step_index - self.last_decision_step < self.config.evaluate_every_steps:
            meta['altitude_reason'] = 'interval_gate'
            return selected, meta
        self.last_decision_step = self.step_index
        # Compare candidate routes against the existing Shield proposal.
        baseline = self._rollout(observation, selected)
        if baseline is None:
            meta['altitude_reason'] = 'baseline_uncertain_or_lethal'
            return selected, meta
        baseline_y = 0.5 * (baseline.mario.top + baseline.mario.bottom)
        best = None
        best_gain = float('-inf')
        for candidate in self._candidates(selected):
            if np.array_equal(candidate, selected):
                continue
            result = self._rollout(observation, candidate)
            meta['altitude_plans'] += 1
            if result is None:
                continue
            # Avoid jumping off the screen edge or selecting stationary air paths.
            if self._clearance(result.mario.bbox) < self.config.min_final_wall_clearance_px:
                continue
            final_y = (result.mario.top + result.mario.bottom) / 2.
            ascent = center_y - final_y
            advantage = baseline_y - final_y
            if ascent < self.config.minimum_ascent_px or advantage < self.config.minimum_advantage_px:
                continue
            # Prefer actual ascent; mild penalty for abrupt horizontal switches.
            score = advantage - 4.0 * abs(float(candidate[0] - selected[0]))
            if score > best_gain:
                best_gain = score
                best = (candidate, ascent, advantage)
        if best is None:
            meta['altitude_reason'] = 'no_verified_better_ascent'
            return selected, meta
        candidate, ascent, advantage = best
        self.escapes_attempted += 1
        self.last_escaped = True
        # A short cooldown prevents oscillating between independent plans.
        self.cooldown = self.config.cooldown_steps
        meta.update(altitude_mode='escape', altitude_overridden=True,
                    altitude_reason='verified_upper_route',
                    predicted_ascent_px=round(ascent, 1),
                    predicted_advantage_px=round(advantage, 1))
        return candidate.copy(), meta
