"""Experimental, observation-only, receding-horizon altitude/acrobatics planner.

NOT a certified collision avoidance system. Only currently observed bullets and
known-world platform physics are simulated. Future random spawns are unknown.
The planner leaves the PPO+Spawn Shield proposal unchanged unless a known-world
unsafe baseline can be improved by a complete and nonlethal route, or a crowded
lower region gives a clearly better *verified* ascent. Never changes weights.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class V2Config:
    horizon_steps: int = 18        # 1.20 s total
    prefix_steps: int = 4          # executed first in simulated sequences
    replan_every: int = 5
    crowd_confirm: int = 3
    lower_min: int = 8
    lower_advantage: int = 5
    target_y: float = 420.0
    min_ascent_px: float = 42.0
    min_advantage_px: float = 35.0
    min_wall_clearance: float = 64.0
    max_candidates: int = 18
    # Do not override an already complete, survivable baseline unless significant
    # additional ascent is verified in lower-half crowd conditions.


class AdaptiveAltitudeV2:
    def __init__(self, simulator, config: V2Config | None = None):
        self.sim = simulator
        self.cfg = config or V2Config()
        if not 0 < self.cfg.prefix_steps < self.cfg.horizon_steps:
            raise ValueError('prefix_steps must be between 1 and horizon_steps')
        self.reset()

    def reset(self):
        self.steps = 0
        self.crowd_streak = 0
        self.last_plan_step = -10**6
        self.overrides = 0
        self.danger_rescues = 0
        self.crowd_escapes = 0
        self.rejections = 0

    @staticmethod
    def _count(obs):
        bb = np.asarray(obs.get('bullet_bills', ()), dtype=np.float32)
        if bb.size == 0:
            return 0, 0
        bb = bb.reshape(-1, bb.shape[-1])
        up = low = 0
        for b in bb:
            if b.size < 5 or not np.isfinite(b[:5]).all() or b[2] <= b[0] or b[3] <= b[1]:
                continue
            if (b[1] + b[3]) / 2 < 432:
                up += 1
            else:
                low += 1
        return up, low

    def _assess(self, obs, actions):
        """None if path is terminated, truncated by unknown star, or incomplete."""
        result = self.sim.simulate_observation(obs, actions, stop_on_star=True)
        self._last_invalid_reason = None
        needed = len(actions) * self.sim.frame_skip
        if result.terminated or result.frames_simulated != needed:
            self._last_invalid_reason = result.termination_reason or 'incomplete'
            return None
        final = result.mario
        clearance = min(float(final.left), float(self.sim.motion.cfg.world_width - final.right))
        if clearance < self.cfg.min_wall_clearance:
            self._last_invalid_reason = 'wall_clearance'
            return None
        return {'y': (final.top + final.bottom) / 2.0, 'clearance': clearance,
                'stomps': sum(e.kind == 'stomp' for e in result.events), 'result': result}

    def _sequences(self, original: np.ndarray):
        """Generate constant and two-phase actions. All actions are legal continuous actions.

        First move is the candidate for the next *one* actual env step; the rest
        is a counterfactual plan re-evaluated next decision.
        """
        c = self.cfg
        first_moves = (float(original[0]), -1.0, -0.5, 0.0, 0.5, 1.0)
        follow_moves = (0.0, -0.7, 0.7)
        seen = set()
        for jump in (float(original[1]), 1.0, 0.75):
            for move in first_moves:
                action = np.asarray([np.clip(move, -1, 1), np.clip(jump, 0, 1)], np.float32)
                if np.array_equal(action, original):
                    continue
                # Keep actual action and branch into a horizontal correction later.
                for follow in (move,) + follow_moves:
                    follow_act = np.asarray([np.clip(follow, -1, 1), 0.0], np.float32)
                    key = (float(action[0]), float(action[1]), float(follow_act[0]))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield action, [action] * c.prefix_steps + [follow_act] * (c.horizon_steps-c.prefix_steps)

    def decide(self, obs: Mapping, proposed_action: Sequence[float]):
        original = np.asarray(proposed_action, np.float32).reshape(2).copy()
        upper, lower = self._count(obs)
        cy = (float(obs['mario'][1]) + float(obs['mario'][3])) / 2
        crowded = lower >= self.cfg.lower_min and lower - upper >= self.cfg.lower_advantage
        self.crowd_streak = self.crowd_streak + 1 if crowded else 0
        self.steps += 1
        info = {'altitude_v2_overridden': False, 'altitude_v2_reason': 'no_trigger',
                'upper_bullets': upper, 'lower_bullets': lower, 'plans_examined': 0,
                'baseline_complete': None, 'route_gain_px': 0.0}
        # Only run expensive planning on confirmed crowding or immediate known-world
        # danger. Check baseline only at each replan interval; avoid per-step overhead.
        if self.steps - self.last_plan_step < self.cfg.replan_every:
            info['altitude_v2_reason'] = 'replan_interval'
            return original, info
        if self.crowd_streak < self.cfg.crowd_confirm or cy <= self.cfg.target_y:
            return original, info
        self.last_plan_step = self.steps
        base_actions = [original] * self.cfg.horizon_steps
        baseline = self._assess(obs, base_actions)
        info['baseline_complete'] = baseline is not None
        # Unknown star pickup is not evidence of danger. Preserve PPO's action,
        # which may intentionally collect a star; never suppress it as a rescue.
        if baseline is None and self._last_invalid_reason == 'unknown_star_pickup':
            info['altitude_v2_reason'] = 'baseline_star_unknown_preserve_ppo'
            return original, info
        # IMPORTANT: even if baseline is lethal, look for a complete safer route.
        best = None
        best_score = float('-inf')
        baseline_y = baseline['y'] if baseline is not None else cy
        for action, seq in self._sequences(original):
            if info['plans_examined'] >= self.cfg.max_candidates:
                break
            info['plans_examined'] += 1
            evaluated = self._assess(obs, seq)
            if evaluated is None:
                continue
            ascent = cy - evaluated['y']
            advantage = baseline_y - evaluated['y']
            if baseline is not None:
                if ascent < self.cfg.min_ascent_px or advantage < self.cfg.min_advantage_px:
                    continue
            else:
                # Require at least nonregression on altitude if the original is
                # predicted lethal. Avoid assuming that no rise is always safe.
                if ascent < -16.0:
                    continue
            # Prioritize intact acrobatic PPO proposal and predicted stomp success.
            deviation = abs(float(action[0] - original[0])) + .3 * abs(float(action[1] - original[1]))
            score = ascent + 6.0 * evaluated['stomps'] - 12.0 * deviation
            if score > best_score:
                best_score = score
                best = (action, evaluated, advantage)
        if best is None:
            self.rejections += 1
            info['altitude_v2_reason'] = ('no_verified_rescue' if baseline is None else 'no_clear_ascent_gain')
            return original, info
        action, evaluated, advantage = best
        self.overrides += 1
        if baseline is None:
            self.danger_rescues += 1
            reason = 'known_world_baseline_incomplete_candidate_complete'
        else:
            self.crowd_escapes += 1
            reason = 'verified_two_phase_ascent'
        info.update(altitude_v2_overridden=True, altitude_v2_reason=reason,
                    predicted_final_y=round(evaluated['y'], 1),
                    route_gain_px=round(advantage, 1),
                    predicted_stomps=evaluated['stomps'])
        return action.copy(), info
