"""Inherited v1 -> v2 -> v3 altitude controller.

v1 selects the default action. v2 runs counterfactually. In `shadow` mode the
returned action is BITWISE identical to v1; in `active` mode a v2 action is
admissible only when a short observed-world rollout of v1 predicts imminent
bullet death/fall and v2 has a complete nonlethal rollout. Unknown star pickups,
incomplete predictions or long-range constant-action failure NEVER justify an
intervention. This remains an experimental heuristic, not a safety guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from models.adaptive_altitude_controller import AdaptiveAltitudeController
from .altitude_v2 import V2Config
from .shadow_planner import ShadowPlanner


@dataclass(frozen=True)
class V3Config:
    mode: str = 'shadow'
    imminent_steps: int = 2  # up to 8 engine frames at frame_skip=4
    audit_horizon_steps: int = 6
    min_remaining_wall_clearance_px: float = 75.0
    require_v1_unmodified: bool = True
    active_cooldown_steps: int = 12


class AdaptiveAltitudeV3:
    def __init__(self, simulator, *, v3_config: V3Config | None = None,
                 v2_config: V2Config | None = None, v1=None):
        self.sim = simulator
        self.config = v3_config or V3Config()
        if self.config.mode not in ('shadow', 'active'):
            raise ValueError('mode must be shadow or active')
        if self.config.imminent_steps < 1 or self.config.audit_horizon_steps < self.config.imminent_steps:
            raise ValueError('invalid horizons')
        self.v1 = v1 if v1 is not None else AdaptiveAltitudeController(simulator)
        self.shadow = ShadowPlanner(simulator, v2_config)
        self.reset()

    def reset(self):
        self.v1.reset()
        self.shadow.reset()
        self.steps = 0
        self.overrides = 0
        self.rejected = 0
        self.cooldown = 0

    def _audit(self, obs, action):
        actions = [np.asarray(action, dtype=np.float32).copy()] * self.config.audit_horizon_steps
        result = self.sim.simulate_observation(obs, actions, stop_on_star=True)
        needed = self.config.audit_horizon_steps * self.sim.frame_skip
        reason = str(getattr(result, 'termination_reason', None) or '')
        frames = int(getattr(result, 'frames_simulated', 0))
        if 'star' in reason or 'unknown' in reason:
            status = 'unknown'
        elif bool(getattr(result, 'terminated', False)) and reason in ('bullet_death', 'death', 'fall', 'water'):
            status = 'lethal'
        elif not bool(getattr(result, 'terminated', False)) and frames == needed:
            status = 'complete'
        else:
            status = 'unknown'
        mario = result.mario
        wall = min(float(mario.left), float(self.sim.motion.cfg.world_width - mario.right))
        return status, reason, frames, wall

    def decide(self, observation, shield_action):
        self.steps += 1
        original = np.asarray(shield_action, dtype=np.float32).reshape(2).copy()
        v1_action, v1_meta = self.v1.decide(observation, original)
        v1_action = np.asarray(v1_action, dtype=np.float32).reshape(2).copy()
        _, proposed, v2_meta = self.shadow.inspect(observation, v1_action, step=self.steps)
        different = not np.array_equal(proposed, v1_action)
        meta = {
            'altitude_overridden': not np.array_equal(v1_action, original),
            'v3_overridden': False, 'v3_reason': 'shadow_only',
            'v1_reason': v1_meta.get('altitude_reason'),
            'v1_overridden': bool(v1_meta.get('altitude_overridden', False)),
            'v2_reason': v2_meta.get('altitude_v2_reason'),
            'v2_proposed': different,
            'v2_plans_examined': v2_meta.get('plans_examined', 0),
            'v2_route_gain_px': v2_meta.get('route_gain_px', 0.0),
            'v1_action': v1_action.tolist(), 'v2_action': proposed.tolist(),
            'v3_active_audit': None,
        }
        if self.config.mode == 'shadow':
            return v1_action, meta
        if self.cooldown > 0:
            self.cooldown -= 1
            meta['v3_reason'] = 'cooldown'
            return v1_action, meta
        if not different:
            meta['v3_reason'] = 'no_v2_proposal'
            return v1_action, meta
        if self.config.require_v1_unmodified and meta['v1_overridden']:
            meta['v3_reason'] = 'protect_v1_intervention'
            self.rejected += 1
            return v1_action, meta
        base = self._audit(observation, v1_action)
        meta['v3_active_audit'] = {'v1': base[:3]}
        if base[0] != 'lethal' or base[2] > self.config.imminent_steps * self.sim.frame_skip:
            meta['v3_reason'] = 'no_verified_imminent_v1_death'
            self.rejected += 1
            return v1_action, meta
        alternate = self._audit(observation, proposed)
        meta['v3_active_audit']['v2'] = alternate[:3]
        if alternate[0] != 'complete' or alternate[3] < self.config.min_remaining_wall_clearance_px:
            meta['v3_reason'] = 'alternative_not_verified'
            self.rejected += 1
            return v1_action, meta
        self.overrides += 1
        self.cooldown = self.config.active_cooldown_steps
        meta.update(altitude_overridden=True, v3_overridden=True,
                    v3_reason='imminent_known_world_rescue')
        return proposed.copy(), meta
