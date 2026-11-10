"""Observation-only tracker/risk/lane integration for V10 hybrid inference.

Risk guard is opt-in; baseline shield and altitude are not modified.
Engine-private collision evidence belongs exclusively to the diagnostic viewer.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .bullet_tracker import BulletTracker
from .lane_manager import LaneManager
from .risk_estimator import analyze_bullets


@dataclass(frozen=True)
class RiskGuardConfig:
    danger_ttc_s: float = 0.8
    min_visible_bullets: int = 3
    min_risk_improvement: float = 0.16
    horizon_steps: int = 12
    cooldown_steps: int = 8
    min_wall_clearance_px: float = 75.0


class V10RiskIntegration:
    def __init__(self, simulator, config: RiskGuardConfig | None = None):
        self.simulator = simulator
        self.config = config or RiskGuardConfig()
        self.tracker = BulletTracker(lane_manager=LaneManager(), fps=60, frame_skip=4)
        self.lanes = LaneManager()
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.cooldown = 0
        self.last_summary = {}
        self.guard_interventions = 0

    def observe(self, obs, time_elapsed=None):
        tracks = self.tracker.update(obs, time_elapsed=time_elapsed)
        hazards = analyze_bullets(obs, tracked_bullets=tracks)
        valid = [x for x in hazards if math.isfinite(x['ttc'])]
        upper = sum((x['lane'] is not None and x['lane'] < 9 for x in hazards))
        lower = len(hazards) - upper
        lane_load = [0] * self.lanes.num_lanes
        lane_risk = [0.0] * self.lanes.num_lanes
        for h in hazards:
            for li in h['occupied_lanes']:
                lane_load[li] += 1
                lane_risk[li] = max(lane_risk[li], float(h['risk']))
        self.last_summary = {
            'visible_bullets': len(hazards),
            'new_bullets': sum(bool(t.is_new) for t in tracks),
            'upper_bullets': upper, 'lower_bullets': lower,
            'min_ttc_s': min((float(x['ttc']) for x in valid), default=None),
            'max_ttc_risk': max((float(x['risk']) for x in hazards), default=0.0),
            'lane_load': lane_load, 'lane_risk': lane_risk,
            'max_lane_load': max(lane_load, default=0),
            'dangerous_tracks': [
                {'track_id': int(x['bullet_id']), 'ttc_s': float(x['ttc']),
                 'risk': float(x['risk']), 'lane': x['lane'],
                 'first_seen_s': x['first_seen_time'],
                 'initial_ttc_s': None if x['initial_ttc'] is None or not math.isfinite(x['initial_ttc']) else float(x['initial_ttc'])}
                for x in hazards[:5]
            ],
        }
        return self.last_summary

    def _route(self, obs, action):
        action = np.asarray(action, dtype=np.float32)
        sim = self.simulator.simulate_observation(obs, [action] * self.config.horizon_steps,
                                                  stop_on_star=True)
        if sim.terminated or sim.frames_simulated != self.config.horizon_steps * self.simulator.frame_skip:
            return None
        width = self.simulator.motion.cfg.world_width
        clearance = min(sim.mario.left, width - sim.mario.right)
        if clearance < self.config.min_wall_clearance_px:
            return None
        # Dynamic lane occupancy: a future path through currently crowded lanes
        # is not guaranteed dangerous, but it is a modest routing penalty.
        lane_penalty = 0.0
        for f in sim.frames[::4]:
            center = (f.mario_bbox[1] + f.mario_bbox[3]) / 2
            li = self.lanes.get_lane(center)
            if li is not None:
                lane_penalty += (self.last_summary['lane_load'][li] +
                                 1.5 * self.last_summary['lane_risk'][li])
        lane_penalty /= max(1, len(sim.frames[::4]))
        return lane_penalty, sim

    def decide(self, obs, proposed_action, *, mode='observe'):
        if mode not in ('observe', 'guard'):
            raise ValueError('risk mode must be observe or guard')
        action = np.asarray(proposed_action, dtype=np.float32).reshape(2).copy()
        meta = {'risk_guard_overridden': False, 'risk_guard_reason': 'observe_only'}
        if mode == 'observe':
            return action, meta
        if self.cooldown:
            self.cooldown -= 1
            meta['risk_guard_reason'] = 'cooldown'
            return action, meta
        s = self.last_summary
        if (s.get('visible_bullets', 0) < self.config.min_visible_bullets or
            s.get('min_ttc_s') is None or s['min_ttc_s'] > self.config.danger_ttc_s):
            meta['risk_guard_reason'] = 'no_congested_imminent_contact'
            return action, meta
        base = self._route(obs, action)
        if base is None:
            meta['risk_guard_reason'] = 'baseline_uncertain_or_lethal'
            return action, meta
        base_cost = base[0]
        best_cost, best_action = base_cost, action
        # Retain the v10 continuous jump amplitude and consider escape horizontally;
        # a single upward-jump alternative is examined only if the original has no jump.
        choices = [np.asarray([x, action[1]], dtype=np.float32)
                   for x in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        if action[1] < 0.05:
            choices.extend([np.asarray([x, 1.0], dtype=np.float32) for x in (-0.5, 0.0, 0.5)])
        for cand in choices:
            if np.array_equal(cand, action):
                continue
            route = self._route(obs, cand)
            if route is None:
                continue
            # Resist needlessly replacing an acrobatic action for tiny score gains.
            cost = route[0] + 0.10 * abs(float(cand[0] - action[0]))
            if cost < best_cost:
                best_cost, best_action = cost, cand
        if best_cost < base_cost - self.config.min_risk_improvement:
            self.guard_interventions += 1
            self.cooldown = self.config.cooldown_steps
            meta = {'risk_guard_overridden': True, 'risk_guard_reason': 'lower_observed_lane_exposure',
                    'risk_cost_before': round(base_cost, 4), 'risk_cost_after': round(best_cost, 4)}
        else:
            meta['risk_guard_reason'] = 'no_verified_better_route'
        return best_action.copy(), meta
