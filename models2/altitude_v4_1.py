"""v4.1 successor: v1->v2->v3->v4, with three requested extensions.

1) Approximate closed-loop PPO continuation on synthetic observations.
2) Multiple plausible future-spawn scenarios (stress score, not probability).
3) Stepwise intervention auditing (wired in runner, not a safety oracle).

Default SHADOW mode cannot alter the inherited action. Active and explore modes
are experimental and do not guarantee a 150-second mean survival.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .altitude_v4 import AdaptiveAltitudeV4, V4Config
from .closed_loop_planner import ClosedLoopPlanner, valid_bullets, valid_stars
from .spawn_aware_risk import SpawnAwareRisk


@dataclass(frozen=True)
class V41Config:
    mode: str = 'shadow'                 # shadow | active | explore
    inherited_v4_mode: str = 'shadow'    # shadow (v1 fallback) | active | explore
    horizon_steps: int = 12             # PPO actions resampled after each imagined step
    max_candidates: int = 8
    replan_every: int = 6
    min_visible_bullets: int = 7
    min_time_for_strategic_s: float = 35.
    urgent_ttc_s: float = 1.3
    crowd_lower_min: int = 9
    crowd_advantage: int = 4
    min_candidate_wall_px: float = 55.
    min_rescue_remaining_steps: int = 2
    max_rescue_spawn_worsening: float = .07
    max_rescue_absolute_spawn_exposure: float = .33
    min_spawn_improvement: float = .15
    min_baseline_spawn_exposure: float = .18
    max_strategic_move_change: float = .72
    cooldown_steps: int = 8
    protect_v1_override: bool = True
    protect_stars: bool = True
    protect_invincible: bool = True


class AdaptiveAltitudeV41:
    def __init__(self, simulator, physics, model=None, *, config=None,
                 v4=None, planner=None, spawn_risk=None, action_predictor=None):
        self.physics = physics
        self.cfg = config or V41Config()
        if self.cfg.mode not in ('shadow','active','explore'):
            raise ValueError('Invalid v4.1 mode')
        if self.cfg.inherited_v4_mode not in ('shadow','active','explore'):
            raise ValueError('Invalid inherited v4 mode')
        if self.cfg.horizon_steps < 2 or self.cfg.max_candidates < 2 or self.cfg.replan_every < 1:
            raise ValueError('Invalid rollout settings')
        self.v4 = v4 or AdaptiveAltitudeV4(
            simulator, physics, config=V4Config(mode=self.cfg.inherited_v4_mode))
        self.planner = planner or ClosedLoopPlanner(
            physics, model, horizon_steps=self.cfg.horizon_steps,
            action_predictor=action_predictor)
        self.spawner = spawn_risk or SpawnAwareRisk(physics)
        self.reset()

    def reset(self):
        self.v4.reset()
        self.step = 0
        self.cooldown = 0
        self.overrides = 0
        self.proposals = 0
        self.rollouts = 0
        self.reason_counts = {}
        self.last_report = {}

    def _reason(self, meta, reason, baseline):
        meta['v41_reason'] = reason
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        self.last_report = meta
        return np.asarray(baseline, np.float32).copy(), meta

    @staticmethod
    def _ttc(m, bullets):
        if not len(bullets):
            return math.inf
        mx = (float(m[0]) + float(m[2])) * .5
        bx = (bullets[:,0] + bullets[:,2]) * .5
        dx = bx - mx
        closing = -np.sign(dx) * (bullets[:,4] - float(m[4]))
        vert = (bullets[:,1] < m[3] + 72) & (bullets[:,3] > m[1] - 72)
        good = (closing > 1e-3) & vert
        return float(np.min(np.maximum(0., np.abs(dx[good])-48.) / closing[good])) if good.any() else math.inf

    def _candidates(self, baseline, v4_meta, grounded):
        # PPO's currently selected jump amplitude is preserved wherever possible.
        a = np.asarray(baseline, np.float32)
        alternatives = [a.copy()]
        if v4_meta.get('v4_proposed'):
            alternatives.append(np.asarray(v4_meta['v4_action'], np.float32))
        for move in (0., -.5, .5, -1., 1., float(np.clip(a[0]-.35,-1,1)),
                     float(np.clip(a[0]+.35,-1,1))):
            alternatives.append(np.asarray([move, a[1]], np.float32))
        if grounded and a[1] <= .01:
            for jump in (.5, .75, 1.):
                alternatives.append(np.asarray([a[0], jump], np.float32))
        uniq, seen = [], set()
        for opt in alternatives:
            opt[0] = np.clip(opt[0], -1., 1.)
            opt[1] = np.clip(opt[1], 0., 1.)
            key = tuple(float(x) for x in opt)
            if key not in seen:
                seen.add(key)
                uniq.append(opt)
        return uniq[:self.cfg.max_candidates]

    def decide(self, observation, shield_action, *, time_s=0., risk=None):
        self.step += 1
        # Inherited v4 runs first, including v1's decision and its state machine.
        baseline, v4meta = self.v4.decide(observation, shield_action, risk=risk)
        baseline = np.asarray(baseline, np.float32).reshape(2).copy()
        meta = {
            'v41_mode': self.cfg.mode, 'v41_overridden': False,
            'v41_reason': 'no_trigger', 'v41_proposed': False,
            'v41_baseline_action': baseline.tolist(),
            'v41_proposal_action': baseline.tolist(),
            'v41_baseline_death_frame': None,
            'v41_candidate_death_frame': None,
            'v41_baseline_spawn_exposure': None,
            'v41_candidate_spawn_exposure': None,
            'v41_candidates_tested': 0,
            'altitude_overridden': bool(v4meta.get('altitude_overridden', False)),
            'v1_reason': v4meta.get('v1_reason',''),
            'v1_overridden': bool(v4meta.get('v1_overridden',False)),
            'inherited_v4_reason': v4meta.get('v4_reason',''),
            'inherited_v4_overridden': bool(v4meta.get('v4_overridden',False)),
        }
        if self.cooldown:
            self.cooldown -= 1
            return self._reason(meta, 'cooldown', baseline)
        if self.step % self.cfg.replan_every:
            return self._reason(meta, 'replan_gate', baseline)
        if float(time_s) + self.cfg.horizon_steps * (4./60.) >= self.spawner.validated_through_s:
            return self._reason(meta, 'spawn_timing_unvalidated_after_180s', baseline)
        m = np.asarray(observation['mario'], np.float64)
        bullets = valid_bullets(observation)
        if len(bullets) < self.cfg.min_visible_bullets:
            return self._reason(meta, 'few_bullets', baseline)
        if self.cfg.protect_invincible and m[6] > .5:
            return self._reason(meta, 'invincibility_unknown', baseline)
        stars = valid_stars(observation)
        if self.cfg.protect_stars and len(stars):
            near = np.any((stars[:,0] < m[2]+80) & (stars[:,2] > m[0]-80) &
                          (stars[:,1] < m[3]+80) & (stars[:,3] > m[1]-80))
            if near:
                return self._reason(meta, 'star_near_uncertain', baseline)
        upper = int(np.sum((bullets[:,1]+bullets[:,3]) / 2. < 432.))
        lower = len(bullets) - upper
        crowd = lower >= self.cfg.crowd_lower_min and lower-upper >= self.cfg.crowd_advantage
        urgent = self._ttc(m, bullets) <= self.cfg.urgent_ttc_s
        upcoming = self.spawner.upcoming_frames(time_s, self.cfg.horizon_steps)
        spawn_near = bool(upcoming and upcoming[0] <= 25)
        if not (urgent or crowd or (spawn_near and time_s >= 50.)):
            return self._reason(meta, 'not_urgent_or_crowded', baseline)
        choices = self._candidates(baseline, v4meta, grounded=m[7] < .5)
        analyzed = []
        try:
            for action in choices:
                rollout = self.planner.rollout(observation, action, time_s=time_s)
                score = self.spawner.assess(rollout, start_time_s=time_s)
                analyzed.append((action.copy(), rollout, score))
        except Exception as exc:
            # No speculative action can be approved after a physics/feature error.
            meta['v41_error'] = f'{type(exc).__name__}: {str(exc)[:180]}'
            return self._reason(meta, 'planner_unavailable', baseline)
        self.rollouts += len(analyzed)
        meta['v41_candidates_tested'] = len(analyzed)
        if len(analyzed) < 2:
            return self._reason(meta, 'no_alternatives', baseline)
        base_action, base_path, base_spawn = analyzed[0]
        meta['v41_baseline_death_frame'] = base_path.known_death_frame
        meta['v41_baseline_spawn_exposure'] = round(base_spawn.exposure_fraction, 4)
        meta['v41_spawn_scenarios'] = base_spawn.scenario_count
        # Never treat a predicted star pickup as a verified safe route.
        feasible = [(a, path, spawn) for a,path,spawn in analyzed[1:]
                    if path.known_death_frame is None and not path.star_uncertain and
                    path.final_wall_clearance >= self.cfg.min_candidate_wall_px and
                    path.steps_survived == self.cfg.horizon_steps]
        if not feasible:
            return self._reason(meta, 'no_known_world_full_horizon_route', baseline)
        # First rank by scenario exposure, then penalize move/jump discontinuity.
        def rank(item):
            a, path, spawn = item
            dev = (abs(float(a[0]-baseline[0])) * .05 +
                   abs(float(a[1]-baseline[1])) * .08)
            edge = max(0., (90. - path.final_wall_clearance) / 90.) * .10
            return spawn.exposure_fraction + dev + edge
        candidate, path, score = min(feasible, key=rank)
        meta['v41_proposed'] = True
        meta['v41_proposal_action'] = candidate.tolist()
        meta['v41_candidate_death_frame'] = path.known_death_frame
        meta['v41_candidate_spawn_exposure'] = round(score.exposure_fraction, 4)
        meta['v41_candidate_spawn_scenarios'] = score.scenario_count
        self.proposals += 1
        # Rescue condition: genuinely predicted known-world death for PPO tail;
        # candidate survives the whole simulated horizon, but future spawns are
        # *still unknown*. A small stress worsening is tolerated for a rescue.
        rescue = (base_path.known_death_frame is not None and
                  base_path.steps_survived + self.cfg.min_rescue_remaining_steps <= self.cfg.horizon_steps and
                  (score.exposure_fraction <= base_spawn.exposure_fraction + self.cfg.max_rescue_spawn_worsening
                   if base_spawn.scenario_count else
                   score.exposure_fraction <= self.cfg.max_rescue_absolute_spawn_exposure))
        # Strategic path: no known-world predicted death; difference is ONLY
        # scenario exposure. This is deliberately strict and experimental.
        strategic = (base_path.known_death_frame is None and
                     not base_path.star_uncertain and
                     self.cfg.mode == 'explore' and
                     time_s >= self.cfg.min_time_for_strategic_s and
                     (crowd or spawn_near) and
                     base_spawn.exposure_fraction >= self.cfg.min_baseline_spawn_exposure and
                     base_spawn.exposure_fraction - score.exposure_fraction >= self.cfg.min_spawn_improvement and
                     abs(float(candidate[0]-baseline[0])) <= self.cfg.max_strategic_move_change and
                     abs(float(candidate[1]-baseline[1])) <= .01)
        if self.cfg.mode == 'shadow':
            return self._reason(meta, 'shadow_proposal', baseline)
        if not (rescue or strategic):
            return self._reason(meta, 'no_verified_improvement', baseline)
        if (self.cfg.protect_v1_override and meta['v1_overridden'] and
            not (rescue and base_path.known_death_frame <= 12)):
            return self._reason(meta, 'v1_override_protected', baseline)
        self.overrides += 1
        self.cooldown = self.cfg.cooldown_steps
        meta['v41_overridden'] = True
        meta['altitude_overridden'] = True
        reason = 'closed_loop_known_world_rescue' if rescue else 'spawn_scenario_strategic_escape'
        meta['v41_reason'] = reason
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        self.last_report = meta
        return np.asarray(candidate, np.float32).copy(), meta
