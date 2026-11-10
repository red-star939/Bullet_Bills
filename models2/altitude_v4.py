"""Inherited v1->v2->v3->v4 experimental controller.

Uses collaborator's frame-accurate Numba rollout to *rank* finite-horizon action
sequences. Only known observed bullets are simulated; unobserved spawns are NOT
covered. v1 is always computed first and is the fallback. Does not change PPO
weights or observation dimensions. No claim of guaranteed improvement.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class V4Config:
    mode: str = 'shadow'  # shadow / active / explore
    horizon_steps: int = 24
    prefix_steps: int = 3
    replan_every: int = 3
    minimum_bullets: int = 6
    urgent_ttc_s: float = 1.25
    urgent_death_frames: int = 72
    crowd_lower_min: int = 11
    crowd_advantage: int = 5
    min_escape_altitude_px: float = 72.0
    min_wall_clearance_px: float = 66.0
    min_survival_gain_frames: int = 24
    cooldown_steps: int = 5
    max_candidates: int = 100
    min_score_advantage: float = 27.0


class AdaptiveAltitudeV4:
    def __init__(self, simulator, ppo_physics, *, config: V4Config | None = None,
                 v1=None, v2=None):
        self.sim = simulator
        self.physics = ppo_physics
        self.cfg = config or V4Config()
        if self.cfg.mode not in ('shadow', 'active', 'explore'):
            raise ValueError('mode: shadow/active/explore')
        if not 1 <= self.cfg.prefix_steps < self.cfg.horizon_steps:
            raise ValueError('invalid rollout dimensions')
        if v1 is None:
            from models.adaptive_altitude_controller import AdaptiveAltitudeController
            v1 = AdaptiveAltitudeController(simulator)
        self.v1 = v1
        self.v2 = v2  # inherited diagnostic, intentionally NOT an authority to override
        self.reset()

    def reset(self):
        self.v1.reset()
        if self.v2 is not None:
            self.v2.reset()
        self.step = 0
        self.cooldown = 0
        self.overrides = 0
        self.proposals = 0
        self.plans_tested = 0
        self.reason_counts = {}
        self.last_report = {}

    @staticmethod
    def _valid_bullets(obs):
        a = np.asarray(obs.get('bullet_bills', []), dtype=np.float64)
        if not a.size:
            return np.zeros((0, 5), np.float64)
        a = a.reshape(-1, a.shape[-1])
        if a.shape[1] < 5:
            return np.zeros((0, 5), np.float64)
        ok = np.isfinite(a[:, :5]).all(axis=1) & (a[:, 2] > a[:, 0]) & (a[:, 3] > a[:, 1])
        return np.ascontiguousarray(a[ok, :5], dtype=np.float64)

    @staticmethod
    def _stars(obs):
        a = np.asarray(obs.get('stars', []), dtype=np.float64)
        if not a.size:
            return np.zeros((0, 4), np.float64)
        a = a.reshape(-1, a.shape[-1])
        ok = np.isfinite(a[:, :4]).all(axis=1) & (a[:, 2] > a[:, 0]) & (a[:, 3] > a[:, 1])
        return np.ascontiguousarray(a[ok, :4], dtype=np.float64)

    @staticmethod
    def _ttc(m, bullets):
        if len(bullets) == 0:
            return float('inf')
        mx = (float(m[0]) + float(m[2])) / 2
        bx = (bullets[:, 0] + bullets[:, 2]) / 2
        dx = bx - mx
        closing = -np.sign(dx) * (bullets[:, 4] - float(m[4]))
        gap = np.maximum(0.0, np.abs(dx) - 48.0)
        overlaps_y = (bullets[:, 1] < float(m[3]) + 72) & (bullets[:, 3] > float(m[1]) - 72)
        possible = (closing > 1e-3) & overlaps_y
        if not possible.any():
            return float('inf')
        return float(np.min(gap[possible] / closing[possible]))

    def _candidates(self, baseline):
        """A branching *two phase* plan; only first step is actually executed."""
        h = self.cfg.horizon_steps
        p = self.cfg.prefix_steps
        moves = (float(baseline[0]), -1., -.65, 0., .65, 1.)
        jumps = (float(baseline[1]), 0., .5, .75, 1.)
        seen = set()
        seqs = []
        firsts = []
        labels = []
        def add(first, follow, n, label):
            first = np.asarray(first, dtype=np.float64)
            follow = np.asarray(follow, dtype=np.float64)
            seq = np.zeros((h, 2), dtype=np.float64)
            seq[:n] = first
            seq[n:] = follow
            key = seq.tobytes()
            if key in seen:
                return
            seen.add(key)
            seqs.append(seq)
            firsts.append(first.copy())
            labels.append(label)
        # PPO first action retained and a neutral continuation for fair reference.
        add(baseline, baseline, h, 'v1_constant')
        add(baseline, (0., 0.), p, 'v1_coast')
        # Existing v1 action then trajectory correction, enabling continued acrobatics.
        for next_move in (-1., -.5, 0., .5, 1.):
            add(baseline, (next_move, 0.), p, 'v1_then_turn')
        # Delayed jump and 2-phase horizontal navigation.
        for mv in moves:
            for jp in jumps:
                a = (mv, jp)
                add(a, (0., 0.), p, 'escape_coast')
                add(a, (mv, 0.), min(p * 2, h-1), 'escape_drift')
                add(a, (-mv, 0.), p, 'escape_brake')
        # Rescue when first action is nonjump: wait 1-2 steps then jump.
        for wait in (1, 2):
            for mv in (-1., 0., 1.):
                seq = np.zeros((h, 2), np.float64)
                seq[:wait, 0] = mv
                seq[wait, :] = (mv, 1.)
                seq[wait+1:wait+p, 0] = mv
                seq[wait+p:, 0] = 0.
                key = seq.tobytes()
                if key not in seen:
                    seen.add(key)
                    seqs.append(seq)
                    firsts.append(seq[0].copy())
                    labels.append('delayed_jump')
        return (np.ascontiguousarray(seqs[:self.cfg.max_candidates], dtype=np.float64),
                firsts[:self.cfg.max_candidates], labels[:self.cfg.max_candidates])

    def _record(self, reason):
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def decide(self, observation, shield_action, *, risk=None):
        self.step += 1
        original = np.asarray(shield_action, dtype=np.float32).reshape(2).copy()
        baseline, v1_meta = self.v1.decide(observation, original)
        baseline = np.asarray(baseline, dtype=np.float32).reshape(2).copy()
        meta = {'v4_mode': self.cfg.mode, 'v4_overridden': False,
                'v4_reason': 'not_triggered', 'v1_action': baseline.tolist(),
                'v1_overridden': bool(v1_meta.get('altitude_overridden', False)),
                'v1_reason': v1_meta.get('altitude_reason', ''),
                'altitude_overridden': not np.array_equal(original, baseline),
                'v4_proposed': False, 'v4_action': baseline.tolist(),
                'plans_tested': 0, 'baseline_death_frame': None,
                'best_death_frame': None, 'score_gain': None}
        if self.cooldown:
            self.cooldown -= 1
            meta['v4_reason'] = 'cooldown'
            self._record('cooldown')
            return baseline, meta
        if self.step % self.cfg.replan_every:
            return baseline, meta
        m = np.asarray(observation['mario'], np.float64)
        bullets = self._valid_bullets(observation)
        if len(bullets) < self.cfg.minimum_bullets:
            return baseline, meta
        upper = int(np.sum((bullets[:, 1] + bullets[:, 3]) / 2 < 432))
        lower = len(bullets) - upper
        cy = (float(m[1]) + float(m[3])) / 2
        ttc = self._ttc(m, bullets)
        congested = lower >= self.cfg.crowd_lower_min and lower - upper >= self.cfg.crowd_advantage and cy > 475
        urgent = ttc <= self.cfg.urgent_ttc_s
        if not (urgent or congested):
            return baseline, meta
        stars = self._stars(observation)
        # Any star near Mario can alter the untracked duration of invincibility.
        # Avoid optimistic rescue claims on those states.
        if len(stars) and np.any((stars[:, 0] < m[2]+72) & (stars[:, 2] > m[0]-72) &
                                 (stars[:, 1] < m[3]+72) & (stars[:, 3] > m[1]-72)):
            meta['v4_reason'] = 'star_nearby_uncertain'
            self._record(meta['v4_reason'])
            return baseline, meta
        sequences, firsts, labels = self._candidates(baseline)
        if len(sequences) == 0:
            return baseline, meta
        v0 = self.physics.restore_vy(float(m[5])) if m[7] > 0.5 else float(m[5])
        m5 = np.asarray([m[0], m[1], m[4], v0, float(m[7] < .5)], np.float64)
        # Conservatively ignore active star protection; don't count on invincibility.
        times, causes = self.physics.simulate_plans(m5, bullets, stars, 0, sequences, self.physics.SOLID)
        self.plans_tested += len(sequences)
        meta['plans_tested'] = len(sequences)
        total_frames = self.cfg.horizon_steps * 4
        survival = np.where(times < 0, total_frames, times)
        # Baseline evaluates both constant and release/coast continuation.
        base_survival = max(int(survival[0]), int(survival[1]))
        meta['baseline_death_frame'] = None if base_survival == total_frames else base_survival
        # Full horizon survivability for alternatives; simulation cannot predict random future spawns.
        survivors = np.where(times < 0)[0]
        if len(survivors) == 0:
            meta['v4_reason'] = 'no_complete_candidate'
            self._record(meta['v4_reason'])
            return baseline, meta
        best_idx = None
        best_score = float('-inf')
        baseline_score = float('-inf')
        for bi in (0, 1):
            if times[bi] < 0:
                bt, d, c = self.physics.simulate_trajectory(m5, bullets, stars, 0, sequences[bi], self.physics.SOLID)
                if d < 0 and np.all(np.isfinite(bt[-1])):
                    f = bt[-1]
                    bcx = cy - (float(f[1])+24.)
                    bw = min(float(f[0]), 1296-float(f[0])-48)
                    baseline_score = max(baseline_score, .25*bcx + .024*bw)
        base_first = baseline.astype(np.float64)
        for i in survivors:
            first = np.asarray(firsts[int(i)], np.float64)
            if np.allclose(first, base_first):
                continue
            trajectory, death, cause = self.physics.simulate_trajectory(m5, bullets, stars, 0,
                                                                          sequences[int(i)], self.physics.SOLID)
            if death >= 0 or not np.all(np.isfinite(trajectory[-1])):
                continue
            last = trajectory[-1]
            final_x, final_y = float(last[0]), float(last[1])
            wall = min(final_x, 1296 - final_x - 48)
            if wall < self.cfg.min_wall_clearance_px:
                continue
            ascent = cy - (final_y + 24.)
            # Avoid stepping into wall / excessive direction changes; prioritize survival.
            deviation = abs(first[0]-baseline[0])*22.0 + abs(first[1]-baseline[1])*23.0
            score = 0.25 * ascent + 0.024*wall - deviation
            # Coherent direction transitions are preferred over violent reversals.
            if np.sign(first[0]) != np.sign(baseline[0]) and abs(float(baseline[0])) > .25:
                score -= 18.0
            # A more elevated terminal position is valuable only in lower congestion.
            if congested and ascent >= self.cfg.min_escape_altitude_px:
                score += 20.0
            if score > best_score:
                best_score, best_idx = score, int(i)
        if best_idx is None:
            meta['v4_reason'] = 'no_verified_better_trajectory'
            self._record(meta['v4_reason'])
            return baseline, meta
        alt = np.asarray(firsts[best_idx], np.float32)
        # If baseline survives complete rollout, don't intervene merely for altitude
        # in normal active mode. explore mode is opt-in and still requires crowding.
        rescue = (base_survival < total_frames and
                  total_frames - base_survival >= self.cfg.min_survival_gain_frames and
                  base_survival <= self.cfg.urgent_death_frames)
        advantage = best_score - baseline_score if np.isfinite(baseline_score) else float('-inf')
        strategic = (self.cfg.mode == 'explore' and congested and base_survival == total_frames
                     and advantage >= self.cfg.min_score_advantage
                     and abs(float(alt[0] - baseline[0])) <= 1.0)
        meta.update(v4_proposed=True, v4_action=alt.tolist(),
                    best_death_frame=None, score_gain=round(advantage, 3) if np.isfinite(advantage) else None,
                    v4_reason='rescue_candidate' if rescue else 'strategic_candidate' if strategic else 'proposal_rejected')
        self.proposals += 1
        if self.cfg.mode == 'shadow' or not (rescue or strategic):
            self._record(meta['v4_reason'])
            return baseline, meta
        # In active mode prevent replacing v1's own corrective action except in
        # a truly imminent, directly predicted rescue.
        if meta['v1_overridden'] and not (rescue and base_survival <= 20):
            meta['v4_reason'] = 'v1_intervention_protected'
            self._record(meta['v4_reason'])
            return baseline, meta
        self.overrides += 1
        self.cooldown = self.cfg.cooldown_steps
        meta.update(v4_overridden=True, v4_reason='known_world_trajectory_rescue' if rescue else 'strategic_upper_escape',
                    altitude_overridden=True)
        self._record(meta['v4_reason'])
        return alt, meta
