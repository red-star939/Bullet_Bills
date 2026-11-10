"""Observation-only approximate closed-loop PPO rollout (v4.1).

Runs one actual proposed action in the collaborator's known-world Numba
simulator, reconstructs a *synthetic* public observation, then queries a
SEPARATE FeatureExtractor and deterministic PPO actor at each next step.
The real agent's FeatureExtractor/weights and environment are untouched.

Important limitations: imagined observations omit unknown future spawns and the
rest of the deployed shield/altitude pipeline; the counterfactual is a model,
not an actual environment branch.  It is never claimed to be ground truth.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


def valid_bullets(observation):
    x = np.asarray(observation.get('bullet_bills', []), dtype=np.float64)
    if x.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    x = x.reshape(-1, x.shape[-1])
    if x.shape[1] < 5:
        return np.empty((0, 5), dtype=np.float64)
    valid = np.isfinite(x[:, :5]).all(1) & (x[:, 2] > x[:, 0]) & (x[:, 3] > x[:, 1])
    return np.ascontiguousarray(x[valid, :5])


def valid_stars(observation):
    x = np.asarray(observation.get('stars', []), dtype=np.float64)
    if x.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    x = x.reshape(-1, x.shape[-1])
    if x.shape[1] < 4:
        return np.empty((0, 4), dtype=np.float64)
    valid = np.isfinite(x[:, :4]).all(1) & (x[:, 2] > x[:, 0]) & (x[:, 3] > x[:, 1])
    return np.ascontiguousarray(x[valid, :4])


@dataclass
class ModelState:
    # mario [left, top, vx, vy64, grounded, remaining_inv_frames]
    mario: np.ndarray
    bullets: np.ndarray               # live [left, top, right, bottom, vx]
    stars: np.ndarray                 # live bbox
    time_s: float

    @classmethod
    def from_observation(cls, obs, physics, time_s=0.0):
        m = np.asarray(obs['mario'], np.float64)
        if len(m) < 8 or not np.isfinite(m[:8]).all():
            raise ValueError('Invalid Mario observation')
        vy64 = (float(physics.restore_vy(float(m[5]))) if m[7] > 0.5
                else float(m[5]))
        # If already invincible, remaining time is private/unknown. A sentinel
        # 1 frame allows diagnostics, while active overrides are gated elsewhere.
        inv = 1.0 if m[6] > .5 else 0.0
        s = np.array([m[0], m[1], m[4], vy64, float(m[7] < .5), inv], np.float64)
        return cls(s, valid_bullets(obs), valid_stars(obs), float(time_s))

    def observation(self):
        m = self.mario
        mario = np.array([m[0], m[1], m[0] + 48., m[1] + 48., m[2], m[3],
                          float(m[5] > 0), float(m[4] < .5)], np.float32)
        # No fake padding tracks required; collaborator's extractor accepts 0 rows.
        return {'mario': mario,
                'bullet_bills': np.ascontiguousarray(self.bullets, np.float32),
                'stars': np.ascontiguousarray(self.stars, np.float32)}


@dataclass
class OneStepResult:
    state: ModelState | None
    death_frame: int | None
    stomp_count: int
    star_collected: bool


class KnownWorldStepper:
    """Use _run_ext specifically to retain stomped-bullet/star state across steps.

    Calling simulate_trajectory one step at a time would resurrect bullets that
    were stomped in preceding imagined steps: that would not be closed loop.
    """
    def __init__(self, physics):
        if not hasattr(physics, '_run_ext') or not hasattr(physics, 'GMAX'):
            raise ValueError('Collaborator v10 physics with _run_ext required')
        self.physics = physics

    def step(self, state: ModelState, action) -> OneStepResult:
        p = self.physics
        a = np.asarray(action, np.float64).reshape(2)
        if not np.isfinite(a).all() or not (-1 <= a[0] <= 1) or not (0 <= a[1] <= 1):
            raise ValueError('Invalid simulated action')
        b = state.bullets
        ns = len(state.stars)
        nb = len(b)
        out_s = np.zeros(6, np.float64)
        out_bl = np.zeros(nb, np.float64)
        out_alive = np.zeros(nb, np.bool_)
        out_g = np.zeros((p.GMAX, 3), np.float64)
        out_gn = np.zeros((1,), np.int64)
        out_stars = np.zeros(ns, np.bool_)
        star_alive = np.ones(ns, np.bool_)
        m = state.mario
        # Spawn handling is intentionally separate. We do not insert phantom
        # bullets as deterministic actual bullets into the known-world simulator.
        death, _ghost = p._run_ext(
            float(m[0]), float(m[1]), float(m[2]), float(m[3]), bool(m[4] > .5),
            int(m[5]),
            np.ascontiguousarray(b[:, 0] if nb else np.zeros(0), np.float64),
            np.ascontiguousarray(b[:, 1] if nb else np.zeros(0), np.float64),
            np.ascontiguousarray(b[:, 4] if nb else np.zeros(0), np.float64),
            np.ones(nb, np.bool_), np.zeros((p.GMAX, 3), np.float64), 0,
            np.ascontiguousarray(state.stars, np.float64), star_alive,
            np.ascontiguousarray(a.reshape(1, 2)), p.SOLID,
            np.zeros(0, np.int64), 0,
            out_s, out_bl, out_alive, out_g, out_gn, out_stars)
        if death >= 0:
            return OneStepResult(None, int(death), 0, False)
        updated = np.ascontiguousarray(b[out_alive].copy())
        if len(updated):
            updated[:, 0] = out_bl[out_alive]
            updated[:, 2] = updated[:, 0] + 48.
        new_stars = np.ascontiguousarray(state.stars[out_stars].copy())
        stomp_count = max(0, nb - int(out_alive.sum()))
        # nb - alive includes offscreen expiration, so this is a disappearance
        # count upper bound, NOT a proven stomping count.
        return OneStepResult(ModelState(out_s.copy(), updated, new_stars,
                                         state.time_s + 4. / 60.),
                             None, stomp_count, len(new_stars) < ns)


@dataclass
class Rollout:
    first_action: tuple[float, float]
    actions: tuple[tuple[float, float], ...]
    # Initial public bbox followed by one bbox per predicted completed step.
    bboxes: tuple[tuple[float, float, float, float], ...]
    steps_survived: int
    known_death_frame: int | None
    star_uncertain: bool
    final_wall_clearance: float
    approximate: bool = True


class SimulatedPPOTail:
    def __init__(self, physics, model=None, action_predictor=None):
        self.physics = physics
        self.model = model
        self.action_predictor = action_predictor
        if self.model is None and self.action_predictor is None:
            raise ValueError('model or action_predictor required')

    def new_extractor(self):
        if self.action_predictor is not None:
            return None
        return self.physics.FeatureExtractor()

    def next_action(self, simulated_obs, simulated_time, extractor):
        if self.action_predictor is not None:
            a = self.action_predictor(simulated_obs, simulated_time)
        else:
            import torch
            # Each rollout owns a clone extractor; no mutation of live agent FE.
            features, _, _ = extractor(simulated_obs, {'time_elapsed': simulated_time})
            device = next(self.model.parameters()).device
            with torch.no_grad():
                j, move, _ = self.model.act(
                    torch.as_tensor(features, device=device).unsqueeze(0), deterministic=True)
            a = self.physics.to_env_action(move.item(), j.item())
        a = np.asarray(a, dtype=np.float64).reshape(2)
        if not np.isfinite(a).all():
            raise ValueError('PPO tail generated a nonfinite action')
        return np.array([np.clip(a[0], -1., 1.), np.clip(a[1], 0., 1.)], np.float64)


class ClosedLoopPlanner:
    def __init__(self, physics, model=None, *, horizon_steps=12, action_predictor=None):
        if horizon_steps < 2:
            raise ValueError('horizon_steps must be >= 2')
        self.physics = physics
        self.horizon_steps = int(horizon_steps)
        self.stepper = KnownWorldStepper(physics)
        self.tail = SimulatedPPOTail(physics, model, action_predictor)

    def rollout(self, observation, first_action, *, time_s=0.) -> Rollout:
        state = ModelState.from_observation(observation, self.physics, time_s)
        a = np.asarray(first_action, np.float64).reshape(2)
        extractor = self.tail.new_extractor()
        if extractor is not None:
            # Warm up synthetic extractor on the real CURRENT public state.
            extractor(state.observation(), {'time_elapsed': float(time_s)})
        actions = []
        bboxes = [(float(state.mario[0]), float(state.mario[1]),
                   float(state.mario[0]+48.), float(state.mario[1]+48.))]
        star_uncertain = False
        death_frame = None
        for k in range(self.horizon_steps):
            if k:
                a = self.tail.next_action(state.observation(), state.time_s, extractor)
            actions.append((float(a[0]), float(a[1])))
            outcome = self.stepper.step(state, a)
            if outcome.death_frame is not None:
                death_frame = k * 4 + outcome.death_frame
                break
            state = outcome.state
            assert state is not None
            star_uncertain |= outcome.star_collected
            bboxes.append((float(state.mario[0]), float(state.mario[1]),
                           float(state.mario[0]+48.), float(state.mario[1]+48.)))
        wall = min(float(state.mario[0]), 1296. - (float(state.mario[0]) + 48.))
        return Rollout((float(first_action[0]), float(first_action[1])),
                       tuple(actions), tuple(bboxes), len(bboxes)-1, death_frame,
                       star_uncertain, wall)
