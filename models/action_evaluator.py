"""Risk-first, known-object action ranking for BulletBill Continuous Stage 2.

This is a deterministic baseline, not a trained policy or a survival-probability
estimator. It ranks a fixed set of candidate actions over a fixed horizon.
No unobserved projectile spawn or future invincibility expiry is predicted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .collision_predictor import BulletState
from .forward_simulator import ForwardResult, ForwardSimulator, RobustForwardResult

Action = tuple[float, float]
DEFAULT_ACTIONS: tuple[Action, ...] = (
    (0., 0.), (-1., 0.), (1., 0.), (0., 1.),
    (-1., 1.), (1., 1.), (-0.5, 0.), (0.5, 0.),
)


@dataclass(frozen=True)
class ActionScore:
    action: Action
    rank: int
    # A class label rather than a probability: the 9 samples have no distribution.
    risk_class: str
    robust_status: str | None
    first_possible_lethal_frame: int | None
    nominal_lethal_frame: int | None
    earliest_known_collision_frame: int | None
    min_horizontal_clearance_px: int | None
    final_x: int
    final_y: int
    simulated_frames: int
    total_horizon_frames: int
    uncertain_environment: bool
    score: tuple[float, ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ActionEvaluation:
    best_action: Action
    rankings: tuple[ActionScore, ...]
    horizon_steps: int
    horizon_seconds: float
    strategy: str
    robust_evaluations: int
    assumptions: tuple[str, ...]

    @property
    def best(self) -> ActionScore:
        return self.rankings[0]


class ActionEvaluator:
    """Rank constant actions by sampled lethal outcomes, then early clearance.

    Risk priority:
      0: no sampled lethal contact and complete known-world prediction
      1: no sampled lethal contact, but prediction stopped at unknown star pickup
      2: at least one sampled trajectory dies (boundary-sensitive or unanimous)
    Any predicted death is prioritized below an action with no sampled death.
    This explicitly makes no quantitative safety guarantee.
    """
    def __init__(self, simulator: ForwardSimulator, *, horizon_steps: int = 8,
                 candidate_actions: Sequence[Sequence[float]] = DEFAULT_ACTIONS,
                 robust_mode: str = 'all', robust_top_k: int = 3):
        if horizon_steps <= 0:
            raise ValueError('horizon_steps must be >= 1')
        if robust_mode not in ('all', 'top_k', 'none'):
            raise ValueError("robust_mode must be 'all', 'top_k' or 'none'")
        if robust_top_k < 1:
            raise ValueError('robust_top_k must be >=1')
        if not candidate_actions:
            raise ValueError('candidate_actions must not be empty')
        self.simulator = simulator
        self.horizon_steps = int(horizon_steps)
        self.robust_mode = robust_mode
        self.robust_top_k = int(robust_top_k)
        actions = []
        for a in candidate_actions:
            checked = self.simulator.motion.action_to_force(a)  # validation
            del checked
            t = tuple(float(v) for v in np.asarray(a, dtype=np.float32))
            if t not in actions:
                actions.append(t)
        self.candidate_actions: tuple[Action, ...] = tuple(actions)

    @staticmethod
    def _clearance(result: ForwardResult) -> int | None:
        """Minimum horizontal separation to vertically-overlapping active shots.

        This is a tie-breaker, not a guarantee of 2D future safety.
        """
        distances = []
        for f in result.frames:
            ml, mt, mr, mb = f.mario_bbox
            for b in f.bullets:
                if b.state != 'active':
                    continue
                bl, bt, br, bb = b.bbox
                if mt < bb and mb > bt:
                    distances.append(max(bl - mr, ml - br, 0))
        return min(distances) if distances else None

    @staticmethod
    def _first_contact(result: ForwardResult) -> int | None:
        e = result.first_collision_event
        return None if e is None else e.frame

    def evaluate(self, observation: Mapping, *,
                 bullets: Iterable[BulletState] | None = None,
                 current_action: Sequence[float] | None = None) -> ActionEvaluation:
        bullet_states = tuple(bullets) if bullets is not None else None
        horizon_frames = self.horizon_steps * self.simulator.frame_skip
        rows = []
        # First compute a cheap nominal screen across *all* choices.
        for action in self.candidate_actions:
            actions = [action] * self.horizon_steps
            pred = self.simulator.simulate_observation(observation, actions,
                                                       bullets=bullet_states)
            rows.append((action, pred))

        def nominal_key(pair):
            a, pred = pair
            unknown = pred.termination_reason == 'unknown_star_pickup'
            lethal = pred.first_lethal_event
            return (2 if lethal else 1 if unknown else 0,
                    -(lethal.frame if lethal else horizon_frames + 1),
                    -pred.frames_simulated,
                    -(self._clearance(pred) or 0),
                    a)

        # Top-k mode is deliberately labeled approximate: unsafe actions among
        # the unexamined candidates may be mis-ranked due to quantization.
        robust_actions = set()
        if self.robust_mode == 'all':
            robust_actions = set(self.candidate_actions)
        elif self.robust_mode == 'top_k':
            robust_actions = {a for a, _ in sorted(rows, key=nominal_key)[:self.robust_top_k]}
        scores: list[ActionScore] = []
        for action, pred in rows:
            robust: RobustForwardResult | None = None
            if action in robust_actions:
                robust = self.simulator.simulate_observation_robust(
                    observation, [action] * self.horizon_steps,
                    bullets=bullet_states)
            lethal_frames = (robust.lethal_frames if robust is not None else
                             (pred.first_lethal_event.frame if pred.first_lethal_event else None,))
            lethal_found = [x for x in lethal_frames if x is not None]
            any_lethal = bool(lethal_found)
            earliest = min(lethal_found) if lethal_found else None
            any_unknown = (pred.termination_reason == 'unknown_star_pickup' or
                           (robust is not None and any(
                               r.termination_reason == 'unknown_star_pickup' for r in robust.samples)))
            risk_class = ('sampled_lethal' if any_lethal else
                          'unknown_future_star' if any_unknown else 'none_sampled')
            tier = 2 if any_lethal else 1 if any_unknown else 0
            # Prefer no lethal; if all actions lethal, prefer the one with
            # *latest earliest* death. For nonlethal prefer longer validated
            # horizon and farther current-bullet clearance.
            clearance = self._clearance(pred)
            sticky = 0.0
            if current_action is not None:
                ca = tuple(float(v) for v in current_action)
                sticky = abs(ca[0] - action[0]) + abs(ca[1] - action[1])
            # Stillness as final tie-break encourages avoiding unnecessary
            # movement when the known world has no significant threat.
            score = (float(tier),
                     -float(earliest if earliest is not None else horizon_frames + 1),
                     -float(pred.frames_simulated),
                     -float(clearance if clearance is not None else 10_000),
                     float(sticky), abs(action[0]) + abs(action[1]),
                     action[0], action[1])
            assumptions = pred.assumptions
            if robust is not None:
                assumptions = tuple(dict.fromkeys(assumptions + robust.assumptions))
            scores.append(ActionScore(
                action=action, rank=0, risk_class=risk_class,
                robust_status=robust.collision_status if robust is not None else None,
                first_possible_lethal_frame=earliest,
                nominal_lethal_frame=(pred.first_lethal_event.frame
                                     if pred.first_lethal_event else None),
                earliest_known_collision_frame=self._first_contact(pred),
                min_horizontal_clearance_px=clearance,
                final_x=pred.mario.left, final_y=pred.mario.top,
                simulated_frames=pred.frames_simulated,
                total_horizon_frames=horizon_frames,
                uncertain_environment=bool(any_unknown), score=score,
                assumptions=assumptions))
        ordered = sorted(scores, key=lambda item: item.score)
        ranked = tuple(__import__('dataclasses').replace(item, rank=i + 1)
                       for i, item in enumerate(ordered))
        return ActionEvaluation(
            best_action=ranked[0].action, rankings=ranked,
            horizon_steps=self.horizon_steps,
            horizon_seconds=horizon_frames / self.simulator.fps,
            strategy=f'constant_action_{self.robust_mode}',
            robust_evaluations=len(robust_actions),
            assumptions=('no_future_random_bullet_spawns',
                         'finite_velocity_samples_not_probabilities',
                         'constant_candidate_action_over_horizon',
                         'not_an_rl_policy'))
