"""Fall-aware conservative extension of the validated constant-action evaluator.

No MPC/beam search.  The v1 action is the default and may only be replaced
when a longer *known-world* trajectory predicts a fall for v1, but not for a
v1-safe alternative.  This is a heuristic, not a safety or survival guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .action_evaluator import Action, ActionEvaluation, ActionEvaluator, DEFAULT_ACTIONS
from .collision_predictor import BulletState
from .forward_simulator import ForwardSimulator


@dataclass(frozen=True)
class CandidateFallAudit:
    action: Action
    short_risk_class: str
    extended_reason: str | None
    extended_frames: int
    extended_horizon_frames: int
    fall_frame: int | None
    bullet_death_frame: int | None
    star_unknown: bool
    final_bottom: int
    final_grounded: bool
    predicted_fall: bool
    verified_no_fall_in_known_world: bool


@dataclass(frozen=True)
class ActionEvaluationV2:
    best_action: Action
    baseline_action: Action
    changed_from_baseline: bool
    change_reason: str
    baseline: ActionEvaluation
    audits: tuple[CandidateFallAudit, ...]
    extra_simulations: int
    extended_horizon_steps: int
    assumptions: tuple[str, ...]

    @property
    def best(self):
        return next((r for r in self.baseline.rankings if r.action == self.best_action),
                    self.baseline.best)


class ActionEvaluatorV2:
    """Evaluate the validated v1 ranking then audit longer-horizon falls.

    Only explicitly predicted FALL events justify overriding v1. Unknown star
    pickups, bullet deaths, and incomplete rollouts do not count as a safe
    alternative. The short-horizon safety class is never degraded.
    """

    def __init__(self, simulator: ForwardSimulator, *, horizon_steps: int = 8,
                 extended_horizon_steps: int = 20, audit_top_k: int = 4,
                 robust_mode: str = 'none', robust_top_k: int = 3,
                 candidate_actions: Sequence[Sequence[float]] = DEFAULT_ACTIONS):
        if extended_horizon_steps < horizon_steps:
            raise ValueError('extended_horizon_steps must be >= horizon_steps')
        if audit_top_k < 1:
            raise ValueError('audit_top_k must be >= 1')
        self.simulator = simulator
        self.extended_horizon_steps = int(extended_horizon_steps)
        self.audit_top_k = int(audit_top_k)
        self.baseline = ActionEvaluator(
            simulator, horizon_steps=horizon_steps, robust_mode=robust_mode,
            robust_top_k=robust_top_k, candidate_actions=candidate_actions)

    def evaluate(self, observation: Mapping, *,
                 bullets: Iterable[BulletState] | None = None,
                 current_action: Sequence[float] | None = None) -> ActionEvaluationV2:
        known_bullets = tuple(bullets) if bullets is not None else None
        base = self.baseline.evaluate(
            observation, bullets=known_bullets, current_action=current_action)
        base_action = base.best_action
        # If v1 already sees death or unknown physics, prefer its established
        # ordering. An extension cannot justify overriding it confidently.
        if base.best.risk_class != 'none_sampled' or self.extended_horizon_steps == base.horizon_steps:
            return ActionEvaluationV2(
                best_action=base_action, baseline_action=base_action,
                changed_from_baseline=False, change_reason='baseline_unmodified',
                baseline=base, audits=(), extra_simulations=0,
                extended_horizon_steps=self.extended_horizon_steps,
                assumptions=('no_future_random_bullet_spawns',
                             'no_extended_audit_when_baseline_uncertain'))

        # Include the selected v1 action and the best few equally short-safe
        # actions; never promote a candidate already predicted lethal by v1.
        finalists = [r for r in base.rankings if r.risk_class == 'none_sampled'][:self.audit_top_k]
        audits = []
        for row in finalists:
            pred = self.simulator.simulate_observation(
                observation, [row.action] * self.extended_horizon_steps,
                bullets=known_bullets)
            fall = next((e.frame for e in pred.events if e.kind == 'fall'), None)
            bullet_death = next((e.frame for e in pred.events if e.kind == 'death'), None)
            unknown = pred.termination_reason == 'unknown_star_pickup'
            # The long rollout must reach the horizon to conclude no fall.
            complete = pred.frames_simulated == self.extended_horizon_steps * self.simulator.frame_skip
            audits.append(CandidateFallAudit(
                action=row.action, short_risk_class=row.risk_class,
                extended_reason=pred.termination_reason,
                extended_frames=pred.frames_simulated,
                extended_horizon_frames=self.extended_horizon_steps * self.simulator.frame_skip,
                fall_frame=fall, bullet_death_frame=bullet_death,
                star_unknown=unknown, final_bottom=pred.mario.bottom,
                final_grounded=pred.mario.grounded, predicted_fall=fall is not None,
                verified_no_fall_in_known_world=complete and fall is None and not unknown))

        selected = base_action
        reason = 'baseline_extended_no_confirmed_fall'
        baseline_audit = next((a for a in audits if a.action == base_action), None)
        if baseline_audit and baseline_audit.predicted_fall:
            # Safe means the entire extended prediction ran, no lethal event,
            # and no unknown star, not merely "did not fall before bullet death".
            alternatives = [a for a in audits if a.action != base_action
                            and a.verified_no_fall_in_known_world
                            and a.bullet_death_frame is None]
            if alternatives:
                # Maintain v1's ranking among the candidates, so the fall
                # adjustment is minimally invasive and deterministic.
                selected = alternatives[0].action
                reason = 'avoided_predicted_extended_fall'
            else:
                reason = 'all_audited_alternatives_unresolved_or_unsafe'
        return ActionEvaluationV2(
            best_action=selected, baseline_action=base_action,
            changed_from_baseline=selected != base_action, change_reason=reason,
            baseline=base, audits=tuple(audits), extra_simulations=len(audits),
            extended_horizon_steps=self.extended_horizon_steps,
            assumptions=('no_future_random_bullet_spawns',
                         'extended_constant_action_is_not_a_recovery_plan',
                         'fall_audit_is_known_world_only',
                         'fallback_to_v1_on_uncertainty',
                         'not_an_rl_policy'))
