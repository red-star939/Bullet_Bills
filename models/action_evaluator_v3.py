"""ActionEvaluator v3: gated, multi-action recovery search on the validated world model.

Preserves the v2 baseline by default. If its longer known-world rollout predicts
lethality or an unstable low-altitude terminal, probes alternative two-phase
plans (initial action, then a different continuation). Only the *first* action
is executed; the controller replans at the next observation (receding horizon).

Not a proof of safety: new projectiles, hidden invulnerability timer and star
transitions are unmodelled. Explicitly report unknown outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .action_evaluator import Action, DEFAULT_ACTIONS
from .action_evaluator_v2 import ActionEvaluatorV2, ActionEvaluationV2
from .collision_predictor import BulletState
from .forward_simulator import ForwardResult, ForwardSimulator
from .platform_safety_evaluator import PlatformSafetyEvaluator, PlatformSafetyReport


@dataclass(frozen=True)
class RecoveryPlan:
    first_action: Action
    follow_action: Action
    switch_after_steps: int
    complete: bool
    first_lethal_frame: int | None
    termination_reason: str | None
    final_supported: bool
    final_grounded: bool
    final_edge_clearance_px: int | None
    first_known_collision_frame: int | None
    score: tuple


@dataclass(frozen=True)
class ActionEvaluationV3:
    best_action: Action
    baseline_action: Action
    changed_from_baseline: bool
    change_reason: str
    baseline: ActionEvaluationV2
    baseline_report: PlatformSafetyReport
    recovery_plans: tuple[RecoveryPlan, ...]
    extra_simulations: int
    assumptions: tuple[str, ...]

    @property
    def best(self):
        """Compatibility with v2 callers using `.best.action` is NOT implied."""
        return next((r for r in self.recovery_plans if r.first_action == self.best_action), None)


class ActionEvaluatorV3:
    def __init__(
        self, simulator: ForwardSimulator, *, horizon_steps: int = 8,
        extended_horizon_steps: int = 20, audit_top_k: int = 4,
        candidate_actions: Sequence[Sequence[float]] = DEFAULT_ACTIONS,
        recovery_first_steps: Sequence[int] = (2, 5),
        recovery_top_k: int = 5,
        continuation_actions: Sequence[Sequence[float]] | None = None,
        min_edge_clearance_px: int = 8,
        low_altitude_bottom_px: int = 700,
    ):
        if extended_horizon_steps < horizon_steps or horizon_steps < 1:
            raise ValueError('Invalid horizon configuration')
        if recovery_top_k < 1 or min_edge_clearance_px < 0:
            raise ValueError('Invalid recovery settings')
        if not recovery_first_steps or any(not (1 <= n < extended_horizon_steps) for n in recovery_first_steps):
            raise ValueError('Recovery switch steps must lie inside horizon')
        self.simulator = simulator
        self.horizon_steps = int(horizon_steps)
        self.extended_horizon_steps = int(extended_horizon_steps)
        self.recovery_first_steps = tuple(dict.fromkeys(map(int, recovery_first_steps)))
        self.recovery_top_k = int(recovery_top_k)
        self.min_edge_clearance_px = int(min_edge_clearance_px)
        self.low_altitude_bottom_px = int(low_altitude_bottom_px)
        self.v2 = ActionEvaluatorV2(
            simulator, horizon_steps=horizon_steps,
            extended_horizon_steps=extended_horizon_steps,
            audit_top_k=audit_top_k, candidate_actions=candidate_actions,
            robust_mode='none')
        self.platform = PlatformSafetyEvaluator(simulator, min_edge_clearance_px=min_edge_clearance_px)
        self.continuations: tuple[Action, ...] = tuple(
            dict.fromkeys(self._action(a) for a in (
                continuation_actions if continuation_actions is not None
                else ((0., 0.), (-1., 0.), (1., 0.), (0., 1.),
                      (-1., 1.), (1., 1.)))))
        if not self.continuations:
            raise ValueError('At least one continuation action required')

    def _action(self, action) -> Action:
        force = self.simulator.motion.action_to_force(action)
        del force
        return (float(action[0]), float(action[1]))

    def _analyze(self, observation, actions, bullets):
        result = self.simulator.simulate_observation(observation, actions, bullets=bullets)
        report = self.platform.analyze(result, horizon_steps=len(actions))
        return result, report

    def _needs_recovery(self, result: ForwardResult, report: PlatformSafetyReport) -> str | None:
        if result.first_lethal_event is not None:
            return 'predicted_lethal'
        if not report.completed_known_world:
            return None  # Do not claim to resolve unknown physics.
        if report.final_supported and report.min_edge_clearance_px is not None:
            if report.min_edge_clearance_px < self.min_edge_clearance_px:
                return 'narrow_support'
        if not report.final_grounded and result.mario.bottom >= self.low_altitude_bottom_px:
            return 'low_airborne_terminal'
        return None

    @staticmethod
    def _rank(result: ForwardResult, report: PlatformSafetyReport,
              first_action: Action, follow_action: Action, baseline_action: Action,
              switch_after: int, horizon_frames: int):
        lethal = result.first_lethal_event
        # Complete predictions without lethal outcome take precedence. Unknown
        # stars cannot be promoted to verified-safe alternatives.
        tier = (0 if report.completed_known_world and lethal is None else
                1 if lethal is None else 2)
        first_lethal = lethal.frame if lethal is not None else None
        edge = report.min_edge_clearance_px
        # Require positive platform support for the best *stable* outcome;
        # airborne trajectories can still be viable in a shorter horizon.
        stable = report.final_supported
        return (tier,
                -int(first_lethal if first_lethal is not None else horizon_frames + 1),
                -result.frames_simulated,
                0 if stable else 1,
                -min(max(edge if edge is not None else -48, -48), 48),
                0 if first_action == baseline_action else 1,
                0 if first_action == follow_action else 1,
                switch_after, first_action, follow_action)

    def evaluate(self, observation: Mapping, *,
                 bullets: Iterable[BulletState] | None = None,
                 current_action: Sequence[float] | None = None) -> ActionEvaluationV3:
        known_bullets = tuple(bullets) if bullets is not None else None
        base = self.v2.evaluate(observation, bullets=known_bullets,
                                current_action=current_action)
        selected = base.best_action
        baseline_actions = [selected] * self.extended_horizon_steps
        base_result, base_report = self._analyze(observation, baseline_actions, known_bullets)
        reason = self._needs_recovery(base_result, base_report)
        caveats = ('known_world_only_no_future_random_bullet_spawns',
                   'future_invincibility_and_star_changes_may_be_unknown',
                   'two_phase_plans_replanned_each_action_step',
                   'finite_candidate_search_not_safety_guarantee')
        if reason is None:
            return ActionEvaluationV3(
                selected, selected, False, 'v2_retained_no_recovery_trigger',
                base, base_report, (), 1, caveats)

        ranked = base.baseline.rankings
        firsts = [selected]
        # Even if every constant action is projected lethal, a first-step
        # rescue might exist. Include sampled-lethal candidates too, but no
        # more than the configured budget.
        for r in ranked:
            if r.action not in firsts and len(firsts) < self.recovery_top_k:
                firsts.append(r.action)
        horizon_frames = self.extended_horizon_steps * self.simulator.frame_skip
        baseline_score = self._rank(base_result, base_report, selected, selected,
                                    selected, self.extended_horizon_steps, horizon_frames)
        plans = []
        for initial in firsts:
            for at in self.recovery_first_steps:
                for follow in self.continuations:
                    if initial == selected and follow == selected:
                        continue  # Already simulated baseline.
                    actions = [initial] * at + [follow] * (self.extended_horizon_steps - at)
                    result, report = self._analyze(observation, actions, known_bullets)
                    lethal = result.first_lethal_event
                    score = self._rank(result, report, initial, follow,
                                       selected, at, horizon_frames)
                    plans.append(RecoveryPlan(
                        first_action=initial, follow_action=follow,
                        switch_after_steps=at, complete=report.completed_known_world,
                        first_lethal_frame=lethal.frame if lethal else None,
                        termination_reason=result.termination_reason,
                        final_supported=report.final_supported,
                        final_grounded=report.final_grounded,
                        final_edge_clearance_px=report.min_edge_clearance_px,
                        first_known_collision_frame=(result.first_collision_event.frame
                                                     if result.first_collision_event else None),
                        score=score))
        plans.sort(key=lambda p: p.score)
        candidate = next((p for p in plans if p.complete and p.first_lethal_frame is None), None)
        # Never override v2 with an unverified continuation. When v2 predicts
        # death, replace it only with a known-world full-horizon survivor.
        # When v2 is complete and alive but precarious, require stable support
        # improvement; mere tie breaking cannot override the baseline.
        chosen = selected
        if candidate is not None:
            if base_result.first_lethal_event is not None:
                chosen = candidate.first_action
                decision = 'recovery_known_world_lethal_avoided'
            elif (base_report.completed_known_world and
                  not base_report.final_supported and candidate.final_supported):
                chosen = candidate.first_action
                decision = 'recovery_final_platform_support'
            elif (base_report.completed_known_world and base_report.final_supported and
                  candidate.final_supported and
                  base_report.min_edge_clearance_px is not None and
                  candidate.final_edge_clearance_px is not None and
                  candidate.final_edge_clearance_px >= self.min_edge_clearance_px and
                  base_report.min_edge_clearance_px < self.min_edge_clearance_px):
                chosen = candidate.first_action
                decision = 'recovery_edge_margin'
            else:
                decision = 'v2_retained_no_material_recovery'
        else:
            decision = 'v2_retained_no_complete_safe_recovery'
        if chosen == selected:
            decision = 'v2_retained_' + reason if decision.startswith('recovery_') else decision
        return ActionEvaluationV3(
            best_action=chosen, baseline_action=selected,
            changed_from_baseline=chosen != selected,
            change_reason=decision, baseline=base, baseline_report=base_report,
            recovery_plans=tuple(plans), extra_simulations=len(plans) + 1,
            assumptions=caveats)
