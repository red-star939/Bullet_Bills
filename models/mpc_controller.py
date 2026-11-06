"""Bounded beam-search MPC for BulletBill Stage 2.

Uses only public observations and the validated known-object ForwardSimulator.
Action sequences are simulated from the SAME initial observation (not from
rounded end-state observations). Execute just the first action, then replan.
No future random bullet spawn or hidden projectile state is inferred.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

from .action_evaluator import DEFAULT_ACTIONS, ActionEvaluator, Action
from .forward_simulator import ForwardResult, ForwardSimulator


@dataclass(frozen=True)
class MPCPlan:
    best_action: Action
    action_sequence: tuple[Action, ...]
    risk_class: str
    first_lethal_frame: int | None
    known_collision_frame: int | None
    min_horizontal_clearance_px: int | None
    evaluated_nodes: int
    beam_width: int
    horizon_steps: int
    unknown_future: bool
    nominal_score: tuple[float, ...]
    assumptions: tuple[str, ...]


class MPCController:
    """Receding-horizon beam search over piecewise-constant action sequences.

    `segment_steps=(2,3,3)` means the action can change at steps 2 and 5.
    Computation scales as beam_width*branch_count*number_of_segments, not
    branch_count**number_of_segments. This heuristic does NOT guarantee a
    globally optimal strategy or real-world safety.
    """
    def __init__(self, simulator: ForwardSimulator, *,
                 segment_steps: Sequence[int] = (2, 3, 3),
                 beam_width: int = 3,
                 candidate_actions: Sequence[Sequence[float]] = DEFAULT_ACTIONS,
                 robust_finalists: int = 0,
                 max_nodes: int = 1000):
        if not segment_steps or any(int(n) != n or n <= 0 for n in segment_steps):
            raise ValueError('segment_steps must be positive integers')
        if beam_width < 1 or max_nodes < 1 or robust_finalists < 0:
            raise ValueError('beam_width, max_nodes and robust_finalists invalid')
        self.simulator = simulator
        self.segment_steps = tuple(int(n) for n in segment_steps)
        self.beam_width = int(beam_width)
        self.robust_finalists = int(robust_finalists)
        self.max_nodes = int(max_nodes)
        self.horizon_steps = sum(self.segment_steps)
        actions = []
        for a in candidate_actions:
            simulator.motion.action_to_force(a)
            act = tuple(map(float,np.asarray(a,dtype=np.float32)))
            if act not in actions: actions.append(act)
        if not actions: raise ValueError('candidate_actions cannot be empty')
        self.candidate_actions = tuple(actions)

    def _score(self, pred: ForwardResult, actions: Sequence[Action],
               prev_action: Sequence[float] | None, expected_frames: int,
               *, final: bool) -> tuple[float, ...]:
        lethal = pred.first_lethal_event
        unknown = pred.termination_reason == 'unknown_star_pickup'
        # Only completed, known-world paths are considered safe at this depth.
        tier = 2 if lethal else 1 if unknown else 0
        clearance = ActionEvaluator._clearance(pred)
        # Incomplete paths (including lethal) are ranked by how long they survive.
        # Preserve the binary safe/unsafe distinction before any clearance bonus.
        transition_cost = sum(abs(a[0]-b[0])+abs(a[1]-b[1])
            for a,b in zip(actions,actions[1:]))
        if prev_action is not None and actions:
            transition_cost += abs(actions[0][0]-float(prev_action[0])) + abs(actions[0][1]-float(prev_action[1]))
        # Danger when unknown future spawns exist is deliberately not scored as zero risk.
        # Favor greater known-shot clearance and fewer direction changes on ties.
        # All simulated prefixes have identical intended depth within a level.
        return (float(tier), -float(lethal.frame if lethal else pred.frames_simulated),
                -float(clearance if clearance is not None else 10000),
                float(transition_cost),
                float(sum(abs(a[0])+abs(a[1]) for a in actions)),
                float(pred.mario.left), float(pred.mario.top))

    def evaluate(self, observation: Mapping, *,
                 current_action: Sequence[float] | None = None) -> MPCPlan:
        # State: (piecewise action segments, expanded action sequence, ForwardResult, score)
        beam = [((), (), None, ())]
        evaluated = 0
        for length in self.segment_steps:
            expanded = []
            for segments, prefix, _, _ in beam:
                for action in self.candidate_actions:
                    if evaluated >= self.max_nodes: break
                    sequence = prefix + (action,) * length
                    result = self.simulator.simulate_observation(observation, sequence)
                    evaluated += 1
                    score = self._score(result, (segments + (action,)), current_action,
                                        len(sequence) * self.simulator.frame_skip,
                                        final=len(sequence) == self.horizon_steps)
                    expanded.append((segments+(action,),sequence,result,score))
                if evaluated >= self.max_nodes: break
            if not expanded:
                raise RuntimeError('max_nodes insufficient for one MPC level')
            expanded.sort(key=lambda x: (x[3],x[0]))
            # Preserve one path per distinct terminal state when possible.
            # Since all sequences have the same action horizon at a level,
            # this is a bounded, reproducible heuristic.
            beam = expanded[:self.beam_width]
        segments,seq,pred,score = beam[0]
        selected_robust = None
        if self.robust_finalists:
            finalists = beam[:self.robust_finalists]
            rescored = []
            for segs, sequence, nominal, base_score in finalists:
                robust = self.simulator.simulate_observation_robust(observation, sequence)
                lethal = [v for v in robust.lethal_frames if v is not None]
                # Re-rank only within same base tier, and never assert safety.
                robust_tier = 2 if lethal else 1 if any(
                    r.termination_reason == 'unknown_star_pickup' for r in robust.samples) else 0
                rescored.append(((robust_tier, -(min(lethal) if lethal else len(sequence)*self.simulator.frame_skip),
                                  *base_score[2:]), segs, sequence, nominal, robust))
            rescored.sort(key=lambda r: (r[0],r[1]))
            _,segments,seq,pred,selected_robust = rescored[0]
            score = rescored[0][0]
        return MPCPlan(
            best_action=seq[0], action_sequence=tuple(seq),
            risk_class=('sampled_lethal' if (selected_robust is not None and
                        any(x is not None for x in selected_robust.lethal_frames)) or pred.first_lethal_event else
                        'unknown_future_star' if pred.termination_reason == 'unknown_star_pickup'
                        else 'none_sampled'),
            first_lethal_frame=(min(x for x in selected_robust.lethal_frames if x is not None)
                if selected_robust is not None and any(x is not None for x in selected_robust.lethal_frames)
                else pred.first_lethal_event.frame if pred.first_lethal_event else None),
            known_collision_frame=(pred.first_collision_event.frame if pred.first_collision_event else None),
            min_horizontal_clearance_px=ActionEvaluator._clearance(pred),
            evaluated_nodes=evaluated, beam_width=self.beam_width,
            horizon_steps=self.horizon_steps,
            unknown_future=pred.termination_reason == 'unknown_star_pickup',
            nominal_score=tuple(score),
            assumptions=tuple(dict.fromkeys(pred.assumptions + (
                'finite_beam_not_globally_optimal','replan_every_action_step',
                'no_unobserved_future_spawns','not_a_learned_policy'))))
