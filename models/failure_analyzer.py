"""Evidence-aware diagnostics for a rule-based BulletBill policy.

A failure report is *not* a proof that a different action would have survived.
Counterfactual labels refer only to the currently observed projectiles and
candidate actions; future random spawns and hidden projectile state are unknown.
No gym/pygame dependency is required in this module.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import math


def _serializable(value: Any) -> Any:
    if hasattr(value, 'tolist'):
        return _serializable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def _bullets(obs: Mapping) -> list[list[float]]:
    return [list(map(float, b[:5])) for b in obs.get('bullet_bills', []) if len(b) >= 5]


def detect_unmatched_bullets(before: Mapping, after: Mapping, *, frame_skip: int = 4,
                             tolerance_px: float = 3.0) -> int:
    """Conservatively detect likely newcomers (never claims exact identities).

    A bullet that moves according to its observed vx is matched greedily to one
    earlier bullet. An unmatched *new* bullet is evidence of an unseen spawn;
    unmatched bullets can also arise from stale/fallen states or rounding.
    """
    old = _bullets(before)
    new = _bullets(after)
    used: set[int] = set()
    newcomers = 0
    for b in new:
        options = []
        for i, a in enumerate(old):
            if i in used or abs(a[1] - b[1]) > tolerance_px or abs(a[3] - b[3]) > tolerance_px:
                continue
            if abs(a[4] - b[4]) > 2.0:
                continue
            # Active bullets move via int(vx/60) per physics frame.
            expected_dx = frame_skip * math.trunc(a[4] / 60.0)
            err = abs((a[0] + expected_dx) - b[0])
            if err <= tolerance_px:
                options.append((err, i))
        if options:
            used.add(min(options)[1])
        else:
            newcomers += 1
    return newcomers


@dataclass(frozen=True)
class FailureConfig:
    history_seconds: float = 2.0
    fps: float = 60.0
    frame_skip: int = 4
    tolerance_position_px: float = 1.0
    tolerance_velocity: float = 0.1


class FailureAnalyzer:
    """Keep bounded episode history and produce auditable terminal reports."""

    def __init__(self, config: FailureConfig = FailureConfig()):
        if config.history_seconds <= 0 or config.fps <= 0 or config.frame_skip < 1:
            raise ValueError('invalid history / FPS / frame skip')
        self.config = config
        self.history = deque(maxlen=max(1, math.ceil(
            config.history_seconds * config.fps / config.frame_skip)))
        self.episode_seed: int | None = None
        self.episode_index: int | None = None

    def reset(self, episode_seed: int, episode_index: int = 0) -> None:
        self.history.clear()
        self.episode_seed = int(episode_seed)
        self.episode_index = int(episode_index)

    def record(self, *, step: int, observation: Mapping, action: Sequence[float],
               evaluation: Any, next_observation: Mapping, info: Mapping,
               terminated: bool, truncated: bool, one_step_prediction: Any = None,
               policy_compute_s: float = 0.0) -> dict[str, Any] | None:
        """Record actual transition; optionally return a report at episode end.

        Evaluation is the pre-action result, NOT an oracle of actual future
        outcomes. The one-step prediction is computed before env.step.
        """
        if self.episode_seed is None:
            raise RuntimeError('call reset() before record()')
        mario_before = [float(v) for v in observation['mario']]
        mario_after = [float(v) for v in next_observation['mario']]
        ranked = []
        for candidate in evaluation.rankings:
            ranked.append({
                'rank': int(candidate.rank),
                'action': list(candidate.action),
                'risk_class': candidate.risk_class,
                'robust_status': candidate.robust_status,
                'first_possible_lethal_frame': candidate.first_possible_lethal_frame,
                'nominal_lethal_frame': candidate.nominal_lethal_frame,
                'earliest_known_collision_frame': candidate.earliest_known_collision_frame,
                'min_horizontal_clearance_px': candidate.min_horizontal_clearance_px,
                'uncertain_environment': candidate.uncertain_environment,
            })
        new_bullets = detect_unmatched_bullets(
            observation, next_observation, frame_skip=self.config.frame_skip)
        prediction_details = None
        if one_step_prediction is not None:
            pred = one_step_prediction
            errors = [float(pred.mario.bbox[i]) - mario_after[i] for i in range(4)]
            velocity_errors = [float(pred.mario.vx) - mario_after[4],
                               float(pred.mario.vy) - mario_after[5]]
            prediction_details = {
                'bbox_error': errors,
                'velocity_error': velocity_errors,
                'predicted_terminated': bool(pred.terminated),
                'predicted_reason': pred.termination_reason,
                'predicted_frames': int(pred.frames_simulated),
                'prediction_assumptions': list(pred.assumptions),
                'position_mismatch': any(abs(v) > self.config.tolerance_position_px for v in errors),
                'velocity_mismatch': any(abs(v) > self.config.tolerance_velocity for v in velocity_errors),
                'terminal_mismatch': bool(pred.terminated) != bool(terminated),
            }
        item = {
            'step': int(step),
            'time_s': float(info.get('time_elapsed', 0.0)),
            'mario_before': mario_before,
            'mario_after': mario_after,
            'bullets_before': _bullets(observation),
            'bullets_after': _bullets(next_observation),
            'new_or_unmatched_bullets': new_bullets,
            'stars_before': _serializable(observation.get('stars', [])),
            'action': list(map(float, action)),
            'rankings': ranked,
            'horizon_steps': int(evaluation.horizon_steps),
            'strategy': str(evaluation.strategy),
            'policy_compute_s': float(policy_compute_s),
            'one_step_prediction': prediction_details,
            'terminated': bool(terminated),
            'truncated': bool(truncated),
            'game_state': str(info.get('game_state', 'unknown')),
        }
        self.history.append(item)
        if terminated or truncated:
            return self.report()
        return None

    def report(self) -> dict[str, Any]:
        history = list(self.history)
        if not history:
            raise RuntimeError('cannot report empty history')
        end = history[-1]
        death = bool(end['terminated'] and end['game_state'] == 'game_over')
        label, confidence, evidence = self._classify(history, death)
        return {
            'schema_version': 1,
            'episode_index': self.episode_index,
            'env_seed': self.episode_seed,
            'last_step': end['step'],
            'end_time_s': end['time_s'],
            'end_state': end['game_state'],
            'terminated': end['terminated'],
            'truncated': end['truncated'],
            'is_death': death,
            'primary_label': label,
            'confidence': confidence,
            'evidence': evidence,
            'history_steps': len(history),
            'history': history,
            'limitations': [
                'Counterfactual actions were not executed in the real environment.',
                'Forward predictions condition on currently observed bullets only.',
                'Projectile active/fallen state and invincibility expiry are not fully observable.',
                'An exhausted candidate set does not prove physical unavoidability.',
            ],
        }

    @staticmethod
    def _classify(history: list[dict[str, Any]], death: bool) -> tuple[str, str, list[str]]:
        end = history[-1]
        if not death:
            return 'censored_or_non_death', 'high', ['episode did not end with game_over death']
        ev = []
        # Unknown-event evidence has priority over explaining the model's own score.
        nearby = history[-min(len(history), 3):]
        if any(s['new_or_unmatched_bullets'] > 0 for s in nearby):
            ev.append('a projectile appeared or could not be matched in the last 3 actions')
            return 'possible_unobserved_projectile', 'low', ev
        pred = end['one_step_prediction']
        if pred is not None and (pred['position_mismatch'] or pred['velocity_mismatch'] or pred['terminal_mismatch']):
            ev.append('one-step nominal prediction disagrees with real transition')
            return 'model_or_hidden_state_discrepancy', 'medium', ev
        chosen = next((r for r in end['rankings'] if r['action'] == end['action']), None)
        if chosen is not None and chosen['nominal_lethal_frame'] is not None:
            ev.append(f"chosen action had predicted lethal frame {chosen['nominal_lethal_frame']}")
            ev.append('other rankings are only known-world counterfactuals')
            return 'predicted_lethal_action', 'high', ev
        # This can be measured without claiming a real alternate would survive.
        if chosen is not None:
            ev.append('chosen action had no nominal lethal prediction at decision time')
        if any(r['risk_class'] == 'none_sampled' and r['action'] != end['action']
               for r in end['rankings']):
            ev.append('other candidate(s) were nominally safe under known-world assumptions')
        return 'unexplained_or_horizon_limited', 'low', ev


def summarize_reports(reports: Sequence[Mapping]) -> dict[str, Any]:
    counts = Counter(str(r['primary_label']) for r in reports)
    deaths = sum(bool(r['is_death']) for r in reports)
    return {'episodes': len(reports), 'deaths': deaths,
            'censored_or_non_death': len(reports) - deaths,
            'label_counts': dict(sorted(counts.items())),
            'mean_observed_time_s': (sum(float(r['end_time_s']) for r in reports) / len(reports)
                                     if reports else None)}
