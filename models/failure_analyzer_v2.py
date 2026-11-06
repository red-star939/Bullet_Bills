"""Failure analysis v2: persistent projectile histories and engine-only contact audit.

The policy NEVER receives engine-private state. EngineCollisionAudit is for
post-action ground-truth annotation, not ActionEvaluator input. Candidate safety
claims remain conditional on the observed world and examined action set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from collections import Counter
import math

from .failure_analyzer import FailureAnalyzer, FailureConfig, _bullets, summarize_reports
from .collision_predictor import overlaps, effective_bullet_vx
from .collision_event_audit import capture_collision_witness


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    vx: float
    first_seen_step: int
    first_seen_s: float
    last_seen_step: int
    last_seen_s: float
    match_quality: str = 'initial'

    def summary(self) -> dict[str, Any]:
        return vars(self).copy()


class ObservedBulletTracker:
    """Persist IDs across observations without trusting array order.

    Only active straight-moving projectiles are confidently matched. Falling
    bullets or unmatched observations receive new provisional IDs. Exact engine
    identity is *never* inferred solely from a provisional track ID.
    """
    def __init__(self, fps: float = 60, frame_skip: int = 4, tolerance: float = 3.):
        self.fps, self.frame_skip, self.tolerance = fps, frame_skip, tolerance
        self.reset()

    def reset(self) -> None:
        self.active: dict[int, Track] = {}
        self.archive: dict[int, Track] = {}
        self.next_id = 0
        self.last_step: int | None = None
        self.last_s: float | None = None

    def observe(self, obs: Mapping, step: int, time_s: float) -> list[dict[str, Any]]:
        rows = _bullets(obs)
        dt_frames = 0 if self.last_step is None else max(0, round((time_s - float(self.last_s)) * self.fps))
        # Coordinates/velocity can be float32 in observation but box is integral.
        pairs = []
        for j, row in enumerate(rows):
            bbox = tuple(int(round(v)) for v in row[:4])
            for tid, prior in self.active.items():
                if abs(prior.vx - row[4]) > 2 or abs(prior.bbox[1] - bbox[1]) > self.tolerance or abs(prior.bbox[3] - bbox[3]) > self.tolerance:
                    continue
                dx = effective_bullet_vx(prior.vx, round(self.fps)) * dt_frames
                e = abs(prior.bbox[0] + dx - bbox[0]) + abs(prior.bbox[2] + dx - bbox[2])
                if e <= 2 * self.tolerance:
                    pairs.append((e, tid, j))
        pairs.sort()
        matched_t, matched_j = set(), set()
        assignments = {}
        for error, tid, j in pairs:
            if tid not in matched_t and j not in matched_j:
                matched_t.add(tid); matched_j.add(j)
                assignments[j] = (tid, 'kinematic')
        next_active = {}
        result = []
        for j, row in enumerate(rows):
            bbox = tuple(int(round(v)) for v in row[:4])
            if j in assignments:
                tid, quality = assignments[j]
                tr = self.active[tid]
                tr.bbox = bbox
                tr.vx = float(row[4])
                tr.last_seen_step, tr.last_seen_s, tr.match_quality = int(step), float(time_s), quality
            else:
                tid = self.next_id; self.next_id += 1
                tr = Track(tid, bbox, float(row[4]), int(step), float(time_s), int(step), float(time_s),
                           'initial' if self.last_step is None else 'unmatched_new_or_state_change')
                self.archive[tid] = tr
            next_active[tid] = tr
            result.append(tr.summary())
        self.active = next_active
        self.last_step, self.last_s = int(step), float(time_s)
        return result

    def associate(self, bbox: Sequence[int], vx: float | None = None) -> tuple[Track | None, str]:
        """Only exact unique post-step bbox is accepted as strong association."""
        exact = [t for t in self.active.values() if tuple(t.bbox) == tuple(bbox)
                 and (vx is None or abs(t.vx - vx) < 2.)]
        if len(exact) == 1:
            return exact[0], 'exact_bbox_unique'
        if len(exact) > 1:
            return None, 'ambiguous_same_bbox'
        return None, 'not_in_observation'


class EngineCollisionAudit:
    """Reversible per-Mario hook. Invoke only in the experiment collector.

    Engine method is called unchanged. Captures pre-collision geometry at each
    internal physics frame; never alters the action or physical simulation.
    """
    def __init__(self, mario):
        self.mario = mario
        self.original = None
        self.contacts: list[dict[str, Any]] = []
        self.frame = 0

    def __enter__(self):
        self.original = self.mario.check_bullet_collision
        def hooked(bullets):
            self.frame += 1
            witness = capture_collision_witness(self.mario, bullets)
            was_alive = bool(self.mario.active_)
            result = self.original(bullets)
            alive = bool(self.mario.active_)
            if was_alive and not alive:
                rect = witness.mario_bbox
                candidates = [b for b in witness.bullets if b.state == 'active' and overlaps(rect, b.bbox)]
                # In engine order, stomp updates velocity, invincible kills bullets;
                # a lethal bullet is the first overlap not handled by either.
                vy = witness.mario_vy
                lethal = []
                for b in candidates:
                    cy = b.bbox[1] + (b.bbox[3] - b.bbox[1]) // 2
                    if vy > 0 and rect[3] < cy:
                        vy = -500.0
                    elif witness.invincible:
                        pass
                    else:
                        lethal.append(b)
                self.contacts.append({
                    'internal_collision_call': self.frame,
                    'mario_bbox_at_collision': list(rect),
                    'mario_vy_at_collision': witness.mario_vy,
                    'mario_invincible_at_collision': witness.invincible,
                    'fatal_candidates': [
                        {'engine_object_id': b.identity, 'bbox': list(b.bbox)} for b in lethal],
                    'active_overlap_count': len(candidates),
                    'status': 'unique_fatal_overlap' if len(lethal) == 1 else
                              'ambiguous_multiple_fatal_overlaps' if len(lethal) > 1 else 'no_fatal_overlap',
                })
            return result
        self.mario.check_bullet_collision = hooked
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original is not None:
            self.mario.check_bullet_collision = self.original
            self.original = None
        return False

    def take(self) -> list[dict[str, Any]]:
        out = self.contacts
        self.contacts = []
        return out


class FailureAnalyzerV2(FailureAnalyzer):
    """Backward-compatible record() plus true-collision and track annotations."""
    def __init__(self, config: FailureConfig = FailureConfig()):
        super().__init__(config)
        self.tracker = ObservedBulletTracker(config.fps, config.frame_skip)

    def reset(self, episode_seed: int, episode_index: int = 0) -> None:
        super().reset(episode_seed, episode_index)
        self.tracker.reset()

    def record(self, *, engine_contacts: Sequence[Mapping] | None = None, **kwargs):
        step = int(kwargs['step'])
        t_after = float(kwargs['info'].get('time_elapsed', 0.))
        elapsed_frames = self.config.frame_skip
        if self.tracker.last_step is None:
            self.tracker.observe(kwargs['observation'], step, t_after - elapsed_frames / self.config.fps)
        before_tracks = [t.summary() for t in self.tracker.active.values()]
        after_tracks = self.tracker.observe(kwargs['next_observation'], step+1, t_after)
        outcome = super().record(**kwargs)
        end = self.history[-1]
        end['tracked_bullets_before'] = before_tracks
        end['tracked_bullets_after'] = after_tracks
        end['engine_contacts'] = [dict(c) for c in (engine_contacts or [])]
        if outcome is not None:
            return self.report()
        return None

    def report(self) -> dict[str, Any]:
        # FailureAnalyzer.report() invokes self._classify() which we override.
        report = super().report()
        report['schema_version'] = 2
        final = self.history[-1]
        death = report['is_death']
        evidence_status = 'no_engine_contact_audit'
        fatal = None
        contacts = final.get('engine_contacts') or []
        if death and contacts:
            # Only the final recorded action's internal collision is used.
            last = contacts[-1]
            evidence_status = last.get('status', 'unknown')
            if last.get('status') == 'unique_fatal_overlap':
                candidate = last['fatal_candidates'][0]
                # Engine witness bbox is post bullet motion; end observation
                # may censor OOB/fallen bullets, in which case keep unmatched.
                tr, match_status = self.tracker.associate(candidate['bbox'])
                fatal = {
                    'engine_object_id_diagnostic_only': candidate['engine_object_id'],
                    'bbox_at_impact': candidate['bbox'],
                    'track_id': tr.track_id if tr else None,
                    'association': match_status,
                    'first_seen_step': tr.first_seen_step if tr else None,
                    'first_seen_s': tr.first_seen_s if tr else None,
                    'visible_seconds_before_death': round(float(report['end_time_s']) - tr.first_seen_s, 8) if tr else None,
                    'was_visible_before_terminal_action': bool(tr and tr.first_seen_step <= final['step']),
                }
        report['engine_evidence_status'] = evidence_status
        report['fatal_projectile'] = fatal
        report['observed_projectile_tracks'] = [t.summary() for t in self.tracker.archive.values()]
        report['counterfactual_diagnostics'] = self._counterfactuals(final)
        report['fatal_visibility_diagnostics'] = self._visibility_diagnostics(report)
        report['primary_label'], report['confidence'], report['evidence'] = self._classify_v2(report)
        report['limitations'].append('Engine-private audit provides diagnostic labels only; never feeds policy decisions.')
        report['limitations'].append('Observed track IDs are provisional; association is based on exact geometry, not engine identity.')
        return report

    @staticmethod
    def _visibility_diagnostics(report):
        """Observational lead time, not proof the projectile was avoidable."""
        fatal = report.get('fatal_projectile')
        if not fatal or fatal['track_id'] is None:
            return {'status': 'unavailable', 'steps_observed_before_death': None,
                    'lead_time_s': None, 'first_visible_history_step': None}
        first = fatal['first_seen_step']
        hist = report['history']
        seen_in_window = [h for h in hist if any(
            t['track_id'] == fatal['track_id'] for t in h.get('tracked_bullets_before', []))]
        return {
            'status': 'observed_track_not_proven_avoidable',
            'steps_observed_before_death': max(0, report['last_step'] - first + 1),
            'lead_time_s': fatal['visible_seconds_before_death'],
            'first_visible_history_step': seen_in_window[0]['step'] if seen_in_window else None,
            'history_window_truncated_first_appearance': bool(not seen_in_window or
                seen_in_window[0]['step'] > first),
        }

    # Avoid v1's recent-spawn priority when super.report() initially classifies.
    @staticmethod
    def _classify(history: list[dict[str, Any]], death: bool):
        if not death:
            return 'censored_or_non_death', 'high', ['not a game_over death']
        return 'cause_unresolved', 'low', ['v2 diagnosis pending collision evidence']

    @staticmethod
    def _counterfactuals(end):
        ranks = end.get('rankings', [])
        chosen = next((r for r in ranks if r['action'] == end['action']), None)
        alternatives = [r for r in ranks if r['action'] != end['action']]
        safe = [r for r in alternatives if r['risk_class'] == 'none_sampled']
        return {
            'chosen_nominal_lethal_frame': chosen['nominal_lethal_frame'] if chosen else None,
            'chosen_rank': chosen['rank'] if chosen else None,
            'known_world_alternatives_without_sampled_death': len(safe),
            'known_world_alternative_actions': [s['action'] for s in safe],
            'evaluated_candidates': len(ranks),
            'conclusion': 'known_world_candidates_only_not_proven_real_world_survival',
        }

    @staticmethod
    def _classify_v2(report):
        if not report['is_death']:
            return 'censored_or_non_death', 'high', ['not a game_over death']
        fatal = report.get('fatal_projectile')
        cf = report['counterfactual_diagnostics']
        end = report['history'][-1]
        if fatal is not None:
            if fatal['track_id'] is None:
                return 'fatal_projectile_unmatched', 'medium', ['real collision observed; fatal bullet not matched to an observed track']
            if fatal['was_visible_before_terminal_action']:
                return 'tracked_fatal_projectile', 'high', [
                    'actual fatal bullet geometrically matched to a tracked projectile',
                    f"first_seen_step={fatal['first_seen_step']}",
                    f"visible_seconds_before_death={fatal['visible_seconds_before_death']}",
                    'visibility and actual avoidability are different questions',
                ]
            return 'fatal_projectile_first_seen_terminal_action', 'medium', [
                'fatal projectile matched only at the terminal action end',
                'unobserved spawn during action is possible but not proven',
            ]
        if report['engine_evidence_status'] == 'ambiguous_multiple_fatal_overlaps':
            return 'ambiguous_multiple_fatal_projectiles', 'low', ['more than one overlapping active projectile at death']
        if report['engine_evidence_status'] == 'no_engine_contact_audit':
            return 'fatal_projectile_unverified', 'low', ['no engine collision witness; recent spawns are not proof of cause']
        pred = end.get('one_step_prediction')
        if pred is not None and (pred['position_mismatch'] or pred['velocity_mismatch'] or pred['terminal_mismatch']):
            return 'model_or_hidden_state_discrepancy', 'medium', ['one-step prediction differed from actual outcome']
        if cf['chosen_nominal_lethal_frame'] is not None:
            return 'predicted_lethal_action', 'medium', ['chosen action was predicted lethal under known-world assumptions']
        return 'cause_unresolved', 'low', ['no unique fatal projectile established']


def summarize_reports_v2(reports: Sequence[Mapping]) -> dict[str, Any]:
    base = summarize_reports(reports)
    base['schema_version'] = 2
    base['engine_evidence_counts'] = dict(sorted(Counter(str(r.get('engine_evidence_status', 'unknown')) for r in reports).items()))
    base['fatal_projectiles_matched'] = sum(bool(r.get('fatal_projectile') and r['fatal_projectile']['track_id'] is not None) for r in reports)
    base['fatal_projectiles_visible_before_last_action'] = sum(bool(r.get('fatal_projectile') and r['fatal_projectile']['was_visible_before_terminal_action']) for r in reports)
    base['known_world_alternative_available_at_terminal'] = sum(bool(r['is_death'] and r.get('counterfactual_diagnostics', {}).get('known_world_alternatives_without_sampled_death', 0) > 0) for r in reports)
    return base
