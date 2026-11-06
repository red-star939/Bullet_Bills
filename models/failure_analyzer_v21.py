"""FailureAnalyzer v2.1: engine death route and engine-object-to-observation identity audit.

All engine-private facts are POST-ACTION diagnostics; policy and simulator must
receive only the public observation. Tracking quality is explicitly reported.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .failure_analyzer_v2 import FailureAnalyzerV2, EngineCollisionAudit, summarize_reports_v2


def snapshot_engine_bullets(core) -> list[dict[str, Any]]:
    """Read-only snapshot. Never serialize Python object identity as a persistent ID."""
    return [dict(engine_object_id=id(b), bbox=[int(b.rect.left), int(b.rect.top),
            int(b.rect.right), int(b.rect.bottom)], state=str(b.state_),
            vx=float(b._velocity)) for b in getattr(core, '_bullet_bills', ())]


class EngineDeathAuditV21(EngineCollisionAudit):
    """Reversible hooks for collision witness AND any internal-frame death.

    Hooks must be installed only during diagnostic collection. Preserve and
    invoke engine methods exactly once, without changing their arguments.
    """
    def __init__(self, core, world_height: int = 864):
        super().__init__(core._mario)
        self.core = core
        self.world_height = int(world_height)
        self.original_step = None
        self.death_events: list[dict[str, Any]] = []
        self.internal_frame = 0

    def __enter__(self):
        super().__enter__()
        self.original_step = self.core._step_internal
        def wrapped(*args, **kwargs):
            self.internal_frame += 1
            alive_before = bool(self.mario.active_)
            contact_before = len(self.contacts)
            answer = self.original_step(*args, **kwargs)
            if alive_before and not bool(self.mario.active_):
                new_contacts = self.contacts[contact_before:]
                if new_contacts:
                    cause = 'bullet_collision'
                elif int(self.mario.rect.bottom) > self.world_height:
                    cause = 'fall_out_of_bounds'
                else:
                    cause = 'other_or_unverified'
                self.death_events.append(dict(
                    cause=cause, internal_frame=self.internal_frame,
                    mario_bbox=[int(self.mario.rect.left), int(self.mario.rect.top),
                                int(self.mario.rect.right), int(self.mario.rect.bottom)],
                    mario_bottom=int(self.mario.rect.bottom),
                    collision_evidence_count=len(new_contacts),
                    collision_contacts=[dict(c) for c in new_contacts],
                ))
            return answer
        self.core._step_internal = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original_step is not None:
            self.core._step_internal = self.original_step
            self.original_step = None
        return super().__exit__(exc_type, exc, tb)

    def take_deaths(self):
        out, self.death_events = self.death_events, []
        return out


class FailureAnalyzerV21(FailureAnalyzerV2):
    """v2-compatible history and new orthogonal physical/policy diagnostics."""
    def reset(self, episode_seed: int, episode_index: int = 0) -> None:
        super().reset(episode_seed, episode_index)
        self._engine_first_seen: dict[int, dict[str, Any]] = {}
        self._engine_track_links: dict[int, set[int]] = {}

    @staticmethod
    def _unique_links(engine_rows, tracks):
        """Unique bbox+vx link only. Identity is from engine snapshots, not bbox."""
        candidates = {}
        for e in engine_rows or ():
            possibilities = [t for t in tracks if tuple(e['bbox']) == tuple(t['bbox'])
                             and abs(float(e['vx']) - float(t['vx'])) <= 2.0]
            if len(possibilities) == 1:
                candidates.setdefault(possibilities[0]['track_id'], []).append(e['engine_object_id'])
        return {ids[0]: tid for tid, ids in candidates.items() if len(ids) == 1}

    def record(self, *, engine_deaths: Sequence[Mapping] | None = None,
               engine_bullets_before: Sequence[Mapping] | None = None,
               engine_bullets_after: Sequence[Mapping] | None = None, **kwargs):
        # The v2 record invokes self.report() on terminal frames. Establish
        # diagnostic facts FIRST so the report sees the completed frame.
        step = int(kwargs['step'])
        t_before = float(kwargs['info'].get('time_elapsed', 0.0)) - self.config.frame_skip / self.config.fps
        if self.tracker.last_step is None:
            self.tracker.observe(kwargs['observation'], step, t_before)
        before_tracks = [t.summary() for t in self.tracker.active.values()]
        before_rows = [dict(e) for e in (engine_bullets_before or [])]
        after_rows = [dict(e) for e in (engine_bullets_after or [])]
        for row in before_rows:
            key = int(row['engine_object_id'])
            self._engine_first_seen.setdefault(key, dict(step=step, time_s=t_before,
                                                        evidence='engine_present_before_action'))
        for row in after_rows:
            key = int(row['engine_object_id'])
            self._engine_first_seen.setdefault(key, dict(step=step+1,
                time_s=float(kwargs['info'].get('time_elapsed', 0.0)),
                evidence='engine_present_after_action'))
        for eid, tid in self._unique_links(before_rows, before_tracks).items():
            self._engine_track_links.setdefault(eid, set()).add(tid)
        # v2 will itself link after-state tracks via heuristic, then report.
        self._pending_v21_before = before_rows
        self._pending_v21 = dict(engine_deaths=[dict(e) for e in (engine_deaths or [])],
                                 engine_bullets_after=after_rows)
        result = super().record(**kwargs)
        return result

    def report(self):
        r = super().report()
        pending = getattr(self, '_pending_v21', {})
        deaths = pending.get('engine_deaths', [])
        r['schema_version'] = '2.1'
        route = deaths[-1]['cause'] if deaths else ('not_a_death' if not r['is_death'] else 'unverified')
        r['physical_death_cause'] = route
        r['engine_death_events'] = deaths
        fatal = r.get('fatal_projectile')
        fatal_engine_id = int(fatal['engine_object_id_diagnostic_only']) if fatal else None
        # Include after-state unique engine link for first terminal observation.
        after_links = self._unique_links(pending.get('engine_bullets_after', []),
            [t.summary() for t in self.tracker.active.values()])
        for eid, tid in after_links.items():
            self._engine_track_links.setdefault(eid, set()).add(tid)
        if fatal_engine_id is not None:
            tracks = self._engine_track_links.get(fatal_engine_id, set())
            ever_engine = self._engine_first_seen.get(fatal_engine_id)
            fatal['engine_first_audited'] = ever_engine
            fatal['engine_linked_track_ids'] = sorted(tracks)
            fatal['identity_link_quality'] = ('unique_engine_identity_and_observed_track'
                if len(tracks) == 1 else 'multiple_observation_track_ids_for_one_engine_object'
                if len(tracks) > 1 else 'engine_object_not_linked_to_observed_track')
            fatal['engine_present_before_terminal_action'] = any(
                e['engine_object_id'] == fatal_engine_id for e in
                (self._pending_v21_before or [])) if hasattr(self, '_pending_v21_before') else None
        # Keep physical and policy axes orthogonal. No causal leap from
        # known-world counterfactual safety to actual avoidability.
        end = r['history'][-1]
        cf = r['counterfactual_diagnostics']
        chosen_lethal = cf['chosen_nominal_lethal_frame'] is not None
        if not r['is_death']:
            policy_label = 'not_a_death'
        elif chosen_lethal:
            policy_label = 'selected_predicted_lethal_action'
        elif cf['known_world_alternatives_without_sampled_death']:
            policy_label = 'alternative_predicted_safe_but_not_verified'
        else:
            policy_label = 'policy_cause_unresolved'
        r['policy_failure_diagnostic'] = policy_label
        # Physical labels replace v2's early 'first seen terminal' confidence.
        r['primary_label'] = route if r['is_death'] else 'censored_or_non_death'
        r['confidence'] = ('high' if route in ('bullet_collision','fall_out_of_bounds')
                           else 'low')
        r['limitations'].extend([
            'Engine object IDs are valid only within one live episode; not stable across runs.',
            'First engine-audited appearance is not necessarily the physical spawn frame.',
            'The counterfactual alternative was not executed; no avoidability proof.',
            'No private engine state is passed to policy or simulator.',
        ])
        return r


def summarize_reports_v21(reports):
    out = summarize_reports_v2(reports)
    out['schema_version'] = '2.1'
    out['physical_death_cause_counts'] = dict(sorted(Counter(
        r.get('physical_death_cause', 'unknown') for r in reports).items()))
    out['policy_failure_diagnostic_counts'] = dict(sorted(Counter(
        r.get('policy_failure_diagnostic', 'unknown') for r in reports).items()))
    out['fatal_engine_identity_link_counts'] = dict(sorted(Counter(
        (r.get('fatal_projectile') or {}).get('identity_link_quality', 'not_available')
        for r in reports).items()))
    return out
