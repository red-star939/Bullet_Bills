"""Per-step model-vs-environment audit and counterfactual proposal logging.

The ACTUAL executed action is checked against a one-step known-world simulator;
proposal predictions remain hypothetical and are never marked ground truth.
No engine-private state is used in controller or audit predictions.
"""
from __future__ import annotations
import csv
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from .closed_loop_planner import ModelState, KnownWorldStepper, valid_bullets


class InterventionAudit:
    def __init__(self, physics, *, diverge_px=12.):
        self.physics = physics
        self.stepper = KnownWorldStepper(physics)
        self.diverge_px = float(diverge_px)
        self.reset()

    def reset(self):
        self.rows = []
        self._pending = None

    def begin(self, observation, executed_action, *, episode, seed, step,
              time_s, baseline_action, proposal_action=None, metadata=None):
        if self._pending is not None:
            raise RuntimeError('complete preceding audit before begin')
        a = np.asarray(executed_action, np.float32).reshape(2)
        base = np.asarray(baseline_action, np.float32).reshape(2)
        p = (np.asarray(proposal_action, np.float32).reshape(2)
             if proposal_action is not None else base.copy())
        metadata = metadata or {}
        row = {
            'episode': int(episode), 'seed': int(seed), 'step': int(step),
            'time_s': round(float(time_s), 6),
            'mario_x': float(observation['mario'][0]),
            'mario_y': float(observation['mario'][1]),
            'visible_bullets': len(valid_bullets(observation)),
            'baseline_move': float(base[0]), 'baseline_jump': float(base[1]),
            'proposal_move': float(p[0]), 'proposal_jump': float(p[1]),
            'executed_move': float(a[0]), 'executed_jump': float(a[1]),
            'changed_from_baseline': int(not np.array_equal(base, a)),
            'reason': str(metadata.get('v41_reason', '')),
            'baseline_known_death_frame': metadata.get('v41_baseline_death_frame'),
            'candidate_known_death_frame': metadata.get('v41_candidate_death_frame'),
            'baseline_spawn_exposure': metadata.get('v41_baseline_spawn_exposure'),
            'candidate_spawn_exposure': metadata.get('v41_candidate_spawn_exposure'),
            'counterfactual_not_observed': 1,
            'predicted_death_frame': None,
            'predicted_left': None, 'predicted_top': None,
            'predicted_vx': None, 'predicted_vy': None,
            'prediction_unavailable': '',
        }
        try:
            if float(observation['mario'][6]) > 0.5:
                raise ValueError('remaining invincibility duration is not observed')
            model = ModelState.from_observation(observation, self.physics, time_s)
            result = self.stepper.step(model, a)
            row['predicted_death_frame'] = result.death_frame
            if result.state is not None:
                m = result.state.mario
                row['predicted_left'] = float(m[0])
                row['predicted_top'] = float(m[1])
                row['predicted_vx'] = float(m[2])
                row['predicted_vy'] = float(m[3])
        except Exception as e:
            row['prediction_unavailable'] = f'{type(e).__name__}: {str(e)[:120]}'
        self._pending = row

    def finish(self, next_observation, *, terminated=False, truncated=False, info=None):
        if self._pending is None:
            raise RuntimeError('audit.finish() without begin()')
        row = self._pending
        self._pending = None
        actual = np.asarray(next_observation['mario'], np.float64)
        row['actual_left'] = float(actual[0])
        row['actual_top'] = float(actual[1])
        row['actual_vx'] = float(actual[4])
        row['actual_vy'] = float(actual[5])
        row['actual_death'] = int(bool(terminated and (info or {}).get('game_state') == 'game_over'))
        row['terminated'] = int(bool(terminated))
        row['truncated'] = int(bool(truncated))
        if row['predicted_left'] is not None:
            dx = abs(row['predicted_left'] - row['actual_left'])
            dy = abs(row['predicted_top'] - row['actual_top'])
            row['position_error_px'] = max(dx, dy)
            row['velocity_error'] = max(abs(row['predicted_vx'] - row['actual_vx']),
                                        abs(row['predicted_vy'] - row['actual_vy']))
            row['model_diverged'] = int(row['position_error_px'] > self.diverge_px)
        else:
            row['position_error_px'] = None
            row['velocity_error'] = None
            row['model_diverged'] = None
        row['unexpected_death'] = int(bool(row['actual_death'] and row['predicted_death_frame'] is None))
        row['predicted_death_not_observed'] = int(bool(row['predicted_death_frame'] is not None
                                                   and not row['actual_death']))
        self.rows.append(row)
        return row

    def summary(self):
        measured = [r['position_error_px'] for r in self.rows if r['position_error_px'] is not None]
        return {
            'audited_steps': len(self.rows),
            'measured_steps': len(measured),
            'mean_position_error_px': round(float(np.mean(measured)), 4) if measured else None,
            'max_position_error_px': round(float(max(measured)), 4) if measured else None,
            'model_divergences': sum(int(r['model_diverged'] or 0) for r in self.rows),
            'unexpected_deaths': sum(r['unexpected_death'] for r in self.rows),
            'predicted_death_not_observed': sum(r['predicted_death_not_observed'] for r in self.rows),
            'actual_interventions': sum(r['changed_from_baseline'] for r in self.rows),
        }

    def save_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text('no_audit_rows\n', encoding='utf-8')
            return
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
