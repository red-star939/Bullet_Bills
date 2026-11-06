"""Collect FailureAnalyzer v2 episodes with optional read-only engine collision audit.

This script NEVER gives private engine data to the policy or ForwardSimulator.
Run from mario_pj root after copying models/failure_analyzer_v2.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from agent_env import create_env
from models.action_evaluator import ActionEvaluator
from models.failure_analyzer import FailureConfig
from models.failure_analyzer_v2 import FailureAnalyzerV2, EngineCollisionAudit, summarize_reports_v2
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    simulator = ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv(Path('map_data/object_tiles.csv')))
    evaluator = ActionEvaluator(simulator, horizon_steps=args.horizon_steps,
                                robust_mode=args.robust_mode, robust_top_k=args.robust_top_k)
    analyzer = FailureAnalyzerV2(FailureConfig(history_seconds=args.history_seconds,
                                               fps=simulator.fps, frame_skip=simulator.frame_skip))
    rng = random.Random(args.seed)
    reports = []
    for ep in range(args.episodes):
        seed = rng.randrange(2**32)
        rng.randrange(2**32)  # preserve baseline paired seed sequence
        env = create_env(render_mode='none')
        analyzer.reset(seed, ep)
        try:
            obs, info = env.reset(seed=seed)
            previous_action = None
            core = env.unwrapped
            mario = getattr(core, '_mario', None)
            audit = EngineCollisionAudit(mario) if args.engine_audit and mario is not None else None
            with audit if audit is not None else nullcontext():
                for step in range(args.steps):
                    t0 = time.perf_counter()
                    evaluation = evaluator.evaluate(obs, current_action=previous_action)
                    compute = time.perf_counter() - t0
                    action = evaluation.best_action
                    prediction = simulator.simulate_observation(obs, [action])
                    next_obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
                    contacts = audit.take() if audit is not None else None
                    report = analyzer.record(step=step, observation=obs, action=action,
                                             evaluation=evaluation, next_observation=next_obs,
                                             info=info, terminated=terminated, truncated=truncated,
                                             one_step_prediction=prediction, policy_compute_s=compute,
                                             engine_contacts=contacts)
                    obs, previous_action = next_obs, action
                    if report is not None:
                        break
                else:
                    report = analyzer.report()
            reports.append(report)
            if args.save_all or report['is_death']:
                name = f"episode_{ep:05d}_seed_{seed}.json"
                (out / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        finally:
            env.close()
        if (ep + 1) % args.report_every == 0:
            s = summarize_reports_v2(reports)
            print('PROGRESS', ep + 1, '/', args.episodes, 'deaths', s['deaths'],
                  'labels', s['label_counts'], flush=True)
    fields = ('episode_index', 'env_seed', 'end_time_s', 'end_state', 'is_death',
              'primary_label', 'confidence', 'last_step', 'history_steps',
              'engine_evidence_status', 'fatal_track_id', 'fatal_first_seen_s',
              'fatal_visible_seconds', 'fatal_seen_before_last_action',
              'known_world_safe_alternatives')
    with (out / 'failure_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in reports:
            fatal = r.get('fatal_projectile') or {}
            d = {k: r.get(k) for k in fields}
            d.update(fatal_track_id=fatal.get('track_id'),
                     fatal_first_seen_s=fatal.get('first_seen_s'),
                     fatal_visible_seconds=fatal.get('visible_seconds_before_death'),
                     fatal_seen_before_last_action=fatal.get('was_visible_before_terminal_action'),
                     known_world_safe_alternatives=r['counterfactual_diagnostics']['known_world_alternatives_without_sampled_death'])
            w.writerow(d)
    s = summarize_reports_v2(reports); s['configuration'] = vars(args)
    (out / 'summary.json').write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FINAL SUMMARY', json.dumps(s, ensure_ascii=False))
    print('OUTPUT', out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--steps', type=int, default=900)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--horizon-steps', type=int, default=8)
    p.add_argument('--robust-mode', choices=('none','top_k','all'), default='none')
    p.add_argument('--robust-top-k', type=int, default=3)
    p.add_argument('--history-seconds', type=float, default=2.)
    p.add_argument('--report-every', type=int, default=10)
    p.add_argument('--save-all', action='store_true')
    p.add_argument('--engine-audit', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--out', default='logs/failure_analyzer_v2')
    args = p.parse_args()
    if args.episodes <= 0 or args.steps <= 0 or args.report_every <= 0:
        p.error('episodes, steps and report-every must be positive')
    run(args)


if __name__ == '__main__':
    main()
