"""Real-engine failure capture for the existing Nominal ActionEvaluator.

Run from project root:
 .\\.venv\\Scripts\\python.exe collect_failure_cases.py --episodes 100 --steps 300
Produces summaries, a per-step trace for terminal episodes and one JSON report
per episode. No engine hidden state is used for policy decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np

from agent_env import create_env
from models.action_evaluator import ActionEvaluator
from models.failure_analyzer import FailureAnalyzer, FailureConfig, summarize_reports
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    simulator = ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv(
        Path('map_data/object_tiles.csv')))
    evaluator = ActionEvaluator(simulator, horizon_steps=args.horizon_steps,
                                robust_mode=args.robust_mode,
                                robust_top_k=args.robust_top_k)
    analyzer = FailureAnalyzer(FailureConfig(history_seconds=args.history_seconds,
                                            fps=simulator.fps,
                                            frame_skip=simulator.frame_skip))
    seeds = random.Random(args.seed)
    reports = []
    for episode_index in range(args.episodes):
        seed = seeds.randrange(2**32)
        # Match the env-seed sequence of test_action_evaluator_env.py:
        # it consumes a second random integer for random_action_seed.
        seeds.randrange(2**32)
        env = create_env(render_mode='none')
        analyzer.reset(seed, episode_index)
        previous_action = None
        try:
            obs, info = env.reset(seed=seed)
            for step in range(args.steps):
                t0 = time.perf_counter()
                evaluation = evaluator.evaluate(obs, current_action=previous_action)
                policy_compute = time.perf_counter() - t0
                action = evaluation.best_action
                # Independently check the *selected action*, one env-step ahead.
                one_step = simulator.simulate_observation(obs, [action])
                new_obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
                report = analyzer.record(step=step, observation=obs, action=action,
                                         evaluation=evaluation, next_observation=new_obs,
                                         info=info, terminated=terminated, truncated=truncated,
                                         one_step_prediction=one_step,
                                         policy_compute_s=policy_compute)
                previous_action, obs = action, new_obs
                if report is not None:
                    break
            else:
                report = analyzer.report()  # censored by --steps limit
            reports.append(report)
            if args.save_all or report['is_death']:
                filename = f"episode_{episode_index:05d}_seed_{seed}.json"
                (out / filename).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        finally:
            env.close()
        if (episode_index + 1) % args.report_every == 0:
            summary = summarize_reports(reports)
            print('PROGRESS', episode_index + 1, '/', args.episodes,
                  'deaths', summary['deaths'], 'labels', summary['label_counts'], flush=True)
    fields = ('episode_index', 'env_seed', 'end_time_s', 'end_state', 'is_death',
              'primary_label', 'confidence', 'last_step', 'history_steps')
    with (out / 'failure_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in reports:
            writer.writerow({key: r[key] for key in fields})
    summary = summarize_reports(reports)
    summary['configuration'] = vars(args)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('FINAL SUMMARY', json.dumps(summary, ensure_ascii=False))
    print('OUTPUT', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--horizon-steps', type=int, default=8)
    parser.add_argument('--robust-mode', choices=('none', 'top_k', 'all'), default='none')
    parser.add_argument('--robust-top-k', type=int, default=3)
    parser.add_argument('--history-seconds', type=float, default=2.)
    parser.add_argument('--report-every', type=int, default=10)
    parser.add_argument('--save-all', action='store_true')
    parser.add_argument('--out', default='logs/failure_analyzer_v1')
    args = parser.parse_args()
    if args.episodes <= 0 or args.steps <= 0 or args.report_every <= 0:
        parser.error('episodes, steps and report-every must be positive')
    run(args)


if __name__ == '__main__':
    main()
