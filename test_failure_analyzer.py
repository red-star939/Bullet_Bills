"""Standalone synthetic tests; no kymnasium needed."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from models.failure_analyzer import FailureAnalyzer, FailureConfig, detect_unmatched_bullets, summarize_reports
from models.action_evaluator import ActionEvaluator
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor, MotionConfig


def obs(x=624, y=720, bullets=None):
    return {'mario': np.asarray([x, y, x + 48, y + 48, 0, 0, 0, 0], dtype=np.float32),
            'bullet_bills': np.asarray(bullets if bullets is not None else [], dtype=np.float32).reshape(-1, 5),
            'stars': np.zeros((0, 4), dtype=np.float32)}


def run():
    original = obs(bullets=[[760, 720, 808, 768, -240.]])
    moved = obs(bullets=[[744, 720, 792, 768, -240.]])
    assert detect_unmatched_bullets(original, moved) == 0
    added = obs(bullets=[[744, 720, 792, 768, -240.], [1250, 100, 1298, 148, -240.]])
    assert detect_unmatched_bullets(original, added) == 1
    sim = ForwardSimulator(MarioMotionPredictor(config=MotionConfig()))
    policy = ActionEvaluator(sim, horizon_steps=8, robust_mode='none',
                             candidate_actions=[(0, 0), (0, 1)])
    decision = policy.evaluate(original)
    analyzer = FailureAnalyzer(FailureConfig(history_seconds=0.2))
    analyzer.reset(123, 9)
    # Four steps history exceeds bounded 3-transition capacity.
    for i in range(3):
        result = analyzer.record(step=i, observation=original, action=decision.best_action,
                                 evaluation=decision, next_observation=moved,
                                 info={'time_elapsed': (i+1)/15, 'game_state':'playing'},
                                 terminated=False, truncated=False)
        assert result is None
    report = analyzer.record(step=3, observation=original, action=decision.best_action,
                             evaluation=decision, next_observation=moved,
                             info={'time_elapsed': 4/15, 'game_state':'game_over'},
                             terminated=True, truncated=False)
    assert report['is_death']
    assert report['history_steps'] == 3
    assert report['primary_label'] in ('unexplained_or_horizon_limited', 'predicted_lethal_action')
    assert len(report['history'][-1]['rankings']) == 2
    assert report['end_time_s'] == 4/15
    analyzer.reset(456, 10)
    spawn_report = analyzer.record(step=0, observation=original, action=decision.best_action,
                                   evaluation=decision, next_observation=added,
                                   info={'time_elapsed': 0.1, 'game_state': 'game_over'},
                                   terminated=True, truncated=False)
    assert spawn_report['primary_label'] == 'possible_unobserved_projectile'
    analyzer.reset(789, 11)
    censored = analyzer.record(step=0, observation=original, action=decision.best_action,
                               evaluation=decision, next_observation=moved,
                               info={'time_elapsed': 0.1, 'game_state': 'playing'},
                               terminated=False, truncated=True)
    assert censored['primary_label'] == 'censored_or_non_death'
    summary = summarize_reports([report, spawn_report, censored])
    assert summary['deaths'] == 2 and summary['episodes'] == 3
    try:
        FailureAnalyzer(FailureConfig(history_seconds=0))
        raise AssertionError('invalid history_seconds accepted')
    except ValueError:
        pass
    print('FailureAnalyzer regression passed!')


if __name__ == '__main__':
    run()
