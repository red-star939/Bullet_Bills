"""Gym-free regression: v2 preservation, deterministic multi-action recovery and integrity."""
import numpy as np
from models.action_evaluator_v3 import ActionEvaluatorV3
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor, MotionConfig


def obs(mario, bullets=()):
    return {'mario': np.asarray(mario, dtype=np.float32),
            'bullet_bills': np.asarray(bullets, dtype=np.float32).reshape(-1, 5),
            'stars': np.empty((0, 4), dtype=np.float32)}


def run():
    sim = ForwardSimulator(MarioMotionPredictor(
        platforms=[(0, 448, 1296, 496)], config=MotionConfig()))
    v3 = ActionEvaluatorV3(sim, horizon_steps=2, extended_horizon_steps=16,
                           audit_top_k=3, recovery_top_k=3,
                           recovery_first_steps=(2, 5),
                           candidate_actions=((0, 0), (1, 0), (-1, 0)),
                           continuation_actions=((0, 0), (1, 0), (-1, 0)))
    quiet = obs([624, 400, 672, 448, 0, 0, 0, 0])
    r = v3.evaluate(quiet)
    assert r.best_action == r.baseline_action
    assert not r.changed_from_baseline
    assert r.extra_simulations == 1
    assert r.change_reason == 'v2_retained_no_recovery_trigger'
    # Off-platform near terminal condition: evaluate multi-action recovery.
    edge_sim = ForwardSimulator(MarioMotionPredictor(
        platforms=[(680, 760, 1296, 808)], config=MotionConfig()))
    rescue = ActionEvaluatorV3(edge_sim, horizon_steps=2, extended_horizon_steps=16,
                               audit_top_k=3, recovery_top_k=3,
                               recovery_first_steps=(2, 5),
                               candidate_actions=((0, 0), (-1, 0), (1, 0)),
                               continuation_actions=((0, 0), (-1, 0), (1, 0)))
    rescue_obs = obs([620, 680, 668, 728, 0, 0, 0, 1])
    outcome = rescue.evaluate(rescue_obs)
    assert outcome.best_action in rescue.v2.baseline.candidate_actions
    assert outcome.extra_simulations <= 1 + 3 * 2 * 3
    assert outcome.baseline.best_action == (1.0, 0.0)
    # Strictly no fabricated success on an uncertain pickup.
    stars = dict(quiet)
    stars['stars'] = np.array([[620, 398, 673, 449]], dtype=np.float32)
    unknown = v3.evaluate(stars)
    assert not unknown.changed_from_baseline
    # Identical call -> deterministic outcome (no dependence on RNG).
    again = rescue.evaluate(rescue_obs)
    assert again.best_action == outcome.best_action
    assert again.change_reason == outcome.change_reason
    for kwargs in ({'recovery_first_steps': (0,)}, {'recovery_first_steps': (16,)},
                   {'recovery_top_k': 0}):
        try:
            ActionEvaluatorV3(sim, horizon_steps=2, extended_horizon_steps=16, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid V3 settings accepted')
    print('ActionEvaluator v3 synthetic regression passed!')


if __name__ == '__main__':
    run()
