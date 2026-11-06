"""Synthetic, Gym-free regressions for conservative fall-aware v2."""
import numpy as np
from models.action_evaluator_v2 import ActionEvaluatorV2
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor, MotionConfig


def obs(mario, bullets=()):
    return {'mario': np.asarray(mario, dtype=np.float32),
            'bullet_bills': np.asarray(bullets, dtype=np.float32).reshape(-1, 5),
            'stars': np.empty((0, 4), dtype=np.float32)}


def run():
    sim = ForwardSimulator(MarioMotionPredictor(config=MotionConfig()))
    evaluator = ActionEvaluatorV2(sim, horizon_steps=2, extended_horizon_steps=16,
                                  audit_top_k=3, candidate_actions=[(0, 0), (1, 0), (-1, 0)])
    safe = obs([624, 400, 672, 448, 0, 0, 0, 1])
    result = evaluator.evaluate(safe)
    assert result.best_action == result.baseline_action
    assert result.extra_simulations <= 3
    assert result.baseline.horizon_steps == 2
    # v2 must reject unsafe short-horizon candidates, independently of
    # extended hypothetical safety.
    immediate = obs([624, 720, 672, 768, 0, 0, 0, 0],
                    [[624, 720, 672, 768, 0]])
    result2 = evaluator.evaluate(immediate)
    assert result2.best_action == result2.baseline_action
    assert result2.extra_simulations == 0
    # A targeted platform case: standing still falls below the screen, while
    # moving right reaches a wide platform and stays supported. Nominal v1
    # prefers stillness in the clear 2-step window, but v2 must rescue it.
    motion_platform = MarioMotionPredictor(
        platforms=[(680, 760, 1296, 808)], config=MotionConfig())
    platform_sim = ForwardSimulator(motion_platform)
    rescue = ActionEvaluatorV2(
        platform_sim, horizon_steps=2, extended_horizon_steps=16,
        audit_top_k=3, candidate_actions=[(0, 0), (-1, 0), (1, 0)])
    rescue_obs = obs([620, 680, 668, 728, 0, 0, 0, 1])
    rescued = rescue.evaluate(rescue_obs)
    assert rescued.baseline_action == (0., 0.)
    assert rescued.best_action == (1., 0.)
    assert rescued.changed_from_baseline
    assert rescued.change_reason == 'avoided_predicted_extended_fall'
    assert rescued.audits[0].fall_frame is not None
    assert rescued.audits[-1].verified_no_fall_in_known_world
    # Guardrail: on the platform, no override is needed.
    supported = rescue.evaluate(obs([900, 712, 948, 760, 0, 0, 0, 0]))
    assert supported.best_action == supported.baseline_action
    # Validations and public-interface consistency.
    for kwargs in ({'horizon_steps': 8, 'extended_horizon_steps': 7},
                   {'horizon_steps': 8, 'audit_top_k': 0}):
        try:
            ActionEvaluatorV2(sim, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid config accepted')
    print('ActionEvaluator v2 conservative fall-audit regression passed!')


if __name__ == '__main__':
    run()
