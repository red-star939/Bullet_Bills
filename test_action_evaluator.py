"""No-Gym deterministic regression for risk-first ActionEvaluator."""
import numpy as np
from models.action_evaluator import ActionEvaluator
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor, MotionConfig


def run():
    motion = MarioMotionPredictor(config=MotionConfig())
    sim = ForwardSimulator(motion)
    # A stationary side-on bullet at Mario's current location kills immediately;
    # jumping cannot evade before the first internal frame in this setup.
    obs = {'mario': np.array([624, 526, 672, 574, 0, -100, 0, 1], dtype=np.float32),
           'bullet_bills': np.array([[624, 526, 672, 574, 0]], dtype=np.float32),
           'stars': np.zeros((1, 4), dtype=np.float32)}
    evaluator = ActionEvaluator(sim, horizon_steps=2,
                                candidate_actions=[(0, 0), (1, 0), (-1, 0)],
                                robust_mode='all')
    result = evaluator.evaluate(obs)
    assert len(result.rankings) == 3
    assert result.robust_evaluations == 3
    assert all(item.risk_class == 'sampled_lethal' for item in result.rankings)
    assert all(item.first_possible_lethal_frame is not None for item in result.rankings)
    assert [x.rank for x in result.rankings] == [1, 2, 3]
    assert result.best_action == result.best.action
    # Clear world: no false threats, no false probability claims.
    obs['bullet_bills'] = np.zeros((0, 5), dtype=np.float32)
    clear = evaluator.evaluate(obs)
    assert all(r.risk_class == 'none_sampled' for r in clear.rankings)
    assert clear.best_action == (0., 0.)
    # A moving bullet can hit a nonjumping action. All nonlethal jump
    # candidates must rank ahead of a known lethal stay-put action.
    approaching = {
        'mario': np.array([624, 720, 672, 768, 0, 0, 0, 0], dtype=np.float32),
        'bullet_bills': np.array([[760, 720, 808, 768, -240]], dtype=np.float32),
        'stars': np.zeros((0, 4), dtype=np.float32),
    }
    evasive = ActionEvaluator(sim, horizon_steps=8,
                              candidate_actions=[(0, 0), (0, 1)],
                              robust_mode='all').evaluate(approaching)
    assert evasive.best_action == (0., 1.)
    assert evasive.rankings[0].risk_class == 'none_sampled'
    assert evasive.rankings[1].risk_class == 'sampled_lethal'
    # Quick nominal mode does not call robust, keeping API/assumptions explicit.
    nominal = ActionEvaluator(sim, horizon_steps=1,
                              candidate_actions=[(0, 0), (1, 0)], robust_mode='none')
    n = nominal.evaluate(obs)
    assert n.robust_evaluations == 0 and all(x.robust_status is None for x in n.rankings)
    # Validate invalid action and invalid horizon.
    try:
        ActionEvaluator(sim, horizon_steps=0)
    except ValueError:
        pass
    else:
        raise AssertionError('zero horizon should fail')
    print('ActionEvaluator regression passed!')


if __name__ == '__main__':
    run()
