"""Unit regressions for candidate validation, finite-beam semantics and first-action replanning."""
import numpy as np
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor,MotionConfig
from models.mpc_controller import MPCController


def obs(mario=(624,720,672,768,0,0,0,0),bullets=()):
    return {'mario':np.asarray(mario,dtype=np.float32),
            'bullet_bills':np.asarray(bullets,dtype=np.float32).reshape(-1,5),
            'stars':np.zeros((0,4),dtype=np.float32)}

def run():
    sim=ForwardSimulator(MarioMotionPredictor(config=MotionConfig()))
    mpc=MPCController(sim,segment_steps=(1,2,2),beam_width=3,
        candidate_actions=((0,0),(-1,0),(1,0),(0,1)),max_nodes=100)
    r=mpc.evaluate(obs())
    assert len(r.action_sequence)==5
    assert r.best_action==r.action_sequence[0]
    assert r.evaluated_nodes<=100
    assert r.risk_class=='none_sampled'
    assert r.horizon_steps==5
    # Repeatability and no mutation of the caller's observation.
    o=obs();v=o['mario'].copy();a=mpc.evaluate(o)
    assert a==mpc.evaluate(o)
    np.testing.assert_array_equal(v,o['mario'])
    # Beam search changes piecewise action but never in the middle of a segment.
    assert a.action_sequence[1]==a.action_sequence[2]
    assert a.action_sequence[3]==a.action_sequence[4]
    # Robust finalists are a non-probabilistic sensitivity sweep.
    mpc2=MPCController(sim,segment_steps=(1,1),beam_width=2,
        candidate_actions=((0,0),(0,1)),robust_finalists=1)
    assert mpc2.evaluate(obs()).best_action in ((0.,0.),(0.,1.))
    print('MPC controller regression passed!')

if __name__=='__main__':run()
