"""Deterministic unit tests of support and trajectory interpretation."""
import numpy as np
from models.platform_safety_evaluator import PlatformSafetyEvaluator
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor, MotionConfig


def obs(x, y, grounded, vy=0):
    return {'mario': np.array([x, y, x + 48, y + 48, 0, vy, 0, not grounded], dtype=np.float32),
            'bullet_bills': np.empty((0, 5), dtype=np.float32),
            'stars': np.empty((0, 4), dtype=np.float32)}


def run():
    motion = MarioMotionPredictor(platforms=[(680, 760, 1296, 808)])
    ev = PlatformSafetyEvaluator(ForwardSimulator(motion), min_overlap_px=12)
    assert ev.supports((680, 712, 728, 760))[0][1] == 48
    assert ev.supports((632, 712, 680, 760)) == ()  # touching edge != overlap
    assert ev.supports((679, 712, 727, 760))[0][1] == 47
    grounded = ev.evaluate_observation(obs(800, 712, True), [(0, 0)] * 4)
    assert grounded.initially_supported and grounded.final_supported
    assert grounded.predicted_fall_frame is None
    assert not grounded.unsupported_grounded_frames
    unsupported = ev.evaluate_observation(obs(620, 680, False), [(0, 0)] * 20)
    assert unsupported.predicted_fall_frame is not None
    assert not unsupported.final_supported
    rescue = ev.evaluate_observation(obs(620, 680, False), [(1, 0)] * 16)
    assert rescue.predicted_fall_frame is None
    assert rescue.completed_known_world
    assert rescue.final_supported
    assert rescue.landings
    for kw in ({'min_overlap_px': 0}, {'min_edge_clearance_px': -1}):
        try:
            PlatformSafetyEvaluator(ForwardSimulator(motion), **kw)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid safety threshold accepted')
    print('PlatformSafetyEvaluator regression passed!')


if __name__ == '__main__':
    run()
