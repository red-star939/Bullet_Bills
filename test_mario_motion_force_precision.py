r"""Test force conversion, including non-binary fractions, without kymnasium.

From project root:
  .\.venv\Scripts\python.exe test_mario_motion_force_precision.py
"""
import math

import numpy as np

from models.mario_motion_predictor import MarioMotionPredictor, MarioState


def main():
    model = MarioMotionPredictor.from_object_tiles_csv('map_data/object_tiles.csv')
    for x, y in [(-1.0, 1.0), (-0.5, 1.0), (0.0, 0.0),
                 (0.1, 0.3), (0.3, 0.7), (0.7, 0.1), (1.0, 0.0)]:
        a = np.asarray([x, y], dtype=np.float32)
        expected = (float(a[0] * model.cfg.ground_force),
                    -float(a[1] * model.cfg.jump_speed))
        actual = model.action_to_force(a)
        assert actual == expected, (a, actual, expected)
        print(f'action=({x:+.7f},{y:.7f}) -> force={actual!r}')

    s = MarioState(624, 720, 48, 48, 0.0, 0.0, grounded=True)
    force = model.action_to_force(np.asarray([0.3, 0.7], dtype=np.float32))
    result, subframes = model.step(s, [0.3, 0.7])
    current = s
    for frame in subframes:
        current = model._internal_step_force(current, force)
        assert current.bbox == frame.bbox
        assert math.isclose(current.vx, frame.vx, abs_tol=1e-10)
        assert math.isclose(current.vy, frame.vy, abs_tol=1e-10)
    assert result.bbox == current.bbox

    for bad in ([1.1, 0], [0, -0.1], [float('nan'), 0], [0], [0, 0, 0]):
        try:
            model.action_to_force(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'should reject {bad}')
    print('Force precision regression tests passed!')


if __name__ == '__main__':
    main()
