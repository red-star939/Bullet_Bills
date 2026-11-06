"""Regression tests for fractional horizontal inputs (no kymnasium required)."""
from math import isclose
from pathlib import Path
from models.mario_motion_predictor import MarioMotionPredictor, MarioState


def main():
    predictor = MarioMotionPredictor.from_object_tiles_csv(
        Path('map_data/object_tiles.csv'))
    # A fractional input must be scaled by 100 BEFORE applying the air cap 75.
    cases = [(-1.0, -75.0), (-0.5, -50.0), (0.0, 0.0),
             (0.5, 50.0), (1.0, 75.0)]
    for x, expected_force in cases:
        # Start airborne, so no platform or jump changes the force mode.
        s = MarioState(624, 400, 48, 48, 0.0, -200.0, grounded=False)
        result = predictor._internal_step(s, (x, 0.0))
        expected_vx = expected_force * predictor.cfg.friction
        assert isclose(result.vx, expected_vx, abs_tol=1e-9), (
            f'x={x}: actual={result.vx} expected={expected_vx}')
        print(f'[AIR] x={x:+.1f} requested={x*100:+.1f} '
              f'applied={expected_force:+.1f} vx={result.vx:+.5f}')

    # On takeoff, the first internal frame applies ground force, the rest air.
    launch = MarioState(624, 720, 48, 48, 0, 0, grounded=True)
    after, frames = predictor.step(launch, (-0.5, 1.0))
    expected = [-45.0, -85.5, -121.95, -154.755]
    assert all(isclose(frame.vx, v, abs_tol=1e-8)
               for frame, v in zip(frames, expected))
    assert after.left == 619, (after.left, 'expected x=619')
    print('[TAKEOFF] x=-0.5, jump=1.0 -> x=', after.left,
          'vx=', round(after.vx, 6))
    print('Fractional-action regression tests passed.')


if __name__ == '__main__':
    main()
