"""Run from project root: python test_mario_motion_predictor.py"""
import math
from pathlib import Path

from models.mario_motion_predictor import MarioMotionPredictor, MarioState, MotionConfig


def main():
    cfg = MotionConfig()
    assert cfg.frame_skip / cfg.fps == 4 / 60
    assert MarioMotionPredictor.overlaps((0, 0, 48, 48), (47, 0, 95, 48))
    assert not MarioMotionPredictor.overlaps((0, 0, 48, 48), (48, 0, 96, 48))

    file = Path('map_data/object_tiles.csv')
    if not file.exists():
        raise FileNotFoundError('map_data/object_tiles.csv not found; run extract_map.py first')
    predictor = MarioMotionPredictor.from_object_tiles_csv(file)
    state = MarioState.from_observation([624, 720, 672, 768, 0, 0, 0, 0])
    assert state.bbox == (624, 720, 672, 768)
    assert predictor._supported(state), 'Initial Mario location should be on the center platform'
    state.grounded = True
    still, frames = predictor.step(state, [0, 0])
    assert still.bbox == state.bbox
    assert len(frames) == 4
    assert still.grounded
    print('[STILL] bbox:', still.bbox)

    jumped, jump_frames = predictor.step(state, [0, 1])
    assert jumped.top < state.top
    assert jumped.vy < 0
    print('[JUMP] bbox:', jumped.bbox, 'vy:', round(jumped.vy, 3))

    moved, _ = predictor.step(state, [1, 0])
    assert moved.left > state.left
    print('[RIGHT] bbox:', moved.bbox, 'vx:', round(moved.vx, 3))

    seq = predictor.predict(state, [[0, 1]] + [[0, 0]] * 4)
    assert len(seq) == 20
    assert math.isclose(seq[-1].time_s, 20 / 60)
    print('[TRAJECTORY] samples:', len(seq))
    print('All MarioMotionPredictor unit tests passed!')


if __name__ == '__main__':
    main()
