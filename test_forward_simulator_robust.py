"""Regression for float32-quantization-sensitive forward collision times.

Run from project root:
  .\\.venv\\Scripts\\python.exe test_forward_simulator_robust.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from models.forward_simulator import ForwardSimulator, _float32_rounding_bin
from models.mario_motion_predictor import MarioMotionPredictor


def run():
    motion = MarioMotionPredictor.from_object_tiles_csv(Path('map_data/object_tiles.csv'))
    sim = ForwardSimulator(motion)

    # Engine event: env_seed=726388407, action_step=53.
    # Observation has float32 velocities, unlike engine's float64 internals.
    obs = {
        'mario': np.array([743, 525, 791, 573, 192.6423797607422,
                           -400.0, 0.0, 1.0], dtype=np.float32),
        'bullet_bills': np.array([
            [808, 458, 856, 506, -271.5428486337053],
            [1182, 473, 1230, 521, -220.64412135220073],
        ], dtype=np.float32),
        'stars': np.zeros((2, 4), dtype=np.float32),
    }
    action = [[1.0, 1.0]]
    old = sim.simulate_observation(obs, action)
    robust = sim.simulate_observation_robust(obs, action)
    assert old.termination_reason == 'bullet'
    assert robust.nominal.termination_reason == old.termination_reason
    assert robust.nominal.first_lethal_event.frame == 4
    assert len(robust.samples) == 9
    assert robust.collision_status == 'boundary_sensitive'
    assert robust.first_possible_lethal_frame == 4
    assert robust.position_spread_px[1] >= 1
    assert set(robust.lethal_frames) == {4, None}, robust.lethal_frames
    assert robust.initial_velocity_bounds[1][0] < -400.0 < robust.initial_velocity_bounds[1][1]
    assert 'finite_grid_not_guaranteed_bound' in robust.assumptions
    print('[BOUNDARY] nominal=death@4; sampled outcomes=', robust.lethal_frames)
    print('[BOUNDARY] vy bin=', robust.initial_velocity_bounds[1],
          'position spread=', robust.position_spread_px)

    # Test next physical action: samples that survived frame 4 can die at 5.
    two_steps = sim.simulate_observation_robust(obs, [[1., 1.], [1., 0.]])
    assert two_steps.collision_status == 'boundary_sensitive'
    assert 4 in two_steps.lethal_frames and 5 in two_steps.lethal_frames
    print('[TIMING] candidate death frames=', two_steps.lethal_frames)

    # No bullets: no phantom lethal events; same observation API preserved.
    clear = dict(obs)
    clear['bullet_bills'] = np.zeros((2, 5), dtype=np.float32)
    safe = sim.simulate_observation_robust(clear, [[0., 0.]])
    assert safe.collision_status == 'none_sampled'
    assert all(frame is None for frame in safe.lethal_frames)
    print('[CLEAR] no sampled collision')

    # The interval bounds enclose the float64 velocity seen in engine log.
    actual_internal_vy = -399.99999999999955
    lo, hi = _float32_rounding_bin(-400.0)
    assert lo <= actual_internal_vy <= hi

    print('Forward Simulator robust quantization regression passed!')


if __name__ == '__main__':
    run()
