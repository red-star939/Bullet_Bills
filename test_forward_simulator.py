"""Deterministic no-Gym regression of the known-object ForwardSimulator."""
from models.mario_motion_predictor import MarioMotionPredictor, MotionConfig, MarioState
from models.collision_predictor import BulletState
from models.forward_simulator import ForwardSimulator


def run():
    cfg = MotionConfig(frame_skip=4)
    motion = MarioMotionPredictor(config=cfg)
    forward = ForwardSimulator(motion)
    start = MarioState(624, 720, 48, 48, 0, 0, True)
    actions = [[0., 0.]] * 3
    pure = motion.predict(start, actions)
    result = forward.simulate(start, [], actions)
    assert result.frames_simulated == 12
    assert result.actions_completed == 3
    assert result.termination_reason is None
    assert [f.mario_bbox for f in result.frames] == [f.bbox for f in pure]
    assert [(f.vx, f.vy) for f in result.frames] == [(f.vx, f.vy) for f in pure]

    # Bullet advances exactly 4 pixels/frame at nominal 285.86 px/s.
    bullet = BulletState(7, (100, 720, 148, 768), 285.86)
    res = forward.simulate(start, [bullet], [[0., 0.]])
    assert res.bullets[0].bbox == (116, 720, 164, 768)
    assert not res.terminated

    # Known stomp at frame 337 from original diagnosis (predict one internal frame).
    one = ForwardSimulator(MarioMotionPredictor(config=MotionConfig(frame_skip=1)))
    falling = MarioState(624, 526, 48, 48, 0., 366.666656, False)
    bullet = BulletState(11, (632, 575, 680, 623), -241.609467)
    res = one.simulate(falling, [bullet], [[0., 0.]])
    assert [(e.frame, e.kind) for e in res.events] == [(1, 'stomp')]
    assert res.mario.vy == -500.0
    assert res.bullets[0].state == 'fallen'
    assert res.bullets[0].bbox == (628, 575, 676, 623)
    follow = one.simulate(res.mario, res.bullets, [[0., 0.]])
    assert follow.bullets[0].bbox == (628, 586, 676, 634)

    # Non-stomp overlap is lethal; death stops inside the action step.
    airborne = MarioState(624, 526, 48, 48, 0., -100., False)
    bullet = BulletState(12, (624, 526, 672, 574), 0.)
    dead = forward.simulate(airborne, [bullet], [[0., 0.]] * 5)
    assert dead.terminated and dead.termination_reason == 'bullet'
    assert dead.first_lethal_event.frame == 1
    assert dead.frames_simulated == 1 and dead.actions_completed == 0
    airborne.invincible = True
    safe = forward.simulate(airborne, [bullet], [[0., 0.]])
    assert not safe.terminated
    assert safe.events[0].kind == 'invincible_hit'
    assert safe.bullets[0].state == 'fallen'
    assert 'invincibility_remaining_time_unknown' in safe.assumptions

    # Per-frame bullet ordering: a stomp makes the next overlap lethal.
    double = one.simulate(falling, [bullet, bullet], [[0., 0.]])
    # This particular pair is lethal without any stomp, proving order handling
    # and active contacts are not silently elided.
    assert double.terminated

    # Observable format uses LTRB (not pygame x/y/width/height).
    obs = {'mario': [624, 720, 672, 768, 0, 0, 0, 0],
           'bullet_bills': [[100, 720, 148, 768, 285.86], [0, 0, 0, 0, 0]],
           'stars': [[0, 0, 0, 0]]}
    seen = forward.bullets_from_observation(obs)
    assert len(seen) == 1 and seen[0].bbox == (100, 720, 148, 768)
    assert 'bullet_states_assumed_active' in forward.simulate_observation(obs, [[0, 0]]).assumptions

    # Unknown star state is explicitly censored rather than declaring survival.
    stars = [(624, 720, 672, 768)]
    censored = forward.simulate(start, [], [[0.,0.]], stars=stars)
    assert censored.terminated and censored.termination_reason == 'unknown_star_pickup'
    assert censored.frames_simulated == 1

    print('Forward Simulator regression passed!')


if __name__ == '__main__':
    run()
