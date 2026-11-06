"""Engine-source collision semantics regression (no kymnasium needed)."""
from models.collision_predictor import (BulletState, engine_contact_outcome,
                                        resolve_engine_contacts)
from models.interaction_simulator import InteractionSimulator
from models.mario_motion_predictor import MarioMotionPredictor, MarioState, MotionConfig


def run():
    m = (0, 0, 48, 48)
    b = (0, 35, 48, 83)  # center Y=59, Mario bottom 48 <59
    assert engine_contact_outcome(m, b, 30) == 'stomp'
    assert engine_contact_outcome(m, b, 30, True) == 'stomp'  # stomp priority
    assert engine_contact_outcome(m, b, -30) == 'death'
    assert engine_contact_outcome(m, b, -30, True) == 'invincible_hit'
    assert engine_contact_outcome(m, b, 30, bullet_state='fallen') == 'ignored'
    assert engine_contact_outcome(m, (48, 35, 96, 83), 30) == 'none'
    # Already-overlapping and horizontally-entering Mario CAN stomp if post-conditions match.
    assert engine_contact_outcome((44, 0, 92, 48), (45, 35, 93, 83), 30) == 'stomp'
    # Below the bullet's center is lethal even when descending.
    assert engine_contact_outcome((0, 15, 48, 63), b, 30) == 'death'
    # Sequential bullet order: first stomp sets vy=-500 -> next bullet becomes lethal.
    two = [BulletState(1, b, 0), BulletState(2, b, 0)]
    outcomes = resolve_engine_contacts(m, 30, two)
    assert [o.outcome for o in outcomes] == ['stomp', 'death']
    cfg = MotionConfig(frame_skip=1)
    sim = InteractionSimulator(MarioMotionPredictor(config=cfg))
    # At frame 337, Mario moves down 6px; bullet moves -4px.
    mario = MarioState(624,526,48,48,0.0,366.666656,False)
    bullet = BulletState(11,(632,575,680,623), -241.609467)
    res = sim.simulate(mario, [bullet], [(0.0, 0.0)])
    assert res.events[0].event_type == 'stomp'
    assert res.mario.vy == -500
    assert res.bullets[0].state == 'fallen'
    assert res.bullets[0].bbox == (628,575,676,623)
    assert not res.terminated
    # The fallen bullet moves vertically by int(700/60)=11 next frame.
    res2 = sim.simulate(res.mario, res.bullets, [(0.0, 0.0)])
    assert res2.bullets[0].bbox == (628,586,676,634)
    assert not res2.terminated
    # Normal side collision terminates, while invincible Mario survives and bullet falls.
    mario = MarioState(624,526,48,48,0.,-100.,False,False)
    bullet = BulletState(12,(624,526,672,574),0.)
    death = sim.simulate(mario,[bullet],[(0.,0.)])
    assert death.terminated and death.termination_reason == 'bullet'
    mario.invincible = True
    safe = sim.simulate(mario,[bullet],[(0.,0.)])
    assert not safe.terminated
    assert safe.events[0].event_type == 'invincible_hit'
    assert safe.bullets[0].state == 'fallen'
    print('Engine collision rule regression passed!')


if __name__ == '__main__':
    run()
