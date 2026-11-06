"""Synthetic regression for first observed Bullet Bill bounce geometry."""
from models.collision_predictor import BulletState, detect_contacts, effective_bullet_vx
from models.interaction_simulator import InteractionSimulator
from models.mario_motion_predictor import MarioMotionPredictor, MarioState


def run():
    assert effective_bullet_vx(-241.609467) == -4
    assert effective_bullet_vx(239.30452) == 3

    # Episode 4, frames 336 -> 337: Mario [624,526,672,574],
    # bullet [632,575,680,623] moves left by 4 pixels.
    m0 = (624, 526, 672, 574)
    m1 = (624, 532, 672, 580)
    bullet = BulletState(11, (632, 575, 680, 623), -241.609467)
    contacts, moved = detect_contacts(m0, m1, 366.666656, [bullet], 337)
    assert len(contacts) == 1 and contacts[0].kind == 'stomp_candidate'
    assert moved[0].bbox == (628, 575, 676, 623)

    # Episode 593, frames 114 -> 115: bullet moves right by 3 pixels.
    bullet = BulletState(12, (36, 574, 84, 622), 239.30452)
    contacts, moved = detect_contacts((56, 525, 104, 573),
                                      (48, 532, 96, 580),
                                      433.333344, [bullet], 115)
    assert len(contacts) == 1 and contacts[0].kind == 'stomp_candidate'
    assert moved[0].bbox == (39, 574, 87, 622)

    # Frame-level integration: starting at ep4's frame336, after one physics
    # frame Mario is still at (624,532,672,580) while vy becomes -500.
    motion = MarioMotionPredictor()
    sim = InteractionSimulator(motion)
    state = MarioState(624, 526, 48, 48, 0.0, 366.666656, False)
    result = sim.simulate(state, [BulletState(11, (632, 575, 680, 623), -241.609467)],
                          [[0.0, 0.0]])
    assert result.events[0].event_type == 'stomp'
    assert result.events[0].frame == 1
    assert result.events[0].mario_bbox == (624, 532, 672, 580)
    assert result.mario.vy > -500  # gravity applies after the bounce in later frames
    assert any(b.bullet_id == 11 and b.state == "fallen" for b in result.bullets)
    print('Collision and interaction synthetic regression passed!')


if __name__ == '__main__':
    run()
