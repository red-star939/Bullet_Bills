"""Collision classification v2 regressions; run without kymnasium installed."""
from models.collision_predictor import (
    BulletState, CollisionPredictor, classify_contact_detail, detect_contacts,
    effective_bullet_vx, overlaps,
)
from models.interaction_simulator import InteractionSimulator
from models.mario_motion_predictor import MarioMotionPredictor, MarioState


def run():
    # Strict Pygame rectangle intersection; adjacent edges are not a hit.
    assert not overlaps((0, 0, 48, 48), (48, 0, 96, 48))
    assert not overlaps((0, 0, 48, 48), (0, 48, 48, 96))
    assert overlaps((0, 0, 48, 48), (47, 47, 95, 95))
    assert effective_bullet_vx(239.9) == 3
    assert effective_bullet_vx(-241.609467) == -4

    # Stomp #1: episode 4, internal frame 337.
    candidate, _ = detect_contacts(
        (624,526,672,574), (624,532,672,580), 366.666656,
        [BulletState(11, (632,575,680,623), -241.609467)], frame=337)
    assert len(candidate) == 1
    assert (candidate[0].kind, candidate[0].entry_face) == ('stomp_candidate', 'top')

    # Stomp #2: episode 593, internal frame 115.
    candidate, _ = detect_contacts(
        (56,525,104,573), (48,532,96,580), 433.333344,
        [BulletState(12, (36,574,84,622), 239.30452)], frame=115)
    assert (candidate[0].kind, candidate[0].entry_face) == ('stomp_candidate', 'top')

    # Episode 4/frame 539: upward collision with a side/underside bullet.
    # Recorded post-frame bullet: [579,500,627,548] moving horizontally,
    # Mario pre [624,538,672,586], post [624,531,672,579].
    # It must NOT be a top-down stomp.
    result = CollisionPredictor().predict_frame(
        (624,538,672,586), (624,531,672,579), -466.666656,
        [BulletState(0, (583,500,631,548), -241.609467)], frame=539)
    assert result.has_unresolved_contact and not result.has_stomp_candidate
    assert result.contacts[0].kind == 'side_or_bottom'

    # Bullet touching the foot but not penetrating cannot trigger stomp.
    detail = classify_contact_detail(
        (0,0,48,48), (0,0,48,48), (0,48,48,96), (0,48,48,96), 100.)
    assert detail.kind == 'none'

    # Already overlapping cannot be treated as a new stomp.
    detail = classify_contact_detail(
        (0,0,48,50), (0,1,48,51), (0,48,48,96), (0,48,48,96), 100.)
    assert detail.kind == 'side_or_bottom' and detail.pre_existing_overlap

    # Horizontal entry is hazardous, not a top stomp, despite descending vy.
    detail = classify_contact_detail(
        (0,15,48,63), (4,16,52,64), (48,0,96,48), (48,0,96,48), 100.)
    assert (detail.kind,detail.entry_face) == ('side_or_bottom','left')

    # Multiple bullet candidates are reported independently.
    result = CollisionPredictor().predict_frame(
        (0,0,48,48), (0,3,48,51), 100.,
        [BulletState(1, (0,50,48,98), 0.),
         BulletState(2, (24,50,72,98), 0.)], 1)
    assert len(result.contacts) == 2 and result.has_stomp_candidate

    # Existing simulation API and bounce semantics must still work.
    sim = InteractionSimulator(MarioMotionPredictor())
    s = MarioState(624,526,48,48,0.0,366.666656,False)
    output = sim.simulate(
        s,[BulletState(11,(632,575,680,623),-241.609467)],[[0.,0.]])
    assert output.events[0].event_type == 'stomp'
    assert output.events[0].frame == 1
    assert not output.unresolved
    print('Collision Predictor v2 precision regression passed!')


if __name__ == '__main__':
    run()
