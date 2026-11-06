"""Regression for engine-witness-based observed collision event labels.
Run: python test_collision_event_audit.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from models.collision_event_audit import (
    CollisionWitness, WitnessBullet, observed_collision_events,
)


def check(witness, after, expected, alive=True, bottom=580):
    result = observed_collision_events(witness, after, alive, bottom)
    assert result == [expected], (result, expected)


def main():
    bullet = WitnessBullet(9, (600, 575, 648, 623), 'active')
    check(CollisionWitness((624, 532, 672, 580), 400.0, False, (bullet,)),
          {9: 'fallen'}, 'stomp')
    check(CollisionWitness((624, 532, 672, 580), 400.0, True, (bullet,)),
          {9: 'fallen'}, 'stomp')  # stomp precedes invincibility
    check(CollisionWitness((624, 532, 672, 580), -500.0, True, (bullet,)),
          {9: 'fallen'}, 'invincible_hit')
    check(CollisionWitness((624, 532, 672, 580), 400.0, False, (bullet,)),
          {9: 'active'}, 'death', alive=False)
    check(CollisionWitness((624, 532, 672, 580), 400.0, False, (bullet,)),
          {9: 'active'}, 'death', alive=False, bottom=580)
    assert observed_collision_events(
        CollisionWitness((624, 532, 672, 580), 400.0, False, (bullet,)),
        {9: 'active'}, False, 900) == []  # out-of-world fall, not bullet death
    b2 = WitnessBullet(10, (605, 575, 653, 623), 'active')
    check(CollisionWitness((624, 532, 672, 580), 400, True, (bullet, b2)),
          {9: 'fallen', 10: 'fallen'}, 'multiple_hit')

    # Optional regression on user-provided old event details. The 27 old
    # mislabels reached -500 from gravity while already moving upwards.
    log = Path(__file__).resolve().parent / 'event_details.csv'
    if log.is_file():
        with log.open(encoding='utf-8-sig', newline='') as f:
            errors = [row for row in csv.DictReader(f) if row['kind'] == 'mismatch']
        assert len(errors) == 27, f'Unexpected legacy log rows: {len(errors)}'
        for row in errors:
            assert (row['pred_events'], row['actual_events']) == ('invincible_hit', 'stomp')
            # The user log encodes numpy scalar repr; its exact numeric value
            # can be read via a literal-safe regex (no code execution).
            import re
            values = re.findall(r'np\.float64\((-?[\d.eE+]+)\)', row['pre_mario'])
            assert len(values) == 6 and float(values[5]) < 0
        print('Legacy 27 false-stomp labels: all start with upward vy.')
    print('Collision event audit regression passed!')


if __name__ == '__main__':
    main()
