"""Regression: pygame.Rect is XYWH when cast to tuple, but predictor uses LTRB.
Run from mario_pj root after copying the v2 validator there.
"""
from test_collision_interaction_multiseed_v2 import bullet_snapshot
from models.collision_predictor import engine_contact_outcome


class Rect:
    def __init__(self, left, top, right, bottom):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom
        self.width, self.height = right-left, bottom-top

    def __iter__(self):  # pygame.Rect's tuple representation is XYWH
        return iter((self.left, self.top, self.width, self.height))


class Bullet:
    def __init__(self):
        self.rect = Rect(600, 575, 648, 623)
        self.state_ = 'active'
        self._velocity = -285.86


class Core:
    def __init__(self):
        self._bullet_bills = [Bullet()]


snapshot = next(iter(bullet_snapshot(Core()).values()))
bbox = snapshot[2]
assert tuple(Core()._bullet_bills[0].rect) == (600, 575, 48, 48)
assert bbox == (600, 575, 648, 623), bbox
assert engine_contact_outcome((624, 532, 672, 580), bbox, 400.0, False, 'active') == 'stomp'
print('Bullet bbox XYWH-to-LTRB contract regression passed!')
