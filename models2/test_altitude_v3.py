"""V3 inheritance, shadow identity and conservative action admission tests."""
import unittest
from types import SimpleNamespace
import numpy as np
from .altitude_v2 import V2Config
from .altitude_v3 import AdaptiveAltitudeV3, V3Config


class FakeV1:
    def __init__(self, changed=False):
        self.changed = changed
        self.count = 0
    def reset(self):
        self.count = 0
    def decide(self, obs, action):
        self.count += 1
        x = np.asarray(action, np.float32).copy()
        if self.changed:
            x[0] = -0.5
        return x, {'altitude_overridden': self.changed, 'altitude_reason': 'fake_v1'}


class FakeSim:
    frame_skip = 4
    motion = SimpleNamespace(cfg=SimpleNamespace(world_width=1296))
    def __init__(self, death_frames=4, star=False):
        self.death_frames = death_frames
        self.star = star
    def simulate_observation(self, obs, actions, stop_on_star=True):
        first = np.asarray(actions[0]); lethal = first[0] < 0.5
        if self.star and lethal:
            reason, terminated, frames = 'unknown_star_pickup', False, 3
        elif lethal:
            reason, terminated, frames = 'bullet_death', True, self.death_frames
        else:
            reason, terminated, frames = None, False, len(actions)*4
        mario = SimpleNamespace(left=600, right=648, top=400, bottom=448)
        return SimpleNamespace(termination_reason=reason, terminated=terminated,
                               frames_simulated=frames, mario=mario, events=[])


class V3Tests(unittest.TestCase):
    def setUp(self):
        self.obs = dict(mario=np.array([600,650,648,698,0,0,0,0], np.float32),
                        bullet_bills=np.array([[100,500,148,548,250]]*10, np.float32))
        self.v2cfg = V2Config(crowd_confirm=1, replan_every=1)

    def make(self, mode='shadow', **kwargs):
        sim = kwargs.pop('sim', FakeSim())
        v1 = kwargs.pop('v1', FakeV1())
        return AdaptiveAltitudeV3(sim, v3_config=V3Config(mode=mode, **kwargs),
                                  v2_config=self.v2cfg, v1=v1)

    def test_shadow_is_identical_to_v1_even_if_v2_proposes(self):
        p = self.make()
        out, m = p.decide(self.obs, [0., 0.])
        self.assertTrue(m['v2_proposed'])
        np.testing.assert_array_equal(out, np.array([0.,0.],np.float32))
        self.assertFalse(m['v3_overridden'])
        self.assertEqual(p.shadow.differences, 1)

    def test_active_rescues_imminent_explicit_death(self):
        p = self.make('active')
        out, m = p.decide(self.obs, [0., 0.])
        self.assertTrue(m['v3_overridden'])
        self.assertGreaterEqual(float(out[0]), 0.5)

    def test_active_does_not_override_distant_constant_action_death(self):
        p = self.make('active', sim=FakeSim(death_frames=16))
        out, m = p.decide(self.obs, [0., 0.])
        self.assertFalse(m['v3_overridden'])
        self.assertEqual(float(out[0]), 0.0)

    def test_unknown_star_is_preserved(self):
        p = self.make('active', sim=FakeSim(star=True))
        out, m = p.decide(self.obs, [0., 0.])
        self.assertFalse(m['v3_overridden'])
        self.assertEqual(float(out[0]), 0.0)

    def test_v1_intervention_is_protected(self):
        p = self.make('active', v1=FakeV1(changed=True))
        out, m = p.decide(self.obs, [0., 0.])
        self.assertFalse(m['v3_overridden'])
        self.assertEqual(float(out[0]), -0.5)

    def test_no_crowd_does_not_change_action(self):
        p = self.make('active')
        obs = dict(self.obs, bullet_bills=self.obs['bullet_bills'][:1])
        out, m = p.decide(obs, [0.3, 0.5])
        np.testing.assert_array_equal(out, np.array([0.3,0.5], np.float32))
        self.assertFalse(m['v2_proposed'])

if __name__ == '__main__':
    unittest.main()
