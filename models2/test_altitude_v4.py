"""Dependency-isolated v4 tests; full game still requires user-installed kymnasium."""
import unittest
from types import SimpleNamespace
import numpy as np
from .altitude_v4 import AdaptiveAltitudeV4, V4Config


class FakeV1:
    def reset(self):
        self.count = 0
    def decide(self, obs, action):
        self.count += 1
        return np.asarray(action, np.float32).copy(), {'altitude_overridden': False, 'altitude_reason':'v1_mock'}


class FakePhysics:
    SOLID = None
    @staticmethod
    def restore_vy(v):
        return v
    def simulate_plans(self, m, bullets, stars, inv, sequences, solid):
        times = np.full(len(sequences), -1, np.int64)
        # On mock rescue setup: constant and coast end in fatal contact; rightward
        # action is complete. Illustrates that rescue is allowed, not real physics.
        if m[0] == 600:
            times[:2] = 10
            for i in range(2, len(sequences)):
                if sequences[i, 0, 0] <= .2:
                    times[i] = 12
        return times, np.where(times < 0, 0, 1)
    def simulate_trajectory(self, m, bullets, stars, inv, seq, solid):
        t = np.zeros((24,5),np.float64)
        t[:,0] = 650
        t[:,1] = 500 if seq[0,1] > .1 else 600
        return t, -1, 0


def obs(x=600):
    bullets = [[90+i*90, 620, 138+i*90, 668, 260.] for i in range(12)]
    bullets += [[0,0,0,0,0]]*18
    return {'mario':np.asarray([x,600,x+48,648,0,0,0,0], np.float32),
            'bullet_bills':np.asarray(bullets,np.float32),
            'stars':np.zeros((2,4),np.float32)}


class TestV4(unittest.TestCase):
    def test_filter_padding(self):
        self.assertEqual(len(AdaptiveAltitudeV4._valid_bullets(obs())), 12)
    def test_shadow_always_retains_original(self):
        a = AdaptiveAltitudeV4(None, FakePhysics(), config=V4Config(mode='shadow',replan_every=1), v1=FakeV1())
        original = np.asarray([0.,0.],np.float32)
        for _ in range(5):
            chosen, report = a.decide(obs(), original)
            np.testing.assert_array_equal(chosen,original)
            self.assertFalse(report['v4_overridden'])
        self.assertEqual(a.overrides,0)
    def test_active_rescue(self):
        a = AdaptiveAltitudeV4(None, FakePhysics(), config=V4Config(mode='active',replan_every=1,
                 min_survival_gain_frames=20), v1=FakeV1())
        chosen, report = a.decide(obs(), [0.,0.])
        self.assertTrue(report['v4_overridden'],report)
        self.assertGreater(chosen[0],.2)
        self.assertEqual(a.overrides,1)
    def test_guard_invalid_config(self):
        with self.assertRaises(ValueError):
            AdaptiveAltitudeV4(None,FakePhysics(),config=V4Config(mode='unsafe'),v1=FakeV1())
    def test_no_bullets(self):
        a=AdaptiveAltitudeV4(None, FakePhysics(),config=V4Config(mode='explore',replan_every=1),v1=FakeV1())
        o=obs(); o['bullet_bills'][:]=0
        chosen,meta=a.decide(o,[.37,.5])
        np.testing.assert_array_equal(chosen,np.asarray([.37,.5],np.float32))
        self.assertEqual(meta['plans_tested'],0)

if __name__=='__main__': unittest.main()
