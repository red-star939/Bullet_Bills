"""Run: python -m unittest test_adaptive_altitude_controller -v"""
import unittest
import numpy as np
from models.mario_motion_predictor import MarioMotionPredictor
from models.forward_simulator import ForwardSimulator
from models.adaptive_altitude_controller import AdaptiveAltitudeController, AltitudeConfig


class AltitudeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = ForwardSimulator(MarioMotionPredictor.from_object_tiles_csv('map_data/object_tiles.csv'))

    def obs(self, lower=0, upper=0, x=550, y=600):
        bullets = []
        for i in range(lower):
            bullets.append([30+i*120, 690+i%2*20, 64+i*120, 720+i%2*20, 400])
        for i in range(upper):
            bullets.append([50+i*120, 220, 84+i*120, 245, -400])
        return dict(mario=np.array([x,y,x+32,y+40,0,0,0,0], dtype=np.float32),
                    bullet_bills=np.asarray(bullets,dtype=np.float32).reshape(-1,5),
                    stars=np.empty((0,4),dtype=np.float32))

    def test_quiet_preserves_continuous_action(self):
        c = AdaptiveAltitudeController(self.sim)
        action = np.array([.37,.65],dtype=np.float32)
        for _ in range(8):
            result, meta = c.decide(self.obs(lower=1), action)
            np.testing.assert_array_equal(result,action)
            self.assertFalse(meta['altitude_overridden'])

    def test_quiet_upper_not_escalated(self):
        c = AdaptiveAltitudeController(self.sim)
        for _ in range(6):
            a, meta = c.decide(self.obs(lower=3,upper=3), [0,0])
            self.assertEqual(meta['altitude_mode'],'acrobat')
            np.testing.assert_array_equal(a,[0,0])

    def test_upper_position_preserves_action(self):
        c = AdaptiveAltitudeController(self.sim)
        for _ in range(5):
            a, meta = c.decide(self.obs(lower=4,y=300),[.2,0])
            np.testing.assert_array_equal(a,np.array([.2,0],dtype=np.float32))
            self.assertEqual(meta['altitude_mode'],'upper_hold')

    def test_trigger_and_reset(self):
        c = AdaptiveAltitudeController(self.sim)
        obs = self.obs(lower=3)
        for _ in range(2):
            _, meta = c.decide(obs,[0,0])
            self.assertFalse(meta['altitude_overridden'])
        _, meta = c.decide(obs,[0,0])
        self.assertTrue(meta['altitude_plans'] > 0 or meta['altitude_reason']=='baseline_uncertain_or_lethal')
        self.assertIn(meta['altitude_reason'],('verified_upper_route','no_verified_better_ascent', 'baseline_uncertain_or_lethal'))
        c.reset()
        self.assertEqual(c.crowd_streak,0)
        self.assertEqual(c.escapes_attempted,0)

    def test_discrete_candidate_is_in_action_table(self):
        from models.ppo_policy import ACTION_TABLE
        c = AdaptiveAltitudeController(self.sim, candidate_actions=ACTION_TABLE)
        for _ in range(3):
            act, meta = c.decide(self.obs(lower=4), ACTION_TABLE[0])
            self.assertTrue(any(np.array_equal(act,a) for a in ACTION_TABLE))

    def test_wall_rejects_escape(self):
        c = AdaptiveAltitudeController(self.sim,
            config=AltitudeConfig(min_final_wall_clearance_px=1000))
        for _ in range(4):
            act, meta = c.decide(self.obs(lower=3),[0,0])
        self.assertFalse(meta['altitude_overridden'])


if __name__=='__main__':
    unittest.main()
