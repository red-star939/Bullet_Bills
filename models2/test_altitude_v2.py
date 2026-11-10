"""Dependency-free regression tests for the planner's decision logic."""
import unittest
import numpy as np
from types import SimpleNamespace
from .altitude_v2 import AdaptiveAltitudeV2, V2Config

class FakeSim:
    frame_skip = 4
    motion = SimpleNamespace(cfg=SimpleNamespace(world_width=1296))
    def simulate_observation(self, obs, actions, stop_on_star=True):
        first = np.asarray(actions[0])
        # Baseline is lethal; rightward jumping follows a complete path.
        lethal = first[0] < 0.5
        mario = SimpleNamespace(left=600, right=648, top=400, bottom=448)
        return SimpleNamespace(termination_reason="bullet_death" if lethal else None, terminated=lethal,
            frames_simulated=len(actions)*4 if not lethal else 1,
            mario=mario, events=[])

class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.obs = dict(mario=np.asarray([600,650,648,698,0,0,0,0],np.float32),
            bullet_bills=np.asarray([[100,500,148,548,250]]*10,np.float32))
    def test_incomplete_baseline_searches_alternatives(self):
        p = AdaptiveAltitudeV2(FakeSim(), V2Config(crowd_confirm=1, replan_every=1))
        action, info = p.decide(self.obs, [0.0,0.0])
        self.assertTrue(info['altitude_v2_overridden'])
        self.assertTrue(action[0] >= .5)
    def test_padding_does_not_count_as_bullets(self):
        p = AdaptiveAltitudeV2(FakeSim(), V2Config(crowd_confirm=1))
        obs = dict(self.obs, bullet_bills=np.zeros((30,5),np.float32))
        action, info = p.decide(obs,[0.0,0.0])
        self.assertEqual(info['lower_bullets'], 0)
        self.assertFalse(info['altitude_v2_overridden'])
    def test_not_crowded_keeps_original(self):
        p = AdaptiveAltitudeV2(FakeSim(), V2Config(crowd_confirm=1))
        obs = dict(self.obs, bullet_bills=self.obs['bullet_bills'][:2])
        action, info = p.decide(obs,[.2,.6])
        np.testing.assert_allclose(action, [.2,.6])

if __name__ == '__main__':
    unittest.main()
