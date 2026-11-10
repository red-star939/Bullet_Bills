import unittest
import numpy as np
from models.v10_risk_integration import V10RiskIntegration
from models.mario_motion_predictor import MarioMotionPredictor
from models.forward_simulator import ForwardSimulator
from models.failure_analyzer_v2 import FailureAnalyzerV2
from models.failure_analyzer import FailureConfig
class DiagnosticEvaluator:
    rankings = ()
    horizon_steps = 0
    strategy = "v10_hybrid_diagnostic_no_action_rankings"


class RiskIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.r = V10RiskIntegration(ForwardSimulator(MarioMotionPredictor()))
        self.obs = {
            'mario': np.array([600, 680, 632, 720, 0, 0, 0, 0], dtype=np.float32),
            'bullet_bills': np.array([[200, 690, 244, 718, 360],
                                      [1000, 690, 1044, 718, -360],
                                      [900, 100, 944, 128, -360]], dtype=np.float32),
            'stars': np.empty((0,4), np.float32),
        }

    def test_tracker_lane_ttc(self):
        a = self.r.observe(self.obs, 0.)
        self.assertEqual(a['visible_bullets'], 3)
        self.assertEqual(a['new_bullets'], 3)
        self.assertEqual(a['lower_bullets'], 2)
        self.assertGreater(a['max_lane_load'], 0)
        self.assertIsNotNone(a['min_ttc_s'])
        b = self.r.observe(self.obs, 4 / 60.)
        self.assertEqual(b['visible_bullets'], 3)

    def test_observe_preserves_exact_action(self):
        self.r.observe(self.obs, 0.)
        a = np.array([0.35, 0.6], dtype=np.float32)
        b, m = self.r.decide(self.obs, a, mode='observe')
        np.testing.assert_array_equal(a, b)
        self.assertFalse(m['risk_guard_overridden'])

    def test_failure_shim(self):
        failure = FailureAnalyzerV2(FailureConfig())
        failure.reset(123, 0)
        next_obs = {'mario': self.obs['mario'].copy(), 'bullet_bills': self.obs['bullet_bills'].copy(),
                    'stars': np.empty((0,4), np.float32)}
        report = failure.record(step=0, observation=self.obs, action=[0,0],
                                evaluation=DiagnosticEvaluator(), next_observation=next_obs,
                                info={'time_elapsed':4/60,'game_state':'running'},
                                terminated=False,truncated=True, engine_contacts=[])
        self.assertIsNotNone(report)
        self.assertEqual(report['schema_version'],2)

if __name__ == '__main__':
    unittest.main()
