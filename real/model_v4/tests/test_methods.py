import unittest
import torch
from methods.curriculum import AdaptiveStartCurriculum, CurriculumConfig
from methods.rewards import SurvivalReward
from methods.evaluation import summarize_survival, make_disjoint_seeds
from methods.teacher_student import distillation_loss
from methods.risk import score_future_scenarios, select_intervention
from methods.ppo_core import gae, clipped_policy_loss

class TestMethods(unittest.TestCase):
    def test_curriculum(self):
        c=AdaptiveStartCurriculum(CurriculumConfig(minimum_s=50, maximum_s=100, window_s=15, min_observations=3, history_size=10))
        for _ in range(3): c.record_warmup(True)
        self.assertEqual(c.adapt(), (55,70))
    def test_seeds(self):
        d=make_disjoint_seeds(100, 20, 30, master_seed=123)
        self.assertEqual(len(set(sum(d.values(),[]))),150)
    def test_eval(self):
        e=summarize_survival([75,85,151])
        self.assertEqual(e.reached_150,1)
    def test_reward(self):
        self.assertLess(SurvivalReward().calculate(terminated=True),0)
    def test_risk(self):
        self.assertEqual(score_future_scenarios([0,1,1,0]).exposure_fraction,.5)
        self.assertTrue(select_intervention(.5,.1))
    def test_distill(self):
        p=torch.tensor([[1.,2.]],requires_grad=True)
        l=distillation_loss(p,torch.zeros_like(p))
        l.backward()
        self.assertIsNotNone(p.grad)
    def test_gae(self):
        r=torch.ones((3,2)); z=torch.zeros_like(r)
        adv,ret=gae(r,z,z,z)
        self.assertEqual(adv.shape,(3,2))
        self.assertTrue(torch.isfinite(clipped_policy_loss(z,z,adv)))
if __name__=='__main__': unittest.main()
