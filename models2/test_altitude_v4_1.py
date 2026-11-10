"""Dependency-isolated regression tests; real gym env still must be run locally."""
import unittest
from types import SimpleNamespace
import numpy as np
from .altitude_v4_1 import AdaptiveAltitudeV41, V41Config
from .spawn_aware_risk import SpawnAwareRisk
from .closed_loop_planner import ClosedLoopPlanner, Rollout, ModelState
from .intervention_audit import InterventionAudit


class FakeV4:
    def __init__(self): self.calls = 0
    def reset(self): self.calls = 0
    def decide(self, observation, action, risk=None):
        self.calls += 1
        return np.asarray(action, np.float32).copy(), {
            'altitude_overridden': False, 'v4_reason':'mock_v4_shadow',
            'v1_reason':'v1_mock', 'v1_overridden':False,
            'v4_overridden':False, 'v4_proposed':False}


class FakePlanner:
    def __init__(self, type_='rescue'):
        self.type = type_
        self.horizon_steps = 12
    def rollout(self, obs, first_action, *, time_s=0):
        a = tuple(float(v) for v in first_action)
        base = a[0] == 0.0 and a[1] == 0.0
        death = 10 if base and self.type == 'rescue' else None
        steps = 2 if death is not None else 12
        x = 625 if not base else 610
        boxes = tuple((x, 650., x+48, 698.) for _ in range(steps+1))
        return Rollout(a, (a,)*steps, boxes, steps, death, False, 150.)


class FakeSpawner:
    validated_through_s = 180.0
    def upcoming_frames(self, t, steps): return (4,)
    def assess(self, rollout, start_time_s=0):
        from .spawn_aware_risk import SpawnScenarioReport
        fraction = .3 if rollout.first_action[0] == 0 else .02
        return SpawnScenarioReport(1, 24, round(fraction*24), fraction, None)


def obs():
    b = np.array([[100+i*50, 660, 148+i*50, 708, 250.] for i in range(12)], np.float32)
    return {'mario':np.asarray([600,660,648,708,0,0,0,0], np.float32),
            'bullet_bills':np.vstack([b,np.zeros((18,5),np.float32)]),
            'stars':np.zeros((2,4),np.float32)}


class TestV41(unittest.TestCase):
    def make(self, mode, fake_type='rescue'):
        return AdaptiveAltitudeV41(None, SimpleNamespace(), config=V41Config(
            mode=mode, replan_every=1, min_visible_bullets=1,
            max_candidates=8, min_candidate_wall_px=40.),
            v4=FakeV4(), planner=FakePlanner(fake_type), spawn_risk=FakeSpawner())
    def test_shadow_retains_baseline_on_rescue(self):
        p=self.make('shadow')
        a,m=p.decide(obs(), [0.,0.], time_s=60.)
        np.testing.assert_array_equal(a, np.array([0.,0.], np.float32))
        self.assertEqual(p.overrides,0)
        self.assertTrue(m['v41_proposed'])
    def test_active_admits_full_horizon_rescue(self):
        p=self.make('active')
        a,m=p.decide(obs(),[0.,0.],time_s=60.)
        self.assertTrue(m['v41_overridden'],m)
        self.assertNotEqual(float(a[0]),0.)
    def test_active_rejects_strategic_only(self):
        p=self.make('active','strategic')
        a,m=p.decide(obs(),[0.,0.],time_s=60.)
        np.testing.assert_array_equal(a,np.array([0.,0.],np.float32))
        self.assertFalse(m['v41_overridden'])
    def test_padding_and_low_bullet_returns_base(self):
        p=self.make('active')
        ob=obs(); ob['bullet_bills'][:]=0
        a,m=p.decide(ob,[.12,.5],time_s=65.)
        self.assertEqual(m['v41_reason'],'few_bullets')
        np.testing.assert_array_equal(a,np.array([.12,.5],np.float32))
    def test_existing_invincibility_protected(self):
        p=self.make('active')
        ob=obs(); ob['mario'][6]=1
        a,m=p.decide(ob,[0.,0.],time_s=65.)
        self.assertEqual(m['v41_reason'],'invincibility_unknown')
    def test_spawn_stress_intersection(self):
        class Physics:
            @staticmethod
            def spawn_frames(game_duration=20): return np.array([4],np.int64)
        f=SpawnAwareRisk(Physics(),offsets=(0.,),pixels_per_frame=(4.,),overlap_margin_px=0.)
        # Mario at x=0, bullet starts at left -48: collision after 1-3 frames.
        r=Rollout((0.,0.),((0.,0.),)*12,((0.,100.,48.,148.),)*13,12,None,False,50.)
        report=f.assess(r,start_time_s=0.)
        self.assertEqual(report.scenario_count,2)
        self.assertGreater(report.exposure_fraction,0.)
    def test_audit_prediction_exception_is_logged(self):
        class BrokenStepper:
            def step(self,*args): raise RuntimeError('physics not ready')
        audit=InterventionAudit.__new__(InterventionAudit)
        audit.physics=SimpleNamespace(restore_vy=lambda a:a)
        audit.stepper=BrokenStepper(); audit.diverge_px=12.; audit.reset()
        audit.begin(obs(),[0,0],episode=1,seed=3,step=4,time_s=0.,baseline_action=[0,0])
        row=audit.finish(obs(),terminated=False,info={'game_state':'playing'})
        self.assertTrue(row['prediction_unavailable'])
        self.assertEqual(audit.summary()['audited_steps'],1)
    def test_stop_using_unverified_schedule_after_180(self):
        p=self.make('explore')
        a,m=p.decide(obs(),[0.,0.],time_s=180.2)
        self.assertEqual(m['v41_reason'],'spawn_timing_unvalidated_after_180s')
        np.testing.assert_array_equal(a,np.array([0.,0.],np.float32))
    def test_closed_loop_requeries_policy_on_each_imagined_observation(self):
        class TinyPhysics:
            GMAX=8
            SOLID=None
            @staticmethod
            def restore_vy(v):return v
            @staticmethod
            def _run_ext(left,top,vx,vy,grounded,inv,bl,bt,bv,ba,g,gn,stars,sa,
                         actions,solid,spawn,offset,out_state,out_bl,out_alive,
                         out_g,out_gn,out_stars):
                out_state[:]=[left+float(actions[0,0])*10,top,0,0,1,0]
                out_bl[:]=bl
                out_alive[:]=ba
                out_stars[:]=sa
                return -1,-1
        seen=[]
        def predictor(o,t):
            seen.append((float(o['mario'][0]),t))
            return np.asarray([.5,0.],np.float32)
        planner=ClosedLoopPlanner(TinyPhysics(),action_predictor=predictor,horizon_steps=4)
        r=planner.rollout(obs(),[0.,0.],time_s=12.)
        self.assertEqual(r.steps_survived,4)
        self.assertEqual(len(seen),3)
        self.assertEqual(r.actions[0],(0.,0.))
        self.assertTrue(all(a[0]==.5 for a in r.actions[1:]))
        self.assertGreater(r.bboxes[-1][0],r.bboxes[0][0])
    def test_config_validation(self):
        with self.assertRaises(ValueError):
            AdaptiveAltitudeV41(None, SimpleNamespace(), config=V41Config(mode='invalid'),
                v4=FakeV4(),planner=FakePlanner(),spawn_risk=FakeSpawner())

if __name__=='__main__': unittest.main()
