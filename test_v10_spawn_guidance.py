"""Offline integration tests. No kymnasium checkpoint or live environment required."""
import numpy as np
from models.v10_spawn_guidance import V10SpawnGuidance
from models.spawn_risk_evaluator import SpawnRiskReport


def fake_report(score, initial, final, approaching=True, min_clearance=10):
    return SpawnRiskReport(initial_clearance_px=initial,
        min_clearance_px=min_clearance,final_clearance_px=final,
        near_wall_frames=1,horizon_frames=1,nearest_wall='left',
        risk_score=score,approaching_wall=approaching)

class FakeRisk:
    caution_px=180
    critical_px=96
    def evaluate(self, obs, action):
        x=float(action[0])
        return fake_report(0.9 if x < 0 else 0.5 if x == 0 else 0.1,
                           60,30 if x < 0 else 60 if x == 0 else 100)

class FakeGuide:
    safety_horizon_steps=12
    def teach(self, obs, prev): return 0, None


def test_raw_action_passthrough():
    g=V10SpawnGuidance.__new__(V10SpawnGuidance)
    action=np.array([0.372,0.75],np.float32)
    output,meta=g.decide({},action,mode='ppo')
    assert np.array_equal(output,action) and not meta['overridden']


def test_spawn_preserves_jump():
    g=V10SpawnGuidance.__new__(V10SpawnGuidance)
    g.spawn=FakeRisk();g.guide=FakeGuide()
    g.min_improvement=0.14;g.min_clearance_gain=24
    g.checked=lambda *args:(True,None,None)
    original=np.array([-0.7,0.6],np.float32)
    chosen,meta=g.decide({},original,mode='spawn_shield')
    assert chosen[0]>0 and chosen[1]==original[1] and meta['spawn_overridden']


def test_no_change_on_safe_central_state():
    g=V10SpawnGuidance.__new__(V10SpawnGuidance)
    g.spawn=FakeRisk();g.guide=FakeGuide()
    g.min_improvement=0.14;g.min_clearance_gain=24
    g.checked=lambda *args:(True,None,None)
    g.spawn.evaluate=lambda *args:fake_report(0.,400,400,False,390)
    raw=np.array([.25,.5],np.float32)
    actual,meta=g.decide({},raw,mode='spawn_shield')
    assert np.array_equal(actual,raw) and not meta['overridden']


def test_v10_original_action_never_discretized():
    g=V10SpawnGuidance.__new__(V10SpawnGuidance)
    g.checked=lambda *args:(True,None,None)
    raw=np.array([.348123,.75],np.float32)
    out,_=g.decide({},raw,mode='shield')
    assert np.array_equal(out,raw)


def test_invalid_inputs():
    for a in ([float('nan'),0.], [1.2,1.], [.5], [0.,-2.]):
        try: V10SpawnGuidance._action(a)
        except ValueError: continue
        raise AssertionError(a)


if __name__=='__main__':
    test_raw_action_passthrough()
    test_spawn_preserves_jump()
    test_no_change_on_safe_central_state()
    test_v10_original_action_never_discretized()
    test_invalid_inputs()
    print('PPO-v10 + Spawn-Aware Shield adapter tests passed!')
