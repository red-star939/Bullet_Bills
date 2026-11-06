"""No-kymnasium regression: physical death routes and identity linkage."""
from types import SimpleNamespace
import numpy as np
from models.failure_analyzer_v21 import FailureAnalyzerV21,EngineDeathAuditV21,snapshot_engine_bullets
from models.failure_analyzer import FailureConfig
from models.action_evaluator import ActionEvaluator
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor,MotionConfig

class Rect:
    def __init__(self,l,t,r,b):self.left,self.top,self.right,self.bottom=l,t,r,b
class Bullet:
    def __init__(self,l,t,r,b):self.rect=Rect(l,t,r,b);self.state_='active';self._velocity=-240.
class Mario:
    def __init__(self):
        self.rect=Rect(600,720,648,768);self.active_=True
        self._velocity=SimpleNamespace(y=0.);self.invincible_=False
    def check_bullet_collision(self,bullets):
        for b in bullets:
            if (self.rect.left<b.rect.right and self.rect.right>b.rect.left and
                self.rect.top<b.rect.bottom and self.rect.bottom>b.rect.top):self.active_=False
class Core:
    def __init__(self,route):self._mario=Mario();self._bullet_bills=[];self.route=route
    def _step_internal(self,force):
        if self.route=='bullet':self._mario.check_bullet_collision(self._bullet_bills)
        if self.route=='fall':self._mario.rect.bottom=865;self._mario.active_=False
        return not self._mario.active_

def obs(bullets=()):
    return {'mario':np.asarray([600,720,648,768,0,0,0,0],dtype=np.float32),
        'bullet_bills':np.asarray(bullets,dtype=np.float32).reshape(-1,5),
        'stars':np.zeros((0,4),dtype=np.float32)}

def run():
    for route,expected in [('fall','fall_out_of_bounds'),('bullet','bullet_collision')]:
        core=Core(route)
        if route=='bullet':core._bullet_bills=[Bullet(610,720,658,768)]
        initial=core._step_internal;original_collision=core._mario.check_bullet_collision
        with EngineDeathAuditV21(core) as audit:
            before=snapshot_engine_bullets(core)
            assert core._step_internal((0,0))
            contacts=audit.take();deaths=audit.take_deaths()
            assert len(deaths)==1 and deaths[0]['cause']==expected, deaths
            assert len(contacts)==(route=='bullet')
        assert core._step_internal==initial
        assert core._mario.check_bullet_collision==original_collision
        if route=='bullet':
            row=before[0];assert row['bbox']==[610,720,658,768]
    sim=ForwardSimulator(MarioMotionPredictor(config=MotionConfig()))
    policy=ActionEvaluator(sim,horizon_steps=1,candidate_actions=[(0,0)],robust_mode='none')
    state=obs([[610,720,658,768,-240]])
    evaluated=policy.evaluate(state)
    for death,cause in [([{'cause':'fall_out_of_bounds'}],'fall_out_of_bounds'),
                        ([{'cause':'bullet_collision'}],'bullet_collision')]:
        ana=FailureAnalyzerV21(FailureConfig(history_seconds=2))
        ana.reset(101)
        after=obs([[594,720,642,768,-240]])
        engine_b=[dict(engine_object_id=555,bbox=[610,720,658,768],vx=-240.,state='active')]
        engine_a=[dict(engine_object_id=555,bbox=[594,720,642,768],vx=-240.,state='active')]
        engine_contacts=[dict(status='unique_fatal_overlap',fatal_candidates=[
            dict(engine_object_id=555,bbox=[594,720,642,768])])] if cause=='bullet_collision' else []
        report=ana.record(step=0,observation=state,action=(0,0),evaluation=evaluated,
            next_observation=after,info={'time_elapsed':4/60,'game_state':'game_over'},
            terminated=True,truncated=False,engine_deaths=death,
            engine_contacts=engine_contacts,engine_bullets_before=engine_b,
            engine_bullets_after=engine_a)
        assert report['physical_death_cause']==cause,report
        if cause=='bullet_collision':
            fatal=report['fatal_projectile']
            assert fatal['engine_present_before_terminal_action'] is True
            assert fatal['identity_link_quality']=='unique_engine_identity_and_observed_track',fatal
            assert fatal['engine_first_audited']['step']==0
    print('FailureAnalyzer v2.1 regression passed!')

if __name__=='__main__':run()
