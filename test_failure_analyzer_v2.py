"""Engine-free regression for persistent tracking, collision evidence and v1 compatibility."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np

from models.failure_analyzer_v2 import ObservedBulletTracker, FailureAnalyzerV2, EngineCollisionAudit, summarize_reports_v2
from models.failure_analyzer import FailureConfig
from models.action_evaluator import ActionEvaluator
from models.forward_simulator import ForwardSimulator
from models.mario_motion_predictor import MarioMotionPredictor, MotionConfig


def obs(bullets=(), mario=(624, 720, 672, 768, 0, 0, 0, 0)):
    return {'mario': np.asarray(mario, dtype=np.float32),
            'bullet_bills': np.asarray(bullets, dtype=np.float32).reshape(-1, 5),
            'stars': np.zeros((0, 4), dtype=np.float32)}


def run():
    tracker = ObservedBulletTracker()
    a = obs([[760, 720, 808, 768, -240.]])
    b = obs([[744, 720, 792, 768, -240.]])
    c = obs([[728, 720, 776, 768, -240.], [1260, 120, 1308, 168, -240.]])
    first = tracker.observe(a, 0, 0.)
    second = tracker.observe(b, 1, 4/60)
    third = tracker.observe(c, 2, 8/60)
    assert first[0]['track_id'] == second[0]['track_id'] == third[0]['track_id']
    assert third[1]['first_seen_step'] == 2
    assert tracker.associate([728,720,776,768])[0].track_id == first[0]['track_id']
    assert tracker.associate([735,720,783,768])[0] is None

    sim = ForwardSimulator(MarioMotionPredictor(config=MotionConfig()))
    policy = ActionEvaluator(sim, horizon_steps=3, robust_mode='none', candidate_actions=[(0,0),(0,1)])
    decision = policy.evaluate(a)
    analyzer = FailureAnalyzerV2(FailureConfig(history_seconds=1.))
    analyzer.reset(11,2)
    # Seed initial tracking at action's starting position; actual bullet moves 16px.
    engine = [{'status':'unique_fatal_overlap', 'fatal_candidates':[
        {'engine_object_id': 999, 'bbox': [744,720,792,768]}]}]
    report = analyzer.record(step=0, observation=a, action=decision.best_action,
                             evaluation=decision, next_observation=b,
                             info={'time_elapsed':4/60, 'game_state':'game_over'},
                             terminated=True, truncated=False, engine_contacts=engine)
    assert report['schema_version'] == 2
    assert report['fatal_projectile']['track_id'] == 0, report['fatal_projectile']
    assert report['primary_label'] == 'tracked_fatal_projectile'
    assert report['fatal_projectile']['was_visible_before_terminal_action']
    assert report['history'][-1]['tracked_bullets_before'][0]['track_id'] == 0

    analyzer.reset(12,3)
    report2 = analyzer.record(step=0, observation=a, action=decision.best_action,
                             evaluation=decision, next_observation=c,
                             info={'time_elapsed':4/60, 'game_state':'game_over'},
                             terminated=True, truncated=False, engine_contacts=[])
    assert report2['primary_label'] == 'fatal_projectile_unverified'
    assert report2['fatal_projectile'] is None
    assert 'possible_unobserved_projectile' not in str(report2['primary_label'])
    assert summarize_reports_v2([report,report2])['fatal_projectiles_matched'] == 1

    # The real collision-audit hook invokes unmodified engine method exactly once.
    class Rect:
        left=744; top=720; right=792; bottom=768
    class FakeBullet:
        rect = Rect(); state_ = 'active'
    class FakeMario:
        rect = Rect()
        _velocity = SimpleNamespace(y=0)
        invincible_ = False
        active_ = True
        def check_bullet_collision(self, bullets):
            self.active_ = False
    fake=FakeMario()
    orig=fake.check_bullet_collision
    with EngineCollisionAudit(fake) as audit:
        fake.check_bullet_collision([FakeBullet()])
        assert audit.take()[0]['status'] == 'unique_fatal_overlap'
    assert fake.check_bullet_collision == orig
    print('FailureAnalyzer v2 regression passed!')

if __name__ == '__main__':
    run()
