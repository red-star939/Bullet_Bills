"""Offline geometry/real-module integration regression; NOT live environment results."""
import numpy as np

from models.mario_motion_predictor import MarioMotionPredictor
from models.spawn_risk_evaluator import SpawnRiskEvaluator
from models.spawn_aware_guidance import SpawnAwareGuidance
from models.ppo_policy import ACTION_TABLE

MAP = 'map_data/object_tiles.csv'


def obs_at(x, *, y=336):
    return {'mario': np.array([x, y, x+48, y+48, 0, 0, 0, 0], dtype=np.float32),
            'bullet_bills': np.zeros((0, 5), dtype=np.float32)}


def test_geometry():
    m = MarioMotionPredictor.from_object_tiles_csv(MAP)
    risk = SpawnRiskEvaluator(m)
    left = obs_at(0)
    center = obs_at(600)
    right = obs_at(1248)
    # Known action table: right/no-jump at [2], left/no-jump at [1].
    l_away = risk.evaluate(left, ACTION_TABLE[2])
    l_toward = risk.evaluate(left, ACTION_TABLE[1])
    r_away = risk.evaluate(right, ACTION_TABLE[1])
    r_toward = risk.evaluate(right, ACTION_TABLE[2])
    assert l_away.risk_score < l_toward.risk_score
    assert r_away.risk_score < r_toward.risk_score
    assert risk.evaluate(center, ACTION_TABLE[0]).risk_score == 0
    assert risk.evaluate(left, ACTION_TABLE[0]).initial_clearance_px == 0
    print('Spawn risk geometry left/right and center passed')


def test_integration():
    guide = SpawnAwareGuidance(map_path=MAP)
    left = obs_at(0)
    c = obs_at(600)
    unchanged, m = guide.decide(c, 0, mode='spawn_shield')
    assert unchanged == 0 and not m['spawn_overridden']
    # Incoming wall-vulnerable state must favor a checked alternative.
    for x, action, away in [(0, 1, 2), (1248, 2, 1)]:
        action_id, meta = guide.decide(obs_at(x), action, mode='spawn_shield')
        assert meta['spawn_overridden'], (x, meta)
        assert meta['source'] == 'spawn_wall_guard'
        assert meta['spawn_score_after'] < meta['spawn_score_before']
        assert action_id != action
        safe, _, pred = guide.check_action(obs_at(x), action_id)
        assert safe and pred.completed_known_world
    # Do not modify old guidance modes.
    for mode in ('ppo','shield','teacher'):
        chosen, _ = guide.decide(c, 2, mode=mode)
        assert 0 <= chosen < len(ACTION_TABLE)
    print('Real ModelGuidance + spawn override and compatibility passed')


if __name__ == '__main__':
    test_geometry()
    test_integration()
    print('SpawnAwareGuidance offline regression passed; LIVE GAME PERFORMANCE NOT TESTED')
