"""Bridge our validated predictive modules to PPO.

Teacher labels support *on-policy* PPO auxiliary imitation; shield operates at
inference only. Prediction is a known-world estimate, NOT a safety guarantee.
"""
from __future__ import annotations

import numpy as np

from .action_evaluator_v2 import ActionEvaluatorV2
from .forward_simulator import ForwardSimulator
from .mario_motion_predictor import MarioMotionPredictor
from .platform_safety_evaluator import PlatformSafetyEvaluator
from .ppo_policy import ACTION_TABLE


def action_index(action):
    action = np.asarray(action, dtype=np.float32)
    matching = np.where(np.all(ACTION_TABLE == action, axis=1))[0]
    if len(matching) != 1:
        raise ValueError(f'Predictive model selected an unrecognized action: {action}')
    return int(matching[0])


class ModelGuidance:
    def __init__(self, map_path='map_data/object_tiles.csv', horizon_steps=8,
                 extended_horizon_steps=20, audit_top_k=4, safety_horizon_steps=12):
        motion = MarioMotionPredictor.from_object_tiles_csv(map_path)
        self.simulator = ForwardSimulator(motion)
        self.teacher = ActionEvaluatorV2(
            self.simulator, horizon_steps=horizon_steps,
            extended_horizon_steps=extended_horizon_steps, audit_top_k=audit_top_k,
            robust_mode='none')
        self.platform = PlatformSafetyEvaluator(self.simulator)
        self.safety_horizon_steps = int(safety_horizon_steps)

    def teach(self, observation, previous_action=None):
        result = self.teacher.evaluate(observation, current_action=previous_action)
        return action_index(result.best_action), result

    def check_action(self, observation, action_id):
        """Check PPO's candidate against forward collision and platform fall.

        Only *predicted lethal events* trigger override, not mere near-edge flags.
        An unknown future star / spawn is not treated as confirmed safe.
        """
        action = ACTION_TABLE[int(action_id)]
        report = self.platform.evaluate_observation(
            observation, [action] * self.safety_horizon_steps)
        if report.predicted_fall_frame is not None:
            return False, 'predicted_fall', report
        if report.predicted_bullet_death_frame is not None:
            return False, 'predicted_bullet_death', report
        return True, 'no_known_lethal_event', report

    def decide(self, observation, ppo_action_id, previous_action=None, mode='shield'):
        """Return (executed_action_id, metadata).

        ppo: pure learned policy; teacher: v2 baseline; shield: override only
        explicit predicted fall/bullet death and only with checked teacher action.
        """
        if mode == 'ppo':
            return int(ppo_action_id), {'source': 'ppo', 'overridden': False}
        teacher_id, result = self.teach(observation, previous_action)
        if mode == 'teacher':
            return teacher_id, {'source': 'teacher_v2', 'overridden': teacher_id != int(ppo_action_id),
                                'teacher_reason': result.change_reason}
        if mode != 'shield':
            raise ValueError(f'Invalid guidance mode: {mode}')
        safe, reason, _ = self.check_action(observation, ppo_action_id)
        if safe:
            return int(ppo_action_id), {'source': 'ppo', 'overridden': False,
                                        'reason': reason, 'teacher_action': teacher_id}
        teacher_safe, teacher_reason, _ = self.check_action(observation, teacher_id)
        if teacher_safe:
            return teacher_id, {'source': 'teacher_shield', 'overridden': True,
                                'reason': reason, 'teacher_action': teacher_id}
        # Never pretend that a predicted-dangerous teacher guarantees safety.
        return int(ppo_action_id), {'source': 'ppo_no_verified_alternative',
                                    'overridden': False, 'reason': reason,
                                    'teacher_reason': teacher_reason, 'teacher_action': teacher_id}
