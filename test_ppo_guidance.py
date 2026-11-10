"""Offline regression: real predictive modules with fake observations, guided on-policy PPO.

No assertion about live game performance.
"""
import tempfile
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from models.model_guidance import ModelGuidance, action_index
from models.ppo_policy import ACTION_TABLE, load_policy
from train_ppo_guided_parallel import guided_ppo_update, train
from test_ppo_parallel_smoke import FakeEnv

ROOT = Path(__file__).resolve().parent

def test_predictor():
    guide = ModelGuidance(ROOT/'map_data/object_tiles.csv')
    obs = FakeEnv()._obs()
    idx, ev = guide.teach(obs)
    assert 0 <= idx < len(ACTION_TABLE)
    assert idx == action_index(ev.best_action)
    safe, reason, report = guide.check_action(obs, idx)
    assert isinstance(safe, bool) and isinstance(report.risk_flags, tuple)
    selected, metadata = guide.decide(obs, idx, mode='shield')
    assert 0 <= selected < len(ACTION_TABLE) and 'source' in metadata
    print('ModelGuidance actual ForwardSimulator / ActionEvaluatorV2 / PlatformSafetyEvaluator passed')


def test_training():
    with tempfile.TemporaryDirectory() as d:
        a = Namespace(num_envs=2, torch_threads=1,
            env_factory='test_ppo_parallel_smoke:make_fake_env',
            total_steps=12, rollout_steps=12, max_episode_steps=10,
            lr=3e-4, epochs=1, minibatch=6, clip=.2, vf_coef=.5, ent_coef=.01,
            max_grad_norm=.5, target_kl=.03, gamma=.995, gae_lambda=.95,
            alive_reward=.01, death_penalty=-1., timeout_bonus=0., success_bonus=1.,
            seed=123, device='cpu', map=str(ROOT/'map_data/object_tiles.csv'),
            out=d, resume=None, log_interval=12, save_interval=12,
            teacher_coef=.5, teacher_decay_steps=100)
        train(a)
        model, ck=load_policy(Path(d)/'mario_ppo_latest.pt')
        assert ck['total_env_steps']==12
        assert ck['extra']['training_method'].endswith('auxiliary_ce')
        w0=model.actor_head.weight.detach().clone()
        a.resume=str(Path(d)/'mario_ppo_latest.pt')
        a.total_steps=24
        train(a)
        model2,ck2=load_policy(Path(d)/'mario_ppo_latest.pt')
        assert ck2['total_env_steps']==24
        assert not torch.equal(w0,model2.actor_head.weight), 'Resume did not update loaded network!'
    print('Guided PPO spawn rollout + CE + real resumed weight update passed')

if __name__=='__main__':
    test_predictor()
    test_training()
