"""Independent Model V4 network, observation adapter, and BulletBill action adapter.
No imports from agent_ppo_v10 or existing trained models.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

OBS_DIM = 256
HIDDEN = 192
JUMP_LEVELS = np.array([0.0, 0.5, 0.6, 0.75, 1.0], dtype=np.float32)


def obs_vector(observation, size=OBS_DIM):
    """Fixed-size numeric encoding from nested Gym observation. Padding/truncation deterministic.

    Prototype encoder; benchmark against stronger physics-aware encoders later.
    """
    arrays = []
    def walk(value):
        if isinstance(value, dict):
            for k in sorted(value):
                walk(value[k])
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        else:
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                arrays.append(arr)
            except (TypeError, ValueError):
                pass
    walk(observation)
    data = np.concatenate(arrays) if arrays else np.zeros(0, np.float32)
    data = np.nan_to_num(data, nan=0., posinf=0., neginf=0.)
    # Large coordinates and velocities bounded while keeping a fixed input range.
    data = np.sign(data) * np.log1p(np.abs(data)) / np.log1p(2048.0)
    out = np.zeros(size, dtype=np.float32)
    n = min(size, len(data))
    out[:n] = data[:n]
    return out


def env_action(move, jump_index):
    return np.array([float(np.clip(move, -1, 1)), float(JUMP_LEVELS[int(jump_index)])], dtype=np.float32)


class ModelV4(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=HIDDEN):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
        self.jump = nn.Linear(hidden, len(JUMP_LEVELS))
        self.move = nn.Linear(hidden, 1)
        self.move_log_std = nn.Parameter(torch.tensor([-0.6]))
        self.value_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.jump.weight, gain=0.01)
        nn.init.zeros_(self.jump.bias)
        nn.init.orthogonal_(self.move.weight, gain=0.01)
        nn.init.zeros_(self.move.bias)

    def distributions(self, x):
        z = self.encoder(x)
        jump_dist = Categorical(logits=self.jump(z))
        move_dist = Normal(self.move(z).squeeze(-1), self.move_log_std.clamp(-4., 1.5).exp())
        value = self.value_head(z).squeeze(-1)
        return jump_dist, move_dist, value

    def act(self, x, deterministic=False):
        jump_dist, move_dist, value = self.distributions(x)
        if deterministic:
            jump = jump_dist.logits.argmax(-1)
            move = move_dist.mean
        else:
            jump, move = jump_dist.sample(), move_dist.sample()
        logprob = jump_dist.log_prob(jump) + move_dist.log_prob(move)
        return jump, move, logprob, value

    def evaluate_actions(self, x, jumps, moves):
        jump_dist, move_dist, value = self.distributions(x)
        logprob = jump_dist.log_prob(jumps) + move_dist.log_prob(moves)
        entropy = jump_dist.entropy() + move_dist.entropy()
        return logprob, entropy, value


def make_payload(model, *, steps, trained, seed, **extra):
    return {'architecture': 'independent_model_v4', 'version': 1,
            'model': {k: v.detach().cpu() for k,v in model.state_dict().items()},
            'obs_dim': model.obs_dim, 'hidden': model.hidden,
            'steps': int(steps), 'trained': bool(trained), 'seed': int(seed), **extra}
