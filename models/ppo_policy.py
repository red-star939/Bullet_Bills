"""PPO discrete-action policy for BulletBill Continuous Stage 2.

Uses the same eight actions as the established ActionEvaluator baseline.
The fixed observation schema must match between train/evaluation/checkpoints.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .action_evaluator import DEFAULT_ACTIONS

ACTION_TABLE = np.asarray(DEFAULT_ACTIONS, dtype=np.float32)
MAX_BULLETS = 8
# Mario 10, 8 bullets * 7, platform context 12, time 1 = 79.
OBS_DIM = 10 + MAX_BULLETS * 7 + 12 + 1
SCHEMA_VERSION = 'ppo_state_v1_m10_b8x7_platform12_t1'


def load_platforms(path: str | Path = 'map_data/object_tiles.csv') -> np.ndarray:
    """Read immutable 48x48 solid tile boxes; no access to future game state."""
    path = Path(path)
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    boxes = [(float(r['x_px']), float(r['y_px']),
              float(r['x_px']) + 48., float(r['y_px']) + 48.)
             for r in rows if r.get('layer') == 'object']
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class StateEncoder:
    """Observation-only, deterministic, finite and normalized state encoder."""

    def __init__(self, platforms: np.ndarray):
        self.platforms = np.asarray(platforms, dtype=np.float32).reshape(-1, 4)

    def encode(self, obs: Mapping, time_elapsed: float = 0.0) -> np.ndarray:
        m = np.asarray(obs['mario'], dtype=np.float32)
        if len(m) < 8:
            raise ValueError('mario must contain 8 observation entries')
        x = float((m[0] + m[2]) / 2)
        y = float((m[1] + m[3]) / 2)
        mario = [x / 1296., y / 864., float(m[0]) / 1296.,
                 float(m[1]) / 864., float(m[4]) / 750.,
                 float(m[5]) / 1000., float(m[6]), float(m[7]),
                 float(m[3]) / 864., (1296. - float(m[2])) / 1296.]

        bullets = np.asarray(obs.get('bullet_bills', []), dtype=np.float32)
        if bullets.size == 0:
            bullets = bullets.reshape(0, 5)
        elif bullets.ndim == 1:
            bullets = bullets.reshape(1, -1)
        if bullets.shape[1] < 5:
            raise ValueError('bullet_bills must contain 5 columns')
        features = []
        for b in bullets:
            if not np.all(np.isfinite(b[:5])) or b[2] <= b[0] or b[3] <= b[1]:
                continue
            bx = float((b[0] + b[2]) / 2)
            by = float((b[1] + b[3]) / 2)
            dx = bx - x
            dy = by - y
            rel_v = float(b[4] - m[4])
            approaching = dx * rel_v < 0
            ttc = abs(dx / rel_v) if approaching and abs(rel_v) > 1e-6 else 10.
            features.append((max(0., ttc), abs(dx),
                             [np.clip(dx / 1296., -2., 2.),
                              np.clip(dy / 864., -2., 2.),
                              float(b[4]) / 500.,
                              np.clip((float(b[0]) - float(m[2])) / 1296., -2., 2.),
                              np.clip((float(m[0]) - float(b[2])) / 1296., -2., 2.),
                              np.clip(ttc / 4., 0., 2.5), 1.]))
        # Imminent contact, then nearest. All extracted solely from obs.
        features.sort(key=lambda it: (it[0], it[1]))
        bvec = []
        for _, _, row in features[:MAX_BULLETS]:
            bvec.extend(row)
        bvec.extend([0.] * (MAX_BULLETS * 7 - len(bvec)))

        # Platform summaries in six local horizontal intervals. Only platforms
        # below Mario's feet and within 300 px vertically are candidates.
        feet = float(m[3])
        bins = np.zeros((6, 2), dtype=np.float32)
        for i in range(6):
            center = x + (i - 2.5) * 96.
            left, right = center - 48., center + 48.
            best_dy = 300.
            best_overlap = 0.
            for pl, pt, pr, pb in self.platforms:
                overlap = max(0., min(right, float(pr)) - max(left, float(pl)))
                dy = float(pt) - feet
                if overlap > 0 and -12. <= dy <= 300. and dy < best_dy:
                    best_dy = dy
                    best_overlap = overlap
            bins[i] = (1. - best_dy / 300., best_overlap / 96.) if best_overlap else (0., 0.)
        vec = np.asarray(mario + bvec + bins.flatten().tolist()
                         + [min(max(float(time_elapsed), 0.), 120.) / 120.], dtype=np.float32)
        if vec.shape != (OBS_DIM,) or not np.all(np.isfinite(vec)):
            raise ValueError('PPO observation is not finite or has wrong dimension')
        return vec


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, actions: int = len(ACTION_TABLE), hidden: int = 128):
        super().__init__()
        self.actor_body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                        nn.Linear(hidden, hidden), nn.Tanh())
        self.actor_head = nn.Linear(hidden, actions)
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.)

    def distribution(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor_head(self.actor_body(states)))

    def forward(self, states: torch.Tensor):
        return self.distribution(states), self.critic(states).squeeze(-1)

    def act(self, states: torch.Tensor, *, deterministic: bool = False):
        dist, values = self(states)
        acts = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        return acts, dist.log_prob(acts), values


def checkpoint_payload(model: ActorCritic, *, trained: bool, total_steps: int,
                       encoder_version: str = SCHEMA_VERSION, optimizer=None, extra=None):
    d = {'model_state_dict': model.state_dict(), 'obs_dim': OBS_DIM,
         'action_table': ACTION_TABLE.tolist(), 'encoder_version': encoder_version,
         'trained': bool(trained), 'total_env_steps': int(total_steps),
         'architecture': 'discrete_ppo_actor_critic_128x128', 'extra': extra or {}}
    if optimizer is not None:
        d['optimizer_state_dict'] = optimizer.state_dict()
    return d


def load_policy(path: str | Path, device: str = 'cpu') -> tuple[ActorCritic, dict]:
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck['encoder_version'] != SCHEMA_VERSION or ck['obs_dim'] != OBS_DIM:
        raise ValueError('Checkpoint state schema mismatch')
    if not np.array_equal(np.asarray(ck['action_table'], np.float32), ACTION_TABLE):
        raise ValueError('Checkpoint action table mismatch')
    net = ActorCritic().to(device)
    net.load_state_dict(ck['model_state_dict'], strict=True)
    net.eval()
    return net, ck
