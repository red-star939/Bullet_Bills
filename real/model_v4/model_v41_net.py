"""Independent Model V4.1: physics-aware observations and discrete horizontal actions.
No PPO v10 dependencies and no assumption that old 500k Gaussian head is compatible.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

OBS_DIM = 64
HIDDEN = 192
JUMP_LEVELS = np.asarray([0.0, 0.5, 0.6, 0.75, 1.0], np.float32)
MOVE_LEVELS = np.asarray([-1.0, 0.0, 1.0], np.float32)
ARCH = 'independent_model_v41_discrete'


def _rows(obs, key, width):
    values = np.asarray(obs.get(key, np.zeros((0, width))), dtype=np.float32)
    if values.size == 0:
        return np.zeros((0, width), dtype=np.float32)
    if values.ndim == 1:
        if values.size % width:
            raise ValueError(f'{key} invalid shape {values.shape}')
        values = values.reshape(-1, width)
    if values.shape[1] < width:
        raise ValueError(f'{key} needs at least {width} columns, got {values.shape}')
    return values[:, :width][np.any(np.abs(values[:, :width]) > 1e-6, axis=1)]


def obs_vector(obs, size=OBS_DIM):
    """Mario and up to 5 bullets sorted by approximate closing time, plus 2 stars.
    All features are bounded and normalized. No unlabelled flatten/truncation.
    """
    if size != OBS_DIM:
        raise ValueError(f'V4.1 requires {OBS_DIM} features, got {size}')
    if not isinstance(obs, dict) or 'mario' not in obs:
        raise ValueError("V4.1 requires observation dictionary containing 'mario'")
    m = np.asarray(obs['mario'], dtype=np.float32).flatten()
    if m.size < 8:
        raise ValueError('mario observation must have left/top/right/bottom/vx/vy/inv/air')
    cx, cy = float((m[0]+m[2])/2), float((m[1]+m[3])/2)
    vx, vy = float(m[4]), float(m[5])
    ground, center, half = 672.0, 648.0, 144.0
    # 12 Mario and global features
    f = [np.clip((cx-center)/half,-3,3), np.clip((cy-550)/400,-3,3),
         np.clip(vx/650,-3,3), np.clip(vy/900,-3,3),
         np.clip((ground-float(m[3]))/250,-3,3), float(m[6]>0.5), float(m[7]>0.5),
         np.clip((cx-(center-half))/half,-3,3), np.clip(((center+half)-cx)/half,-3,3),
         np.clip((cx + vx/60-center)/half,-3,3), 0.0, 0.0]
    bb = _rows(obs, 'bullet_bills', 5)
    count = len(bb)
    if count:
        bcx = (bb[:,0]+bb[:,2])/2; bcy = (bb[:,1]+bb[:,3])/2
        dx = bcx-cx; dvx = bb[:,4]-vx
        closing = np.maximum(0.0, -np.sign(dx)*dvx)
        ttc = np.where(closing>1e-4, np.maximum(0,np.abs(dx)-48)/np.maximum(closing,1e-4), 3.0)
        ttc = np.clip(ttc,0,3)
        danger = np.exp(-ttc/0.7) * (np.abs(bcy-cy) < 72)
        f[10] = float(np.max(danger))
        f[11] = float(min(count,10)/10)
        order = np.argsort(ttc + np.abs(dx)*1e-5)
    else:
        order = []
    # 5*8=40 bullet features
    for ix in list(order)[:5]:
        row = bb[ix]
        bx=(row[0]+row[2])/2; by=(row[1]+row[3])/2
        ddx=float(bx-cx); ddy=float(by-cy); rv=float(row[4]-vx)
        close = max(0.0,-np.sign(ddx)*rv)
        ttc = min(3.0, max(0,abs(ddx)-48)/close) if close>1e-4 else 3.0
        f.extend([1.0,np.clip(ddx/600,-3,3),np.clip(ddy/400,-3,3),
                  np.clip(row[4]/350,-3,3),np.clip(rv/650,-3,3),
                  ttc/3.0,float(abs(ddy)<70),float(close>0)])
    f.extend([0.0] * (40-8*min(count,5)))
    stars = _rows(obs,'stars',4)
    for s in stars[:2]:
        f.extend([1.0,np.clip(((s[0]+s[2])/2-cx)/600,-3,3),np.clip(((s[1]+s[3])/2-cy)/400,-3,3)])
    f.extend([0.]*(6-3*min(len(stars),2)))
    f.extend([np.clip(vx/650,-3,3), np.clip((center-cx)/half,-3,3),
              float(abs(cx-center)>half-48), float(m[3]>ground+2), 0., 0.])
    out = np.nan_to_num(np.asarray(f,np.float32),nan=0,posinf=0,neginf=0)
    if out.size != OBS_DIM:
        raise RuntimeError(f'Feature count mismatch {out.size} != {OBS_DIM}')
    return out


def env_action(move_index, jump_index):
    return np.asarray([MOVE_LEVELS[int(move_index)],JUMP_LEVELS[int(jump_index)]],dtype=np.float32)


class ModelV41(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=HIDDEN):
        super().__init__()
        self.obs_dim=int(obs_dim); self.hidden=int(hidden)
        self.encoder=nn.Sequential(nn.Linear(self.obs_dim,self.hidden),nn.LayerNorm(self.hidden),nn.SiLU(),nn.Linear(self.hidden,self.hidden),nn.SiLU())
        self.move=nn.Linear(self.hidden,3)
        self.jump=nn.Linear(self.hidden,5)
        self.value_head=nn.Linear(self.hidden,1)
        for head in (self.move,self.jump):
            nn.init.orthogonal_(head.weight,gain=0.01);nn.init.zeros_(head.bias)

    def distributions(self,x):
        z=self.encoder(x)
        return Categorical(logits=self.move(z)),Categorical(logits=self.jump(z)),self.value_head(z).squeeze(-1)

    def act(self,x,deterministic=False):
        md,jd,v=self.distributions(x)
        if deterministic:
            # For equal logits prefer neutral move and no jump, not arbitrary first index.
            move=(md.logits+torch.as_tensor([0.,1.e-6,0.],device=x.device)).argmax(-1)
            jump=(jd.logits+torch.as_tensor([1.e-6,0.,0.,0.,0.],device=x.device)).argmax(-1)
        else:
            move=md.sample();jump=jd.sample()
        return jump,move,md.log_prob(move)+jd.log_prob(jump),v

    def evaluate_actions(self,x,jumps,moves):
        md,jd,v=self.distributions(x)
        return md.log_prob(moves)+jd.log_prob(jumps),md.entropy()+jd.entropy(),v


def make_payload(model, *, steps, trained, seed, **extra):
    return {'architecture':ARCH,'version':1,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'obs_dim':model.obs_dim,'hidden':model.hidden,'steps':int(steps),'trained':bool(trained), 'seed':int(seed),**extra}
