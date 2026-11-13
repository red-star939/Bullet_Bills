"""Framework-independent on-policy PPO math primitives (torch expected).

Does NOT define network shape, environment action distribution, or trainer loop.
Log probabilities must correspond to the actual sampled actions at collection.
"""

def gae(rewards, values, next_values, dones, *, gamma=0.995, lam=0.95):
    """Compute GAE with tensor inputs of shape [time, environments]."""
    import torch
    if not (rewards.shape == values.shape == next_values.shape == dones.shape):
        raise ValueError('all GAE arrays must have equal shapes')
    adv = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0]-1, -1, -1):
        mask = 1.0-dones[t].to(rewards.dtype)
        delta = rewards[t] + gamma*next_values[t]*mask-values[t]
        carry = delta + gamma*lam*mask*carry
        adv[t] = carry
    return adv, adv + values


def clipped_policy_loss(new_logprob, old_logprob, advantage, *, clip_eps=0.2):
    import torch
    ratio = (new_logprob-old_logprob).exp()
    return -torch.minimum(ratio*advantage,
                          ratio.clamp(1-clip_eps, 1+clip_eps)*advantage).mean()
