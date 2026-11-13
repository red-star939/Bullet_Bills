"""Teacher-student action imitation with behavior preservation.

The caller supplies student model, teacher actions, and optionally original-policy
parameters. This is supervised imitation, NOT on-policy PPO data collection.
"""
from dataclasses import dataclass

@dataclass
class TeacherBatch:
    observations: object
    teacher_actions: object
    weights: object | None = None


def distillation_loss(student_pred, teacher_action, *, weights=None, base_pred=None,
                      preserve_coef=0.0):
    """Continuous-action weighted mean-squared-error loss (PyTorch tensors)."""
    import torch
    diff = (student_pred - teacher_action).square().mean(dim=-1)
    if weights is not None:
        w = torch.as_tensor(weights, device=diff.device, dtype=diff.dtype)
        if w.shape != diff.shape:
            raise ValueError('weights must match per-sample loss shape')
        loss = (diff*w).sum() / w.sum().clamp_min(1e-8)
    else:
        loss = diff.mean()
    if base_pred is not None and preserve_coef:
        loss = loss + float(preserve_coef) * (student_pred-base_pred.detach()).square().mean()
    return loss
