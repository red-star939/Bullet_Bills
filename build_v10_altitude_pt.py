"""Package an existing trained PPO v10 checkpoint and exact inference-overlay settings.

This is a deployment checkpoint, NOT a newly trained or distilled neural policy.
Run locally where models/bullet_bill_ppo_v10_best.pt exists.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

FORMAT = 'bulletbill_v10_spawn_altitude_bundle_v1'

def package(src: str, dst: str, *, map_path='map_data/object_tiles.csv',
            caution_px=180, critical_px=96, safety_horizon_steps=12,
            spawn_horizon_steps=12):
    source = Path(src).resolve()
    target = Path(dst).resolve()
    if source == target:
        raise ValueError('Input and output must differ; never overwrite source model')
    if target.exists():
        raise FileExistsError(f'Target already exists: {target}')
    # v10's original checkpoint contains just tensors and simple Python metadata.
    ck = torch.load(source, map_location='cpu', weights_only=True)
    if not isinstance(ck, dict) or not {'model', 'obs_dim', 'hidden'} <= set(ck):
        raise ValueError('Input is not the original PPO v10 checkpoint format')
    if not isinstance(ck['model'], dict) or not ck['model']:
        raise ValueError('Missing network parameters')
    if not all(isinstance(t, torch.Tensor) for t in ck['model'].values()):
        raise ValueError('Network state_dict must contain tensors')
    if 'hybrid_controller' in ck:
        raise ValueError('Source already contains hybrid metadata; use an original checkpoint')
    # Retain the original v10 checkpoint keys for compatibility with BulletBillAgent.load.
    out = dict(ck)
    out['hybrid_controller'] = {
        'format': FORMAT,
        'policy': 'PPO v10 (unchanged weights)',
        'guidance': 'altitude_shield',
        'pipeline': ['v10_policy', 'known_world_shield', 'spawn_shield', 'adaptive_altitude'],
        'map_path': str(map_path),
        'caution_px': int(caution_px),
        'critical_px': int(critical_px),
        'safety_horizon_steps': int(safety_horizon_steps),
        'spawn_horizon_steps': int(spawn_horizon_steps),
        'altitude_config': {
            'midline_y': 432.0, 'crowd_lower_min': 3, 'crowd_advantage': 2,
            'trigger_confirm_steps': 3, 'evaluate_every_steps': 5,
            'rollout_steps': 22, 'minimum_ascent_px': 45.0,
            'minimum_advantage_px': 24.0, 'min_final_wall_clearance_px': 65.0,
            'ascent_target_y': 415.0, 'cooldown_steps': 25,
        },
        'note': 'Requires external Python safety-controller modules at inference; not a standalone neural policy.',
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, target)
    reloaded = torch.load(target, map_location='cpu', weights_only=True)
    assert reloaded['hybrid_controller']['format'] == FORMAT
    assert reloaded['model'].keys() == ck['model'].keys()
    assert all(torch.equal(reloaded['model'][k], ck['model'][k]) for k in ck['model'])
    print(f'CREATED {target}')
    print(f'MODEL_TENSORS {len(ck["model"])} | unchanged=True | guidance=altitude_shield')
    print('NOTE: This bundles policy weights and controller metadata. It is not distilled.')
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='models/bullet_bill_ppo_v10_best.pt')
    p.add_argument('--out', default='models/bullet_bill_ppo_v10_altitude_hybrid.pt')
    p.add_argument('--map', default='map_data/object_tiles.csv')
    p.add_argument('--caution-px', type=int, default=180)
    p.add_argument('--critical-px', type=int, default=96)
    p.add_argument('--safety-horizon-steps', type=int, default=12)
    p.add_argument('--spawn-horizon-steps', type=int, default=12)
    a = p.parse_args()
    package(a.model, a.out, map_path=a.map, caution_px=a.caution_px,
            critical_px=a.critical_px, safety_horizon_steps=a.safety_horizon_steps,
            spawn_horizon_steps=a.spawn_horizon_steps)

if __name__ == '__main__':
    main()
