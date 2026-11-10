"""Play a packaged PPO v10 + Spawn Shield + Adaptive Altitude .pt in Pygame.

The model and overlay are loaded together, but safety logic remains external Python.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from build_v10_altitude_pt import FORMAT
from watch_ppo_v10_altitude import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='models/bullet_bill_ppo_v10_altitude_hybrid.pt')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--steps', type=int, default=9000,
                   help='9000=maximum 600 simulated seconds; 4500=300 seconds')
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--fps', type=float, default=0)
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--profile-every', type=int, default=100)
    p.add_argument('--stochastic', action='store_true')
    a = p.parse_args()
    if a.episodes < 1 or a.steps < 1 or a.fps < 0 or a.torch_threads < 1:
        p.error('Invalid run arguments')
    ck = torch.load(Path(a.model), map_location='cpu', weights_only=True)
    meta = ck.get('hybrid_controller')
    if not isinstance(meta, dict) or meta.get('format') != FORMAT:
        p.error('Not an altitude hybrid bundle. Create it with build_v10_altitude_pt.py.')
    from models.adaptive_altitude_controller import AltitudeConfig
    if meta.get('altitude_config') != vars(AltitudeConfig()):
        p.error('Stored altitude configuration differs from this installed controller version. Refusing silent mismatch.')
    a.guidance = 'altitude_shield'
    a.map = meta['map_path']
    a.caution_px = meta['caution_px']
    a.critical_px = meta['critical_px']
    a.safety_horizon_steps = meta['safety_horizon_steps']
    a.spawn_horizon_steps = meta['spawn_horizon_steps']
    print(f'HYBRID_BUNDLE {a.model} | guidance forced to altitude_shield', flush=True)
    run(a)

if __name__ == '__main__':
    main()
