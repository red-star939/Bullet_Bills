"""Matched-seed, headless comparison of PPO, original Shield and Spawn Shield.

Writes each result immediately. Actual BulletBill environment needed to run.
No guarantee of identical future bullet spawn draws under different policies.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from models.ppo_policy import ACTION_TABLE, StateEncoder, load_platforms, load_policy
from models.spawn_aware_guidance import SpawnAwareGuidance

FIELDS = ('seed', 'mode', 'survival_s', 'decisions', 'terminal', 'wall_step_fraction',
          'shield_overrides', 'spawn_overrides', 'final_wall_clearance_px', 'runtime_s')


def run_episode(model, encoder, guide, seed, mode, steps, caution_px):
    from agent_env import create_env
    env = create_env(render_mode='none')
    previous = None
    wall_steps = shield_overrides = spawn_overrides = 0
    last_t = 0.
    final_clearance = -1
    start = time.perf_counter()
    terminal = 'step_cap'
    try:
        obs, info = env.reset(seed=seed)
        for n in range(1, steps + 1):
            # The actual wall exposure is measured from the CURRENT observation,
            # not from hypothetical future predictions.
            left, _, right, _ = obs['mario'][:4]
            final_clearance = max(0, min(float(left), 1296. - float(right)))
            wall_steps += int(final_clearance < caution_px)
            state = encoder.encode(obs, float(info.get('time_elapsed', 0.)))
            with torch.inference_mode():
                action_id, _, _ = model.act(torch.as_tensor(state).unsqueeze(0),
                                           deterministic=True)
            action_id = int(action_id.item())
            chosen, meta = guide.decide(obs, action_id, previous_action=previous,
                                        mode=mode)
            shield_overrides += int(meta.get('overridden', False))
            spawn_overrides += int(meta.get('spawn_overridden', False))
            action = ACTION_TABLE[chosen].copy()
            previous = tuple(float(v) for v in action)
            obs, _, terminated, truncated, info = env.step(action)
            last_t = float(info.get('time_elapsed', n * 4 / 60.))
            if terminated or truncated:
                terminal = 'terminated' if terminated else 'truncated'
                break
        return dict(seed=seed, mode=mode, survival_s=round(last_t, 5), decisions=n,
                    terminal=terminal, wall_step_fraction=round(wall_steps/n, 5),
                    shield_overrides=shield_overrides, spawn_overrides=spawn_overrides,
                    final_wall_clearance_px=round(final_clearance, 2),
                    runtime_s=round(time.perf_counter()-start, 2))
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--episodes', type=int, default=30)
    ap.add_argument('--steps', type=int, default=900)
    ap.add_argument('--seed', type=int, default=20261110)
    ap.add_argument('--map', default='map_data/object_tiles.csv')
    ap.add_argument('--out', default='logs/spawn_shield_comparison')
    ap.add_argument('--caution-px', type=int, default=180)
    ap.add_argument('--critical-px', type=int, default=96)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.episodes < 1 or args.steps < 1:
        ap.error('--episodes and --steps must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'episode_results.csv'
    rows = []
    if args.resume and path.exists():
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    elif path.exists():
        raise FileExistsError(f'{path} exists; select a fresh --out or --resume')
    seeds = random.Random(args.seed).sample(range(2**32), args.episodes)
    modes = ('ppo', 'shield', 'spawn_shield')
    expected = {(str(s), m) for s in seeds for m in modes}
    completed = {(str(r['seed']), r['mode']) for r in rows}
    if not completed.issubset(expected):
        raise RuntimeError('Saved CSV does not match current seeds/configuration')
    torch.set_num_threads(1)
    model, _ = load_policy(args.model, device='cpu')
    model.eval()
    encoder = StateEncoder(load_platforms(args.map))
    guide = SpawnAwareGuidance(map_path=args.map,
                               caution_px=args.caution_px,
                               critical_px=args.critical_px)
    for idx, seed in enumerate(seeds, start=1):
        for mode in modes:
            if (str(seed), mode) in completed:
                continue
            row = run_episode(model, encoder, guide, seed, mode, args.steps, args.caution_px)
            rows.append(row)
            with path.open('a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if f.tell() == 0:
                    w.writeheader()
                w.writerow(row)
        grouped = {m: [float(r['survival_s']) for r in rows if r['mode'] == m] for m in modes}
        print('PROGRESS', idx, '/', len(seeds),
              ' '.join(f'{m}={statistics.mean(grouped[m]):.3f}s' for m in modes if grouped[m]), flush=True)
    stats = {}
    for mode in modes:
        selected = [r for r in rows if r['mode'] == mode]
        stats[mode] = dict(episodes=len(selected),
                           mean_capped_survival_s=statistics.mean(float(r['survival_s']) for r in selected),
                           mean_wall_fraction=statistics.mean(float(r['wall_step_fraction']) for r in selected),
                           total_spawn_overrides=sum(int(r['spawn_overrides']) for r in selected))
    (out/'summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
