"""Independent Model V4 evaluation: pygame visualization or headless simulation.

Requires model_v4_net.py from the independent Model V4 training package.
Does not import PPO v10, v4.1, or their checkpoint loaders.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from model_v4_net import ModelV4, OBS_DIM, HIDDEN, obs_vector, env_action

ROOT = Path(__file__).resolve().parent
ENV_ID = 'kymnasium/BulletBill-Continuous-Hard-Stage-2'


def load_model(path: Path, device: torch.device):
    """Accept independent final or resume checkpoints, reject older PPO formats."""
    if not path.is_file():
        raise FileNotFoundError(f'Model not found: {path}')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(ck, dict) or ck.get('architecture') != 'independent_model_v4':
        raise ValueError('Checkpoint is not independent Model V4 (.pt); do not use PPO v10 checkpoints')
    if 'model' not in ck:
        raise ValueError('Model V4 checkpoint missing model state_dict')
    model = ModelV4(obs_dim=int(ck.get('obs_dim', OBS_DIM)), hidden=int(ck.get('hidden', HIDDEN)))
    model.load_state_dict(ck['model'], strict=True)
    model.to(device).eval()
    return model, ck


def make_env(render: str):
    try:
        import gymnasium as gym
        import kymnasium  # noqa: F401 register BulletBill
    except ImportError as exc:
        raise RuntimeError('This requires gymnasium and kymnasium in your project virtualenv') from exc
    return gym.make(id=ENV_ID, render_mode=render, bgm=False)


def get_time(info, step, fps):
    if isinstance(info, dict):
        for name in ('time_elapsed', 'elapsed_time'):
            try:
                value = float(info[name])
                if np.isfinite(value) and value >= 0:
                    return value, name
            except (KeyError, TypeError, ValueError):
                pass
    # Only a fallback, not an authoritative simulation clock.
    return step / float(fps), 'estimated_steps_per_fps'


def run(args, env_factory=make_env):
    if args.episodes < 1 or args.max_steps < 1 or args.fps < 0 or args.game_fps <= 0:
        raise ValueError('episodes, max-steps and game-fps must be positive; fps must be nonnegative')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    torch.set_num_threads(1)
    model, ck = load_model(args.model, device)
    print(f'MODEL {args.model} steps={ck.get("steps", "unknown")} trained={ck.get("trained", "unknown")} device={device}', flush=True)
    if not ck.get('trained', False):
        print('WARNING: This checkpoint is untrained (0-step); simulation is for interface testing, not survival performance.', flush=True)
    if args.mode == 'human':
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError('Human rendering requires pygame') from exc
    else:
        pygame = None
    if args.seed_file is not None:
        stored = json.loads(args.seed_file.read_text(encoding='utf-8'))
        if isinstance(stored, dict):
            stored = stored.get('seeds', stored.get('eval_seeds', stored.get('final_holdout_seeds')))
        if not isinstance(stored, list) or len(stored) < args.episodes:
            raise ValueError('seed-file must contain a list (or a dictionary with seeds) of at least --episodes integers')
        seeds = [int(x) for x in stored[:args.episodes]]
    else:
        generator = random.Random(args.seed)
        seeds = generator.sample(range(1, 2**31 - 1), args.episodes)
    env = env_factory(args.mode)
    clock = pygame.time.Clock() if pygame else None
    results = []
    try:
        for i, seed in enumerate(seeds, 1):
            obs, info = env.reset(seed=seed)
            start = time.perf_counter()
            steps, stopped, terminated, truncated = 0, False, False, False
            if args.mode == 'human':
                print('WINDOW: press ESC or close window to stop evaluation', flush=True)
            for step in range(args.max_steps):
                if pygame:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            stopped = True
                    if stopped:
                        break
                features = torch.as_tensor(obs_vector(obs, size=model.obs_dim), dtype=torch.float32, device=device).unsqueeze(0)
                with torch.inference_mode():
                    jump, move, _, _ = model.act(features, deterministic=not args.stochastic)
                action = env_action(float(move.item()), int(jump.item()))
                obs, reward, terminated, truncated, info = env.step(action)
                steps = step + 1
                if terminated or truncated:
                    break
                if clock and args.fps:
                    clock.tick(args.fps)
            survival, time_source = get_time(info, steps, args.game_fps)
            status = 'user_quit' if stopped else 'terminated' if terminated else 'truncated' if truncated else 'step_limit'
            record = {'episode': i, 'seed': seed, 'survival_s': survival, 'time_source': time_source,
                      'steps': steps, 'status': status, 'wall_s': time.perf_counter() - start}
            results.append(record)
            print(f'EPISODE {i}/{len(seeds)} seed={seed} survival={survival:.3f}s steps={steps} status={status} wall={record["wall_s"]:.1f}s', flush=True)
            if stopped:
                break
    finally:
        env.close()
        if pygame:
            pygame.quit()
    times = [r['survival_s'] for r in results if r['status'] != 'user_quit']
    summary = {'model': str(args.model), 'model_steps': int(ck.get('steps', 0)), 'trained': bool(ck.get('trained', False)),
               'mode': args.mode, 'device': str(device), 'stochastic': args.stochastic,
               'requested_episodes': args.episodes, 'completed_episodes': len(times),
               'mean_survival_s': float(np.mean(times)) if times else None,
               'min_survival_s': float(np.min(times)) if times else None,
               'max_survival_s': float(np.max(times)) if times else None,
               'reached_80': sum(t >= 80 for t in times), 'reached_100': sum(t >= 100 for t in times),
               'reached_120': sum(t >= 120 for t in times), 'reached_150': sum(t >= 150 for t in times),
               'results': results, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        with (args.out / 'episodes.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode','seed','survival_s','time_source','steps','status','wall_s'])
            writer.writeheader()
            writer.writerows(results)
        print(f'SAVED {args.out / "summary.json"} {args.out / "episodes.csv"}', flush=True)
    print(f'SUMMARY completed={len(times)} mean_survival_s={summary["mean_survival_s"]} reached_150={summary["reached_150"]}/{len(times)}', flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Independent Model V4 game simulation, human or headless')
    parser.add_argument('--model', type=Path, default=ROOT / 'model_v4_initial_50k.pt')
    parser.add_argument('--mode', choices=('human', 'none'), default='none')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261112, help='Generates reproducible random episode seeds')
    parser.add_argument('--seed-file', type=Path, help='Optional JSON seed file to compare two models')
    parser.add_argument('--max-steps', type=int, default=18000)
    parser.add_argument('--fps', type=int, default=30, help='Human rendering cap; 0=unlimited')
    parser.add_argument('--game-fps', type=float, default=50, help='Only used if environment does not provide time_elapsed')
    parser.add_argument('--stochastic', action='store_true', help='Sample actions instead of deterministic policy')
    parser.add_argument('--device', choices=('cpu','cuda','auto'), default='cpu')
    parser.add_argument('--out', type=Path, default=ROOT / 'eval_results')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
