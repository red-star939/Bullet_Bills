r"""Watch our PPO checkpoint controlling the *real* BulletBill Stage 2 environment.

No training, no model updates.  Uses the same StateEncoder / ACTION_TABLE as
train_ppo.py and eval_ppo.py.  Visual output requires a graphical desktop.

Examples (Windows PowerShell):
  .\.venv\Scripts\python.exe watch_ppo.py --model models/ppo_runs_parallel16/mario_ppo_latest.pt
  .\.venv\Scripts\python.exe watch_ppo.py --model models/ppo_runs_parallel16/mario_ppo_latest.pt --render rgb_array
  .\.venv\Scripts\python.exe watch_ppo.py --model models/mario_ppo_init.pt --allow-untrained
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from models.ppo_policy import ACTION_TABLE, StateEncoder, load_platforms, load_policy


def make_env(render_mode: str):
    """Use the ORIGINAL game's renderer for human mode, as manual_play.py does."""
    if render_mode == 'human':
        try:
            import gymnasium as gym
            import kymnasium  # noqa: F401; registers BulletBill environment
        except ImportError as exc:
            raise RuntimeError('Install gymnasium and kymnasium in the project .venv.') from exc
        return gym.make(
            id='kymnasium/BulletBill-Continuous-Hard-Stage-2',
            render_mode='human',
            bgm=False,
        )

    # Observation drawing and optional RGB-array support remain as alternatives.
    try:
        from agent_env import create_env
    except ImportError as exc:
        raise RuntimeError('agent_env.py is required for schematic/rgb_array modes.') from exc
    return create_env(render_mode=render_mode)


def schematic_frame(obs, platforms, *, width=1296, height=864):
    """Visualize the REAL environment observation; not the game's sprite renderer.

    Bounding boxes are in engine pixels. A missing observation entity cannot be
    displayed; background graphics and unobserved future bullets are not invented.
    """
    import cv2

    image = np.full((height, width, 3), (38, 27, 22), dtype=np.uint8)
    # Subtle grid makes velocities and relative positions easier to judge.
    for x in range(0, width, 96):
        cv2.line(image, (x, 0), (x, height - 1), (53, 42, 36), 1)
    for y in range(0, height, 96):
        cv2.line(image, (0, y), (width - 1, y), (53, 42, 36), 1)
    for l, t, r, b in np.asarray(platforms, dtype=np.float32).reshape(-1, 4):
        cv2.rectangle(image, (int(l), int(t)), (int(r) - 1, int(b) - 1), (84, 123, 97), -1)
        cv2.rectangle(image, (int(l), int(t)), (int(r) - 1, int(b) - 1), (125, 171, 135), 1)

    bullets = np.asarray(obs.get('bullet_bills', []), dtype=np.float32)
    if bullets.size:
        bullets = bullets.reshape(-1, bullets.shape[-1])
        for bullet in bullets:
            if len(bullet) < 4 or not np.all(np.isfinite(bullet[:4])):
                continue
            l, t, r, b = (int(v) for v in bullet[:4])
            if r <= l or b <= t:
                continue
            cv2.rectangle(image, (l, t), (r, b), (83, 84, 239), -1)
            if len(bullet) > 4:
                vx = float(bullet[4])
                direction = 1 if vx >= 0 else -1
                cx, cy = (l+r)//2, (t+b)//2
                cv2.arrowedLine(image, (cx, cy), (cx + direction*30, cy),
                                (255, 236, 235), 2, tipLength=0.4)

    m = np.asarray(obs['mario'], dtype=np.float32)
    if m.size >= 4 and np.all(np.isfinite(m[:4])):
        l, t, r, b = (int(v) for v in m[:4])
        cv2.rectangle(image, (l, t), (r, b), (245, 181, 64), -1)
        cv2.rectangle(image, (l, t), (r, b), (255, 255, 255), 2)
        if m.size >= 6:
            cx, cy = (l+r)//2, (t+b)//2
            vx, vy = float(m[4]), float(m[5])
            if np.isfinite(vx) and np.isfinite(vy):
                cv2.arrowedLine(image, (cx, cy),
                                (cx + int(np.clip(vx / 20, -55, 55)),
                                 cy + int(np.clip(vy / 20, -55, 55))),
                                (255, 255, 255), 2, tipLength=0.35)
    return image


def show_frame(image, *, title: str, hud: str, already_bgr=True):
    import cv2

    if not already_bgr:
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError(f'Unexpected RGB frame shape {frame.shape!r}')
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR if frame.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    image = np.ascontiguousarray(image.copy())
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 1200), 36), (15, 15, 15), -1)
    cv2.putText(image, hud, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(title, image)
    key = cv2.waitKey(1) & 0xFF
    return key in (27, ord('q'))


def render_rgb(env, *, title: str, hud: str):
    frame = env.render()
    if isinstance(frame, (list, tuple)) and len(frame):
        frame = frame[-1]
    if frame is None:
        return None  # signal caller to switch to observation-based schematic
    return show_frame(frame, title=title, hud=hud, already_bgr=False)


def run(args):
    if args.episodes < 1 or args.steps < 1 or args.fps <= 0:
        raise ValueError('--episodes/--steps/--fps must be positive')
    if not Path(args.model).is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {args.model}')
    torch.set_num_threads(1)
    model, ck = load_policy(args.model, device='cpu')
    if not ck.get('trained', False):
        if not args.allow_untrained:
            raise RuntimeError(
                'This checkpoint has trained=False (initial/random or BC-only weights). '
                'Use a real PPO training checkpoint, or add --allow-untrained to see initial behavior.'
            )
        print('WARNING: This checkpoint is NOT PPO-trained; behavior is not representative of a trained agent.', flush=True)

    print('Model:', args.model)
    print('PPO trained:', bool(ck.get('trained')), '| saved env steps:', ck.get('total_env_steps', 0))
    print('Render:', args.render, '| actions:', 'stochastic' if args.stochastic else 'deterministic argmax')
    print('Ctrl+C to stop; press ESC or close the original game window to quit')
    platforms = load_platforms(args.map)
    encoder = StateEncoder(platforms)
    # Schematic mode uses actual game observations with no dependency on env.render().
    env = make_env('none' if args.render == 'schematic' else args.render)
    actual_render = args.render
    rng = random.Random(args.seed)
    # A decision corresponds to four internal frames; 15 actions/s approximates 60 FPS.
    frame_period = 1.0 / args.fps
    interrupted = False
    try:
        for ep in range(args.episodes):
            seed = rng.randrange(2**32)
            obs, info = env.reset(seed=seed)
            start = time.perf_counter()
            end_state = 'step_cap'
            last_t = 0.0
            for step in range(args.steps):
                wall_t = time.perf_counter()
                state = encoder.encode(obs, float(info.get('time_elapsed', 0.0)))
                with torch.inference_mode():
                    act_id, _, _ = model.act(torch.from_numpy(state).unsqueeze(0),
                                             deterministic=not args.stochastic)
                action_id = int(act_id.item())
                action = ACTION_TABLE[action_id].copy()
                obs, _, terminated, truncated, info = env.step(action)
                last_t = float(info.get('time_elapsed', (step + 1) * 4 / 60))
                hud = f'Episode {ep+1}/{args.episodes} | t={last_t:.2f}s | action={action_id} {action.tolist()} | steps={ck.get("total_env_steps", 0)}'
                if actual_render == 'rgb_array':
                    result = render_rgb(env, title='BulletBill PPO Watch (Q / Esc to quit)', hud=hud)
                    if result is None:
                        print('NOTICE: env.render() returned None; using observed-state schematic instead. '
                              'This is real environment motion, but not original game graphics.', flush=True)
                        actual_render = 'schematic'
                    else:
                        interrupted = result
                if actual_render == 'schematic':
                    image = schematic_frame(obs, platforms)
                    interrupted = show_frame(image,
                        title='BulletBill PPO - Observation View (Q / Esc)', hud=hud)
                elif actual_render == 'human':
                    # Like manual_play.py, the game's human renderer draws during
                    # env.step(); render() does not return an RGB frame.
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                        ):
                            interrupted = True
                            break
                if args.show_actions and (step % args.print_every == 0):
                    print(hud, flush=True)
                if terminated or truncated:
                    end_state = str(info.get('game_state', 'terminated' if terminated else 'truncated'))
                    break
                if interrupted:
                    end_state = 'user_exit'
                    break
                dt = time.perf_counter() - wall_t
                if dt < frame_period:
                    time.sleep(frame_period - dt)
            print(f'EPISODE {ep+1} seed={seed} survival={last_t:.3f}s '
                  f'steps={step+1} result={end_state} wall={time.perf_counter()-start:.1f}s', flush=True)
            if interrupted:
                break
    except KeyboardInterrupt:
        print('\nStopped by user.')
    finally:
        env.close()
        if actual_render in ('rgb_array', 'schematic'):
            import cv2
            cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='Path to PPO .pt checkpoint')
    ap.add_argument('--render', choices=['human', 'rgb_array', 'schematic'], default='human',
                    help='human shows original Pygame screen (default); schematic uses OpenCV' )
    ap.add_argument('--map', default='map_data/object_tiles.csv')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--steps', type=int, default=900, help='Maximum actions per episode (900 = 60s)')
    ap.add_argument('--seed', type=int, default=20261111)
    ap.add_argument('--fps', type=float, default=15, help='Agent decisions/second; default matches 4-frame skip at 60 FPS')
    ap.add_argument('--stochastic', action='store_true', help='Sample actions; default argmax for reproducibility')
    ap.add_argument('--allow-untrained', action='store_true', help='Allow viewing randomly initialized or BC-only checkpoints')
    ap.add_argument('--show-actions', action='store_true', help='Periodically print chosen actions')
    ap.add_argument('--print-every', type=int, default=15)
    args = ap.parse_args(argv)
    if args.render in ('rgb_array', 'schematic'):
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            ap.error('For --render schematic/rgb_array install OpenCV: .\\.venv\\Scripts\\python.exe -m pip install opencv-python')
    try:
        run(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
