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
    try:
        from agent_env import create_env
    except ImportError as exc:
        raise RuntimeError('agent_env.py / kymnasium is required to display the actual game. Run from mario_pj project root.') from exc
    try:
        return create_env(render_mode=render_mode)
    except Exception as exc:
        raise RuntimeError(
            f'Could not create a {render_mode!r} environment. '
            'Try --render rgb_array or --render human, and verify agent_env.create_env supports that render mode.'
        ) from exc


def render_rgb(env, *, title: str, hud: str):
    import cv2

    frame = env.render()
    if isinstance(frame, (list, tuple)) and len(frame):
        frame = frame[-1]
    if frame is None:
        raise RuntimeError('env.render() returned None for rgb_array. Check the render mode implementation in agent_env.py.')
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise RuntimeError(f'Unexpected rgb_array frame shape: {frame.shape!r}')
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[2] == 4:
        image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 1100), 34), (15, 15, 15), -1)
    cv2.putText(image, hud, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.57, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(title, image)
    # cv2.waitKey is mandatory for drawing and window event processing.
    key = cv2.waitKey(1) & 0xFF
    return key in (27, ord('q'))


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
    print('Ctrl+C to stop' + ('; press Q or ESC in the game window to quit' if args.render == 'rgb_array' else ''))
    encoder = StateEncoder(load_platforms(args.map))
    env = make_env(args.render)
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
                if args.render == 'rgb_array':
                    interrupted = render_rgb(env, title='BulletBill PPO Watch (Q / Esc to quit)', hud=hud)
                else:
                    # Some human render modes paint during env.step; others need render().
                    env.render()
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
        if args.render == 'rgb_array':
            import cv2
            cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='Path to PPO .pt checkpoint')
    ap.add_argument('--render', choices=['human', 'rgb_array'], default='human')
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
    if args.render == 'rgb_array':
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            ap.error('For --render rgb_array install OpenCV: .\\.venv\\Scripts\\python.exe -m pip install opencv-python')
    try:
        run(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
