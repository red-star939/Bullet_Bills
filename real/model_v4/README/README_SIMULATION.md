# Independent Model V4 simulation

Copy the **three** new Python files into `mario_pj/real/model_v4/`, alongside `model_v4_net.py` and `train_initial_50k.py` from the initial-50k package. This patch does **not** overwrite the model or training script.

The evaluation imports `model_v4_net.ModelV4`, not any PPO v10 code. It accepts both a final Model V4 `.pt` and a Model V4 resume checkpoint (for observation only).

## Quick test before 50k training

```powershell
.\.venv\Scripts\python.exe real/model_v4/watch_model_v4.py --model real/model_v4/model_v4_initial_untrained.pt --episodes 1 --fps 30
```

This **untrained** model only tests whether state encoding, action application and rendering work. Low survival is expected.

## Human visualization after training

```powershell
.\.venv\Scripts\python.exe real/model_v4/watch_model_v4.py --model real/model_v4/model_v4_initial_50k.pt --episodes 5 --seed 500123 --fps 30 --out real/model_v4/eval_results/visible
```

Press `Esc` or close window to stop. `--fps 0` runs without a frame cap.

## Headless (no window)

```powershell
.\.venv\Scripts\python.exe real/model_v4/simulate_model_v4.py --model real/model_v4/model_v4_initial_50k.pt --episodes 30 --seed 500123 --out real/model_v4/eval_results/headless
```

Output is `summary.json` + `episodes.csv` in `--out` directory, **not** in `checkpoints/`.

## More options

- `--stochastic`: sample actions instead of deterministic action selection
- `--device cpu|cuda|auto`: neural-network inference device (environment always CPU)
- `--seed-file path.json`: reuse exact episode seeds for fair model comparison; accepts `[1,2,...]` or `{ "seeds": [1,2,...] }`
- `--max-steps`: safety limit for each episode (default 18,000)
- `--game-fps`: used only for fallback time estimate when the game does not expose `info["time_elapsed"]`

Both scripts instantiate the real `kymnasium/BulletBill-Continuous-Hard-Stage-2` environment. No real-game runtime is available in the package builder; test it on your Windows machine. The model's `obs_vector` is a prototype encoder that uses sorted numeric observations and fixed-size 256D encoding. Actual game observation compatibility needs validation.

If the model has not trained to 50k yet, its file does not exist. Use `model_v4_initial_untrained.pt` until 50k training completes.
