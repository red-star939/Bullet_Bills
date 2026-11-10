# PPO v10 + Spawn + Adaptive Altitude: true Windows multiprocess training

This replaces the **sequential** `train_v10_altitude_guided.py` with real 16-worker subprocess training. It uses **the same** v10 network, rewards, PPO/BC objectives, frozen teacher, action encoding and checkpoint model schema. Each subprocess owns its environment, feature extractor, v10 frozen teacher, Spawn Shield, and Adaptive Altitude controller. The parent owns the trainable neural policy and Adam optimizer, sends each worker a sampled student action, and receives teacher imitation targets plus environment transition. Worker computations happen concurrently.

## Requirements

Unzip into `C:\Users\USER\Desktop\HONG_LB\mario_pj`. Required: `agent_ppo_v10.py`, `models/v10_spawn_guidance.py`, `models/adaptive_altitude_controller.py` and their existing dependent modules, `map_data/object_tiles.csv`, `models/bullet_bill_ppo_v10_best.pt`, installed `kymnasium`, `gymnasium`, `torch`, `numpy`. **Numba is strongly recommended**. No original source or checkpoint files are overwritten.

Check Numba:

```powershell
.\.venv\Scripts\python.exe -c "import numba; print(numba.__version__)"
```

Install if missing:

```powershell
.\.venv\Scripts\python.exe -m pip install numba
```

## First: 4-worker smoke test (64 transitions)

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe train_v10_altitude_guided_parallel.py `
  --init models/bullet_bill_ppo_v10_best.pt `
  --steps 64 --envs 4 --rollout 16 `
  --threads 1 --save-every 1 --eval-every 0 `
  --out models/v10_altitude_parallel_smoke
```

A successful smoke test prints `STARTING 4 independent Windows-spawn game workers...`, `STEPS 64 ...`, and `CHECKPOINT ...`. First start may take longer due to imports and Numba JIT compilation in each worker. Do not read the tiny-smoke throughput as a steady-state benchmark.

## Full training: 2,000,000 steps, 16 workers

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe train_v10_altitude_guided_parallel.py `
  --init models/bullet_bill_ppo_v10_best.pt `
  --steps 2000000 `
  --envs 16 `
  --rollout 250 `
  --threads 1 `
  --lr 0.0001 `
  --teacher-coef-start 0.3 `
  --teacher-coef-end 0.1 `
  --log-every 1 --save-every 5 `
  --out models/v10_altitude_guided_2m_parallel
```

One PPO update is 16*250=4,000 transitions; each `--save-every 5` writes checkpoint every 20,000 steps. It runs 500 updates. By default `--eval-every 125` triggers a deterministic single-process **student-only** 10-seed evaluation, each capped at 9000 actions (600 game seconds), every 500,000 training steps. Evaluation can be slow, and its wall time is included in reported throughput. To disable while measuring throughput use `--eval-every 0`; do **not** confuse a 600s evaluation cap with actual death time.

For 16 cores / 16 threads, 16 envs + learner compete for CPU; test 12 workers if throughput is worse. For 12 workers choose `--rollout 250 --steps 2001000` or choose another divisible target: 2,000,000 is NOT divisible by 12*250. An easy 12-worker 2M setting is `--envs 12 --rollout 200 --steps 1999200` (or `--steps 2001600`); note the exact step total. For exactly 2M at 12 workers, `--rollout 125` yields 1500 per update, not divisible; `--rollout 100` yields 1200, not divisible. Use `--rollout 250 --envs 10` (2500 divides 2M) or `--envs 16` as configured.

## Resuming

```powershell
.\.venv\Scripts\python.exe train_v10_altitude_guided_parallel.py `
  --init models/bullet_bill_ppo_v10_best.pt `
  --resume models/v10_altitude_guided_2m_parallel/bullet_bill_ppo_v10_altitude_latest.pt `
  --steps 2000000 --envs 16 --rollout 250 `
  --out models/v10_altitude_guided_2m_parallel
```

`--resume` also accepts compatible checkpoints written by the older sequential script; restore weights, optimizer and total step, but **not** in-progress episode or RNG trajectories. Resume counts `--steps` as the cumulative target. Keep same original `--init` to preserve teacher, and same BC/loss hyperparameters.

## Playback

```powershell
.\.venv\Scripts\python.exe watch_ppo_v10_altitude.py `
  --model models/v10_altitude_guided_2m_parallel/bullet_bill_ppo_v10_altitude_latest.pt `
  --guidance altitude_shield --episodes 5 --steps 9000 --fps 0
```

All 16 workers execute the PPO **student action**; the frozen hybrid controller only supplies BC targets. `.pt` remains a neural v10 checkpoint and does NOT embed shield Python code. Actual 100-second survival from the frozen hybrid does not guarantee the trained student reaches 100 seconds.

## Limitations

- Windows spawn launches 16 new Python interpreters, each with its own environment, v10 frozen network and simulator. Startup (and Numba compilation) is initially expensive. CPU usage will depend on processor scheduling and workload.
- Numba compilation in each process can briefly delay the first rollout. Throughput gain is not guaranteed; compare after startup with the previous 86.1 steps/sec.
- In-process evaluation is intentionally sequential, so training throughput averages decrease during eval; use `--eval-every 0` for training-speed comparison.
- `LAST100_SURVIVAL nan` until an episode completes, as in original script.
- Existing trainer's GAE treats all terminations/truncations as boundaries. This implementation retains that behavior.
- The teacher is a deterministic frozen v10 + overlays and depends on private controller history (e.g. altitude cooldown); the standalone neural policy cannot perfectly represent this history from one snapshot.
