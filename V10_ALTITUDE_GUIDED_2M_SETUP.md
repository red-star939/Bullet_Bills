# PPO v10 + Spawn Shield + Adaptive Altitude: 2M-step fine-tuning

This trains a *neural student* with **on-policy PPO** and **auxiliary supervised imitation** of a frozen PPO v10 + Spawn + Altitude teacher. It does not put Python-based safety logic inside the neural network and does not guarantee any improvement in survival.

## Required project files

Put `train_v10_altitude_guided.py` in the project root. Existing `agent_ppo_v10.py`, `models/v10_spawn_guidance.py`, `models/adaptive_altitude_controller.py`, `models/model_guidance.py`, `models/spawn_risk_evaluator.py`, all other current predictive `models` files and `map_data/object_tiles.csv` must be present.

Run in Windows PowerShell:

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.\.venv\Scripts\python.exe -m pip install numba
.\.venv\Scripts\python.exe train_v10_altitude_guided.py `
  --init models/bullet_bill_ppo_v10_best.pt `
  --steps 2000000 `
  --envs 16 `
  --rollout 250 `
  --threads 1 `
  --lr 0.0001 `
  --teacher-coef-start 0.3 `
  --teacher-coef-end 0.1 `
  --out models/v10_altitude_guided_2m
```

This is exactly 500 updates × 4,000 transitions. Teacher coefficient decays from 0.3 to just above 0.1 and never reaches zero. Checkpoints are written every 100,000 steps and tested at 500k increments on 10 fixed evaluation seeds, with a **600-second maximum** per episode. Evaluation is **pure student PPO** (not shield-assisted). The original model is not overwritten.

## Evaluation of final student

Student only:

```powershell
.\.venv\Scripts\python.exe watch_ppo_v10_altitude.py `
  --model models/v10_altitude_guided_2m/bullet_bill_ppo_v10_altitude_latest.pt `
  --guidance ppo --episodes 5 --steps 9000 --fps 0
```

Student + Spawn Shield + Altitude:

```powershell
.\.venv\Scripts\python.exe watch_ppo_v10_altitude.py `
  --model models/v10_altitude_guided_2m/bullet_bill_ppo_v10_altitude_latest.pt `
  --guidance altitude_shield --episodes 5 --steps 9000 --fps 0
```

Resume interrupted training (not identical RNG trajectories because fresh environments are created; optimizer is restored):

```powershell
.\.venv\Scripts\python.exe train_v10_altitude_guided.py `
  --init models/bullet_bill_ppo_v10_best.pt `
  --resume models/v10_altitude_guided_2m/bullet_bill_ppo_v10_altitude_latest.pt `
  --steps 2000000 --envs 16 --rollout 250 `
  --out models/v10_altitude_guided_2m
```

## Important technical distinctions

- PPO rollout actions are always sampled by student and executed in environment. Shield is NEVER passed off as an on-policy student action.
- Frozen teacher computes an action for the same observed state, modified by the existing shield and altitude controller. This becomes the auxiliary BC label (categorical jump + continuous horizontal move).
- Model action space remains **5 jump levels + Gaussian move**; not our 8 discrete-action PPO.
- The saved `.pt` retains `model`, `obs_dim`, `hidden` and can load using original `BulletBillAgent.load`.
- Pure student may not replicate controller; teacher features and hidden temporal control state are not perfectly representable in a single Markov neural output. Actual survival must be measured, ideally on seeds excluded from training.
- 16 environments are executed sequentially in one process, not Windows subprocess parallel. Heavy frozen teacher path is likely to be computation intensive. `numba` is strongly recommended.
- Checkpoints preserve optimizer state, but restart begins fresh environment episodes, not bitwise-continuous experience.
