# BulletBill Model-Guided PPO v2 (16 Windows workers)

## Important distinction

The original PPO v1 used the **same 8 actions** as ActionEvaluator but **did not use the evaluator at training or inference**. This update integrates the actual existing predictive components. The previous `mario_ppo_latest.pt` is *not retroactively a model-guided policy*.

`models/model_guidance.py` directly instantiates and calls:

- `MarioMotionPredictor.from_object_tiles_csv`
- `ForwardSimulator`
- `ActionEvaluatorV2` (which calls `ActionEvaluator`)
- `PlatformSafetyEvaluator`

**Learning** (`train_ppo_guided_parallel.py`): each of 16 environment processes computes a teacher action on the current observation using ActionEvaluator v2. The rollout **executes the PPO sampled action**, logs its true log probability and GAE returns, and uses the teacher action *only as an auxiliary cross-entropy target*: `L = L_PPO + vf_coef L_value - ent_coef H + teacher_coef L_teacher`. The teacher coefficient linearly decays to zero by `--teacher-decay-steps`. The `teacher_agreement` metric tracks predicted action agreement, not game success. Do not replace rollout actions with the teacher unless correcting the off-policy logprob/importance weights.

**Playback** (`watch_ppo_guided.py`):

- `--guidance teacher`: ActionEvaluator v2 directly controls Mario. This is a **teacher baseline**, not PPO performance.
- `--guidance shield` (default): PPO selects an action; `ForwardSimulator` and `PlatformSafetyEvaluator` look for a known-world predicted death/fall for that candidate. If one is predicted, teacher action is applied **only if separately checked without such a predicted lethal event**. If no verified alternative, retains PPO action and marks `ppo_no_verified_alternative`. `overrides` count is printed. This is a hybrid inference controller, **not pure PPO**; future bullet spawns remain unknown. It may not fix policy inactivity when no impending death is predicted.
- `--guidance ppo`: unmodified PPO (fair checkpoint performance benchmark).

## Windows commands

Copy `train_ppo_guided_parallel.py`, `watch_ppo_guided.py`, `test_ppo_guidance.py` into project root, and `models/model_guidance.py` into `models/`.

Test:

```powershell
.\.venv\Scripts\python.exe test_ppo_guidance.py
```

Compare the current unmodified PPO against the established v2 behavior, using the original game display:

```powershell
.\.venv\Scripts\python.exe watch_ppo_guided.py --model models/ppo_runs_parallel16/mario_ppo_latest.pt --render human --guidance ppo --episodes 3
.\.venv\Scripts\python.exe watch_ppo_guided.py --model models/ppo_runs_parallel16/mario_ppo_latest.pt --render human --guidance teacher --episodes 3
.\.venv\Scripts\python.exe watch_ppo_guided.py --model models/ppo_runs_parallel16/mario_ppo_latest.pt --render human --guidance shield --episodes 3 --show-actions
```

**Recommended teacher warm-start**: collect demonstrations, clone the actor, then run guided PPO. Start the BC run in its own output directory. For fair comparisons, evaluate on held-out seeds not used to collect demonstrations.

```powershell
.\.venv\Scripts\python.exe collect_ppo_teacher.py --episodes 100 --steps 900 --out data/ppo_teacher_v2.npz
.\.venv\Scripts\python.exe pretrain_ppo_bc.py --dataset data/ppo_teacher_v2.npz --epochs 20 --out models/mario_ppo_bc_init.pt

$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe train_ppo_guided_parallel.py `
  --num-envs 16 --torch-threads 1 `
  --resume models/mario_ppo_bc_init.pt `
  --total-steps 1000000 --rollout-steps 4096 `
  --teacher-coef 0.5 --teacher-decay-steps 500000 `
  --log-interval 4096 --save-interval 16384 `
  --out models/ppo_guided_bc16
```

To continue the *existing* PPO without BC (less likely to improve rapidly, but preserves existing model):

```powershell
.\.venv\Scripts\python.exe train_ppo_guided_parallel.py `
  --num-envs 16 --torch-threads 1 `
  --resume models/ppo_runs_parallel16/mario_ppo_latest.pt `
  --total-steps 1000000 --rollout-steps 4096 `
  --teacher-coef 0.5 --teacher-decay-steps 500000 `
  --out models/ppo_guided_from_old
```

Playback of the newly trained model:

```powershell
.\.venv\Scripts\python.exe watch_ppo_guided.py --model models/ppo_guided_bc16/mario_ppo_latest.pt --render human --guidance shield --episodes 5
```

## Important regression fixed

The original `train_ppo_parallel.py` constructed its Adam optimizer **before** replacing `net` when resuming a checkpoint. The optimizer could then update the discarded model instead of the reloaded policy. This guided trainer constructs Adam again **after** loading the model, then loads optimizer state, and includes a test asserting that actor parameters change on resumed training. This may explain failed progress during previous `--resume` runs; it does not by itself prove the cause of the reported stationary-jump behavior.

## Verification status

- Python compilation: passed.
- Actual MarioMotionPredictor/ForwardSimulator/ActionEvaluatorV2/PlatformSafetyEvaluator composition on synthetic observations: passed.
- `spawn` synthetic game worker rollout, teacher labels, PPO+CE optimization, checkpoint save, **resumed actor parameter change**: passed.
- Actual BulletBill environment and 16-core throughput: **not tested here** (kymnasium unavailable).
- This update does **not** claim to have trained a new improved `.pt` or surpassed ActionEvaluator v2. Run the commands above on the project Windows machine and evaluate new seeds.
