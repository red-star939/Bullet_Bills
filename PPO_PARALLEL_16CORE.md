# BulletBill PPO v1 — 16개 CPU 환경 병렬 학습

## 구조

- `train_ppo_parallel.py`: **한 개 PPO Actor-Critic**이 16개 별도 게임 프로세스에서 롤아웃을 받아 학습
- Windows `multiprocessing` **spawn** 방식. 게임 환경은 각 워커 안에서 생성
- 행동 선택은 메인 프로세스에서 배치로 추론, 워커에는 8개 행동 중 인덱스만 전달
- `rollout_steps`는 **워커 전체 합산 환경 스텝 수**. 기본 4096이면 16개 워커당 보통 256스텝
- GAE는 워커·에피소드 경계를 구분하고 종료/시간 제한을 다르게 처리
- `.pt` 체크포인트는 기존 `models/ppo_policy.py`와 호환

## 설치

프로젝트 루트에 ZIP을 풀어 `agent_env.py`, `models/action_evaluator.py`, `map_data/object_tiles.csv`가 그대로 있게 하세요.
PyTorch, NumPy 및 기존 게임 패키지(`kymnasium` 등)가 설치되어야 합니다.

## 검증

```powershell
.\.venv\Scripts\python.exe test_ppo_parallel_smoke.py
```

합성 게임으로 4개 `spawn` 프로세스와 PPO 업데이트, 체크포인트 재개를 검사합니다. 실제 게임 성능 증거는 아닙니다.

## 16워커 실제 학습

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.\.venv\Scripts\python.exe train_ppo_parallel.py `
  --num-envs 16 `
  --torch-threads 1 `
  --total-steps 1000000 `
  --rollout-steps 4096 `
  --max-episode-steps 900 `
  --save-interval 25000 `
  --log-interval 10000 `
  --out models/ppo_runs_parallel16
```

**주의:** 16개 워커는 16개 물리 코어 사용을 *보장하는 설정*이 아니라 16개 CPU 프로세스를 생성하는 설정입니다. 운영체제가 프로세스를 코어에 배치합니다. PC에 논리 CPU가 16개뿐이면 메인 학습 프로세스까지 더해 과부하가 생길 수 있습니다. 자원 부족·오히려 느려지는 경우 `--num-envs 12` 또는 `8`로 줄여 측정하세요.

첫 시험은 `--total-steps 32768 --save-interval 8192`로 수행하는 것이 안전합니다.

## 사전학습 모델에서 시작

```powershell
.\.venv\Scripts\python.exe train_ppo_parallel.py `
  --num-envs 16 `
  --resume models/mario_ppo_bc_init.pt `
  --total-steps 1000000 `
  --out models/ppo_runs_parallel_bc
```

학습 완료 후:

```powershell
.\.venv\Scripts\python.exe eval_ppo.py `
  --model models/ppo_runs_parallel16/mario_ppo_latest.pt `
  --episodes 100 --steps 900 --seed 20261111
```

## 체크포인트

- `models/ppo_runs_parallel16/mario_ppo_latest.pt`
- `models/ppo_runs_parallel16/mario_ppo_step_XXXXXXXXX.pt`
- `models/ppo_runs_parallel16/latest_metrics.json`

같은 경로로 `--resume`하면 모델/옵티마이저/랜덤 생성기 상태를 복원합니다. 단, **진행 중이던 게임의 환경 내부 상태는 저장하지 않습니다**. 재개 시 각 워커에서 새로운 에피소드를 시작합니다. 따라서 완전한 비트 단위 재현은 아닙니다.

## 성능 관리

- `ENV_STEPS_PER_S`: 실제 환경 수집+PPO 업데이트를 포함한 평균 속도
- `LAST100_SURVIVAL`: 최근 완료 100개 에피소드 평균 생존 시간
- CPU 100%에 가깝고 처리량이 낮으면 워커 수를 줄입니다.
- FPS/게임 스텝 수는 16배 늘어도 학습 품질이 16배 좋아지는 것은 아닙니다. PPO는 한 모델을 공유하고 on-policy 롤아웃을 모읍니다.
- CPU 환경 워커는 병렬이지만 정책 추론과 업데이트는 한 메인 프로세스에서 처리합니다. GPU를 사용하려면 `--device cuda` (CUDA가 설치된 경우) 설정이 가능하지만, 작은 MLP에서는 CPU가 빠를 수도 있습니다.
- 실제 게임 `agent_env.create_env(render_mode='none')`가 작업자 프로세스 안에서 실행되는지 확인하세요.
