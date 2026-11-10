# BulletBill PPO v1 — 설치·학습·평가

## 현재 상태

- `models/mario_ppo_init.pt`는 **미학습 초기 가중치**입니다. `trained=False`, `total_env_steps=0`으로 명확하게 표시합니다.
- `models/ppo_policy.py`는 별도의 이산 Actor-Critic 신경망과 관측 인코더를 제공합니다.
- `train_ppo.py`를 실제 Windows 게임 환경에서 실행해야 **게임으로 학습한** `mario_ppo_latest.pt`가 생성됩니다.
- `collect_ppo_teacher.py`와 `pretrain_ppo_bc.py`는 검증된 ActionEvaluator v2의 지식을 초기 정책에 전달하기 위한 **선택적 사전학습** 경로입니다.

## 파일 배치

```text
mario_pj/
  agent_env.py                         # 기존 프로젝트에 있는 환경 팩토리
  map_data/object_tiles.csv            # 기존 파일
  models/
    action_evaluator.py                # 기존 파일, DEFAULT_ACTIONS 제공
    action_evaluator_v2.py             # 기존 파일, Teacher로 사용
    ppo_policy.py                      # 신규
    mario_ppo_init.pt                  # 신규, 아직 학습되지 않음
  train_ppo.py                         # 신규
  eval_ppo.py                          # 신규
  collect_ppo_teacher.py               # 신규
  pretrain_ppo_bc.py                   # 신규
  test_ppo.py                          # 신규
  test_ppo_training_smoke.py           # 신규
```

## 1. 기본 기능 테스트

```powershell
.\.venv\Scripts\python.exe test_ppo.py
.\.venv\Scripts\python.exe test_ppo_training_smoke.py
```

두 번째 테스트의 가짜 환경 성공은 **실제 게임 생존 성능을 증명하지 않습니다.**

## 2A. PPO 직접 학습 — 가장 간단한 첫 .pt 생성 경로

```powershell
.\.venv\Scripts\python.exe train_ppo.py `
  --total-steps 100000 `
  --rollout-steps 2048 `
  --max-episode-steps 900 `
  --save-interval 10000 `
  --out models/ppo_runs
```

학습 중 `models/ppo_runs/mario_ppo_latest.pt`와 스텝별 버전 `.pt`가 생성됩니다.

**주의:** `--total-steps`는 행동 스텝 수이며 내부 물리 프레임 수가 아닙니다.

## 2B. 권장: ActionEvaluator v2 교사 초기화

```powershell
.\.venv\Scripts\python.exe collect_ppo_teacher.py `
  --episodes 100 `
  --steps 900 `
  --seed 20261112 `
  --out data/ppo_teacher_v2.npz

.\.venv\Scripts\python.exe pretrain_ppo_bc.py `
  --dataset data/ppo_teacher_v2.npz `
  --epochs 12 `
  --out models/mario_ppo_bc_init.pt

.\.venv\Scripts\python.exe train_ppo.py `
  --resume models/mario_ppo_bc_init.pt `
  --total-steps 100000 `
  --save-interval 10000 `
  --out models/ppo_runs_bc
```

`mario_ppo_bc_init.pt`는 **BC만 학습된 가중치**입니다. PPO 경험으로 업데이트된 가중치는 그 다음 `train_ppo.py`에서 생성됩니다.

## 3. 독립 시드 평가

```powershell
.\.venv\Scripts\python.exe eval_ppo.py `
  --model models/ppo_runs/mario_ppo_latest.pt `
  --episodes 100 `
  --steps 900 `
  --seed 20261111 `
  --out logs/ppo_v1_eval
```

비교는 반드시 동일 시드의 ActionEvaluator v2와 실행하고, 60초까지 살아남은 에피소드는 **우측 검열된 시간**으로 표시합니다. `eval_ppo.py`는 미학습 초기 .pt 평가를 거부합니다.

## 핵심 설계

- **환경 행동:** 연속 입력 `np.float32([move, jump])`를 유지하며 내부 정책의 정수 ID를 기존 8개 후보에 매핑합니다.
- **입력:** `mario` 8개 값, 포탄 최대 8개의 상대 위치·속도, 플랫폼 충돌 타일 요약, 경과 시간. 고정 79차원.
- **네트워크:** actor / critic 각각 2×128 Tanh MLP, 8-class Categorical 정책.
- **학습:** PPO clipped objective, GAE(λ), 가치함수 손실, entropy 보너스, KL 조기 중단, gradient clipping.
- **보상:** 행동당 +0.01 생존, 실제 `game_over` -1, 명시적 성공 시 +1. 초기 하이퍼파라미터이며 성능 검증 필요.
- **경계:** 시간 제한 `truncation` / 수동 에피소드 절단 시 가치 부트스트랩을 유지하되 GAE는 다음 에피소드로 이어지지 않도록 처리.
- **Teacher:** ActionEvaluator v2의 관측→행동을 기록하여 PPO actor를 사전학습합니다. 실제 게임의 내부 오라클 값은 입력에 포함하지 않습니다.

## 현재 한계

이산화된 8개 후보 외에 미세 연속 행동을 출력하지 않습니다. 초기 보상과 인코더는 최초 버전이며 최적화되지 않았습니다. `--resume`은 모델/옵티마이저/RNG 상태를 로딩하지만 중단된 게임의 남은 에피소드 위치는 복원하지 않으므로, 비트 단위로 완벽한 세션 재현은 보장하지 않습니다. 기존의 `models/` 디렉터리가 패키지로 접근되도록 **프로젝트 루트에서 실행**하세요.
