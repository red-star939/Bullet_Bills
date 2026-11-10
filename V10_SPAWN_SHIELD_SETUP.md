# PPO v10 + Spawn-Aware Shield: 실행 및 검증

## 무엇이 달라졌나

- 모델: 협업자 `models/bullet_bill_ppo_v10_best.pt` (원본 `agent_ppo_v10.py`의 `BulletBillAgent.load`).
- 특징: 원본 v10 FeatureExtractor 148차원, Numba 기반 계획 특징 그대로 사용.
- 행동: 수평 **연속 행동**과 5단계 점프 강도를 그대로 전달. PPO 출력의 행동을 8개 이산 행동으로 강제 변환하지 않음.
- `ppo`: 순수 v10 정책 (안전 제어 미개입).
- `shield`: v10 행동에 대해 **우리 ForwardSimulator+PlatformSafetyEvaluator**로 예측 사망/추락 검사. 대안이 검증될 때만 ActionEvaluator v2의 이산 안전 행동으로 교체.
- `spawn_shield`: 위 검사에 더해 **SpawnRiskEvaluator**를 사용해 벽 접근 위험도가 높을 때 수평 이동을 조정. 이 경우 점프 강도는 v10이 선택한 값을 유지.
- `teacher`: ActionEvaluator v2 정책 단독. PPO v10 자체 평가가 아님.

**주의:** 벽 위험은 임의적인 기하학적 노출 지표이며 포탄 생성 확률이 아닙니다. 미래 신규 포탄은 정확하게 예측하지 못합니다. Shield는 무조건 안전을 보장하지 않습니다.

## 설치

새 파일을 프로젝트 루트 기준으로 배치:

```text
mario_pj/
  agent_ppo_v10.py                 # 기존 협업자 원본 유지
  watch_ppo_v10_spawn.py           # 신규
  benchmark_ppo_v10_spawn.py       # 신규
  test_v10_spawn_guidance.py       # 신규
  models/
    bullet_bill_ppo_v10_best.pt    # 사용자 보유 체크포인트
    v10_spawn_guidance.py          # 신규
    model_guidance.py              # 기존
    spawn_risk_evaluator.py       # 기존
    ...                            # 기존 ForwardSimulator 등
```

실행 전 의존성 검사:

```powershell
.\.venv\Scripts\python.exe -c "import torch, pygame, gymnasium, kymnasium, numba; print('OK')"
.\.venv\Scripts\python.exe test_v10_spawn_guidance.py
```

Numba가 없다면 `python -m pip install numba`를 실행합니다. Windows PowerShell의 가상환경 Python 경로를 사용하세요.

## 원본 게임 화면

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.\.venv\Scripts\python.exe watch_ppo_v10_spawn.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --guidance spawn_shield `
  --episodes 5 `
  --seed 20261110 `
  --fps 0 `
  --profile-every 100
```

원본 v10과 비교할 때 `--guidance ppo`, 기존 Shield는 `--guidance shield`로 변경. `--fps 0`은 **추가 대기만 제거**하며 실제 연산 병목은 개선하지 않습니다. `--fps 15`로 실시간 재생을 시도할 수 있습니다. 처음에는 Numba JIT 컴파일 시간이 커질 수 있습니다.

## 같은 시드 정량 비교

```powershell
.\.venv\Scripts\python.exe benchmark_ppo_v10_spawn.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --episodes 20 `
  --steps 900 `
  --seed 20261110 `
  --out logs/v10_spawn_20
```

`episodes.csv`를 각 에피소드 종료 즉시 기록하며, 완료 후 `summary.json`을 생성합니다. 중단 후 동일 인자로 `--resume` 지정 가능. **재개 시 모델·평가 설정·지도 파일을 변경하지 마세요**(CSV에서 시드/모드만 검증).

정량 비교는 v10 순수 정책, 기존 Shield, Spawn Shield를 동일 시드로 수행하지만 환경 내부 난수 소비가 서로 달라질 수 있어 완전히 동일한 미래 포탄 경로가 보장되지는 않습니다.

## 성능 병목 확인

뷰어 `PERF`의 `v10_act_ms`는 v10 모델 특징 추출+신경망 추론, `guidance_ms`는 추가 안전 평가의 평균 결정 시간입니다. 과도하게 느리다면 먼저 순수 `ppo` 모드와 비교하여 어떤 모듈이 병목인지 확인합니다.

## 검증 범위

Python 구문 검사, 안전 모드의 연속 행동 보존, 양쪽 벽 복귀 후보 중 점프 세기 유지 등의 **오프라인 테스트만 수행**했습니다. 협업자 `.pt` 파일 및 실제 kymnasium 게임은 사용자 PC에 있으므로 실환경 성공 및 생존 시간 개선은 아직 미검증입니다.
