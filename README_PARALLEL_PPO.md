# models2 — Windows CPU 병렬 PPO v4.1 추가 학습기

기존 `train_models2_v41_ppo.py`는 `--envs 4`를 주더라도 게임 환경을 **단일 Python 프로세스에서 순차 실행**합니다. 이 패키지의 `train_models2_v41_ppo_parallel.py`는 환경과 148차원 특징 추출기를 **독립 Worker 프로세스**로 옮깁니다. PyTorch의 정책 추론·PPO 업데이트는 중앙 프로세스가 수행합니다.

- **입력**: 기존 `models2/bullet_bill_ppo_v10_v41_distilled.pt` 또는 호환 PPO v10 `.pt`
- **선택 입력**: `logs/models2_distill/teacher.npz` (v4.1 교사 행동의 보조 BC 손실)
- **출력**: `models2/bullet_bill_ppo_v10_v41_parallel_best.pt`, `..._last.pt`, `..._resume.pt`
- **학습**: 무작위 에피소드 시드, v10 원본 보상, PPO clipped objective + GAE. v4.1 교사 데이터는 보조 손실이며 환경 내 v4.1 컨트롤러를 매번 구동하는 것은 아닙니다.

## 1. 설치

ZIP을 `C:\Users\USER\Desktop\HONG_LB\mario_pj`에 풀고 **프로젝트 루트**에서 명령을 실행하세요. 기존 `train_models2_v41_ppo.py`, `agent_ppo_v10.py`, `models2` 체크포인트가 있어야 합니다. 기존 파일은 덮어쓰지 않습니다.

```powershell
Test-Path .\train_models2_v41_ppo.py
Test-Path .\agent_ppo_v10.py
Test-Path .\models2\bullet_bill_ppo_v10_v41_distilled.pt
Test-Path .\logs\models2_distill\teacher.npz
```

마지막 `teacher.npz`가 없다면 **`--teacher-data` 줄을 제거**할 수 있습니다. 그 경우 PPO만 추가 학습하며 v4.1 보조 행동 증류는 실행되지 않습니다. 증류 모델 자체가 없다면 `--init models2/bullet_bill_ppo_v10_best.pt`로 변경할 수 있지만 그 경우 기존 v4.1 증류 결과는 포함되지 않습니다.

## 2. 간단한 실제 환경 사전 점검 (먼저 2만 스텝)

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe train_models2_v41_ppo_parallel.py `
  --init models2/bullet_bill_ppo_v10_v41_distilled.pt `
  --teacher-data logs/models2_distill/teacher.npz `
  --out models2/bullet_bill_ppo_v41_parallel_smoke_best.pt `
  --total-steps 20000 `
  --envs 4 `
  --batch-steps 128 `
  --epochs 2 `
  --eval-episodes 2 `
  --final-episodes 2 `
  --eval-interval 10000 `
  --checkpoint-interval 5000 `
  --log-dir logs/models2_v41_parallel_smoke
```

이 실행은 **별도 예비 실험**이므로, 본 학습에 사용할 출력 경로·로그 경로와 구분했습니다.

## 3. 500만 스텝 본 학습 — 12 Workers 권장 시작점

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe train_models2_v41_ppo_parallel.py `
  --init models2/bullet_bill_ppo_v10_v41_distilled.pt `
  --teacher-data logs/models2_distill/teacher.npz `
  --out models2/bullet_bill_ppo_v10_v41_parallel_best.pt `
  --total-steps 5000000 `
  --envs 12 `
  --batch-steps 256 `
  --lr 0.00002 `
  --eval-interval 50000 `
  --checkpoint-interval 10000 `
  --eval-episodes 10 `
  --final-episodes 50 `
  --log-dir logs/models2_v41_ppo_parallel_5m
```

CPU와 RAM에 여유가 있으면 **본 학습을 시작하기 전에** `--envs 16`을 시험해 보세요. 12와 16은 실행 처리량(예: `steps / elapsed`)으로 비교하세요. Worker마다 Python·Numba·PyTorch 라이브러리가 로드되므로 메모리 사용량이 커질 수 있습니다. 물리 코어가 16개인지, 논리 프로세서가 16개인지도 확인하세요. 사용률 100%가 목적이 아니라 **초당 게임 스텝 처리량**이 목적입니다. `--envs 16`은 Worker 16개를 생성하지만 스케일링 성능을 보장하지 않습니다.

학습 처리량 측정은 `UPDATE` 줄의 스텝 진행과 `wall` 값을 참고하세요. `wall`은 이전 로그 이후의 경과 시간입니다. `EVAL`은 별도로 단일 환경에서 수행돼 CPU 사용률이 낮아질 수 있습니다.

**참고:** 남은 스텝이 Worker 수로 나누어떨어지지 않으면 최대 `envs-1` 환경 스텝 초과 수행할 수 있습니다. 예: 5,000,000 / 12는 나누어떨어지지 않아 마지막 보고값은 5,000,004가 될 수 있습니다.

## 4. 체크포인트와 학습 재개

다음 세 파일을 저장합니다.

- `..._best.pt`: 검증 시드에서 생존 평균이 가장 높았던 정책; 기존 `BulletBillAgent.load()` 호환
- `..._last.pt`: 가장 최근 평가에서의 정책; 일반 `.pt` 로더 호환
- `..._resume.pt`: **PPO 학습 전용 전체 체크포인트**. 모델·Adam 옵티마이저·난수 상태·시드·완료 스텝 포함. `BulletBillAgent.load()`로 직접 실행하는 파일이 아님.

중간 Resume 체크포인트는 10,000스텝 간격(정확히는 가장 가까운 rollout 경계), 평가 시점에도 저장됩니다. 프로세스가 비정상 종료되면 마지막 저장 지점 이후의 미저장 진행은 재생되지 않습니다. 게임 내부 에피소드 상태는 복원할 수 없으므로 재개 직후 각 Worker는 **새 무작위 에피소드로 시작**하며, 이는 완전히 동일한 중단 없는 실행과 다릅니다.

동일한 실험을 500만에서 **누적 1,000만 스텝으로 확대**할 때:

```powershell
.\.venv\Scripts\python.exe train_models2_v41_ppo_parallel.py `
  --init models2/bullet_bill_ppo_v10_v41_distilled.pt `
  --teacher-data logs/models2_distill/teacher.npz `
  --out models2/bullet_bill_ppo_v10_v41_parallel_best.pt `
  --resume models2/bullet_bill_ppo_v10_v41_parallel_best_resume.pt `
  --total-steps 10000000 `
  --envs 12 `
  --batch-steps 256 `
  --lr 0.00001 `
  --eval-interval 50000 `
  --checkpoint-interval 10000 `
  --eval-episodes 10 `
  --final-episodes 50 `
  --log-dir logs/models2_v41_ppo_parallel_5m
```

**`--total-steps`는 추가 스텝이 아니라 누적 목표 스텝**입니다. Resume 중에는 `--envs`, `--batch-steps`, `--max-episode-steps`, 출력 경로 및 교사 데이터 경로가 일치해야 합니다. 이미 저장한 최고 정책과 로그 폴더도 그대로 보존하세요.

중요: 이 학습기의 `--resume`은 이 학습기가 생성한 `_resume.pt`만 받습니다. 기존 단일 프로세스 학습기의 일반 `.pt`로부터는 `--init` 재시작만 가능합니다.

## 5. 테스트와 시뮬레이션

게임이 설치된 환경에서 단위 테스트:

```powershell
.\.venv\Scripts\python.exe -m unittest test_parallel_ppo -v
```

테스트는 **모의 게임 환경**에 대해 Windows `spawn`을 통한 실제 Worker 생성, PPO 스텝, 옵티마이저 저장·재개를 검증합니다. 실제 `kymnasium` 게임에서 생존 성능이나 16 Worker의 처리량까지 검증하는 테스트는 아닙니다.

실제 게임 시뮬레이션(기존 실행기 필요):

```powershell
.\.venv\Scripts\python.exe eval_models2_pt_random.py `
  --model models2/bullet_bill_ppo_v10_v41_parallel_best.pt `
  --seed-file logs/models2_v41_ppo_parallel_5m/final_holdout_seeds.json `
  --episodes 5 `
  --steps 9000 `
  --render human `
  --out logs/models2_v41_ppo_parallel_5m/visual
```

## 6. 제약 및 실험 해석

- 새 구조는 Worker를 **실제 멀티프로세스로 실행**합니다. 중앙 PPO 추론·경사 계산은 단일 프로세스입니다.
- Windows에서는 `spawn` 시작 비용이 발생하고, Numba JIT 초기화/첫 실행이 느릴 수 있습니다.
- `--envs 16`과 PyTorch의 내부 스레드 16개 사용은 별개입니다. 기존 OpenMP 1개 제한은 Worker 간 경쟁 방지에 유리할 수 있습니다.
- PPO는 on-policy 상태를 유지하며, 교사 데이터는 `auxiliary_bc()`에서만 보조 손실로 사용됩니다.
- 초기 모델이 `.pt`이고 출력도 `.pt`이지만, 500만 스텝이 생존 150초 평균을 보장하지는 않습니다.
- 환경 스텝 단위와 프레임 단위는 다릅니다(게임 1스텝 약 4프레임).
- 평가 시드와 최종 홀드아웃 시드는 학습 에피소드 시드와 구분됩니다. **평가와 홀드아웃 자체는 동일한 고정 집합**이어서 체크포인트의 비교가 가능합니다.
- 500만 스텝을 돌리기 전에 20,000스텝 예비 학습과 출력 파일 저장을 꼭 확인하세요.
