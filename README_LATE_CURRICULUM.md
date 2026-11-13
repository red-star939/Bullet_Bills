# models2: 120~180초 구간 PPO 후반부 커리큘럼

## 핵심 동작

이 학습기는 120~180초 **새 게임 상태를 조작하거나 시간을 순간 이동하는 기능이 아닙니다**. 게임의 실제 물리·포탄·난수 상태를 보존하기 위해 환경을 0초에서 시작한 뒤, **현재 정책의 결정론적 행동으로 실제 게임을 진행**해 목표 시간에 도달했을 때만 PPO 수집을 시작합니다.

- 기본 시작 시간은 **120~165초 사이 무작위 선택**입니다. 해당 시작 지점부터 종료 시각인 약 180초까지 PPO 전이를 수집합니다. `--curriculum-max-s 175`로 늦출 수 있습니다.
- 120초 이전 워밍업 전이는 PPO 업데이트 스텝에 **포함하지 않습니다**. 따라서 `--total-steps`는 **후반부 학습 전이** 수이고, 실제 시뮬레이터 실행 횟수는 더 많습니다.
- 학습 보상은 기존 `agent_ppo_v10.py`의 v10 보상 함수와 동일합니다.
- 시작 시간 전 사망하면 새로운 학습 시드로 0초부터 재시도합니다. 한 워커에서 `--warmup-max-attempts`번 실패하면 중단합니다. 120초 도달이 불가능한 정책으로는 이 방법을 학습할 수 없습니다.
- 평가와 최종 홀드아웃은 **0초에서 시작하는 전체 게임**으로 측정합니다. `best.pt`는 전체 생존 평균을 기준으로 선택됩니다.
- Windows `spawn` 병렬 Worker 16개를 지원합니다. 워밍업에도 같은 Worker를 사용합니다.
- 기존 일반 PPO `best.pt`에서 시작해 **새 실험**으로 2백만~5백만 후반부 전이를 학습합니다. 기존 일반 PPO `..._resume.pt`로 바로 `--resume`하지 마세요. 새 학습기의 Resume만 지원합니다.

## 설치

이 ZIP에서 `train_models2_v41_ppo_late.py`를 `mario_pj` 폴더에 복사합니다. 기존 `train_models2_v41_ppo.py`와 `agent_ppo_v10.py` 및 `gymnasium`, `kymnasium` 환경이 필요합니다. 원본 파일을 덮어쓰지 않습니다.

```powershell
Test-Path .\models2\bullet_bill_ppo_v10_v41_parallel_best.pt
Test-Path .\train_models2_v41_ppo.py
Test-Path .\agent_ppo_v10.py
```

## 120~165초부터 200만 PPO 스텝 학습 (16 Worker)

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe train_models2_v41_ppo_late.py `
  --init models2/bullet_bill_ppo_v10_v41_parallel_best.pt `
  --out models2/bullet_bill_ppo_v41_late_best.pt `
  --total-steps 2000000 `
  --envs 16 `
  --batch-steps 256 `
  --lr 0.00001 `
  --curriculum-min-s 120 `
  --curriculum-max-s 165 `
  --warmup-max-attempts 25 `
  --eval-interval 50000 `
  --checkpoint-interval 10000 `
  --eval-episodes 10 `
  --final-episodes 50 `
  --log-dir logs/models2_v41_late_2m
```

`--teacher-data`는 생략했습니다. 이전 teacher.npz가 초반 시간대 위주라면 후반부 정책 개선을 방해할 수 있기 때문입니다. 후반부 경험 데이터가 있다면 별도 실험에서 넣을 수 있습니다.

**중요:** 모델 평균 생존이 120초 미만이면 16개 Worker가 120초를 넘기기 전에 반복 사망하고 결국 학습이 시작되지 않을 수 있습니다. 이때에는 80~110초, 100~130초, 120~165초로 점진적으로 시작 시간을 높이는 커리큘럼이 권장됩니다. 각 단계는 별도 새 `--init` 학습으로 이어가세요.

## 후반부 200만 → 누적 500만 전이 학습

이번에 생성된 Resume 파일만 이용합니다. 기존과 같은 `--envs 16`, `--batch-steps 256`, 로그 및 출력 경로를 유지합니다.

```powershell
.\.venv\Scripts\python.exe train_models2_v41_ppo_late.py `
  --resume models2/bullet_bill_ppo_v41_late_best_resume.pt `
  --out models2/bullet_bill_ppo_v41_late_best.pt `
  --total-steps 5000000 `
  --envs 16 `
  --batch-steps 256 `
  --lr 0.00001 `
  --curriculum-min-s 120 `
  --curriculum-max-s 165 `
  --warmup-max-attempts 25 `
  --eval-interval 50000 `
  --checkpoint-interval 10000 `
  --eval-episodes 10 `
  --final-episodes 50 `
  --log-dir logs/models2_v41_late_2m
```

## 시뮬레이션 / 평가

```powershell
.\.venv\Scripts\python.exe eval_models2_pt_random.py `
  --model models2/bullet_bill_ppo_v41_late_best.pt `
  --episodes 5 `
  --steps 9000 `
  --render human `
  --out logs/models2_v41_late_visual
```

시뮬레이션은 **0초부터 시작**하며, 이게 후반부 학습 이후의 전체 게임 생존력 평가입니다.

## 제한사항 / 확인해야 할 점

- 작업에 필요한 `kymnasium` 실제 게임은 이 패키지를 만든 환경에 설치되어 있지 않습니다. **모의 환경의 Windows식 spawn 처리·PPO 업데이트·Resume·워밍업 성공/실패 테스트만 검증**했습니다.
- `--total-steps`는 워밍업을 제외한 후반부 PPO 학습 스텝입니다. 후반부 진입 성공률이 낮으면 전체 실행 시간이 매우 길어질 수 있습니다.
- 한 학습 에피소드는 실제 0초에서 시작했으므로 운동 상태, 포탄 속도/위치, 무적 상태, 시드 진행은 실제 게임에 의해 구성됩니다.
- `best.pt`는 기존 최고 모델과 비교되지 않으며 **새 실험의 초기 모델부터** 비교합니다. 원본 초기 체크포인트는 유지됩니다.
- Resume 체크포인트는 모델·Adam·난수 상태 및 누적 후반부 학습 스텝을 저장하지만, 중단 순간의 게임 엔진 상태는 복원하지 않습니다.
- 학습 데이터 편향: 후반부 진입에 성공하는 시드만 학습하므로, 0~120초 생존력은 별도의 전체 에피소드 평가로 확인해야 합니다.
