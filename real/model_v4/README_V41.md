# Independent Model V4.1: discrete motion + risk features

이 패키지는 PPO v10 및 기존 Model V4 가중치와 **독립적**입니다. `model_v4_500k.pt`를 덮어쓰지 않습니다.

## 구성

- `model_v41_net.py`: 64차원 물리·위험 상태(마리오 위치/속도, 발판 경계, 가까운 포탄 상대 위치/속도/TTC, 별)와 수평 이산 행동 3개(-1, 0, +1), 점프 5개.
- `model_v41_untrained.pt`: 새 구조의 **0스텝 초기 가중치**, 학습된 생존 모델이 아님.
- `train_v41.py`: 16 Worker PPO, 체크포인트는 `real/model_v4/checkpoints/model_v41_resume.pt`, 최종 모델은 `real/model_v4/model_v41_50k.pt`.
- `watch_v41.py`: 렌더링 있는 게임 실행.
- `simulate_v41.py`: 화면 없는 평가.
- `diagnose_v41.py`: 시드 5개 x deterministic/stochastic 비교; 실제 X 이동량과 좌/중/우 선택 횟수.
- `test_v41.py`: 모의 환경 기본 동작 테스트.

**환경 요구사항:** 기존 프로젝트의 `gymnasium`, `kymnasium`, `pygame`, `numpy`, `torch`. 실제 게임 통합 실행은 사용자 PC에서 확인.

## 1. 5만 스텝 예비 PPO 학습 (PowerShell / mario_pj 프로젝트 루트)

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe real/model_v4/train_v41.py `
  --init real/model_v4/model_v41_untrained.pt `
  --total-steps 50000 `
  --envs 16 `
  --batch-steps 256 `
  --device cpu `
  --ppo-epochs 2 `
  --checkpoint-every 16384 `
  --checkpoint-dir real/model_v4/checkpoints `
  --out real/model_v4/model_v41_50k.pt
```

추가 학습은 위 명령의 `--init`을 `--resume real/model_v4/checkpoints/model_v41_resume.pt`로 바꾸고 `--total-steps`를 누적 목표로 지정합니다. `--init`과 `--resume`은 동시에 지정하지 마세요.

## 2. 화면 표시 실행

```powershell
.\.venv\Scripts\python.exe real/model_v4/watch_v41.py `
  --model real/model_v4/model_v41_50k.pt `
  --episodes 5 --fps 30 `
  --out real/model_v4/eval_results/v41_visible
```

## 3. 화면 없는 평가

```powershell
.\.venv\Scripts\python.exe real/model_v4/simulate_v41.py `
  --model real/model_v4/model_v41_50k.pt `
  --episodes 30 `
  --out real/model_v4/eval_results/v41_headless
```

## 4. 좌/우 실제 움직임 검증 (권장)

```powershell
.\.venv\Scripts\python.exe real/model_v4/diagnose_v41.py `
  --model real/model_v4/model_v41_50k.pt `
  --episodes 5 `
  --out real/model_v4/eval_results/v41_action_audit
```

`summary.json`에서 `move_counts`의 좌/정지/우 비율과 `mean_abs_dx`를 확인합니다. deterministic 0이면 현재 정책이 움직이지 않는 것입니다. 학습 전 `model_v41_untrained.pt`는 확률적인 `--stochastic`으로 인터페이스 확인 가능하나 좋은 회피 전략을 의미하지 않습니다.

## 5. 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s real/model_v4 -p test_v41.py -v
```

## 제약 및 유의사항

- 포탄의 위치와 속도는 실제 `obs['bullet_bills']` 형식 `[left, top, right, bottom, vx]`를 전제로 합니다.
- 점프 입력은 `obs['mario']` 형식 `[left,top,right,bottom,vx,vy,inv,air,...]`를 전제로 합니다. 실제 스키마 변경 시 인코더를 수정해야 합니다.
- 위험 인식형 특징은 **관측 인코딩 개선**입니다. 보상은 기존 생존 0.1/사망 -10 기준이며, 회피 성능 보장은 아닙니다.
- **중요: 50만 스텝 V4 가중치는 3방향 이산 수평 정책과 아키텍처가 달라 직접 로드할 수 없습니다.** 새로운 0스텝 모델에서 학습합니다.
- 검증용 5개 가짜 환경 테스트를 수행했지만 실제 BulletBill 패키지가 이 실행 환경에 설치되어 있지 않아 실게임 시뮬레이션 결과는 제공할 수 없습니다.
