# models2 PPO v10 + v4.1: 실제 PPO 추가 학습

이 패키지는 **기존 148차원 PPO v10 체크포인트의 가중치를 불러와 실제 게임 생존 보상으로 on-policy PPO 업데이트**합니다. 기존 `models2/bullet_bill_ppo_v10_best.pt`는 변경하지 않습니다. 새 `.pt`는 기존 `BulletBillAgent.load()`와 호환되며 추론할 때 v4.1을 실행할 필요가 없습니다.

> **주의:** 현재 제공된 실행 환경에는 `kymnasium`이 없어 실제 BulletBill 게임으로 학습/150초 생존 검증을 할 수 없었습니다. 가짜 환경으로 전체 PPO 루프를 포함한 4개 테스트를 통과했습니다. 150초는 목표이지 보장된 성능이 아닙니다.

## 0. 설치

이 ZIP을 `C:\Users\USER\Desktop\HONG_LB\mario_pj`에 압축 해제하세요. 이전 설치의 `agent_ppo_v10.py`, `models2/*`, `models/*`를 그대로 유지합니다. Windows의 `.venv`에 `torch`, `numpy`, `gymnasium`, `kymnasium`, `numba`가 설치되어 있어야 합니다. 데이터 수집부터 하려면 이전 **v4.1 제어기**와 `collect_models2_v41_teacher.py`, `train_models2_v41_distill.py`가 설치되어 있어야 합니다.

## 1. 기존 모델에서 바로 PPO 추가 학습 — 가장 간단한 실험

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe train_models2_v41_ppo.py `
  --init models2/bullet_bill_ppo_v10_best.pt `
  --out models2/bullet_bill_ppo_v10_v41_finetuned_best.pt `
  --total-steps 100000 `
  --envs 4 `
  --batch-steps 256 `
  --eval-interval 25000 `
  --eval-episodes 5 `
  --final-episodes 30 `
  --log-dir logs/models2_v41_ppo_pilot
```

이 명령은 기존 **v10을 이어 PPO 학습**하지만, **v4.1 행동을 추가로 가르치지는 않습니다**. v4.1을 실제로 반영하려면 2절을 실행하세요.

## 2. v4.1 전략을 반영한 학습 — 권장

(1) **무작위 교사 시드 생성** (예: 100개):

```powershell
.\.venv\Scripts\python.exe make_models2_random_seeds.py `
  --count 100 --out logs/models2_v41_teacher/train_seeds.json
```

(2) **기존 PPO + v4.1의 실제 행동 수집**:

```powershell
.\.venv\Scripts\python.exe collect_models2_v41_teacher.py `
  --model models2/bullet_bill_ppo_v10_best.pt `
  --altitude v4.1 --v41-mode explore --v41-base-mode shadow `
  --seed-file logs/models2_v41_teacher/train_seeds.json `
  --episodes 100 --steps 9000 --render none `
  --teacher-dataset logs/models2_v41_teacher/teacher.npz `
  --out logs/models2_v41_teacher/run
```

(3) **기존 모델을 초기화로 하여 v4.1 행동 증류**:

```powershell
.\.venv\Scripts\python.exe train_models2_v41_distill.py `
  --model models2/bullet_bill_ppo_v10_best.pt `
  --data logs/models2_v41_teacher/teacher.npz `
  --out models2/bullet_bill_ppo_v10_v41_distilled.pt `
  --epochs 12 --lr 0.00002 --override-weight 4 --preserve-kl 0.25
```

(4) **증류 모델에서 실제 PPO 추가 학습** (보조 교사 손실은 학습 초기에만 적용되고 감소):

```powershell
.\.venv\Scripts\python.exe train_models2_v41_ppo.py `
  --init models2/bullet_bill_ppo_v10_v41_distilled.pt `
  --teacher-data logs/models2_v41_teacher/teacher.npz `
  --out models2/bullet_bill_ppo_v10_v41_finetuned_best.pt `
  --total-steps 100000 `
  --envs 4 --batch-steps 256 --epochs 4 `
  --lr 0.00002 --bc-coef 0.02 --bc-decay 0.5 `
  --eval-interval 25000 --eval-episodes 5 --final-episodes 30 `
  --log-dir logs/models2_v41_ppo_pilot
```

초기 실험에서 개선이 확인되면 `--total-steps 1000000`, `--eval-episodes 10` 정도로 확대하세요. **오래 학습했다고 150초가 자동으로 달성되지는 않습니다.**

## 3. 평가와 직접 시뮬레이션

학습기는 실행마다 평가용 무작위 시드 5개와 최종 홀드아웃 30개를 **서로 다르게** 생성하고 `logs/.../eval_seeds.json`, `final_holdout_seeds.json`에 보존합니다. 훈련 중 새 에피소드마다 별도의 무작위 시드를 사용합니다.

성능을 비교할 때는 원본과 신규 모델을 **동일한 홀드아웃 시드**로 평가하세요:

```powershell
.\.venv\Scripts\python.exe eval_models2_pt_random.py `
  --model models2/bullet_bill_ppo_v10_best.pt `
  --seed-file logs/models2_v41_ppo_pilot/final_holdout_seeds.json `
  --episodes 30 --steps 9000 --render none `
  --out logs/models2_v41_ppo_pilot/baseline_holdout

.\.venv\Scripts\python.exe eval_models2_pt_random.py `
  --model models2/bullet_bill_ppo_v10_v41_finetuned_best.pt `
  --seed-file logs/models2_v41_ppo_pilot/final_holdout_seeds.json `
  --episodes 30 --steps 9000 --render none `
  --out logs/models2_v41_ppo_pilot/new_holdout
```

새 모델을 **눈으로 보기**:

```powershell
.\.venv\Scripts\python.exe eval_models2_pt_random.py `
  --model models2/bullet_bill_ppo_v10_v41_finetuned_best.pt `
  --seed-file logs/models2_v41_ppo_pilot/final_holdout_seeds.json `
  --episodes 1 --steps 9000 --render human `
  --out logs/models2_v41_ppo_pilot/visual
```

## 4. 결과물

- `models2/bullet_bill_ppo_v10_v41_finetuned_best.pt`: 평가 평균이 개선될 때만 갱신되는 최선 체크포인트. 처음에는 입력 모델을 복사한 기준 가중치가 저장되며, **개선이 전혀 없으면 PPO 훈련 후에도 원본과 동일할 수 있습니다.**
- `models2/bullet_bill_ppo_v10_v41_finetuned_best_last.pt`: PPO 학습이 적용된 마지막 체크포인트. 최선 체크포인트와 별도로 보존됩니다.
- `logs/models2_v41_ppo_pilot/history.json`: 학습 단계별 손실, 개입 데이터의 보조 손실 계수, 최근 생존 시간.
- `logs/models2_v41_ppo_pilot/eval_initial.csv`, `eval_step_*.csv`: 모델 선택용 검증 성능.
- `logs/models2_v41_ppo_pilot/final_holdout.csv`, `summary.json`: 모델 선택 완료 후 최종 평가.
- `logs/models2_v41_ppo_pilot/train_episodes.json`, `train_seed_history.json`: 실제 사용한 무작위 학습 시드와 생존 기록.

## 5. 모델 설계상의 주의점

- PPO는 **실제로 모델이 샘플링한** 점프/이동을 게임에서 실행하고 그 정책 로그 확률로 학습합니다. v4.1 대체 행동을 게임에 직접 끼워 넣어 수집한 경험을 PPO on-policy 표본이라고 잘못 처리하지 않습니다.
- 기존 PPO v10 보상 (`SURVIVE`, `DEATH`, potential shaping, 점프 비용, 별 조기 획득 비용)을 계승합니다.
- v4.1 교사 데이터는 옵션이며 보조 행동 모방 손실로만 사용합니다. 학습 초기 `--bc-coef`에서 시작하여 `--bc-decay` 비율 동안 선형으로 0까지 감소합니다.
- **정책 증류만으로 v4.1의 Closed-loop 알고리즘 전체가 신경망 내부에 구현되는 것은 아닙니다.** 실제 게임에서 PPO로 추가 최적화해야 합니다.
- 이번 패키지에는 환경 내부 상태 복원이나 60초부터 시작하는 진짜 커리큘럼이 들어 있지 않습니다. 해당 기능은 게임 엔진의 상태 설정 인터페이스 검증 후 추가해야 합니다.
- 180초 이상 포탄 생성 예측은 기존 v10 코드에서 아직 검증되지 않았습니다. 이것은 성공적으로 150초 이상 생존할 때 별도로 확인해야 할 핵심 영역입니다.

## 6. 오프라인 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest test_models2_v41_ppo -v
```

실제 게임 없이 검증 가능한 체크포인트 생성/복원, 보조 손실 미분, 시드 분리, 가짜 환경 전체 학습 루프를 점검합니다.
