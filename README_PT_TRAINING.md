# models2 — PPO v10 + v4.1 정책 증류로 새 `.pt` 만들기

## 목적

입력: `models2/bullet_bill_ppo_v10_best.pt` (사용자 PC에 있는 원본)

출력: `models2/bullet_bill_ppo_v10_v41_distilled.pt` (새 가중치)

v4.1 제어기를 `.pt`에 코드째로 넣는 것이 아닙니다. 기존 PPO v10의 148차원 관측 특성으로부터 v4.1 개입 행동을 **모방 학습**시켜 별도 모델을 만듭니다. 결과물은 `agent_ppo_v10.BulletBillAgent.load()`로 읽을 수 있는 독립 실행 체크포인트입니다. 이는 강화학습(PPO 리워드 업데이트)이 아니라 **행동 복제(behavior cloning)/정책 증류**입니다. 성능 개선 및 150초 평균 생존은 보장하지 않습니다.

## 필요한 기존 파일

프로젝트 루트 `C:\Users\USER\Desktop\HONG_LB\mario_pj`에 `agent_ppo_v10.py`, `models/` 폴더의 기존 가이드/추적 코드, `models2/altitude_v2.py`~`altitude_v4_1.py`, `models2/closed_loop_planner.py`, `models2/spawn_aware_risk.py`, `models2/intervention_audit.py`, `map_data/object_tiles.csv`, 설치된 `.venv` 환경이 필요합니다. 기존 `models2` v4.1 실험이 실행된다면 대부분 준비돼 있습니다.

이 패키지는 **증분 설치**이며 원본 `.pt`와 원본 구현 파일은 덮어쓰지 않습니다. 압축을 프로젝트 루트에 푸세요.

## PowerShell 명령: 무작위 시드 학습부터 모델 생성까지

프로젝트 루트에서 실행하세요.

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

# 1. 학습용 무작위 시드 생성 (예: 20개)
.\.venv\Scripts\python.exe make_models2_random_seeds.py `
  --count 20 --out logs/models2_distill/train_seeds.json

# 2. 실제 게임으로 v4.1 교사 데이터 수집
.\.venv\Scripts\python.exe collect_models2_v41_teacher.py `
  --model models2/bullet_bill_ppo_v10_best.pt `
  --altitude v4.1 --v41-mode explore --v41-base-mode shadow `
  --seed-file logs/models2_distill/train_seeds.json `
  --episodes 20 --steps 9000 --render none `
  --teacher-dataset logs/models2_distill/teacher.npz `
  --out logs/models2_distill/teacher_run

# 3. 가중치 추가 학습하여 새 .pt 생성
.\.venv\Scripts\python.exe train_models2_v41_distill.py `
  --model models2/bullet_bill_ppo_v10_best.pt `
  --data logs/models2_distill/teacher.npz `
  --out models2/bullet_bill_ppo_v10_v41_distilled.pt `
  --epochs 12 --lr 0.00002 --override-weight 8 --preserve-kl 0.25

# 4. 학습 데이터와 겹치지 않는 무작위 평가 시드 10개 생성
.\.venv\Scripts\python.exe make_models2_random_seeds.py `
  --count 10 --exclude logs/models2_distill/train_seeds.json `
  --out logs/models2_distill/test_seeds.json

# 5. 신형 PT 단독 평가 (외부 v4.1 제어기를 호출하지 않음)
.\.venv\Scripts\python.exe eval_models2_pt_random.py `
  --model models2/bullet_bill_ppo_v10_v41_distilled.pt `
  --seed-file logs/models2_distill/test_seeds.json `
  --episodes 10 --steps 9000 --render none `
  --out logs/models2_distill/new_pt_eval

# 6. 동일 평가 시드에서 원본 PT 비교 (공정한 비교를 위한 페어링)
.\.venv\Scripts\python.exe eval_models2_pt_random.py `
  --model models2/bullet_bill_ppo_v10_best.pt `
  --seed-file logs/models2_distill/test_seeds.json `
  --episodes 10 --steps 9000 --render none `
  --out logs/models2_distill/base_pt_eval
```

## 새 모델을 눈으로 확인하기

```powershell
.\.venv\Scripts\python.exe eval_models2_pt_random.py `
  --model models2/bullet_bill_ppo_v10_v41_distilled.pt `
  --seed-file logs/models2_distill/test_seeds.json `
  --episodes 1 --steps 9000 --render human `
  --out logs/models2_distill/visual
```

## 출력

- `models2/bullet_bill_ppo_v10_v41_distilled.pt`: 새 PPO v10 형식 체크포인트
- `models2/bullet_bill_ppo_v10_v41_distilled.training.json`: 학습 손실·변경 텐서 수
- `logs/models2_distill/teacher.npz`: 수집한 관측, 행동, 개입 여부, 에피소드 구분
- `logs/models2_distill/*_seeds.json`: 재현 가능한 난수 시드 목록
- `logs/models2_distill/new_pt_eval/episodes.csv`: 새 PT 평가
- `logs/models2_distill/base_pt_eval/episodes.csv`: 원본 PT 평가

## 유의점

1. `--v41-mode explore`라도 실제 v4.1 행동 개입이 매우 적을 수 있습니다. `TEACHER_DATASET ... altitude_overrides=N`을 확인하세요. **N=0이면 v4.1 고유 개선을 배울 신호가 없습니다.**
2. v4.1이 개입하지 않은 상태도 원래 행동을 보존하도록 함께 학습합니다. 원래 PPO의 연속형 출력에는 분포가 존재하지만, 여기서는 실제 실행 행동을 대상으로 감독 학습합니다.
3. `--out`에 기존 파일이 있으면 덮어쓰지 않고 중단합니다. 재학습 시 새 파일명을 사용하세요.
4. `--seed-file`은 `--episodes` 수 이상이어야 합니다. 같은 테스트 시드를 원본과 신규 모델에 사용해야 공정하게 비교할 수 있습니다. 학습 시드와 테스트 시드는 겹치지 않습니다.
5. 평균 150초 목표에는 10개를 넘는 별도의 홀드아웃 시드, 생존 분포, 최악 성능까지 확인해야 합니다. 결과가 악화되면 새 `.pt`를 채택하지 마세요.
6. 외부 v4.1과 **완전히 동일한 의사결정**을 원한다면 단독 `.pt`가 아니라 모델+제어기 배포가 필요합니다. 이 산출물은 가중치에 행동 패턴을 학습한 근사 모델입니다.
