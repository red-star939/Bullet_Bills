# 독립 PPO V14 모델 — real/model_v4

새 V14 모델은 업로드된 `agent_ppo_v14.py`에서 파생되며 기존 V4/V4.1 및 PPO v10 **가중치는 사용하지 않습니다**.

- 상태: 178차원, 마리오·포탄 특징과 미래 계획별 15개 행동의 위험 점수 포함
- 행동: 점프 5단계 × 이동 3방향 = 15개 이산 조합
- 학습: PPO + GAE + DAgger 방식의 교사 정답 집합 손실
- 네트워크: 독립 Actor/Critic, 은닉층 256 × 2

## 파일

- `model_v14_initial.pt`: **0스텝, 무작위 초기화**, 즉시 로드 가능한 실물 체크포인트
- `agent_ppo_v14_model_v4.py`: 기존 V14 로직의 독립 복사본. 출력 경로 변경, 학습 에피소드 랜덤 시드 적용
- `train_v14.py`: 학습 실행 진입점
- `watch_v14.py`: 화면 보이는/없는 시뮬레이션
- `checkpoints/`: 최종이 아닌 마지막 모델과 옵티마이저 스냅샷
- `model_v14_best.pt`, `model_v14_final.pt`: 실제 학습 후 `real/model_v4` 루트에 생성

## 준비

1. ZIP을 `mario_pj` 루트에 풀어 `real/model_v4/...`에 배치
2. 기존 가상환경에 `torch`, `numpy`, `numba`, `gymnasium`, `kymnasium` 설치 확인
3. 학습은 가능하면 Numba 설치 상태에서 진행. V14의 계획 점수 계산이 매우 무겁습니다.

## 먼저 미학습 모델 화면 확인

```powershell
.\.venv\Scripts\python.exe real/model_v4/watch_v14.py `
  --model real/model_v4/model_v14_initial.pt `
  --episodes 2 --render human --out real/model_v4/eval_results/v14_initial
```

## 신규 모델 처음부터 학습 — 약 5만 환경 스텝

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe real/model_v4/train_v14.py `
  --init real/model_v4/model_v14_initial.pt `
  --steps 50000 --envs 16 --eval-interval 16384 --eval-episodes 10
```

**주의**: 원본 V14 학습 루프는 16환경 × 256 = 4096스텝 단위로 실행합니다. `--steps 50000`을 지정하면 **49,152스텝**(12 업데이트)을 실제 수행합니다. 정확한 49,152스텝을 원하면 `--steps 49152`; 5만 이상을 원하면 `--steps 53248`으로 지정하세요.

원본 V14 학습은 **한 Python 프로세스에서 16개 환경을 순차 호출**하며 Windows 멀티프로세스 16-Worker 버전이 아닙니다. 또한 DAgger 교사 점수 계산 비용이 커서 짧은 완료시간을 보장할 수 없습니다. 중간 체크포인트는 평가 시점마다 생성하지만 **optimizer 자동 Resume는 구현되어 있지 않습니다**.

## 학습 후 시뮬레이션

```powershell
.\.venv\Scripts\python.exe real/model_v4/watch_v14.py `
  --model real/model_v4/model_v14_best.pt `
  --episodes 5 --render human --out real/model_v4/eval_results/v14_visual
```

```powershell
.\.venv\Scripts\python.exe real/model_v4/watch_v14.py `
  --model real/model_v4/model_v14_best.pt `
  --episodes 30 --render none --out real/model_v4/eval_results/v14_headless
```

평가 스크립트는 CSV와 JSON에 에피소드별 시드와 생존 시간을 저장합니다. 이전 V4 평가 스크립트는 네트워크 형식이 다르므로 사용하지 마세요.

## 검증 범위

- 현재 제공된 미학습 `.pt`는 실제로 생성했고 동일 아키텍처로 재로드했습니다.
- Python AST 구문 검사 및 텐서 크기 검사를 수행했습니다.
- 이 실행 환경에는 `gymnasium`/`kymnasium`이 없어 실제 BulletBill 게임 및 GPU 학습은 수행하지 못했습니다.
