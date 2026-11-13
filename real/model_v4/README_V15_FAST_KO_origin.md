# PPO V15 고속 추가 학습 — V14 10M 이어서

이 패키지는 `agent_ppo_v15.py`의 **178D 입력·15-action 구조, DAgger+PPO, 600초 학습 제한, 60~120초 시간 이동 커리큘럼, 엔트로피 감쇠, 점프 비용 제거**를 이전 병렬 V14 학습기(CPU Worker + CUDA PPO)에 통합한 것입니다.

## 설치

프로젝트 루트(`C:\Users\USER\Desktop\HONG_LB\mario_pj`)에서 ZIP 압축을 푸세요. `real/model_v4` 아래 다음 파일이 추가됩니다.

- `agent_ppo_v15_model_v4.py` — 업로드한 V15 소스 복사본
- `train_v15_fast.py` — 고속 병렬 V15 추가 학습기

`v/model_v14_10m.pt`는 그대로 보존하며, 새로운 `.pt`는 `v/` 아래 저장됩니다.

## 호환성 확인

```powershell
Test-Path v/model_v14_10m.pt
.\.venv\Scripts\python.exe -c "import torch; x=torch.load('v/model_v14_10m.pt',map_location='cpu',weights_only=False); print(x['obs_dim'],x.get('hidden',256),x['model']['pair_head.weight'].shape)"
```

기대 출력: `178 256 torch.Size([15, 256])`.

## 빠른 연결 시험 — 추가 8,192스텝

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe real/model_v4/train_v15_fast.py `
  --init v/model_v14_10m.pt `
  --steps 8192 --envs 16 --batch-steps 256 --device cuda `
  --epochs 2 --minibatch 1024 --lr 0.0001 `
  --p-late 0.30 --eval-every 0 `
  --checkpoint-dir v/checkpoints/test_v15 `
  --out v/model_v15_test_8k.pt
```

## V15 본 학습 — 추가 약 800만 스텝

V14 10M + V15 8,003,584 **추가** = 약 1,800만 누적 스텝입니다. `--init`은 가중치만 승계하며 Adam 상태를 승계하지 않습니다.

```powershell
.\.venv\Scripts\python.exe real/model_v4/train_v15_fast.py `
  --init v/model_v14_10m.pt `
  --steps 8003584 --envs 16 --batch-steps 256 `
  --device cuda --epochs 4 --minibatch 1024 --lr 0.0001 `
  --p-late 0.30 --t-lo 60 --t-hi 120 `
  --imit0 0.05 --imit1 0.05 --ent0 0.01 --ent1 0.003 `
  --eval-every 1000000 --eval-episodes 30 --eval-seed 20000 `
  --checkpoint-every 16384 --checkpoint-dir v/checkpoints `
  --best-out v/model_v15_from_v14_best.pt `
  --out v/model_v15_from_v14_8m.pt
```

실행 중 `UPDATE`에 정상 시작/늦은 시작 평균을 별도로 표시합니다. `EVAL`은 **원래 180초 게임을 처음부터** 평가합니다. 평가 시점마다 `v/checkpoints/model_v15_step{누적스텝}.pt` 모델 스냅샷이 저장됩니다. `--best-out`은 이번 V15 실행에서 평가한 최고 모델입니다.

원본 V15의 교사 계수는 새로 시작하면 1.0→0.05이지만, 이미 훈련된 V14 10M를 보호하기 위해 여기서는 **기본 0.05 고정**입니다. 원본과 동일한 감쇠가 필요하면 `--imit0 1.0 --imit1 0.05 --imit-steps 5000000`을 지정할 수 있습니다.

## 중단 후 재개

주기적으로 저장되는 `v/checkpoints/model_v15_fast_resume.pt`에는 Adam 상태·학습 일정 진행량이 들어갑니다. **`--steps`에는 남은 추가 스텝 수**를 지정합니다. 예: 본 실험 8,003,584 중 2,002,944스텝을 마쳤다면 남은 6,000,640스텝을 `--steps`에 지정합니다. 나머지 옵션은 원래 실험과 같게 유지합니다.

```powershell
.\.venv\Scripts\python.exe real/model_v4/train_v15_fast.py `
  --resume v/checkpoints/model_v15_fast_resume.pt `
  --steps 6000640 --envs 16 --batch-steps 256 --device cuda `
  --epochs 4 --minibatch 1024 --lr 0.0001 `
  --p-late 0.30 --imit0 0.05 --imit1 0.05 `
  --ent0 0.01 --ent1 0.003 `
  --eval-every 1000000 --eval-episodes 30 --eval-seed 20000 `
  --checkpoint-dir v/checkpoints `
  --out v/model_v15_from_v14_8m.pt
```

학습 환경의 진행 중 게임 상태는 저장되지 않으며 Worker 재시작 시 새 판부터 시작합니다. 재개 시 일부 랜덤 시퀀스가 달라질 수 있습니다.

## 화면 없는 평가

기존 V14 `watch_v14.py`와 체크포인트 형식(178D, 15 action)이 호환되므로 아래와 같이 검증할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe real/model_v4/watch_v14.py `
  --model v/model_v15_from_v14_8m.pt --episodes 100 --render none `
  --out v/eval_results/v15_100ep
```

## 중요한 실험적 한계

- **V15의 `late_start()`는 60~120초로 게임 시계를 이동하지만, 그 시점에 실제 있어야 할 포탄 30개의 분포를 재구성하지 않습니다.** 시간이 앞선 상태에서 포탄이 비어 있게 시작하고 이후 점차 포탄이 생성됩니다. 따라서 본 구현의 '포화 구간 집중'은 현실적인 포화 상태로의 완전한 초기화가 아닙니다. `--p-late 0` 대비 성능을 반드시 비교하세요.
- CUDA는 중앙 PPO에 사용되며 Numba 위험 예측은 CPU Worker에서 실행합니다.
- 실제 BulletBill 게임 + GPU 장시간 학습은 이 환경에서 실행/벤치마크하지 못했습니다. Python 구문 및 합성 배치 PPO 테스트만 확인했습니다.
