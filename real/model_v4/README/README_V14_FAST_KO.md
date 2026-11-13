# PPO V14 CPU 병렬 수집 + CUDA 업데이트

원본 `agent_ppo_v14_model_v4.py`를 그대로 사용합니다. 기존 `train_v14.py`는 변경하지 않고 새 `train_v14_fast.py`만 추가합니다. 기존 V4/V4.1 가중치는 사용하지 않습니다.

## 적용 구조

- 16개의 **Windows multiprocessing spawn** CPU Worker: 각 Worker는 `FeatureExtractor`의 178차원 입력, Numba 기반 3.2초 계획 예측 및 15개 교사 점수를 계산합니다.
- 각 Worker는 256 환경 스텝씩 수집하며, 중앙 학습기에 4096-transition 배치를 전달합니다.
- GPU(CUDA)가 있으면 중앙 **PPO + DAgger** 신경망 업데이트를 GPU에서 수행합니다. Worker의 행동 추론은 CPU에서 수행합니다.
- PPO 보상: 생존 + 위험도 potential shaping + 사망 패널티 + 점프 비용 + 별 조기 사용 패널티. 원본 V14와 같은 요소입니다.
- 모방 정답: 특징 인덱스 148~162의 실제 포탄 기준 조합 점수 최대 집합. 원본과 같은 손실입니다.
- 출력: `real/model_v4/model_v14_fast_final.pt`; 재개 체크포인트: `real/model_v4/checkpoints/model_v14_fast_resume.pt`.

## 설치

`mario_pj` 루트에서 ZIP을 압축 해제합니다. *기존 파일 보존을 위해 별도 파일만 추가해도 됩니다.*

```powershell
cd C:\Users\USER\Desktop\HONG_LB\mario_pj
Expand-Archive -Path "$env:USERPROFILE\Downloads\model_v14_parallel_cuda.zip" -DestinationPath . -Force
Test-Path real/model_v4/train_v14_fast.py
Test-Path real/model_v4/model_v14_initial.pt
```

PyTorch CUDA 버전 확인:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`False`인 경우 GPU가 활성화되지 않았으므로 `--device cuda`는 오류를 냅니다. 해당 PC의 CUDA 대응 PyTorch 환경을 먼저 확인하거나 `--device cpu`를 사용하세요.

## 먼저 8192스텝 속도 측정

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe real/model_v4/train_v14_fast.py `
  --init real/model_v4/model_v14_initial.pt `
  --steps 8192 --envs 16 --batch-steps 256 `
  --device cuda --epochs 2 `
  --checkpoint-every 4096 `
  --out real/model_v4/model_v14_fast_8k.pt
```

**주의**: 각 Worker의 Numba 최초 JIT 컴파일 때문에 첫 업데이트는 오래 걸릴 수 있습니다. 이후 `UPDATE`의 `steps_per_s`와 원본 학습 속도를 비교해야 실제 가속 효과를 판단할 수 있습니다.

## 새 모델로 53,248스텝 학습

```powershell
.\.venv\Scripts\python.exe real/model_v4/train_v14_fast.py `
  --init real/model_v4/model_v14_initial.pt `
  --steps 53248 --envs 16 --batch-steps 256 `
  --device cuda --epochs 4 --minibatch 1024 `
  --checkpoint-every 16384 `
  --out real/model_v4/model_v14_fast_final.pt
```

16프로세스에서 메모리 문제가 있거나 오히려 느리다면 `--envs 8`로 변경해 비교해 보세요.

## 이어서 학습

```powershell
.\.venv\Scripts\python.exe real/model_v4/train_v14_fast.py `
  --resume real/model_v4/checkpoints/model_v14_fast_resume.pt `
  --steps 53248 --envs 16 --batch-steps 256 --device cuda `
  --out real/model_v4/model_v14_fast_106k.pt
```

`--steps`는 **이번 실행에서 추가 학습할 스텝 수**입니다. Resume 시 Worker가 실행하던 에피소드의 중간 상태와 난수 스트림은 복원되지 않고 새로 생성됩니다. 모델·Adam 상태 및 누적 스텝은 유지됩니다.

## 화면에서 실제 움직임 확인

```powershell
.\.venv\Scripts\python.exe real/model_v4/watch_v14.py `
  --model real/model_v4/model_v14_fast_final.pt `
  --episodes 5 --render human `
  --out real/model_v4/eval_results/v14_fast_visual
```

30판 무화면 평가:

```powershell
.\.venv\Scripts\python.exe real/model_v4/watch_v14.py `
  --model real/model_v4/model_v14_fast_final.pt `
  --episodes 30 --render none `
  --out real/model_v4/eval_results/v14_fast_headless
```

`--eval-every 16384 --eval-episodes 10` 설정 시 학습 중 정해진 시드로 정책을 평가해 `model_v14_fast_best.pt`도 저장합니다. 평가 시간은 학습 처리량에 포함되므로 기본값은 `--eval-every 0`입니다.

## 테스트 범위

- 178D 특징/15행동 PPO 손실, DAgger 손실, GAE, 체크포인트 저장/재로드를 합성 배치로 검증했습니다.
- 현재 제작 환경에는 `gymnasium`과 `kymnasium`이 없고 CUDA 장치가 없어 **실제 BulletBill 환경 병렬 실행 또는 CUDA 속도 향상을 실측하지 못했습니다**.
- 원본 V14의 `FeatureExtractor`/물리 예측 정확성은 이번 패치에서 바꾸지 않았습니다.
