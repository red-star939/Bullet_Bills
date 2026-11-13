# BulletBill v4.1 Late PPO: CUDA 시험 버전

## 설치

ZIP에서 `train_models2_v41_ppo_late_gpu.py`와 `train_models2_v41_ppo.py`를 `mario_pj` 폴더에 복사합니다. 기존 `train_models2_v41_ppo_late.py`는 수정하거나 덮어쓰지 않습니다. 기존 `agent_ppo_v10.py`, `gymnasium`, `kymnasium`이 필요합니다.

## 0. CUDA 확인

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch=',torch.__version__, 'cuda=',torch.cuda.is_available(), 'gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

CUDA가 False이면 CUDA 지원 PyTorch와 NVIDIA 드라이버 상태를 확인하세요. GPU가 없는 장치에서는 CUDA 실행이 불가능합니다.

## 1. 먼저 기존 최고 모델로 GPU 예비 학습 20,480스텝

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe train_models2_v41_ppo_late_gpu.py `
  --init models2/bullet_bill_ppo_v41_late_60_75_best.pt `
  --out models2/bullet_bill_ppo_v41_gpu_trial_best.pt `
  --log-dir logs/models2_v41_gpu_trial `
  --device cuda `
  --inference-device cpu `
  --total-steps 20480 `
  --envs 16 `
  --batch-steps 256 `
  --lr 0.00001 `
  --curriculum-min-s 60 `
  --curriculum-max-s 75 `
  --warmup-max-attempts 100 `
  --eval-interval 20480 `
  --checkpoint-interval 4096 `
  --eval-episodes 10 `
  --final-episodes 10 `
  --log-every 1
```

가장 중요한 출력은 `DEVICE train=cuda ...`, `UPDATE ... collect=... h2d=... ppo=...` 입니다. 16 환경 시뮬레이션과 특징 추출은 여전히 CPU Worker가 수행하고, 중앙 PPO 모델 학습 및 전체 롤아웃 배치는 GPU에서 처리합니다. `--inference-device cpu`는 환경 진행 중 매 프레임의 PCIe 왕복을 피하기 위한 기본 권장값입니다. GPU 사용량은 역전파 때만 짧게 증가할 수 있습니다.

### GPU 추론 비교

같은 명령에서 `--inference-device train`으로 바꾸고 **출력 및 로그 경로도 다르게** 지정하세요. 이 경우 매 프레임 `[16,148]` 관측이 GPU로 전송되고 행동이 CPU Worker로 돌아옵니다. 이 비교는 GPU 추론이 실제로 유리한지 확인하기 위함입니다.

### CPU 비교

같은 명령에서 `--device cpu --inference-device cpu`로 바꾸고 별도의 `--out`, `--log-dir`를 사용하세요.

## 2. 기존 진행 중인 60~75초 학습을 GPU로 Resume

**GPU 예비 학습이 정상 종료된 후에만 권장합니다.** 기존 학습이 실행 중인 동안 같은 체크포인트를 사용해서 두 실험을 동시에 수행하지 마세요. 기존 `--out`은 Resume 메타데이터 검증을 위해 동일하게 유지해야 합니다.

```powershell
.\.venv\Scripts\python.exe train_models2_v41_ppo_late_gpu.py `
  --resume models2/bullet_bill_ppo_v41_late_60_75_best_resume.pt `
  --out models2/bullet_bill_ppo_v41_late_60_75_best.pt `
  --total-steps 2000000 `
  --envs 16 `
  --batch-steps 256 `
  --lr 0.00001 `
  --device cuda `
  --inference-device cpu `
  --curriculum-min-s 60 `
  --curriculum-max-s 75 `
  --warmup-max-attempts 100 `
  --eval-interval 50000 `
  --checkpoint-interval 10000 `
  --eval-episodes 10 `
  --final-episodes 50 `
  --log-dir logs/models2_v41_late_60_75
```

기존 `.pt` 정책 형식 및 기존 Resume 체크포인트 형식을 읽습니다. GPU에서 학습한 뒤에도 제출용 `.pt`는 CPU에서도 로드 가능합니다. 환경 내부 상태 자체를 저장하는 것은 아니므로 Resume 시 새 워밍업을 수행합니다.

## 3. 로그 해석

- `collect`: 환경 진행, 워밍업, 워커 통신을 포함한 롤아웃 수집 시간
- `h2d`: 수집된 PPO 롤아웃 배치를 학습 장치로 전송하는 시간. CUDA에서는 전송 완료까지 동기화 측정
- `ppo`: PPO 학습 epoch 전체 시간. CUDA에서는 동기화 포함
- `wall`: 최근 로그부터 경과 시간(평가·워밍업 등의 비용 포함)
- `WARMUP steps_excluded`: 학습 전이가 아닌 게임 진행 스텝. CUDA로 바꾸더라도 대부분의 워밍업 비용은 남습니다.

정확한 CPU/GPU 비교는 비슷한 시드·같은 환경 수·배치·모델로 여러 업데이트를 실행하고 `collect/h2d/ppo` 및 전체 steps/sec를 함께 비교해야 합니다.

## 4. 구현 한계

CPU에서 계산한 관측 특징은 반드시 CPU에 먼저 존재합니다. `--inference-device train`은 16개 관측을 한 번에 GPU로 전송하므로 작은 전송 수를 줄이지만 **CPU 경유를 제거하지는 않습니다**. 완전한 GPU 환경은 시뮬레이터의 물리, 충돌, 포탄 생성, 난수, 특징 추출까지 별도 GPU 백엔드로 재구현해야 합니다.

현재 작성·검증된 부분은 CPU 경로 및 GPU 코드의 문법입니다. 실제 NVIDIA CUDA 장치에서의 실행 성능은 사용자 PC에서 시험해야 합니다.
