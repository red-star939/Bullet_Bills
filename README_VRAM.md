# PPO v4.1: CUDA VRAM-resident rollout 실험

## 정확한 범위
이 버전은 **중앙 학습기가 소유한 데이터**(정책/가치 모델, Adam 상태, BC 교사 텐서, PPO rollout S/J/M/LP/V/R/D, GAE, return, 미니배치)를 CUDA 메모리에 둡니다. 매 업데이트마다 CPU NumPy rollout 전체를 GPU에 다시 올리는 과정을 제거했습니다. 정책 추론도 CUDA에서 실행합니다.

**절대로 바뀌지 않은 것:** BulletBill `gymnasium` 게임 엔진, Worker 16개, Worker 내부 `FeatureExtractor`, 환경 상태, 워밍업, multiprocessing Pipe는 CPU 메모리에서 실행됩니다. `env.step`은 CPU 코드이므로 게임 전체를 VRAM으로 옮길 수 없습니다. 워커 행동(점프·이동)은 GPU→CPU로, 관측값·보상·종료 플래그는 CPU→GPU로 매 스텝 전달됩니다. 이 비용이 `collect` 시간에 포함됩니다. GPU 사용 메모리가 6GB에 가까워지지 않아도 정상입니다. VRAM 사용률을 높이기 위해 불필요한 텐서를 생성하지 않습니다.

## 설치
기존 `mario_pj` 루트 폴더에 ZIP 내부의 다음 두 파일을 복사하세요.

- `train_models2_v41_ppo_late_vram.py` (신규 파일; 기존 `train_models2_v41_ppo_late.py`, `train_models2_v41_ppo_late_gpu.py`를 덮어쓰지 않음)
- `train_models2_v41_ppo.py` (기존 PPO helper; 없다면 같이 복사. 원본과 충돌이 우려되면 기존 파일 백업)

게임 환경 및 `agent_ppo_v10.py`는 기존 프로젝트에 설치되어 있어야 합니다.

## GPU 20,480스텝 예비 실행 (PowerShell)

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe train_models2_v41_ppo_late_vram.py `
  --init models2/bullet_bill_ppo_v41_late_60_75_best.pt `
  --out models2/bullet_bill_ppo_v41_vram_trial_best.pt `
  --log-dir logs/models2_v41_vram_trial `
  --device cuda `
  --inference-device train `
  --total-steps 20480 `
  --envs 16 `
  --batch-steps 256 `
  --epochs 4 `
  --minibatches 4 `
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

새 로그 형식:

```
DEVICE train=cuda:... rollout_inference=train rollout_storage=cuda gae=cuda ...
UPDATE ... collect=...s h2d_update=0.000s ppo=...s vram=<allocated>/<reserved>MiB peak=<peak>MiB
```

`h2d_update=0`은 **PPO 업데이트 직전 전체 NumPy 배치 전송이 없어졌다는 뜻**이며, 매 스텝 관측값 전달이 사라진다는 뜻은 아닙니다. `collect`에는 GPU 추론 왕복과 CPU 게임 실행, 워밍업이 포함됩니다.

`best.pt`, `best_last.pt`는 시뮬레이션용 모델 가중치입니다. `best_resume.pt`는 학습 재개용으로 CPU 직렬화되며 정상입니다. 이전 GPU 실험의 `_resume.pt`를 쓰고 싶다면 `--resume`과 같은 `--out` 경로와 동일한 Worker/커리큘럼 설정이 필요합니다.

## 학습 시간·VRAM 해석

148차원 × 16환경 × 256스텝 float32 관측값은 약 2.31MiB입니다. 이 정도 모델/배치에서 VRAM은 수백 MB 이하일 수 있습니다. RTX 4050 6GB 중 적게 사용한다고 구현 오류는 아닙니다. 다른 GPU 작업에 VRAM을 남기는 것이 오히려 안정적입니다.

## 검증 범위

Windows RTX 4050 실기기 검증은 미실시. CPU fake environment에서 3개 spawn Worker, PPO rollout, 체크포인트 저장, Resume를 테스트했습니다. 실제 GPU에서는 오류 로그, `UPDATE`의 `collect`, `ppo`, `vram`, `peak`를 확인하세요.
