# Model V4 — 1,000판 독립 강화학습 (Windows 16 Worker)

이 학습기는 **PPO v10의 코드·모델 가중치를 사용하지 않습니다.** `model_v4_net.py`의 `ModelV4`와 `obs_vector`/`env_action`을 그대로 이용합니다. 사전에 실행에 성공했던 `model_v4` 환경을 사용하며, 게임 화면은 띄우지 않습니다.

## 설치

압축파일을 `mario_pj` 프로젝트 루트에 압축 해제하세요. 기존 `model_v4_net.py`, `train_initial_50k.py`, `watch_model_v4.py`, `simulate_model_v4.py`는 덮어쓰지 않습니다. 새 파일인 `real/model_v4/train_parallel_1000ep.py`와 이 가이드, 테스트만 추가됩니다.

모델이 존재하는지 확인:

```powershell
Test-Path real/model_v4/model_v4_initial_50k.pt
```

`False`이면 5만 스텝 훈련을 먼저 완료하거나, `--init real/model_v4/model_v4_initial_untrained.pt`로 **미학습 모델에서 시작**하세요. `model_v4_initial_50k.pt`가 학습 완료 모델인지 확인하는 것이 좋습니다.

## 정식 학습: 랜덤 시드 1,000판, 16개 Worker

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe real/model_v4/train_parallel_1000ep.py `
  --init real/model_v4/model_v4_initial_50k.pt `
  --episodes 1000 `
  --envs 16 `
  --max-steps 9000 `
  --device cpu `
  --ppo-epochs 2 `
  --max-train-transitions 32768 `
  --checkpoint-every 64 `
  --checkpoint-dir real/model_v4/checkpoints `
  --out real/model_v4/model_v4_1000ep.pt
```

`--device cpu`는 16개 Worker의 게임 실행과 작은 신경망 추론에 대한 안전한 기준 설정입니다. `--device cuda`로 바꾸면 중앙 PPO 업데이트만 GPU에 올립니다. **Worker의 게임 실행과 정책 추론은 CPU**입니다. 학습 속도 향상을 보장하지 않으므로 `--device cpu`와 비교하세요.

### 속도에 대한 주의

- Worker 16개가 각각 완전한 게임 에피소드 1개를 처리합니다. 매 프레임 데이터를 보내지 않고 **에피소드 완료 후 궤적을 전달**하여 IPC 부담을 줄였습니다.
- 16개 에피소드가 끝나야 PPO 업데이트가 수행됩니다. Worker 완료 시간 차이로 마지막 Worker를 기다리는 시간이 발생할 수 있습니다.
- `--episodes 1000`은 **완료된 학습 에피소드 수**입니다. `--max-steps 9000`에서 끝난 판도 계산됩니다. 총 환경 스텝 수는 실행 결과에 따라 달라집니다.
- 수집한 전이는 모두 에피소드 통계에 포함하지만 PPO 업데이트에는 각 16판 묶음에서 최대 `32768`개를 **무작위 선택**하여 사용합니다. 더 많은 학습 데이터를 사용하려면 `--max-train-transitions 65536` 등으로 올리되 업데이트가 느려질 수 있습니다.
- 훈련 도중 실시간 렌더링·중간 평가를 수행하지 않아 추가 비용을 줄였습니다.
- 1,000판의 전체 소요 시간은 실제 생존 시간, 시뮬레이터 속도, CPU 성능에 따라 달라지므로 미리 보장할 수 없습니다.

## 16판 예비 실행

실제 게임에서 병렬 워커와 학습 루프가 작동하는지 먼저 확인하려면 다음과 같이 작은 별도 실험을 수행할 수 있습니다. 정식 1,000판 파일을 덮어쓰지 않도록 출력 경로를 나눕니다.

```powershell
.\.venv\Scripts\python.exe real/model_v4/train_parallel_1000ep.py `
  --init real/model_v4/model_v4_initial_50k.pt `
  --episodes 16 --envs 16 --max-steps 9000 `
  --device cpu --checkpoint-every 16 `
  --checkpoint-dir real/model_v4/checkpoints/pilot `
  --out real/model_v4/model_v4_pilot_16ep.pt
```

## 중단 후 재개

학습은 16판 단위로 정책을 업데이트합니다. `--checkpoint-every 64`를 지정하면 대략 64판마다 다음 파일이 저장됩니다.

`real/model_v4/checkpoints/model_v4_1000ep_resume.pt`

```powershell
.\.venv\Scripts\python.exe real/model_v4/train_parallel_1000ep.py `
  --resume real/model_v4/checkpoints/model_v4_1000ep_resume.pt `
  --episodes 1000 --envs 16 --max-steps 9000 `
  --device cpu --checkpoint-every 64 `
  --checkpoint-dir real/model_v4/checkpoints `
  --out real/model_v4/model_v4_1000ep.pt
```

Resume는 정책·Adam 상태·학습 에피소드 수·시드 계획을 복구하지만 이미 실행 중이던 16개 판은 마지막 저장된 파동부터 다시 실행됩니다. 같은 학습 설정을 사용하세요. 확장 학습에서는 `--episodes`를 늘릴 수 있으며 기존 평가 시드와 중복되지 않는 새 학습 시드를 추가합니다.

## 결과 파일

- `real/model_v4/model_v4_1000ep.pt`: **최종 추론 모델** (기존 `simulate_model_v4.py`에서 읽을 수 있음)
- `real/model_v4/checkpoints/model_v4_1000ep_resume.pt`: 옵티마이저가 포함된 중간/최종 재개 파일
- `real/model_v4/checkpoints/seed_plan.json`: 학습 및 평가 시드 분리 기록
- `real/model_v4/checkpoints/training_waves.jsonl`: 16판 단위 PPO 손실, 완료 판 수, 스텝, 경과 시간
- `real/model_v4/checkpoints/training_episodes.csv`: 판별 시드, 생존 시간, 상태

## 화면 없이 새 모델 평가 (30판)

```powershell
.\.venv\Scripts\python.exe real/model_v4/simulate_model_v4.py `
  --model real/model_v4/model_v4_1000ep.pt `
  --seed-file real/model_v4/checkpoints/seed_plan.json `
  --episodes 30 `
  --out real/model_v4/eval_results/parallel_1000ep
```

평가기는 `seed_plan.json` 안의 `eval_seeds` 30개를 사용하므로 학습 시드와 겹치지 않습니다.

## 화면으로 새 모델 확인 (5판)

```powershell
.\.venv\Scripts\python.exe real/model_v4/watch_model_v4.py `
  --model real/model_v4/model_v4_1000ep.pt `
  --seed 12345 --episodes 5 --fps 30
```

## 검증 범위

별도 프로세스 3개를 이용한 모의 환경에서 에피소드 수, 학습, 저장, Resume, 시드 비중복 테스트를 통과했습니다. 실제 BulletBill 게임에서 16 Worker 실행/성능은 사용자 PC에서 검증해야 합니다.
